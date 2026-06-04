"""Animal detector using YOLOv8n. Batched inference via queue, N parallel worker threads.
Pre-processing (PIL→tensor) happens in caller threads; batch threads only run the GPU kernel."""

import logging
import os
import queue
import threading

import numpy as np
import torch
from PIL import Image

from device import get_torch_device

import intel_extension_for_pytorch as ipex

log = logging.getLogger("detector")

YOLO_BATCH_SIZE = int(os.environ.get("YOLO_BATCH_SIZE", 32))
YOLO_WORKERS = int(os.environ.get("GPU_WORKERS", 2))
YOLO_INPUT_SIZE = int(os.environ.get("YOLO_INPUT_SIZE", 640))

ANIMAL_CLASS_IDS = {
    14,  # bird
    15,  # cat
    16,  # dog
    17,  # horse
    18,  # sheep
    19,  # cow
    20,  # elephant
    21,  # bear
    22,  # zebra
    23,  # giraffe
}


class _YoloReq:
    __slots__ = ("tensor", "event", "result")
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
        self.event = threading.Event()
        self.result: list | None = None


_yolo_queue: queue.Queue[_YoloReq] = queue.Queue()
_yolo_worker_threads: list[threading.Thread] = []
_yolo_worker_lock = threading.Lock()

yolo_batch_total = 0
yolo_batch_count = 0
_yolo_stats_lock = threading.Lock()


def _yolo_batch_loop(worker_id: int) -> None:
    global yolo_batch_total, yolo_batch_count
    from ultralytics import YOLO
    device = get_torch_device()
    log.info(f"YOLO worker {worker_id} loading on {device}...")
    try:
        model = YOLO("yolov8n.pt")
        log.debug(f"YOLO worker {worker_id} model loaded, moving to {device}")
        model.to(device)
        if device == "xpu":
            model = ipex.optimize(model.model, dtype=torch.float32)
        log.info(f"YOLO worker {worker_id} ready on {device}")
    except Exception as e:
        log.error(f"YOLO worker {worker_id} failed to load on {device}: {e}", exc_info=True)
        log.warning(f"YOLO worker {worker_id} falling back to CPU")
        model = YOLO("yolov8n.pt")
        model.to("cpu")

    while True:
        first = _yolo_queue.get()
        batch = [first]
        try:
            while len(batch) < YOLO_BATCH_SIZE:
                batch.append(_yolo_queue.get_nowait())
        except queue.Empty:
            pass

        with _yolo_stats_lock:
            yolo_batch_total += len(batch)
            yolo_batch_count += 1

        try:
            # Tensors are already preprocessed by caller threads: B×C×H×W, float32, [0,1], RGB.
            # Ultralytics skips PIL/numpy conversion when given a tensor directly.
            stacked = torch.stack([req.tensor for req in batch])
            model_device = getattr(next(model.parameters(), None), "device", device)
            log.debug(
                f"YOLO worker {worker_id} infer batch={len(batch)} stacked_shape={stacked.shape} stacked_device={stacked.device} model_device={model_device}"
            )
            results_list = model(stacked, verbose=False, imgsz=YOLO_INPUT_SIZE)
            for req, result in zip(batch, results_list):
                boxes = []
                for box in result.boxes:
                    cls = int(box.cls[0])
                    if cls not in ANIMAL_CLASS_IDS:
                        continue
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxyn[0].tolist()
                    boxes.append((conf, x1, y1, x2, y2))
                boxes.sort(reverse=True)
                req.result = [(x1, y1, x2, y2) for _, x1, y1, x2, y2 in boxes]
                req.event.set()
        except Exception as e:
            extra = []
            if hasattr(torch, "cuda"):
                try:
                    extra.append(f"cuda_available={torch.cuda.is_available()}")
                    if torch.cuda.is_available():
                        extra.append(f"cuda_device_count={torch.cuda.device_count()}")
                        current = torch.cuda.current_device()
                        extra.append(f"cuda_current_device={current}")
                        extra.append(
                            f"cuda_allocated={torch.cuda.memory_allocated(current)}"
                        )
                        extra.append(
                            f"cuda_reserved={torch.cuda.memory_reserved(current)}"
                        )
                except Exception as inner:
                    extra.append(f"cuda_status_error={inner}")
            if hasattr(torch, "xpu"):
                try:
                    extra.append(f"xpu_available={torch.xpu.is_available()}")
                except Exception as inner:
                    extra.append(f"xpu_status_error={inner}")
            log.error(
                f"YOLO worker {worker_id} batch error device={device} batch_size={len(batch)} "
                f"stacked_shape={stacked.shape if 'stacked' in locals() else 'unknown'} "
                f"model_device={model_device if 'model_device' in locals() else 'unknown'} "
                f"queue_size={_yolo_queue.qsize()} {' '.join(extra)}: {e}",
                exc_info=True,
            )
            for req in batch:
                req.result = []
                req.event.set()


def _ensure_yolo_workers() -> None:
    with _yolo_worker_lock:
        alive = [t for t in _yolo_worker_threads if t.is_alive()]
        for i in range(len(alive), YOLO_WORKERS):
            t = threading.Thread(target=_yolo_batch_loop, args=(i,), daemon=True, name=f"yolo-batch-{i}")
            t.start()
            _yolo_worker_threads.append(t)


def detect_animals(img: Image.Image) -> list[tuple[float, float, float, float]]:
    """Returns (x1, y1, x2, y2) normalized bboxes for detected animals, sorted by confidence."""
    _ensure_yolo_workers()
    # Pre-process in caller's thread (parallel across all scan workers).
    small = img.resize((YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), Image.BILINEAR)
    arr = np.array(small, dtype=np.float32) / 255.0  # H×W×3, RGB, [0,1]
    tensor = torch.from_numpy(arr.transpose(2, 0, 1))  # C×H×W
    req = _YoloReq(tensor)
    _yolo_queue.put(req)
    req.event.wait()
    return req.result
