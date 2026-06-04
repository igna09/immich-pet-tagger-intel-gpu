import logging
import os
import queue
import threading
import time
import numpy as np
import openvino as ov
from PIL import Image

log = logging.getLogger("detector")

YOLO_INPUT_SIZE = int(os.environ.get("YOLO_INPUT_SIZE", 640))
OPENVINO_NUM_REQUESTS = int(os.environ.get("OPENVINO_NUM_REQUESTS", 4)) 

ANIMAL_CLASS_IDS = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}

# --- [RIPRISTINO STATS PER POLLER] ---
# Aggiungiamo i componenti richiesti da poller.py per evitare l'AttributeError
_yolo_stats_lock = threading.Lock()
_yolo_total_ms = 0.0
_yolo_count = 0

class _YoloTask:
    __slots__ = ("array", "event", "result", "start_time")
    def __init__(self, array: np.ndarray):
        self.array = array
        self.event = threading.Event()
        self.result: list | None = None
        self.start_time = time.perf_counter() # Tracciamo il tempo iniziale per task

# Coda per i thread chiamanti
_task_queue: queue.Queue[_YoloTask] = queue.Queue()
_initialized = False
_init_lock = threading.Lock()

def _letterbox(img: Image.Image, new_shape=(640, 640)):
    """Pre-processing standard per YOLO mantenendo il ratio"""
    small = img.resize(new_shape, Image.BILINEAR)
    arr = np.array(small, dtype=np.uint8)
    return arr

def _ov_worker_loop():
    global _yolo_total_ms, _yolo_count
    core = ov.Core()
    
    device = "GPU" if "GPU" in core.available_devices else "CPU"
    log.info(f"OpenVINO loading model on {device}...")
    
    model_path = "yolov8n_openvino_model/yolov8n.xml"
    
    if not os.path.exists(model_path):
        log.error(f"Modello OpenVINO non trovato in {model_path}. Esportalo prima!")
        return

    model = core.read_model(model_path)
    compiled_model = core.compile_model(model, device)
    
    infer_queue = ov.AsyncInferQueue(compiled_model, OPENVINO_NUM_REQUESTS)

    # Callback di fine inferenza
    def completion_callback(request, task: _YoloTask):
        global _yolo_total_ms, _yolo_count
        try:
            output_tensor = request.get_output_tensor(0)
            output_data = output_tensor.data[0] 
            
            boxes = []
            output_data = output_data.T 
            
            for pred in output_data:
                scores = pred[4:]
                cls = np.argmax(scores)
                conf = scores[cls]
                
                if conf > 0.25 and cls in ANIMAL_CLASS_IDS:
                    cx, cy, w, h = pred[0:4]
                    x1 = (cx - w / 2) / YOLO_INPUT_SIZE
                    y1 = (cy - h / 2) / YOLO_INPUT_SIZE
                    x2 = (cx + w / 2) / YOLO_INPUT_SIZE
                    y2 = (cy + h / 2) / YOLO_INPUT_SIZE
                    boxes.append((float(conf), float(x1), float(y1), float(x2), float(y2)))
            
            boxes.sort(reverse=True, key=lambda x: x[0])
            task.result = [(x[1], x[2], x[3], x[4]) for x in boxes]
            
            # Aggiorniamo le metriche usando il lock atteso da poller.py
            duration_ms = (time.perf_counter() - task.start_time) * 1000.0
            with _yolo_stats_lock:
                _yolo_total_ms += duration_ms
                _yolo_count += 1

        except Exception as e:
            log.error(f"Errore nel post-processing OpenVINO: {e}", exc_info=True)
            task.result = []
        finally:
            task.event.set()

    infer_queue.set_callback(completion_callback)

    while True:
        task = _task_queue.get()
        
        input_data = np.expand_dims(task.array, axis=0).astype(np.float32) / 255.0
        input_data = input_data.transpose(0, 3, 1, 2) 
        
        infer_queue.start_async({0: input_data}, userdata=task)

def _ensure_worker():
    global _initialized
    with _init_lock:
        if not _initialized:
            t = threading.Thread(target=_ov_worker_loop, daemon=True, name="openvino-manager")
            t.start()
            _initialized = True

def detect_animals(img: Image.Image) -> list[tuple[float, float, float, float]]:
    """Invia l'immagine alla pipeline asincrona OpenVINO e attende il risultato"""
    _ensure_worker()
    
    arr = _letterbox(img, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))
    
    task = _YoloTask(arr)
    _task_queue.put(task)
    
    task.event.wait()
    return task.result