import logging
import os
import queue
import threading
import numpy as np
import openvino as ov
from PIL import Image

log = logging.getLogger("detector")

YOLO_INPUT_SIZE = int(os.environ.get("YOLO_INPUT_SIZE", 640))
# Con OpenVINO definiamo quanti slot paralleli dare all'hardware (es. 2 o 4)
OPENVINO_NUM_REQUESTS = int(os.environ.get("OPENVINO_NUM_REQUESTS", 4)) 

ANIMAL_CLASS_IDS = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}

class _YoloTask:
    __slots__ = ("array", "event", "result")
    def __init__(self, array: np.ndarray):
        self.array = array
        self.event = threading.Event()
        self.result: list | None = None

# Coda leggera per i thread chiamanti
_task_queue: queue.Queue[_YoloTask] = queue.Queue()
_initialized = False
_init_lock = threading.Lock()

def _letterbox(img: Image.Image, new_shape=(640, 640)):
    """Pre-processing standard per YOLO mantenendo il ratio (o resize diretto se preferisci)"""
    # Per semplicità usiamo il resize diretto che usavi tu, ma convertito in un formato digeribile da OpenVINO
    small = img.resize(new_shape, Image.BILINEAR)
    arr = np.array(small, dtype=np.uint8)
    # OpenVINO di solito si aspetta [B, C, H, W], se esporti il modello in quel modo.
    # Se esporti il modello con Ultralytics, accetta l'input NHWC o NCHW a seconda di come viene compilato.
    return arr

def _ov_worker_loop():
    core = ov.Core()
    
    # Seleziona il device (es. "GPU" o "CPU"). Se hai una iGPU Intel usa "GPU"
    device = "GPU" if "GPU" in core.available_devices else "CPU"
    log.info(f"OpenVINO loading model on {device}...")
    
    # Assicurati di aver esportato il modello con: yolo export model=yolov8n.pt format=openvino
    model_path = "yolov8n_openvino_model/yolov8n.xml"
    
    if not os.path.exists(model_path):
        log.error(f"Modello OpenVINO non trovato in {model_path}. Esportalo prima!")
        return

    model = core.read_model(model_path)
    compiled_model = core.compile_model(model, device)
    
    # Creiamo una coda di inferenza asincrona nativa
    infer_queue = ov.AsyncInferQueue(compiled_model, OPENVINO_NUM_REQUESTS)

    # Callback eseguita automaticamente quando la GPU/CPU termina un'inferenza
    def completion_callback(request, task: _YoloTask):
        try:
            # Recupera i risultati (output_tensor 0 sono i boxes/scores)
            output_tensor = request.get_output_tensor(0)
            output_data = output_tensor.data[0] # Shape tipica YOLOv8: [84, 8400]
            
            # --- Parsing dei Box (Post-processing standard YOLOv8) ---
            # Nota: YOLOv8 sputa fuori [cx, cy, w, h, class_0, class_1, ...]
            # Di seguito una logica semplificata per estrarre le classi degli animali
            boxes = []
            
            # Trasponiamo per avere i detection index sulla prima dimensione -> [8400, 84]
            output_data = output_data.T 
            
            for pred in output_data:
                scores = pred[4:]
                cls = np.argmax(scores)
                conf = scores[cls]
                
                if conf > 0.25 and cls in ANIMAL_CLASS_IDS:
                    cx, cy, w, h = pred[0:4]
                    # Conversione in coordinate normalizzate x1, y1, x2, y2
                    # (Dividendo per 640 se l'output non è già normalizzato)
                    x1 = (cx - w / 2) / YOLO_INPUT_SIZE
                    y1 = (cy - h / 2) / YOLO_INPUT_SIZE
                    x2 = (cx + w / 2) / YOLO_INPUT_SIZE
                    y2 = (cy + h / 2) / YOLO_INPUT_SIZE
                    boxes.append((float(conf), float(x1), float(y1), float(x2), float(y2)))
            
            boxes.sort(reverse=True, key=lambda x: x[0])
            task.result = [(x[1], x[2], x[3], x[4]) for x in boxes]
        except Exception as e:
            log.error(f"Errore nel post-processing OpenVINO: {e}", exc_info=True)
            task.result = []
        finally:
            task.event.set()

    # Assegna la callback alla coda di inferenza
    infer_queue.set_callback(completion_callback)

    while True:
        task = _task_queue.get()
        
        # Prepara l'input per OpenVINO. 
        # Di solito YOLOv8 OpenVINO si aspetta float32 normalizzato [1, 3, 640, 640]
        input_data = np.expand_dims(task.array, axis=0).astype(np.float32) / 255.0
        input_data = input_data.transpose(0, 3, 1, 2) # da NHWC a NCHW
        
        # Invia in modo non bloccante alla coda di OpenVINO. 
        # Se tutti gli slot (OPENVINO_NUM_REQUESTS) sono pieni, questo metodo blocca 
        # finché uno slot non si libera, garantendo contropressione (backpressure).
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
    
    # Pre-processing leggero nel thread chiamante
    arr = _letterbox(img, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))
    
    task = _YoloTask(arr)
    _task_queue.put(task)
    
    # Attende il completamento della callback di OpenVINO
    task.event.wait()
    return task.result