# ==========================================
# 1. BUILD STAGE
# ==========================================
FROM python:3.12-slim AS builder

# Intel GPU (XPU) support flag
ARG XPU=false

# Installiamo i pacchetti direttamente nel sistema (senza venv) usando --user
ENV PIP_USER=true
ENV PATH="/root/.local/bin:$PATH"

# Install PyTorch with Intel GPU support or CPU-only
RUN if [ "$XPU" = "true" ]; then \
      pip install --no-cache-dir \
        torch==2.8.0 \
        torchvision \
        intel_extension_for_pytorch \
        openvino \
        --extra-index-url https://download.pytorch.org/whl/xpu; \
    else \
      pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu; \
    fi

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir transformers onnx onnxruntime-openvino \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir opencv-python-headless \
    && pip uninstall -y triton 2>/dev/null || true

# ==========================================
# 2. RUNTIME STAGE
# ==========================================
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Installiamo Python 3.12 e i driver Intel
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3-pip \
    gpg \
    wget \
    ca-certificates \
    && wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble client" > /etc/apt/sources.list.d/intel-gpu.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    intel-opencl-icd \
    intel-level-zero-gpu \
    libze1 \
    && rm -rf /var/lib/apt/lists/*

# Creiamo i link simbolici per far rispondere Python correttamente
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

# Copiamo le librerie Python installate dal builder direttamente nella cartella locale di Ubuntu
COPY --from=builder /root/.local /root/.local

# Config वडिलाiamo il PATH in modo che Python veda i pacchetti copiati
ENV PATH="/root/.local/bin:$PATH"
ENV PYTHONPATH="/root/.local/lib/python3.12/site-packages"

WORKDIR /app

# Usiamo python -c per non aver bisogno di file di script esterni
RUN python -c "from ultralytics import YOLO; model = YOLO('yolov8s.pt'); model.export(format='openvino', half=True)"

RUN mkdir -p clip_vit_b_16_openvino_model && \
    python - <<'PY'
from pathlib import Path
import torch
from transformers import CLIPModel

model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16')
model.eval()

class ClipEncoder(torch.nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.vision = clip.vision_model
        self.projection = clip.visual_projection

    def forward(self, pixel_values):
        outputs = self.vision(pixel_values)
        pooled = outputs.pooler_output
        return self.projection(pooled)

wrapper = ClipEncoder(model)
output_path = Path('clip_vit_b_16_openvino_model') / 'model.onnx'
dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    wrapper,
    dummy,
    output_path,
    opset_version=16,
    input_names=['pixel_values'],
    output_names=['image_embeds'],
    dynamic_axes={
        'pixel_values': {0: 'batch'},
        'image_embeds': {0: 'batch'},
    },
)
PY

VOLUME ["/data"]
EXPOSE 8000

COPY VERSION .
COPY app/ .
COPY debug_device.py .

CMD ["python", "main.py"]