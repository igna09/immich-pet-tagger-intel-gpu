# ==========================================
# 1. BUILD STAGE
# ==========================================
FROM python:3.12-slim AS builder

ARG CUDA=false
ARG CUDA_LEGACY=false
ARG ROCM=false
ARG XPU=false

# Installiamo i pacchetti direttamente nel sistema (senza venv) usando --user
ENV PIP_USER=true
ENV PATH="/root/.local/bin:$PATH"

# RUN if [ "$CUDA" = "true" ] && [ "$CUDA_LEGACY" = "true" ]; then \
#       pip install --no-cache-dir torch==2.7.0+cu126 torchvision==0.22.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126; \
#     elif [ "$CUDA" = "true" ]; then \
#       pip install --no-cache-dir torch==2.7.0+cu128 torchvision==0.22.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128; \
#     elif [ "$ROCM" = "true" ]; then \
#       pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/rocm6.3; \
#     elif [ "$XPU" = "true" ]; then \
RUN if [ "$XPU" = "true" ]; then \
      # pip install --no-cache-dir --upgrade pip && \
      # pip install --no-cache-dir \
      #   torch==2.6.0 torchvision==0.21.0 \
      #   --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/; \
      pip install --no-cache-dir \
        torch==2.8.0 \
        torchvision \
        intel_extension_for_pytorch \
        --extra-index-url https://download.pytorch.org/whl/xpu; \
    else \
      pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu; \
    fi

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
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
VOLUME ["/data"]
EXPOSE 8000

COPY VERSION .
COPY app/ .
COPY debug_device.py .

CMD ["python", "main.py"]