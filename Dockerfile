# ==========================================
# 1. BUILD STAGE (Resta invariato)
# ==========================================
FROM python:3.12-slim AS builder

ARG CUDA=false
ARG CUDA_LEGACY=false
ARG ROCM=false
ARG XPU=false

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN if [ "$CUDA" = "true" ] && [ "$CUDA_LEGACY" = "true" ]; then \
      pip install --no-cache-dir torch==2.7.0+cu126 torchvision==0.22.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126; \
    elif [ "$CUDA" = "true" ]; then \
      pip install --no-cache-dir torch==2.7.0+cu128 torchvision==0.22.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128; \
    elif [ "$ROCM" = "true" ]; then \
      pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/rocm6.3; \
    elif [ "$XPU" = "true" ]; then \
      pip install --no-cache-dir --upgrade pip && \
      pip install --no-cache-dir \
        torch==2.6.0 torchvision==0.21.0 \
        --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/; \
    else \
      pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu; \
    fi

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir opencv-python-headless \
    && pip uninstall -y triton 2>/dev/null || true

# ==========================================
# 2. RUNTIME STAGE (Modificato con Ubuntu 24.04)
# ==========================================
FROM ubuntu:24.04

# Evita prompt interattivi durante l'installazione
ENV DEBIAN_FRONTEND=noninteractive

# Installiamo Python 3.12 e i driver Intel ufficiali per Ubuntu Noble
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
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

# Copiamo il virtualenv dal builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
VOLUME ["/data"]
EXPOSE 8000

COPY VERSION .
COPY app/ .
COPY debug_device.py .

CMD ["python3", "main.py"]