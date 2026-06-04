# Build stage: install all Python deps then strip inference-irrelevant packages.
# Using a venv so the runtime stage snapshots only the final cleaned-up filesystem.
FROM python:3.12-slim AS builder

# GPU support:
#   NVIDIA (default, Turing+ incl. Blackwell):       set CUDA=true
#   NVIDIA legacy (Maxwell/Pascal/Volta, no Blackwell): set CUDA=true and CUDA_LEGACY=true
#   AMD:    set ROCM=true  (requires ROCm drivers on the host)
#   Intel integrated GPU: set XPU=true (requires Intel GPU drivers and /dev/dri access)
#   None:   leave all false (CPU-only, slow but works)
ARG CUDA=false
ARG CUDA_LEGACY=false
ARG ROCM=false
ARG XPU=false

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install torch first so it gets its own cached layer.
# cu128 wheels (default) drop sm_50/60/70 to fit PyPI size limits; cu126 wheels keep
# Maxwell through Hopper but lack Blackwell (sm_100/120). See pytorch/pytorch#145544.
RUN if [ "$CUDA" = "true" ] && [ "$CUDA_LEGACY" = "true" ]; then \
      pip install --no-cache-dir \
        torch==2.7.0+cu126 \
        torchvision==0.22.0+cu126 \
        --extra-index-url https://download.pytorch.org/whl/cu126; \
    elif [ "$CUDA" = "true" ]; then \
      pip install --no-cache-dir \
        torch==2.7.0+cu128 \
        torchvision==0.22.0+cu128 \
        --extra-index-url https://download.pytorch.org/whl/cu128; \
    elif [ "$ROCM" = "true" ]; then \
      pip install --no-cache-dir \
        torch==2.7.0 \
        torchvision==0.22.0 \
        --index-url https://download.pytorch.org/whl/rocm6.3; \
    elif [ "$XPU" = "true" ]; then \
      pip install --no-cache-dir \
        torch==2.7.0+xpu \
        torchvision \
        intel_extension_for_pytorch \
        --extra-index-url https://download.pytorch.org/whl/xpu; \
    else \
      pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 \
        --index-url https://download.pytorch.org/whl/cpu; \
    fi

COPY requirements.txt .
# All nvidia-*-cu12 packages except triton are hard-required by torch at import time:
# torch.__init__.py preloads them via ctypes before loading torch._C, and libtorch_cuda.so
# has them in its NEEDED list. Triton is only used by torch.compile(), not inference.
# opencv-python (GUI variant, installed by ultralytics) is replaced by headless;
# explicit uninstall removes the orphaned opencv_python.libs directory.
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
