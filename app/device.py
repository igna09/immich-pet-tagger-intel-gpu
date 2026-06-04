"""Device helpers for CPU/GPU selection.

Supports CUDA, Intel XPU, and CPU fallback.
"""

import logging
import torch

log = logging.getLogger("device")


def get_torch_device() -> str:
    log.debug(f"Torch location: {torch.__file__}")
    log.debug(f"Torch version: {torch.__version__}")
    log.debug(f"Checking device availability...")
    log.debug(f"  torch.cuda exists: {hasattr(torch, 'cuda')}")
    
    if hasattr(torch, "cuda"):
        try:
            cuda_available = torch.cuda.is_available()
            log.debug(f"  torch.cuda.is_available(): {cuda_available}")
            if cuda_available:
                log.info("Using CUDA device")
                return "cuda"
        except Exception as e:
            log.debug(f"  Error checking CUDA: {e}")
    
    log.debug(f"  torch.xpu exists: {hasattr(torch, 'xpu')}")
    if hasattr(torch, "xpu"):
        try:
            xpu_available = torch.xpu.is_available()
            log.debug(f"  torch.xpu.is_available(): {xpu_available}")
            if xpu_available:
                log.info("Using Intel XPU device")
                return "xpu"
        except Exception as e:
            log.debug(f"  Error checking xpu availability: {e}")
    
    log.info("Using CPU device (no GPU detected)")
    return "cpu"


def create_torch_stream(device: str):
    if device == "cuda" and hasattr(torch.cuda, "Stream"):
        return torch.cuda.Stream()
    if device == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "Stream"):
        return torch.xpu.Stream()
    return None
