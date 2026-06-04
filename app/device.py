"""Device helpers for CPU/GPU selection.

Supports CUDA, Intel XPU, and CPU fallback.
"""

import torch


def get_torch_device() -> str:
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def create_torch_stream(device: str):
    if device == "cuda" and hasattr(torch.cuda, "Stream"):
        return torch.cuda.Stream()
    if device == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "Stream"):
        return torch.xpu.Stream()
    return None
