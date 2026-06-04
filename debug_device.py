#!/usr/bin/env python3
"""Debug script to check GPU device availability."""

import sys
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

log = logging.getLogger("debug_device")

log.info("=" * 60)
log.info("GPU Device Detection Debug")
log.info("=" * 60)

# Check torch
try:
    import torch
    log.info(f"✓ torch installed: {torch.__version__}")
    log.info(f"  Location: {torch.__file__}")
except ImportError as e:
    log.error(f"✗ torch not available: {e}")
    sys.exit(1)

# Check CUDA
log.info("")
log.info("CUDA Support:")
if hasattr(torch, "cuda"):
    log.info("  ✓ torch.cuda module exists")
    try:
        cuda_available = torch.cuda.is_available()
        log.info(f"  torch.cuda.is_available() = {cuda_available}")
        if cuda_available:
            log.info(f"  torch.cuda.device_count() = {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                log.info(f"    Device {i}: {torch.cuda.get_device_name(i)}")
    except Exception as e:
        log.error(f"  Error querying CUDA: {e}")
else:
    log.warning("  ✗ torch.cuda module not available")

# Check XPU
log.info("")
log.info("Intel XPU Support:")
if hasattr(torch, "xpu"):
    log.info("  ✓ torch.xpu module exists")
    try:
        xpu_available = torch.xpu.is_available()
        log.info(f"  torch.xpu.is_available() = {xpu_available}")
        if xpu_available:
            log.info(f"  torch.xpu.device_count() = {torch.xpu.device_count()}")
            for i in range(torch.xpu.device_count()):
                log.info(f"    Device {i}: {torch.xpu.get_device_name(i)}")
    except Exception as e:
        log.error(f"  Error querying XPU: {e}")
else:
    log.warning("  ✗ torch.xpu module not available")

# Check intel_extension_for_pytorch
log.info("")
log.info("Intel Extension for PyTorch:")
try:
    import intel_extension_for_pytorch as ipex
    log.info(f"  ✓ intel_extension_for_pytorch installed: {ipex.__version__}")
except ImportError:
    log.warning("  ✗ intel_extension_for_pytorch not installed")
except Exception as e:
    log.error(f"  Error importing ipex: {e}")

# Check device selection
log.info("")
log.info("Selected Device (from device.py):")
sys.path.insert(0, "app")
try:
    from device import get_torch_device
    device = get_torch_device()
    log.info(f"  Device: {device}")
except Exception as e:
    log.error(f"  Error getting device: {e}")

log.info("")
log.info("=" * 60)
