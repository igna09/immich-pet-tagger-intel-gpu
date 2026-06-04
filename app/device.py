"""Device helpers for CPU/GPU selection.

Uses OpenVINO to detect and manage available hardware accelerators.
"""

import logging
import openvino as ov

log = logging.getLogger("device")

# Cache the detected device
_detected_device = None
_available_devices = None


def _detect_devices():
    """Detect available OpenVINO devices."""
    global _available_devices
    if _available_devices is not None:
        return
    try:
        core = ov.Core()
        _available_devices = core.available_devices
        log.debug(f"Available OpenVINO devices: {_available_devices}")
    except Exception as e:
        log.warning(f"Error detecting OpenVINO devices: {e}")
        _available_devices = []


def get_openvino_device() -> str:
    """
    Get the best available OpenVINO device.
    Prefers GPU > XPU > CPU in that order.
    """
    global _detected_device
    if _detected_device is not None:
        return _detected_device
    
    _detect_devices()
    
    log.debug(f"Checking device availability...")
    log.debug(f"  Available devices: {_available_devices}")
    
    # Check for discrete/integrated GPU
    for device in _available_devices:
        if "GPU" in device:
            _detected_device = "GPU"
            log.info(f"Using OpenVINO GPU device: {device}")
            return _detected_device
    
    # Fallback to CPU
    _detected_device = "CPU"
    log.info("Using OpenVINO CPU device")
    return _detected_device


# Keep get_torch_device as alias for backward compatibility
def get_torch_device() -> str:
    """Deprecated: Use get_openvino_device() instead. Kept for backward compatibility."""
    device = get_openvino_device()
    # Map OpenVINO device names to common names for logging
    if device == "GPU":
        return "gpu"
    return "cpu"
