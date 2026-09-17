"""Pluggable feature-extraction backends.

A TensorRT backend for the Jetson Orin Nano slots in here later without any
change to callers.
"""

from backends.base import FeatureExtractor, l2_normalise, pool_tokens
from backends.torch_backend import MODEL_SOURCES, TorchDinoBackend

AVAILABLE_MODELS = tuple(sorted(MODEL_SOURCES))

__all__ = [
    "AVAILABLE_MODELS",
    "FeatureExtractor",
    "TorchDinoBackend",
    "get_backend",
    "l2_normalise",
    "pool_tokens",
]


def get_backend(name, device=None, use_fp16=None):
    """Build the feature extractor for `name`, defaulting from config."""
    import config

    device = config.DEVICE if device is None else device
    use_fp16 = config.USE_FP16 if use_fp16 is None else use_fp16

    return TorchDinoBackend(name, device=device, use_fp16=use_fp16)
