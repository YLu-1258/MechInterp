"""Model registry for loading and caching TransformerLens models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from transformer_lens import HookedTransformer

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model mapping: API name → TransformerLens identifier
# ---------------------------------------------------------------------------

MODEL_MAP: dict[str, str] = {
    "gpt2-small": "gpt2-small",
    "gpt2-medium": "gpt2-medium",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B-Base",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
}

# ---------------------------------------------------------------------------
# In-memory model cache
# ---------------------------------------------------------------------------

_model_cache: dict[str, HookedTransformer] = {}


def _get_device() -> str:
    """Detect the best available device."""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("CUDA device detected: %s", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Apple MPS device detected")
    else:
        device = "cpu"
        logger.info("No GPU detected, using CPU")
    return device


def get_supported_models() -> list[str]:
    """Return the list of supported API model names."""
    return list(MODEL_MAP.keys())


def get_model(model_name: str) -> HookedTransformer:
    """Load a TransformerLens model by API name, with lazy loading and caching.

    Args:
        model_name: One of the supported API model names (e.g. 'gpt2-small').

    Returns:
        A loaded HookedTransformer instance ready for inference.

    Raises:
        ValueError: If the model name is not in MODEL_MAP.
        RuntimeError: If the model fails to load (network error, OOM, etc.).
    """
    if model_name not in MODEL_MAP:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Supported models: {get_supported_models()}"
        )

    # Return cached model if available
    if model_name in _model_cache:
        logger.info("Using cached model: %s", model_name)
        return _model_cache[model_name]

    # Load the model
    tl_name = MODEL_MAP[model_name]
    device = _get_device()
    logger.info("Loading model '%s' (TransformerLens ID: '%s') on %s...", model_name, tl_name, device)

    try:
        model = HookedTransformer.from_pretrained(tl_name, device=device)
        model.eval()
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError(
            f"GPU out of memory loading '{model_name}'. "
            f"Try a smaller model (e.g. 'gpt2-small') or free GPU memory."
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model '{model_name}': {e}"
        )

    _model_cache[model_name] = model
    logger.info("Model '%s' loaded and cached successfully", model_name)
    return model
