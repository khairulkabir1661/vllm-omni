"""
Utilities for resolving real models to their tiny model equivalents.
"""

import logging

from tests.model_tests.diffusion.model_settings import MODEL_SETTINGS
from vllm_omni.diffusion.data import resolve_model_class_name

logger = logging.getLogger(__name__)


def resolve_tiny_model_path(model: str) -> str:
    """Given a real model name/path, resolve it to a tiny model path.

    Tries two strategies:
      1. Resolve via diffusion pipeline class name (for diffusion models).
      2. Match by HF model ID against MODEL_SETTINGS (for omni/tts models).

    Returns the original model path if no tiny builder exists yet."""
    # Strategy 1: diffusion pipeline class name
    pipeline_class = resolve_model_class_name(model)
    if pipeline_class is not None:
        test_opts = MODEL_SETTINGS.get(pipeline_class)
        if test_opts is not None:
            return test_opts.builder()

    # Strategy 2: match by HF model ID
    for _name, opts in MODEL_SETTINGS.items():
        if opts.model == model:
            return opts.builder()

    logger.warning(
        "No tiny model builder for model: %s. Using original model.", model
    )
    return model
