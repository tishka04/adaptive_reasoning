"""
Shared LLM cache — loads the model once and shares it across components.

Both GoalDecomposer and StrategyGenerator can use the same model/tokenizer
instance instead of loading weights every time a new game starts.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

_cached_model: Optional[Any] = None
_cached_tokenizer: Optional[Any] = None
_cached_model_name: Optional[str] = None


def get_shared_llm(
    model_name: str, device: str = "cpu"
) -> Tuple[Optional[Any], Optional[Any]]:
    """Return (model, tokenizer), loading only on first call."""
    global _cached_model, _cached_tokenizer, _cached_model_name

    if _cached_model is not None and _cached_model_name == model_name:
        return _cached_model, _cached_tokenizer

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading shared LLM: {model_name} (one-time)")
        _cached_tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        _cached_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device,
            trust_remote_code=True,
        )
        _cached_model.eval()
        _cached_model_name = model_name
        logger.info("Shared LLM loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load LLM: {e}")
        _cached_model = None
        _cached_tokenizer = None

    return _cached_model, _cached_tokenizer


def clear_cache() -> None:
    """Free the cached model (e.g. to reclaim VRAM)."""
    global _cached_model, _cached_tokenizer, _cached_model_name
    _cached_model = None
    _cached_tokenizer = None
    _cached_model_name = None
