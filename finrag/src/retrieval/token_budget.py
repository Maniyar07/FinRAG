"""Token counting for request-scoped evidence budgets."""

import logging
from functools import lru_cache

import tiktoken

from src.config import LLM_MODEL

@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding | None:
    try:
        return tiktoken.encoding_for_model(LLM_MODEL)
    except Exception as error:
        logging.getLogger(__name__).warning(
            "Exact token counting unavailable (%s); using conservative UTF-8 byte counts. "
            "Preload the tokenizer vocabulary before evaluating or serving requests.", type(error).__name__,
        )
        return None


def count_tokens(text: str) -> int:
    """Count tokens exactly when available, else use a safe byte upper bound.

    Tiktoken downloads its model vocabulary on first use. A restricted runtime
    may have no network/cache, so UTF-8 bytes provide a conservative upper bound
    instead of making retrieval fail or silently undercounting its budget.
    """
    encoding = _encoding()
    return (
        len(encoding.encode(text, disallowed_special=()))
        if encoding is not None
        else len(text.encode("utf-8"))
    )


def token_counter_name() -> str:
    """Describe the active counter so retrieval traces explain budget decisions."""
    encoding = _encoding()
    return (
        f"tiktoken:{encoding.name}"
        if encoding is not None
        else "utf8_bytes_conservative_fallback"
    )
