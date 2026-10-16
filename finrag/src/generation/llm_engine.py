from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    OPENAI_API_KEY,
    OPENAI_MAX_RETRIES,
    OPENAI_REQUEST_TIMEOUT_SECONDS,
    validate_runtime_config,
)
from src.telemetry import UsageCallbackHandler
from src.request_runtime import CancellationCallback


def get_llm_engine(
    *,
    model: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    max_tokens: int | None = None,
    parallel_tool_calls: bool | None = None,
) -> ChatOpenAI:
    request_timeout = (
        OPENAI_REQUEST_TIMEOUT_SECONDS if timeout is None else timeout
    )
    retry_count = OPENAI_MAX_RETRIES if max_retries is None else max_retries
    if request_timeout <= 0:
        raise ValueError("timeout must be positive.")
    if retry_count < 0:
        raise ValueError("max_retries cannot be negative.")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive when provided.")
    selected_model = (model or LLM_MODEL).strip()
    if not selected_model:
        raise ValueError("LLM model name cannot be empty.")

    validate_runtime_config(require_openai=LLM_BASE_URL is None)
    return ChatOpenAI(
        # Never send the embedding provider's OpenAI key to a model server.
        api_key=(LLM_API_KEY or "local") if LLM_BASE_URL else OPENAI_API_KEY,
        base_url=LLM_BASE_URL,
        model=selected_model,
        temperature=0.0,
        timeout=request_timeout,
        max_retries=retry_count,
        max_tokens=max_tokens,
        callbacks=[UsageCallbackHandler(selected_model), CancellationCallback()],
        model_kwargs=(
            {"parallel_tool_calls": parallel_tool_calls}
            if parallel_tool_calls is not None
            else {}
        ),
        streaming=False,
    )
