from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.config import LLM_MODEL, OPENAI_API_KEY, validate_runtime_config
from src.telemetry import UsageCallbackHandler


def get_llm_engine(
    *,
    model: str | None = None,
    timeout: float = 60.0,
    max_retries: int = 2,
    max_tokens: int | None = None,
    parallel_tool_calls: bool | None = None,
) -> ChatOpenAI:
    if timeout <= 0:
        raise ValueError("timeout must be positive.")
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative.")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive when provided.")
    selected_model = (model or LLM_MODEL).strip()
    if not selected_model:
        raise ValueError("LLM model name cannot be empty.")

    validate_runtime_config(require_openai=True)
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=selected_model,
        temperature=0.0,
        timeout=timeout,
        max_retries=max_retries,
        max_tokens=max_tokens,
        callbacks=[UsageCallbackHandler(selected_model)],
        model_kwargs=(
            {"parallel_tool_calls": parallel_tool_calls}
            if parallel_tool_calls is not None
            else {}
        ),
        streaming=False,
    )
