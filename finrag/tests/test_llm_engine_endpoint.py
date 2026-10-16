from __future__ import annotations

from unittest.mock import patch

from src.generation.llm_engine import get_llm_engine


def test_default_chat_client_uses_existing_openai_configuration() -> None:
    with (
        patch("src.generation.llm_engine.LLM_BASE_URL", None),
        patch("src.generation.llm_engine.OPENAI_API_KEY", "openai-test-key"),
        patch("src.generation.llm_engine.ChatOpenAI") as constructor,
        patch("src.generation.llm_engine.validate_runtime_config") as validate,
    ):
        get_llm_engine()
    validate.assert_called_once_with(require_openai=True)
    assert constructor.call_args.kwargs["api_key"] == "openai-test-key"
    assert constructor.call_args.kwargs["base_url"] is None


def test_model_endpoint_never_receives_openai_embedding_key() -> None:
    with (
        patch("src.generation.llm_engine.LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        patch("src.generation.llm_engine.LLM_API_KEY", None),
        patch("src.generation.llm_engine.OPENAI_API_KEY", "do-not-send-this"),
        patch("src.generation.llm_engine.ChatOpenAI") as constructor,
        patch("src.generation.llm_engine.validate_runtime_config") as validate,
    ):
        get_llm_engine(model="finrag-lora")
    validate.assert_called_once_with(require_openai=False)
    assert constructor.call_args.kwargs["api_key"] == "local"
    assert constructor.call_args.kwargs["base_url"] == "http://127.0.0.1:8000/v1"
    assert constructor.call_args.kwargs["model"] == "finrag-lora"
