from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Decimal
from threading import Lock
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from src.config import (
    LLM_CACHED_INPUT_USD_PER_MILLION,
    LLM_INPUT_USD_PER_MILLION,
    LLM_OUTPUT_USD_PER_MILLION,
    LLM_PRICING_MODEL,
)


@dataclass(frozen=True)
class RequestMetrics:
    """Safe, public metrics for one FastAPI chat request."""

    latency_ms: float
    llm_calls: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float | None
    models: tuple[str, ...]


@dataclass(frozen=True)
class _TokenUsage:
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    total_tokens: int


class UsageCollector:
    """Thread-safe accumulator shared only within the current request context."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._llm_calls = 0
        self._input_tokens = 0
        self._cached_input_tokens = 0
        self._output_tokens = 0
        self._total_tokens = 0
        self._estimated_cost = Decimal("0")
        self._cost_complete = True
        self._models: set[str] = set()

    def record(self, model: str, usage: _TokenUsage | None) -> None:
        normalized_model = str(model or "").strip()
        with self._lock:
            self._llm_calls += 1
            if normalized_model:
                self._models.add(normalized_model)
            if usage is None:
                self._cost_complete = False
                return

            self._input_tokens += usage.input_tokens
            self._cached_input_tokens += usage.cached_input_tokens
            self._output_tokens += usage.output_tokens
            self._total_tokens += usage.total_tokens

            if not _matches_pricing_model(normalized_model):
                self._cost_complete = False
                return

            uncached_input = usage.input_tokens - usage.cached_input_tokens
            self._estimated_cost += (
                Decimal(uncached_input) * LLM_INPUT_USD_PER_MILLION
                + Decimal(usage.cached_input_tokens)
                * LLM_CACHED_INPUT_USD_PER_MILLION
                + Decimal(usage.output_tokens) * LLM_OUTPUT_USD_PER_MILLION
            ) / Decimal(1_000_000)

    def snapshot(self, *, latency_ms: float) -> RequestMetrics:
        with self._lock:
            if self._llm_calls == 0:
                estimated_cost: float | None = 0.0
            elif self._cost_complete:
                estimated_cost = round(float(self._estimated_cost), 8)
            else:
                estimated_cost = None
            return RequestMetrics(
                latency_ms=round(max(0.0, float(latency_ms)), 2),
                llm_calls=self._llm_calls,
                input_tokens=self._input_tokens,
                cached_input_tokens=self._cached_input_tokens,
                output_tokens=self._output_tokens,
                total_tokens=self._total_tokens,
                estimated_cost_usd=estimated_cost,
                models=tuple(sorted(self._models)),
            )


_ACTIVE_COLLECTOR: ContextVar[UsageCollector | None] = ContextVar(
    "finrag_active_usage_collector", default=None
)


@contextmanager
def collect_usage() -> Iterator[UsageCollector]:
    """Collect LLM usage in this context without leaking across requests."""

    collector = UsageCollector()
    reset_token = _ACTIVE_COLLECTOR.set(collector)
    try:
        yield collector
    finally:
        _ACTIVE_COLLECTOR.reset(reset_token)


class UsageCallbackHandler(BaseCallbackHandler):
    """Forward LangChain/OpenAI usage metadata to the active API request."""

    raise_error = False

    def __init__(self, configured_model: str) -> None:
        self.configured_model = configured_model

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        del kwargs
        collector = _ACTIVE_COLLECTOR.get()
        if collector is None:
            return
        try:
            message = _first_message(response)
            model = _response_model(response, message) or self.configured_model
            collector.record(model, _response_usage(response, message))
        except Exception:
            # Telemetry must never make an otherwise valid financial answer fail.
            collector.record(self.configured_model, None)


def _matches_pricing_model(model: str) -> bool:
    candidate = model.strip().casefold()
    priced = LLM_PRICING_MODEL.casefold()
    return bool(candidate) and (
        candidate == priced or candidate.startswith(f"{priced}-20")
    )


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    return {}


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _first_message(response: Any) -> Any | None:
    generations = getattr(response, "generations", None)
    if not generations:
        return None
    try:
        generation = generations[0][0]
    except (IndexError, TypeError):
        return None
    return getattr(generation, "message", None)


def _response_model(response: Any, message: Any | None) -> str:
    llm_output = _as_mapping(getattr(response, "llm_output", None))
    model = _first(llm_output, "model_name", "model")
    if model:
        return str(model)
    response_metadata = _as_mapping(getattr(message, "response_metadata", None))
    model = _first(response_metadata, "model_name", "model")
    return str(model or "")


def _response_usage(response: Any, message: Any | None) -> _TokenUsage | None:
    llm_output = _as_mapping(getattr(response, "llm_output", None))
    usage = _as_mapping(_first(llm_output, "token_usage", "usage"))

    if not usage and message is not None:
        usage = _as_mapping(getattr(message, "usage_metadata", None))
    if not usage and message is not None:
        response_metadata = _as_mapping(
            getattr(message, "response_metadata", None)
        )
        usage = _as_mapping(
            _first(response_metadata, "token_usage", "usage")
        )

    token_keys = {
        "prompt_tokens",
        "input_tokens",
        "completion_tokens",
        "output_tokens",
        "total_tokens",
    }
    if not token_keys.intersection(usage):
        return None

    input_tokens = _non_negative_int(
        _first(usage, "prompt_tokens", "input_tokens")
    )
    output_tokens = _non_negative_int(
        _first(usage, "completion_tokens", "output_tokens")
    )
    details = _as_mapping(
        _first(usage, "prompt_tokens_details", "input_token_details")
    )
    cached_input_tokens = min(
        input_tokens,
        _non_negative_int(
            _first(details, "cached_tokens", "cache_read", "cache_read_tokens")
        ),
    )
    total_tokens = max(
        input_tokens + output_tokens,
        _non_negative_int(usage.get("total_tokens")),
    )
    return _TokenUsage(
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )
