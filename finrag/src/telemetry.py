from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from decimal import Decimal
from threading import Lock
from time import perf_counter
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from src.config import (
    EMBEDDING_INPUT_USD_PER_MILLION,
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
    stages: dict[str, dict] = field(default_factory=dict)
    embedding_calls: int = 0
    embedding_input_tokens: int = 0
    embedding_cost_usd: float | None = 0.0


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
        self._stages: dict[str, dict] = {}

    def record(self, model: str, usage: _TokenUsage | None) -> None:
        normalized_model = str(model or "").strip()
        with self._lock:
            stage_name = _ACTIVE_STAGE.get()
            if stage_name == "finrag.chat_request":
                stage_name = "answer_generation"
            defaults = {
                "runs": 0, "latency_ms": 0.0, "llm_calls": 0,
                "input_tokens": 0, "cached_input_tokens": 0,
                "output_tokens": 0, "total_tokens": 0,
                "estimated_cost_usd": 0.0, "usage_complete": True,
            }
            stage = self._stages.setdefault(stage_name, {})
            for key, value in defaults.items():
                stage.setdefault(key, value)
            stage["llm_calls"] += 1
            self._llm_calls += 1
            if normalized_model:
                self._models.add(normalized_model)
            if usage is None:
                stage["usage_complete"] = False
                stage["estimated_cost_usd"] = None
                self._cost_complete = False
                return

            self._input_tokens += usage.input_tokens
            self._cached_input_tokens += usage.cached_input_tokens
            self._output_tokens += usage.output_tokens
            self._total_tokens += usage.total_tokens
            for key in ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens"):
                stage[key] += getattr(usage, key)

            if not _matches_pricing_model(normalized_model):
                stage["estimated_cost_usd"] = None
                self._cost_complete = False
                return

            uncached_input = usage.input_tokens - usage.cached_input_tokens
            cost = (
                Decimal(uncached_input) * LLM_INPUT_USD_PER_MILLION
                + Decimal(usage.cached_input_tokens)
                * LLM_CACHED_INPUT_USD_PER_MILLION
                + Decimal(usage.output_tokens) * LLM_OUTPUT_USD_PER_MILLION
            ) / Decimal(1_000_000)
            self._estimated_cost += cost
            if stage["estimated_cost_usd"] is not None:
                stage["estimated_cost_usd"] = round(stage["estimated_cost_usd"] + float(cost), 8)

    def record_stage(self, name: str, latency_ms: float) -> None:
        with self._lock:
            stage = self._stages.setdefault(name, {"runs": 0, "latency_ms": 0.0, "llm_calls": 0})
            stage["runs"] += 1
            stage["latency_ms"] = round(stage["latency_ms"] + latency_ms, 2)

    def record_embeddings(self, payload: Mapping, stage_name: str) -> None:
        usage = payload.get("usage", {})
        tokens = usage.get("prompt_tokens", usage.get("total_tokens"))
        with self._lock:
            stage = self._stages.setdefault(stage_name, {"runs": 0, "latency_ms": 0.0, "llm_calls": 0})
            stage["embedding_calls"] = stage.get("embedding_calls", 0) + 1
            stage["embedding_input_tokens"] = stage.get("embedding_input_tokens", 0) + _non_negative_int(tokens)
            stage.setdefault("embedding_usage_complete", True)
            if tokens is None:
                stage["embedding_usage_complete"] = False
            stage["embedding_cost_usd"] = (
                round(float(Decimal(stage["embedding_input_tokens"]) * EMBEDDING_INPUT_USD_PER_MILLION / Decimal(1_000_000)), 8)
                if EMBEDDING_INPUT_USD_PER_MILLION is not None and stage["embedding_usage_complete"] else None
            )

    def record_model_latency(self, name: str, latency_ms: float) -> None:
        with self._lock:
            stage = self._stages.setdefault(name, {"runs": 0, "latency_ms": 0.0, "llm_calls": 0})
            stage["model_latency_ms"] = round(stage.get("model_latency_ms", 0.0) + latency_ms, 2)

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
                stages={key: dict(value) for key, value in self._stages.items()},
                embedding_calls=sum(value.get("embedding_calls", 0) for value in self._stages.values()),
                embedding_input_tokens=sum(value.get("embedding_input_tokens", 0) for value in self._stages.values()),
                embedding_cost_usd=(None if any(value.get("embedding_calls") and value.get("embedding_cost_usd") is None for value in self._stages.values())
                                    else round(sum(value.get("embedding_cost_usd") or 0 for value in self._stages.values()), 8)),
            )


_ACTIVE_COLLECTOR: ContextVar[UsageCollector | None] = ContextVar(
    "finrag_active_usage_collector", default=None
)
_ACTIVE_STAGE: ContextVar[str] = ContextVar("finrag_usage_stage", default="answer_generation")


@contextmanager
def usage_stage(name: str):
    token = _ACTIVE_STAGE.set(name)
    started = perf_counter()
    try:
        yield
    finally:
        collector = _ACTIVE_COLLECTOR.get()
        if collector is not None:
            collector.record_stage(name, (perf_counter() - started) * 1000)
        _ACTIVE_STAGE.reset(token)


async def record_provider_response(response) -> None:
    """Capture non-streaming judge responses; never interfere with SDK parsing."""
    collector = _ACTIVE_COLLECTOR.get()
    if collector is None or response.status_code >= 400:
        return
    if "text/event-stream" in response.headers.get("content-type", ""):
        return
    try:
        await response.aread()
        payload = response.json()
        if "/embeddings" in str(response.request.url):
            collector.record_embeddings(payload, "ragas_embeddings")
            return
        usage = payload.get("usage", {})
        from types import SimpleNamespace
        message = SimpleNamespace(llm_output={"usage": usage, "model": payload.get("model", "")})
        collector.record(payload.get("model", ""), _response_usage(message, None))
    except Exception:
        collector.record("", None)


def record_embedding_response(response) -> None:
    collector = _ACTIVE_COLLECTOR.get()
    if collector is None or response.status_code >= 400:
        return
    try:
        response.read()
        collector.record_embeddings(response.json(), _ACTIVE_STAGE.get())
    except Exception:
        collector.record_embeddings({}, _ACTIVE_STAGE.get())


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
        self._starts: dict[str, tuple[str, float]] = {}
        self._lock = Lock()

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        stage = _ACTIVE_STAGE.get()
        with self._lock:
            self._starts[str(run_id)] = ("answer_generation" if stage == "finrag.chat_request" else stage, perf_counter())

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        collector = _ACTIVE_COLLECTOR.get()
        with self._lock:
            started = self._starts.pop(str(kwargs.get("run_id")), None)
        if collector is not None and started is not None:
            collector.record_model_latency(started[0], (perf_counter() - started[1]) * 1000)
        if collector is None:
            return
        try:
            message = _first_message(response)
            model = _response_model(response, message) or self.configured_model
            collector.record(model, _response_usage(response, message))
        except Exception:
            # Telemetry must never make an otherwise valid financial answer fail.
            collector.record(self.configured_model, None)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        with self._lock:
            self._starts.pop(str(kwargs.get("run_id")), None)
        collector = _ACTIVE_COLLECTOR.get()
        if collector is not None:
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
