from __future__ import annotations

import asyncio
import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

from src.telemetry import UsageCallbackHandler, collect_usage


def _response(
    *,
    model: str = "gpt-4o-mini-2024-07-18",
    input_tokens: int = 1_000,
    cached_tokens: int = 400,
    output_tokens: int = 200,
):
    return SimpleNamespace(
        llm_output={
            "model_name": model,
            "token_usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "prompt_tokens_details": {"cached_tokens": cached_tokens},
            },
        },
        generations=[],
    )


class TelemetryTests(unittest.TestCase):
    def test_callback_aggregates_tokens_and_cached_input_cost(self) -> None:
        handler = UsageCallbackHandler("gpt-4o-mini")
        pricing = {
            "LLM_PRICING_MODEL": "gpt-4o-mini",
            "LLM_INPUT_USD_PER_MILLION": Decimal("0.15"),
            "LLM_CACHED_INPUT_USD_PER_MILLION": Decimal("0.075"),
            "LLM_OUTPUT_USD_PER_MILLION": Decimal("0.60"),
        }
        with patch.multiple("src.telemetry", **pricing):
            with collect_usage() as collector:
                handler.on_llm_end(_response())
                handler.on_llm_end(
                    _response(input_tokens=500, cached_tokens=0, output_tokens=100)
                )
            metrics = collector.snapshot(latency_ms=123.456)

        self.assertEqual(metrics.latency_ms, 123.46)
        self.assertEqual(metrics.llm_calls, 2)
        self.assertEqual(metrics.input_tokens, 1_500)
        self.assertEqual(metrics.cached_input_tokens, 400)
        self.assertEqual(metrics.output_tokens, 300)
        self.assertEqual(metrics.total_tokens, 1_800)
        self.assertEqual(metrics.estimated_cost_usd, 0.000375)
        self.assertEqual(metrics.models, ("gpt-4o-mini-2024-07-18",))

    def test_usage_metadata_fallback_and_nested_collectors_are_isolated(self) -> None:
        handler = UsageCallbackHandler("gpt-4o-mini")
        message = SimpleNamespace(
            usage_metadata={
                "input_tokens": 20,
                "output_tokens": 5,
                "total_tokens": 25,
                "input_token_details": {"cache_read": 10},
            },
            response_metadata={"model_name": "gpt-4o-mini"},
        )
        response = SimpleNamespace(
            llm_output=None,
            generations=[[SimpleNamespace(message=message)]],
        )

        with collect_usage() as outer:
            handler.on_llm_end(response)
            with collect_usage() as inner:
                handler.on_llm_end(response)
                inner_metrics = inner.snapshot(latency_ms=1)
            handler.on_llm_end(response)
        outer_metrics = outer.snapshot(latency_ms=2)

        self.assertEqual(inner_metrics.llm_calls, 1)
        self.assertEqual(inner_metrics.total_tokens, 25)
        self.assertEqual(outer_metrics.llm_calls, 2)
        self.assertEqual(outer_metrics.total_tokens, 50)

    def test_unknown_model_keeps_tokens_but_omits_cost(self) -> None:
        handler = UsageCallbackHandler("custom-model")
        with collect_usage() as collector:
            handler.on_llm_end(_response(model="custom-model"))
        metrics = collector.snapshot(latency_ms=10)

        self.assertEqual(metrics.total_tokens, 1_200)
        self.assertIsNone(metrics.estimated_cost_usd)

    def test_collector_context_reaches_fastapi_worker_thread(self) -> None:
        handler = UsageCallbackHandler("gpt-4o-mini")

        async def invoke_in_worker() -> None:
            await asyncio.to_thread(handler.on_llm_end, _response())

        with collect_usage() as collector:
            asyncio.run(invoke_in_worker())
        metrics = collector.snapshot(latency_ms=5)

        self.assertEqual(metrics.llm_calls, 1)
        self.assertEqual(metrics.total_tokens, 1_200)

    def test_request_without_llm_calls_has_zero_cost(self) -> None:
        with collect_usage() as collector:
            pass
        metrics = collector.snapshot(latency_ms=0)

        self.assertEqual(metrics.llm_calls, 0)
        self.assertEqual(metrics.total_tokens, 0)
        self.assertEqual(metrics.estimated_cost_usd, 0.0)


if __name__ == "__main__":
    unittest.main()
