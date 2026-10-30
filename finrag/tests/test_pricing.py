import json
from decimal import Decimal
from unittest.mock import patch

import pytest

from evals.evaluate_ragas import summarize_costs, summarize_generation
from src.pricing import load_prices
from src.telemetry import UsageCollector, _TokenUsage


def catalog(tmp_path, models):
    path = tmp_path / "prices.json"
    path.write_text(json.dumps({"currency": "USD", "rate_unit": "per_million_tokens", "providers": models}))
    return load_prices(path)


@pytest.mark.parametrize("price", [-1, "NaN", "Infinity", True, "not-a-rate"])
def test_invalid_rates_are_rejected_before_an_evaluation_can_spend_requests(tmp_path, price):
    with pytest.raises(ValueError, match="Invalid"):
        catalog(tmp_path, {"provider": {"model": {"input": price}}})


def test_aliases_are_explicit_and_provider_specific(tmp_path):
    rates = catalog(tmp_path, {"one": {"model": {"input": "1", "output": "2", "aliases": ["snapshot"]}},
                               "two": {"model": {"input": "3", "output": "4"}}})
    assert rates[("one", "snapshot")] is rates[("one", "model")]
    assert rates[("two", "model")]["input"] == Decimal("3")
    assert ("one", "model-2099") not in rates
    with patch("src.telemetry.TOKEN_PRICES", rates):
        usage = UsageCollector()
        usage.record("snapshot", _TokenUsage(1000, 0, 100, 1100), provider="one")
        usage.record("model", _TokenUsage(1000, 0, 100, 1100), provider="two")
    assert usage.snapshot(latency_ms=1).total_token_cost_usd == 0.0046


def test_alias_collisions_and_wrong_rate_units_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        catalog(tmp_path, {"provider": {"a": {"aliases": ["b"]}, "b": {"input": "1"}}})
    path = tmp_path / "wrong-unit.json"
    path.write_text(json.dumps({"currency": "USD", "rate_unit": "per_thousand_tokens", "providers": {}}))
    with pytest.raises(ValueError, match="million"):
        load_prices(path)


def test_missing_output_rate_preserves_known_input_subtotal(tmp_path):
    rates = catalog(tmp_path, {"custom": {"model": {"input": "2"}}})
    with patch("src.telemetry.TOKEN_PRICES", rates):
        usage = UsageCollector()
        usage.record("model", _TokenUsage(1000, 0, 100, 1100), provider="custom")
    metrics = usage.snapshot(latency_ms=1)
    assert metrics.total_token_cost_usd is None and not metrics.cost_complete
    assert metrics.known_token_cost_usd == 0.002
    assert "custom/model: missing output rate" in metrics.cost_issues


def test_zero_price_is_distinct_from_an_unknown_price(tmp_path):
    rates = catalog(tmp_path, {"custom": {"free": {"input": "0", "cached_input": "0", "output": "0"}}})
    with patch("src.telemetry.TOKEN_PRICES", rates):
        usage = UsageCollector()
        usage.record("free", _TokenUsage(100, 50, 10, 110), provider="custom")
    metrics = usage.snapshot(latency_ms=1)
    assert metrics.cost_complete and metrics.total_token_cost_usd == 0


def test_embedding_models_are_priced_per_call_and_missing_usage_remains_visible(tmp_path):
    rates = catalog(tmp_path, {"custom": {"small": {"input": "1"}, "large": {"input": "3"}}})
    with patch("src.telemetry.TOKEN_PRICES", rates):
        usage = UsageCollector()
        usage.record_embeddings({"model": "small", "usage": {"prompt_tokens": 1000}}, "retrieval", provider="custom")
        usage.record_embeddings({"model": "large", "usage": {"prompt_tokens": 1000}}, "retrieval", provider="custom")
        assert usage.snapshot(latency_ms=1).embedding_cost_usd == 0.004
        usage.record_embeddings({"model": "small"}, "retrieval", provider="custom")
        usage.record_embeddings({"model": "small", "usage": {"prompt_tokens": 1000}}, "retrieval", provider="custom")
    metrics = usage.snapshot(latency_ms=1)
    assert metrics.embedding_cost_usd is None and not metrics.cost_complete
    assert metrics.known_token_cost_usd == 0.005 and metrics.embedding_calls == 4


def test_application_and_judge_costs_include_embeddings_without_double_counting():
    result = summarize_costs([{"llm_usage": {"estimated_cost_usd": 0.004, "embedding_cost_usd": 0.0001,
        "total_token_cost_usd": 0.0041, "known_token_cost_usd": 0.0041},
        "judge_usage": {"estimated_cost_usd": 0.008, "embedding_cost_usd": 0.0002},
        "scoring_latency_ms": 2, "decision": "answered"}])
    assert result["complete"] and result["total_usd"] == 0.0123
    assert result["application_usd"] == 0.0041 and result["judging_usd"] == 0.0082


def test_unobserved_http_usage_does_not_become_a_zero_cost_success():
    rows = [{"llm_usage": None}, {"llm_usage": {"estimated_cost_usd": 0.01}}]
    result = summarize_costs(rows)
    assert not result["complete"] and result["total_usd"] is None
    assert result["known_subtotal_usd"] == 0.01 and result["missing_generation_usage_rows"] == 1
    usage = summarize_generation(rows)["llm_usage"]
    assert usage["total_token_cost_usd"] is None and not usage["token_cost_complete"]
    assert usage["estimated_cost_usd"] is None and not usage["cost_complete"]


def test_missing_judge_usage_keeps_the_known_application_cost_separate():
    result = summarize_costs([{"decision": "answered", "scoring_latency_ms": 10,
        "llm_usage": {"estimated_cost_usd": 0.01}}])
    assert result["application_usd"] == 0.01 and result["judging_usd"] is None
    assert not result["complete"] and result["known_subtotal_usd"] == 0.01
