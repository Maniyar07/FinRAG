from __future__ import annotations

import json

from src.observability import summarize_trace_value, trace_component


def test_trace_summary_redacts_secrets_and_bounds_content() -> None:
    secret = "lsv2_test_secret_value_123456789"
    summary = summarize_trace_value(
        {
            "api_key": secret,
            "authorization": f"Bearer {secret}",
            "context": "confidential evidence body",
            "question": "What was annual revenue?",
        }
    )

    assert summary["api_key"] == "[REDACTED]"
    assert summary["authorization"] == "[REDACTED]"
    assert summary["context"] == {"chars": len("confidential evidence body")}
    assert summary["question"]["preview"] == "What was annual revenue?"
    assert secret not in json.dumps(summary)
    assert "confidential evidence body" not in json.dumps(summary)


def test_trace_component_preserves_function_result_when_tracing_is_disabled() -> None:
    @trace_component("finrag.test.add")
    def add(left: int, right: int) -> int:
        return left + right

    assert add(2, 3) == 5


def test_short_scope_tuples_are_readable_arrays() -> None:
    summary = summarize_trace_value(
        {"available_keys": (("MSFT", "2025", "10K"),)}
    )

    assert summary["available_keys"]["items"] == [["MSFT", "2025", "10K"]]


def test_component_records_do_not_hide_normal_query_fields() -> None:
    payload = {f"field_{index}": index for index in range(24)}

    summary = summarize_trace_value(payload)

    assert "_omitted_fields" not in summary
    assert summary["field_23"] == 23
