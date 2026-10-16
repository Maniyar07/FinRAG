import json
import sys

import pytest

from fine_tuning.inference.benchmark_endpoint import (
    DEFAULT_DATASET,
    INSUFFICIENT_EVIDENCE_RESPONSE,
    check_output,
    load_examples,
    main,
    summarize,
)


def test_held_out_split_and_gold_format():
    examples = load_examples(DEFAULT_DATASET, 40)
    assert len(examples) == 40
    checks = [
        check_output(row["gold"][0]["content"], row["prompt"], "stop")
        for row in examples
    ]
    assert all(item["json_valid"] and item["schema_valid"] for item in checks)
    assert sum(item["source_ids_exact"] is True for item in checks) == 38
    assert sum(item["source_ids_exact"] is None for item in checks) == 2


def test_invalid_citation_is_detected():
    prompt = [{"role": "user", "content": "[SOURCE S1] Revenue was 10."}]
    output = json.dumps(
        {"answer": "Revenue was 10. [S99]", "claims": [], "source_ids": ["S99"]}
    )
    checks = check_output(output, prompt, "stop")
    assert checks["schema_valid"]
    assert checks["source_ids_exact"] is False


def test_canonical_insufficient_evidence_response_is_valid_without_sources():
    prompt = [{"role": "user", "content": "[SOURCE S1] Revenue was 10."}]
    output = json.dumps(
        {"answer": INSUFFICIENT_EVIDENCE_RESPONSE, "claims": [], "source_ids": []}
    )
    checks = check_output(output, prompt, "stop")
    assert checks["schema_valid"]
    assert checks["source_ids_exact"] is None


def test_empty_answer_and_claims_is_invalid():
    output = json.dumps({"answer": "", "claims": [], "source_ids": []})
    checks = check_output(output, [], "stop")
    assert not checks["schema_valid"]
    assert "exactly_one_output_mode_required" in checks["schema_errors"]


def test_answer_and_claims_cannot_both_be_populated():
    output = json.dumps(
        {
            "answer": "Revenue was 10. [S1]",
            "claims": [{"text": "Revenue was 10.", "source_ids": ["S1"]}],
            "source_ids": ["S1"],
        }
    )
    checks = check_output(output, [], "stop")
    assert not checks["schema_valid"]
    assert "exactly_one_output_mode_required" in checks["schema_errors"]


def test_extra_top_level_field_is_invalid():
    output = json.dumps(
        {
            "answer": "Revenue was 10. [S1]",
            "claims": [],
            "source_ids": ["S1"],
            "unexpected": True,
        }
    )
    checks = check_output(output, [], "stop")
    assert not checks["schema_valid"]
    assert "top_level_fields" in checks["schema_errors"]


def test_claim_sources_must_match_top_level_sources():
    prompt = [
        {"role": "user", "content": "[SOURCE S1] A.\n[SOURCE S2] B."},
    ]
    output = json.dumps(
        {
            "answer": "",
            "claims": [{"text": "A.", "source_ids": ["S1"]}],
            "source_ids": ["S2"],
        }
    )
    checks = check_output(output, prompt, "stop")
    assert not checks["schema_valid"]
    assert "claim_source_ids_mismatch" in checks["schema_errors"]


def test_claim_text_cannot_contain_inline_citation():
    output = json.dumps(
        {
            "answer": "",
            "claims": [{"text": "Revenue was 10. [S1]", "source_ids": ["S1"]}],
            "source_ids": ["S1"],
        }
    )
    checks = check_output(output, [], "stop")
    assert not checks["schema_valid"]
    assert "claim_contains_citation" in checks["schema_errors"]


def test_answer_citations_must_match_declared_sources():
    output = json.dumps(
        {"answer": "Revenue was 10. [S1]", "claims": [], "source_ids": ["S2"]}
    )
    checks = check_output(output, [], "stop")
    assert not checks["schema_valid"]
    assert "answer_citations_mismatch" in checks["schema_errors"]


def test_source_ids_must_be_uppercase_unique_s_numbers():
    output = json.dumps(
        {
            "answer": "Revenue was 10. [s1]",
            "claims": [],
            "source_ids": ["s1", "s1"],
        }
    )
    checks = check_output(output, [], "stop")
    assert not checks["schema_valid"]
    assert "invalid_source_id_format" in checks["schema_errors"]
    assert "duplicate_source_ids" in checks["schema_errors"]


def test_summary_uses_only_applicable_source_checks():
    rows = [
        {"latency_ms": 10.0, "error": None, "completion_tokens": 2,
         "checks": {"json_valid": True, "schema_valid": True,
                    "source_ids_exact": True, "source_ids_normalized": True,
                    "possibly_truncated": False, "schema_errors": []}},
        {"latency_ms": 20.0, "error": None, "completion_tokens": 3,
         "checks": {"json_valid": True, "schema_valid": True,
                    "source_ids_exact": None, "source_ids_normalized": None,
                    "possibly_truncated": False,
                    "schema_errors": ["example_error"]}},
    ]
    report = summarize(rows, 1.0)
    assert report["source_id_checks_applicable"] == 1
    assert report["format_counts"]["source_ids_exact"] == 1
    assert report["output_tokens_per_second"] == 5.0
    assert report["schema_error_counts"] == {"example_error": 1}


def test_main_rejects_concurrency_above_server_capacity(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_endpoint.py",
            "--concurrency",
            "2",
            "--server-max-num-seqs",
            "1",
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
