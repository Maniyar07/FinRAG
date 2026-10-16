"""Run the live FinRAG pipeline and score it with five RAGAS metrics.

Place this file in ``FinRAG_Phase1_Corrected/finrag/evals/`` beside
``ragas_gold.json``. Retrieved contexts and generated answers are collected
from ``ChatService.ask`` at runtime; they are not stored in the gold file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schemas import Decision
from src.runtime_errors import classify_runtime_error
from src.observability import trace_component
from src.telemetry import collect_usage, usage_stage, record_provider_response

DEFAULT_DATASET = Path(__file__).with_name("ragas_gold.json")
DEFAULT_OUTPUT = Path(__file__).parent / "results" / "ragas_results.json"
DEFAULT_GENERATION_OUTPUT = Path(__file__).parent / "results" / "fixed_answers.json"
METRIC_NAMES = (
    "faithfulness",
    "context_precision",
    "context_recall",
    "response_relevancy",
    "factual_correctness",
)
VALID_DOC_TYPES = {"10K", "TRANSCRIPT"}
SOURCE_SECTION_RE = re.compile(
    r"\n{2,}\*\*Sources\*\*\s*\n.*\Z",
    re.IGNORECASE | re.DOTALL,
)
CITATION_RE = re.compile(
    r"\s*\[S\d+(?::[^\]]*)?\]",
    re.IGNORECASE,
)
CITATION_ID_RE = re.compile(r"\[(S\d+)(?:\s*:[^\]]*)?\]", re.IGNORECASE)
VALID_EXPECTED_DECISIONS = {
    decision.value
    for decision in Decision
    if decision not in {Decision.SEARCH, Decision.ERROR}
}


def load_gold(path: Path) -> list[dict[str, Any]]:
    """Load and strictly validate the human-authored gold questions."""
    if not path.is_file():
        raise ValueError(f"Gold dataset not found: {path}")
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path.name}: {error}") from error
    if not isinstance(records, list) or not records:
        raise ValueError("Gold dataset must be a non-empty JSON array.")

    errors: list[str] = []
    seen_ids: set[str] = set()
    for position, record in enumerate(records, 1):
        if not isinstance(record, dict):
            errors.append(f"record {position}: must be an object")
            continue
        record_id = str(record.get("id", "")).strip()
        if not record_id:
            errors.append(f"record {position}: missing id")
        elif record_id in seen_ids:
            errors.append(f"record {position}: duplicate id {record_id}")
        seen_ids.add(record_id)

        for field in ("category", "question"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                errors.append(f"record {record_id or position}: missing {field}")

        raw_expected_decision = record.get("expected_decision")
        if raw_expected_decision is not None and not isinstance(raw_expected_decision, str):
            errors.append(
                f"record {record_id or position}: expected_decision must be a string"
            )
        expected_decision = str(
            raw_expected_decision or Decision.ANSWERED.value
        ).strip().casefold()
        if expected_decision not in VALID_EXPECTED_DECISIONS:
            errors.append(
                f"record {record_id or position}: invalid expected_decision "
                f"{expected_decision!r}; expected one of "
                f"{sorted(VALID_EXPECTED_DECISIONS)}"
            )
        raw_reference = record.get("reference")
        if raw_reference is not None and not isinstance(raw_reference, str):
            errors.append(f"record {record_id or position}: reference must be a string")
        reference = raw_reference.strip() if isinstance(raw_reference, str) else ""
        raw_expected_outcome = record.get("expected_outcome")
        if raw_expected_outcome is not None and not isinstance(raw_expected_outcome, str):
            errors.append(
                f"record {record_id or position}: expected_outcome must be a string"
            )
        expected_outcome = (
            raw_expected_outcome.strip()
            if isinstance(raw_expected_outcome, str)
            else ""
        )
        if expected_decision == Decision.ANSWERED.value and not reference:
            errors.append(f"record {record_id or position}: missing reference for an answered expectation")
        if expected_decision != Decision.ANSWERED.value and not expected_outcome:
            errors.append(
                f"record {record_id or position}: expected_outcome is required when "
                f"expected_decision is {expected_decision!r}"
            )

        scope = record.get("scope")
        if not isinstance(scope, dict):
            errors.append(f"record {record_id or position}: missing scope")
            continue
        scope_lists: dict[str, list[Any]] = {}
        for field in ("tickers", "years", "doc_types"):
            values = scope.get(field, [])
            if not isinstance(values, list):
                errors.append(f"record {record_id or position}: scope.{field} must be a list")
                scope_lists[field] = []
            else:
                scope_lists[field] = values
                if any(not isinstance(value, str) or not value.strip() for value in values):
                    errors.append(
                        f"record {record_id or position}: scope.{field} values must be non-empty strings"
                    )
                if expected_decision == Decision.ANSWERED.value and not values:
                    errors.append(f"record {record_id or position}: scope.{field} is empty")
        invalid_types = {
            str(value).upper().replace("-", "")
            for value in scope_lists.get("doc_types", [])
            if isinstance(value, str)
        } - VALID_DOC_TYPES
        if invalid_types:
            errors.append(
                f"record {record_id or position}: invalid document types {sorted(invalid_types)}"
            )

    if errors:
        raise ValueError("Gold validation failed:\n- " + "\n- ".join(errors))
    return records


def _expected_decision(record: dict[str, Any]) -> str:
    """Legacy answerable gold rows default to answered; special outcomes are explicit."""
    return str(record.get("expected_decision") or Decision.ANSWERED.value).strip().casefold()


def _expected_outcome(record: dict[str, Any]) -> str:
    outcome = record.get("expected_outcome") or record.get("reference")
    if isinstance(outcome, str) and outcome.strip():
        return outcome.strip()
    return f"Expected decision: {_expected_decision(record)}"


def load_generated_answers(
    path: Path,
    gold_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Load fixed runtime answers and verify that they match the selected gold rows."""
    if not path.is_file():
        raise ValueError(f"Generated-answer file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path.name}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("Generated-answer file must contain a results array.")
    if not isinstance(payload.get("run"), dict) or payload["run"].get("mode") != "generation":
        raise ValueError(
            "--score-input requires a file created by --generate-only."
        )

    by_id: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(payload["results"], 1):
        if not isinstance(row, dict):
            raise ValueError(f"Generated result {position} must be an object.")
        row_id = str(row.get("id", "")).strip()
        if not row_id:
            raise ValueError(f"Generated result {position} is missing id.")
        if row_id in by_id:
            raise ValueError(f"Generated-answer file contains duplicate id {row_id}.")
        by_id[row_id] = row

    selected: list[dict[str, Any]] = []
    for record in gold_records:
        record_id = str(record["id"])
        row = by_id.get(record_id)
        if row is None:
            raise ValueError(f"Generated-answer file is missing gold question id {record_id}.")
        for field in ("category", "question"):
            if row.get(field) != record[field]:
                raise ValueError(
                    f"Generated result {record_id} does not match gold field {field}."
                )
        if row.get("reference", "") != str(record.get("reference") or ""):
            raise ValueError(
                f"Generated result {record_id} does not match gold field reference."
            )
        expected_decision = _expected_decision(record)
        if row.get("expected_decision", expected_decision) != expected_decision:
            raise ValueError(
                f"Generated result {record_id} does not match gold field expected_decision."
            )
        expected_outcome = _expected_outcome(record)
        if row.get("expected_outcome", expected_outcome) != expected_outcome:
            raise ValueError(
                f"Generated result {record_id} does not match gold field expected_outcome."
            )
        contexts = row.get("retrieved_contexts")
        sources = row.get("retrieved_sources")
        has_generation_error = bool(str(row.get("error", "")).strip())
        if not isinstance(row.get("decision"), str) or not row["decision"].strip():
            raise ValueError(f"Generated result {record_id} has an invalid decision.")
        if row.get("scope_match") is not None and not isinstance(row.get("scope_match"), bool):
            raise ValueError(f"Generated result {record_id} has invalid scope_match.")
        if not isinstance(row.get("answer"), str):
            raise ValueError(f"Generated result {record_id} has an invalid answer.")
        if not isinstance(contexts, list) or any(
            not isinstance(context, str) for context in contexts
        ):
            raise ValueError(
                f"Generated result {record_id} has invalid retrieved_contexts."
            )
        if not isinstance(sources, list) or any(
            not isinstance(source, dict) for source in sources
        ):
            raise ValueError(
                f"Generated result {record_id} has invalid retrieved_sources."
            )
        if not has_generation_error and not row["answer"].strip():
            raise ValueError(
                f"Generated result {record_id} needs a non-empty answer."
            )
        if not has_generation_error and row["decision"] == "answered" and not any(
            context.strip() for context in contexts
        ):
            raise ValueError(
                f"Answered result {record_id} needs a non-empty answer and context."
            )
        normalized_row = dict(row)
        normalized_row.setdefault("reference", str(record.get("reference") or ""))
        normalized_row.setdefault("expected_decision", expected_decision)
        normalized_row.setdefault("expected_outcome", _expected_outcome(record))
        normalized_row.setdefault("errors", [])
        normalized_row.setdefault(
            "tool_trace", {"calls": [], "results": [], "calculator_results": []}
        )
        normalized_row.setdefault("citations", [])
        normalized_row.setdefault("citation_validation", "missing")
        normalized_row["decision_match"] = (
            normalized_row.get("decision") == expected_decision
            and not has_generation_error
            and not _has_generation_errors(normalized_row)
        )
        normalized_row["passed"] = normalized_row["decision_match"]
        selected.append(normalized_row)
    return selected


def _number(value: Any) -> float | None:
    raw = getattr(value, "value", value)
    try:
        result = float(raw)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _clean_answer(answer: str) -> str:
    """Remove citation presentation while preserving the generated answer."""
    without_sources = SOURCE_SECTION_RE.sub("", answer)
    return CITATION_RE.sub("", without_sources).strip()


def _scope_matches(expected: dict[str, Any], actual: Any) -> bool | None:
    comparisons: list[bool] = []
    expected_tickers = {str(item).upper() for item in expected.get("tickers", ())}
    expected_years = {str(item) for item in expected.get("years", ())}
    expected_types = {
        str(item).upper().replace("-", "") for item in expected.get("doc_types", ())
    }
    if expected_tickers:
        actual_tickers = {str(item).upper() for item in getattr(actual, "tickers", ()) or ()}
        comparisons.append(actual_tickers == expected_tickers)
    if expected_years:
        actual_years = {str(item) for item in getattr(actual, "years", ()) or ()}
        comparisons.append(actual_years == expected_years)
    if expected_types:
        actual_types = set(getattr(actual, "required_doc_types", ()) or ())
        if not actual_types and getattr(actual, "doc_type", None):
            actual_types = {actual.doc_type}
        actual_types = {str(item).upper().replace("-", "") for item in actual_types}
        comparisons.append(actual_types == expected_types)
    return all(comparisons) if comparisons else None


def _citation_diagnostics(
    answer: str, sources: list[dict[str, Any]]
) -> dict[str, Any]:
    source_by_id = {
        str(source.get("id") or "").strip().upper(): source
        for source in sources
        if str(source.get("id") or "").strip()
    }
    citation_ids = list(
        dict.fromkeys(match.group(1).upper() for match in CITATION_ID_RE.finditer(answer))
    )
    citations = [
        {
            "id": citation_id,
            "valid": citation_id in source_by_id,
            "source": source_by_id.get(citation_id),
        }
        for citation_id in citation_ids
    ]
    invalid_ids = [citation["id"] for citation in citations if not citation["valid"]]
    return {
        "citations": citations,
        "citation_ids": citation_ids,
        "valid_citation_ids": [
            citation["id"] for citation in citations if citation["valid"]
        ],
        "invalid_citation_ids": invalid_ids,
        "citation_validation": (
            "missing" if not citation_ids else "invalid" if invalid_ids else "valid"
        ),
    }


def _tool_trace(trace: dict[str, Any]) -> dict[str, Any]:
    agent = trace.get("agent") if isinstance(trace.get("agent"), dict) else {}
    tool_results = agent.get("tool_results") or []
    return {
        "calls": agent.get("tool_call_details") or agent.get("tool_calls") or [],
        "results": tool_results,
        "calculator_results": agent.get("calculator_results") or [
            result.get("calculator_result")
            for result in tool_results
            if isinstance(result, dict) and isinstance(result.get("calculator_result"), dict)
        ],
    }


def _compact_runtime_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """Keep routing/runtime diagnostics while storing tool payloads only once."""
    compact = dict(trace)
    agent = compact.get("agent")
    if isinstance(agent, dict):
        compact["agent"] = {
            key: value
            for key, value in agent.items()
            if key not in {
                "tool_calls", "tool_call_details", "tool_results", "calculator_results"
            }
        }
    return compact


def _collect_errors(trace: dict[str, Any], row_error: str | None) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if row_error:
        errors.append({"component": "evaluation", "message": row_error})
    if trace.get("error_type"):
        detail = trace.get("error_detail") or trace.get("error_type")
        entry: dict[str, Any] = {
            "component": "chat_service",
            "message": str(detail)[:500],
        }
        for field in ("error_code", "error_type", "retryable"):
            if field in trace:
                entry[field] = trace[field]
        errors.append(entry)
    for result in _tool_trace(trace)["results"]:
        if isinstance(result, dict) and result.get("error"):
            entry = {
                "component": str(result.get("name") or "tool"),
                "message": str(result["error"])[:500],
            }
            for field in ("error_code", "error_type", "retryable", "error_detail", "resolved_in_final_answer"):
                if field in result:
                    entry[field] = result[field]
            errors.append(entry)
    return errors


def _has_generation_errors(row: dict[str, Any]) -> bool:
    """Treat runtime and tool failures as generation failures, not valid answers."""
    if row.get("error"):
        return True
    errors = row.get("errors") or []
    if any(
        not str(error.get("component", "")).startswith("ragas.") and not error.get("resolved_in_final_answer")
        if isinstance(error, dict)
        else bool(error)
        for error in errors
    ):
        return True
    tool_trace = row.get("tool_trace")
    results = tool_trace.get("results", ()) if isinstance(tool_trace, dict) else ()
    return any(
        isinstance(result, dict)
        and not result.get("resolved_in_final_answer")
        and (result.get("error") or str(result.get("status", "")).casefold() == "error")
        for result in results
    )


def _decision_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_counts: dict[str, int] = {}
    actual_counts: dict[str, int] = {}
    for row in rows:
        expected = row.get("expected_decision")
        actual = row.get("decision")
        if expected:
            key = str(expected)
            expected_counts[key] = expected_counts.get(key, 0) + 1
        if actual:
            key = str(actual)
            actual_counts[key] = actual_counts.get(key, 0) + 1

    evaluated = [row for row in rows if row.get("expected_decision")]
    passed = sum(row.get("decision_match") is True for row in evaluated)
    expected_clarifications = [
        row for row in evaluated if row.get("expected_decision") == Decision.CLARIFY.value
    ]
    latency_summary: dict[str, dict[str, float | int | None]] = {}
    for field in ("generation_latency_ms", "scoring_latency_ms", "total_latency_ms"):
        values = [
            float(row[field])
            for row in rows
            if isinstance(row.get(field), (int, float))
            and math.isfinite(float(row[field]))
        ]
        ordered = sorted(values)
        p95 = ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)] if ordered else None
        latency_summary[field] = {
            "count": len(values),
            "mean": round(fmean(values), 2) if values else None,
            "max": round(max(values), 2) if values else None,
            "p95": round(p95, 2) if p95 is not None else None,
        }

    citation_ids = [
        citation for row in rows for citation in (row.get("citation_ids") or [])
    ]
    invalid_ids = [
        citation
        for row in rows
        for citation in (row.get("invalid_citation_ids") or [])
    ]
    return {
        "expected_decision_counts": expected_counts,
        "actual_decision_counts": actual_counts,
        "decision_expectations_evaluated": len(evaluated),
        "decision_expectations_passed": passed,
        "decision_expectations_failed": len(evaluated) - passed,
        "decision_expectation_match_rate": round(passed / len(evaluated), 4) if evaluated else None,
        "expected_clarifications": len(expected_clarifications),
        "expected_clarifications_correct": sum(
            row.get("decision_match") is True for row in expected_clarifications
        ),
        "citation_count": len(citation_ids),
        "invalid_citation_count": len(invalid_ids),
        "rows_with_missing_citations": sum(
            row.get("citation_validation") == "missing" for row in rows
        ),
        "latency_ms": latency_summary,
    }


def build_metrics(judge_model: str, embedding_model: str, api_key: str):
    """Create RAGAS 0.4 collection metrics using the existing OpenAI key."""
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for a RAGAS run.")
    try:
        from openai import AsyncOpenAI
        from ragas.embeddings.base import embedding_factory
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
            FactualCorrectness,
        )
    except ImportError as error:
        raise RuntimeError(
            "Evaluation packages are missing. Run: "
            "python -m pip install -r requirements-eval.txt"
        ) from error

    import httpx
    client = AsyncOpenAI(api_key=api_key, http_client=httpx.AsyncClient(
        event_hooks={"response": [record_provider_response]},
    ))
    llm = llm_factory(judge_model, client=client)
    embeddings = embedding_factory(
        "openai", model=embedding_model, client=client, interface="modern"
    )
    return {
        "faithfulness": Faithfulness(llm=llm),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
        "response_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "factual_correctness": FactualCorrectness(llm=llm, mode="f1"),
    }, client


@trace_component("finrag.evaluation.ragas_score")
async def score_one(
    metrics: dict[str, Any],
    *,
    question: str,
    answer: str,
    contexts: list[str],
    reference: str,
) -> tuple[dict[str, float | None], dict[str, str]]:
    """Score one runtime result; one failed metric does not erase the others."""
    async def score_with_retry(name: str, **kwargs: Any) -> Any:
        for attempt in range(3):
            try:
                return await metrics[name].ascore(**kwargs)
            except Exception as error:
                if "Error code: 429" not in str(error) or attempt == 2:
                    raise
                match = re.search(r"try again in ([\d.]+)(ms|s)", str(error), re.I)
                delay = 2.0
                if match:
                    delay = float(match.group(1))
                    if match.group(2).lower() == "ms":
                        delay /= 1000
                await asyncio.sleep(min(10.0, delay + 1.0))

    calls = {
        "faithfulness": score_with_retry(
            "faithfulness", user_input=question, response=answer, retrieved_contexts=contexts
        ),
        "context_precision": score_with_retry(
            "context_precision", user_input=question, reference=reference, retrieved_contexts=contexts
        ),
        "context_recall": score_with_retry(
            "context_recall", user_input=question, reference=reference, retrieved_contexts=contexts
        ),
        "response_relevancy": score_with_retry(
            "response_relevancy", user_input=question, response=answer
        ),
        "factual_correctness": score_with_retry(
            "factual_correctness", response=answer, reference=reference
        ),
    }
    completed = await asyncio.gather(*calls.values(), return_exceptions=True)
    scores: dict[str, float | None] = {}
    errors: dict[str, str] = {}
    for name, result in zip(calls, completed):
        if isinstance(result, BaseException):
            scores[name] = None
            errors[name] = f"{type(result).__name__}: {result}"
        else:
            scores[name] = _number(result)
    return scores, errors


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answered_rows = [
        row
        for row in rows
        if row.get("decision") == "answered" and not row.get("error")
    ]
    means: dict[str, float | None] = {}
    counts: dict[str, int] = {}
    for name in METRIC_NAMES:
        values = [
            float(row[name])
            for row in answered_rows
            if isinstance(row.get(name), (int, float))
            and math.isfinite(float(row[name]))
        ]
        counts[name] = len(values)
        means[name] = round(fmean(values), 4) if values else None

    available = [value for value in means.values() if value is not None]
    answered = sum(row.get("decision") == "answered" for row in rows)
    scored_rows = sum(
        any(isinstance(row.get(name), (int, float)) for name in METRIC_NAMES)
        for row in answered_rows
    )
    final_score = round(fmean(available), 4) if available else None
    answer_success_rate = round(answered / len(rows), 4) if rows else None
    return {
        "questions_evaluated": len(rows),
        "answered": answered,
        "answer_success_rate": answer_success_rate,
        "scored_answered_rows": scored_rows,
        "metric_coverage_rate": (
            round(scored_rows / len(rows), 4) if rows else None
        ),
        "metric_means": means,
        "metric_counts": counts,
        "final_ragas_score": final_score,
        "coverage_adjusted_score": (
            round(final_score * answer_success_rate, 4)
            if final_score is not None and answer_success_rate is not None
            else None
        ),
        "final_score_note": (
            "Unweighted mean of the five requested RAGAS metrics over answered "
            "rows only; answer coverage and the coverage-adjusted score are "
            "reported separately."
        ),
        "scope_match_rate": (
            round(sum(row.get("scope_match") is True for row in rows) / len(rows), 4)
            if rows
            else None
        ),
        "rows_with_errors": sum(
            bool(row.get("error") or row.get("errors") or row.get("metric_errors"))
            for row in rows
        ),
        "generation_usage": summarize_usage(rows, "llm_usage"),
        "judge_usage": summarize_usage(rows, "judge_usage"),
        **_decision_diagnostics(rows),
    }


def summarize_usage(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    usage = [row[key] for row in rows if isinstance(row.get(key), dict)]
    costs = [item["estimated_cost_usd"] for item in usage if isinstance(item.get("estimated_cost_usd"), (int, float))]
    embedding_costs = [item["embedding_cost_usd"] for item in usage if isinstance(item.get("embedding_cost_usd"), (int, float))]
    stages: dict[str, dict] = {}
    for item in usage:
        for name, metrics in item.get("stages", {}).items():
            aggregate = stages.setdefault(name, {})
            for field, value in metrics.items():
                if isinstance(value, bool):
                    aggregate[field] = aggregate.get(field, True) and value
                elif isinstance(value, (int, float)):
                    if field not in aggregate or aggregate[field] is not None:
                        aggregate[field] = round(aggregate.get(field, 0) + value, 8)
                elif value is None:
                    aggregate[field] = None
    return {
        **{field: sum(int(item.get(field) or 0) for item in usage)
           for field in ("llm_calls", "input_tokens", "cached_input_tokens", "output_tokens", "total_tokens", "embedding_calls", "embedding_input_tokens")},
        "rows_recorded": len(usage), "stages": stages,
        "estimated_cost_usd": round(sum(costs), 8) if usage and len(costs) == len(usage) else None,
        "cost_complete": bool(usage) and len(costs) == len(usage),
        "embedding_cost_usd": round(sum(embedding_costs), 8) if len(embedding_costs) == len(usage) and usage else None,
    }


def summarize_generation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answered = sum(row.get("decision") == "answered" for row in rows)
    passed = sum(row.get("passed") is True for row in rows)
    successful_answers = sum(
        row.get("decision") == "answered" and row.get("passed") is True
        for row in rows
    )
    usage_summary = summarize_usage(rows, "llm_usage")
    if usage_summary["rows_recorded"] != len(rows):
        usage_summary.update(estimated_cost_usd=None, cost_complete=False)
    return {
        "questions_generated": len(rows),
        "answered": answered,
        "answered_successfully": successful_answers,
        "answer_success_rate": round(successful_answers / len(rows), 4) if rows else None,
        "rows_with_errors": sum(bool(row.get("error") or row.get("errors")) for row in rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "llm_usage": usage_summary,
        **_decision_diagnostics(rows),
    }


def save_checkpoint(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    index_version: str,
    judge_model: str,
    dataset: Path,
    mode: str = "live",
    score_input: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "2.0",
        "run": {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "index_version": index_version,
            "judge_model": judge_model,
            "dataset": str(dataset),
            "mode": mode,
            "evaluation_path": (
                "ChatService.ask"
                if mode in {"live", "generation"}
                else "saved answers"
            ),
            "score_input": str(score_input) if score_input is not None else None,
        },
        "summary": summarize_generation(rows) if mode == "generation" else summarize(rows),
        "results": rows,
    }
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _select_gold(
    records: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    if args.ids:
        selected = set(args.ids)
        unknown = selected - {int(record["id"]) for record in records}
        if unknown:
            raise ValueError(f"Unknown gold question IDs: {sorted(unknown)}")
        records = [record for record in records if int(record["id"]) in selected]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be greater than zero.")
        records = records[: args.limit]
    return records


def _generate_row(service: Any, record: dict[str, Any]) -> dict[str, Any]:
    expected_decision = _expected_decision(record)
    row: dict[str, Any] = {
        "id": record["id"],
        "category": record["category"],
        "question": record["question"],
        "reference": str(record.get("reference") or ""),
        "expected_decision": expected_decision,
        "expected_outcome": _expected_outcome(record),
        "decision": "error",
        "decision_match": False,
        "passed": False,
        "answer": "",
        "retrieved_contexts": [],
        "retrieved_sources": [],
        "scope_match": False,
        "citations": [],
        "citation_ids": [],
        "valid_citation_ids": [],
        "invalid_citation_ids": [],
        "citation_validation": "missing",
        "tool_trace": {"calls": [], "results": [], "calculator_results": []},
        "errors": [],
        "llm_usage": None,
    }
    started = perf_counter()
    with collect_usage() as usage:
        try:
            result = service.ask(str(record["question"]))
            decision = getattr(result.decision, "value", str(result.decision))
            trace = getattr(result, "trace", {})
            trace = trace if isinstance(trace, dict) else {}
            contexts = [
                str(
                    source.get("evidence_text")
                    or source.get("full_evidence_text")
                    or ""
                ).strip()
                for source in (result.sources or [])
                if isinstance(source, dict)
            ]
            contexts = [context for context in contexts if context]
            sources = [
                {
                    key: source.get(key)
                    for key in (
                        "id", "ticker", "fiscal_year", "doc_type",
                        "source", "section", "pdf_page", "parent_id",
                        "dense_score", "lexical_score", "rrf_score",
                        "rerank_score", "score",
                    )
                }
                for source in (result.sources or [])
                if isinstance(source, dict)
            ]
            answer = str(result.answer or "")
            citation_data = _citation_diagnostics(answer, sources)
            tool_trace = _tool_trace(trace)
            row.update(
                {
                    "decision": decision,
                    "answer": answer,
                    "retrieved_contexts": contexts,
                    "retrieved_sources": sources,
                    "scope_match": _scope_matches(record["scope"], result.scope),
                    **citation_data,
                    "tool_trace": tool_trace,
                    "trace": _compact_runtime_trace(trace),
                }
            )
            row["errors"] = _collect_errors(trace, None)
            row["decision_match"] = (
                decision == expected_decision and not _has_generation_errors(row)
            )
            row["passed"] = row["decision_match"]
            if not _clean_answer(row["answer"]):
                raise RuntimeError("ChatService returned an empty answer.")
            if decision == "answered" and not contexts:
                raise RuntimeError("ChatService returned no retrieved evidence text.")
        except Exception as error:
            error_info = classify_runtime_error(error)
            row["error"] = f"{type(error).__name__}: {error}"[:500]
            row["errors"] = _collect_errors(row.get("trace", {}), None)
            failure = {
                "component": "evaluation",
                "message": row["error"],
                "error_code": error_info.code,
                "error_type": type(error).__name__,
                "retryable": error_info.retryable,
            }
            if error_info.detail:
                failure["error_detail"] = error_info.detail
            row["errors"].insert(0, failure)
            row["decision_match"] = False
            row["passed"] = False
        finally:
            generation_latency_ms = round((perf_counter() - started) * 1000, 2)
            row["generation_latency_ms"] = generation_latency_ms
            row["llm_usage"] = asdict(
                usage.snapshot(latency_ms=generation_latency_ms)
            )
            row["scoring_latency_ms"] = None
            row["total_latency_ms"] = generation_latency_ms
    return row


def _fresh_score_row(saved: dict[str, Any]) -> dict[str, Any]:
    """Copy generation fields without carrying scores from another run."""
    fields = (
        "id", "category", "question", "reference", "expected_decision",
        "expected_outcome", "decision", "decision_match", "passed", "answer",
        "retrieved_contexts", "retrieved_sources", "scope_match", "citations",
        "citation_ids", "valid_citation_ids", "invalid_citation_ids",
        "citation_validation", "tool_trace", "trace", "errors",
        "generation_latency_ms", "scoring_latency_ms", "total_latency_ms",
        "llm_usage",
    )
    row = {field: saved.get(field) for field in fields}
    row["expected_decision"] = row.get("expected_decision") or Decision.ANSWERED.value
    row["expected_outcome"] = row.get("expected_outcome") or row.get("reference") or (
        f"Expected decision: {row['expected_decision']}"
    )
    if saved.get("error"):
        row["error"] = str(saved["error"])
    row["errors"] = [
        error
        for error in (row.get("errors") or [])
        if not (
            isinstance(error, dict)
            and str(error.get("component", "")).startswith("ragas.")
        )
    ]
    row["tool_trace"] = row.get("tool_trace") or {
        "calls": [], "results": [], "calculator_results": []
    }
    if row.get("citation_ids") is None:
        citation_data = _citation_diagnostics(
            str(row.get("answer") or ""), row.get("retrieved_sources") or []
        )
        row.update(citation_data)
    row["decision_match"] = (
        row.get("decision") == row["expected_decision"]
        and not _has_generation_errors(row)
    )
    row["passed"] = row["decision_match"]
    return row


async def _score_row(
    row: dict[str, Any], metrics: dict[str, Any]
) -> dict[str, Any]:
    if _has_generation_errors(row):
        row["score_status"] = "not_scored_generation_error"
        return row
    expected_decision = str(row.get("expected_decision") or Decision.ANSWERED.value)
    if expected_decision != Decision.ANSWERED.value:
        row["score_status"] = f"not_scored_expected_decision:{expected_decision}"
        return row
    if row.get("decision") != "answered":
        row["score_status"] = f"not_scored_decision:{row.get('decision')}"
        return row
    if not str(row.get("reference") or "").strip():
        row["score_status"] = "not_scored_missing_reference"
        return row
    answer = _clean_answer(str(row["answer"]))
    contexts = [
        context.strip()
        for context in row["retrieved_contexts"]
        if isinstance(context, str) and context.strip()
    ]
    if not answer:
        row["error"] = "RuntimeError: Saved answer is empty after cleaning."
        return row
    if not contexts:
        row["error"] = "RuntimeError: Saved result has no retrieved evidence text."
        return row
    scoring_started = perf_counter()
    with collect_usage() as judge_usage, usage_stage("ragas_judge"):
        scores, metric_errors = await score_one(
            metrics, question=str(row["question"]), answer=answer,
            contexts=contexts, reference=str(row["reference"]),
        )
    row.update(scores)
    row["score_status"] = "scored"
    row["scoring_latency_ms"] = round((perf_counter() - scoring_started) * 1000, 2)
    row["judge_usage"] = asdict(judge_usage.snapshot(latency_ms=row["scoring_latency_ms"]))
    generation_latency = row.get("generation_latency_ms")
    row["total_latency_ms"] = round(
        float(generation_latency or 0) + row["scoring_latency_ms"], 2
    )
    if metric_errors:
        row["metric_errors"] = metric_errors
        row.setdefault("errors", []).extend(
            {"component": f"ragas.{name}", "message": message}
            for name, message in metric_errors.items()
        )
    return row


@trace_component("finrag.evaluation.ragas_run", root=True)
async def run(args: argparse.Namespace) -> int:
    if args.validate_only and (args.generate_only or args.score_input):
        raise ValueError(
            "--validate-only cannot be combined with --generate-only or --score-input."
        )
    dataset = Path(args.dataset).resolve()
    records = _select_gold(load_gold(dataset), args)
    print(f"PASS: validated {len(records)} unique gold questions.")
    if args.validate_only:
        return 0

    output = Path(
        args.output
        or (DEFAULT_GENERATION_OUTPUT if args.generate_only else DEFAULT_OUTPUT)
    ).resolve()
    score_input = Path(args.score_input).resolve() if args.score_input else None
    if score_input is not None and output == score_input:
        raise ValueError("--output must be different from --score-input.")

    if score_input is not None:
        saved_rows = load_generated_answers(score_input, records)
        from src.config import EMBEDDING_MODEL, OPENAI_API_KEY

        metrics, judge_client = build_metrics(
            args.judge_model,
            EMBEDDING_MODEL,
            OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", ""),
        )
        rows: list[dict[str, Any]] = []
        try:
            for position, saved in enumerate(saved_rows, 1):
                print(f"[{position}/{len(saved_rows)}] Q{int(saved['id']):03d}")
                row = await _score_row(_fresh_score_row(saved), metrics)
                rows.append(row)
                save_checkpoint(
                    output,
                    rows=rows,
                    index_version=args.index_version,
                    judge_model=args.judge_model,
                    dataset=dataset,
                    mode="score_saved",
                    score_input=score_input,
                )
        finally:
            await judge_client.close()
        print(json.dumps(summarize(rows), indent=2))
        print(f"Detailed results: {output}")
        return 0 if all(row.get("decision_match") is True for row in rows) else 1

    # Import the application only when new runtime answers are required.
    from src.app.chat_service import ChatService

    service = None
    metrics = None
    judge_client = None
    rows = []
    mode = "generation" if args.generate_only else "live"
    try:
        service = ChatService(index_version=args.index_version)
        if not args.generate_only:
            from src.config import EMBEDDING_MODEL, OPENAI_API_KEY

            metrics, judge_client = build_metrics(
                args.judge_model,
                EMBEDDING_MODEL,
                OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", ""),
            )
        for position, record in enumerate(records, 1):
            print(f"[{position}/{len(records)}] Q{int(record['id']):03d}")
            row = _generate_row(service, record)
            if metrics is not None:
                row = await _score_row(row, metrics)
            rows.append(row)
            save_checkpoint(
                output,
                rows=rows,
                index_version=args.index_version,
                judge_model=args.judge_model,
                dataset=dataset,
                mode=mode,
            )
    finally:
        if service is not None:
            service.close()
        if judge_client is not None:
            await judge_client.close()

    summary = summarize_generation(rows) if args.generate_only else summarize(rows)
    print(json.dumps(summary, indent=2))
    print(
        f"Generated answers: {output}"
        if args.generate_only
        else f"Detailed results: {output}"
    )
    return 0 if all(row.get("decision_match") is True for row in rows) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate live FinRAG answers with RAGAS 0.4.")
    parser.add_argument(
        "--index-version",
        default=os.getenv("FINRAG_INDEX_VERSION", "phase1-v2"),
        help="Existing FinRAG index version (default: FINRAG_INDEX_VERSION or phase1-v2).",
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument(
        "--output",
        help=(
            "Output JSON path. Defaults to fixed_answers.json for --generate-only "
            "and ragas_results.json otherwise."
        ),
    )
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--limit", type=int, help="Run only the first N questions.")
    parser.add_argument("--ids", nargs="+", type=int, help="Run only selected question IDs.")
    parser.add_argument("--validate-only", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate and save answers/contexts without building or running RAGAS metrics.",
    )
    modes.add_argument(
        "--score-input",
        metavar="PATH",
        help="Score a file created by --generate-only without running ChatService.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(run(parse_args())))
    except (RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
