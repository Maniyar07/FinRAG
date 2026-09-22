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
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

        for field in ("category", "question", "reference"):
            if not str(record.get(field, "")).strip():
                errors.append(f"record {record_id or position}: missing {field}")

        scope = record.get("scope")
        if not isinstance(scope, dict):
            errors.append(f"record {record_id or position}: missing scope")
            continue
        for field in ("tickers", "years", "doc_types"):
            values = scope.get(field)
            if not isinstance(values, list) or not values:
                errors.append(f"record {record_id or position}: scope.{field} is empty")
        invalid_types = set(scope.get("doc_types") or []) - VALID_DOC_TYPES
        if invalid_types:
            errors.append(
                f"record {record_id or position}: invalid document types {sorted(invalid_types)}"
            )

    if errors:
        raise ValueError("Gold validation failed:\n- " + "\n- ".join(errors))
    return records


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
        for field in ("category", "question", "reference"):
            if row.get(field) != record[field]:
                raise ValueError(
                    f"Generated result {record_id} does not match gold field {field}."
                )
        contexts = row.get("retrieved_contexts")
        sources = row.get("retrieved_sources")
        has_generation_error = bool(str(row.get("error", "")).strip())
        if not isinstance(row.get("decision"), str) or not row["decision"].strip():
            raise ValueError(f"Generated result {record_id} has an invalid decision.")
        if not isinstance(row.get("scope_match"), bool):
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
        if not has_generation_error and (
            not row["answer"].strip() or not any(context.strip() for context in contexts)
        ):
            raise ValueError(
                f"Generated result {record_id} needs a non-empty answer and context."
            )
        selected.append(row)
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


def _scope_matches(expected: dict[str, Any], actual: Any) -> bool:
    expected_types = set(expected["doc_types"])
    actual_types = set(getattr(actual, "required_doc_types", ()) or ())
    if not actual_types and getattr(actual, "doc_type", None):
        actual_types = {actual.doc_type}
    return (
        set(getattr(actual, "tickers", ()) or ()) == set(expected["tickers"])
        and set(getattr(actual, "years", ()) or ()) == set(expected["years"])
        and actual_types == expected_types
    )


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

    client = AsyncOpenAI(api_key=api_key)
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
    means: dict[str, float | None] = {}
    for name in METRIC_NAMES:
        values = [
            float(row[name])
            for row in rows
            if isinstance(row.get(name), (int, float))
            and math.isfinite(float(row[name]))
        ]
        means[name] = round(fmean(values), 4) if values else None

    available = [value for value in means.values() if value is not None]
    answered = sum(row.get("decision") == "answered" for row in rows)
    return {
        "questions_evaluated": len(rows),
        "answered": answered,
        "answer_success_rate": round(answered / len(rows), 4) if rows else None,
        "metric_means": means,
        "final_ragas_score": round(fmean(available), 4) if available else None,
        "final_score_note": "Unweighted mean of the five requested RAGAS metrics.",
        "scope_match_rate": (
            round(sum(row.get("scope_match") is True for row in rows) / len(rows), 4)
            if rows
            else None
        ),
        "rows_with_errors": sum(bool(row.get("error") or row.get("metric_errors")) for row in rows),
    }


def summarize_generation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answered = sum(row.get("decision") == "answered" for row in rows)
    return {
        "questions_generated": len(rows),
        "answered": answered,
        "answer_success_rate": round(answered / len(rows), 4) if rows else None,
        "rows_with_errors": sum(bool(row.get("error")) for row in rows),
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
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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
    row: dict[str, Any] = {
        "id": record["id"],
        "category": record["category"],
        "question": record["question"],
        "reference": record["reference"],
        "decision": "error",
        "answer": "",
        "retrieved_contexts": [],
        "retrieved_sources": [],
        "scope_match": False,
    }
    try:
        result = service.ask(str(record["question"]))
        decision = getattr(result.decision, "value", str(result.decision))
        contexts = [
            str(source.get("evidence_text") or source.get("full_evidence_text") or "").strip()
            for source in result.sources
        ]
        contexts = [context for context in contexts if context]
        row.update(
            {
                "decision": decision,
                "answer": str(result.answer),
                "retrieved_contexts": contexts,
                "retrieved_sources": [
                    {
                        key: source.get(key)
                        for key in (
                            "id", "ticker", "fiscal_year", "doc_type",
                            "source", "section", "pdf_page", "parent_id",
                            "dense_score", "lexical_score", "rrf_score",
                            "rerank_score", "score",
                        )
                    }
                    for source in result.sources
                ],
                "scope_match": _scope_matches(record["scope"], result.scope),
            }
        )
        if not _clean_answer(row["answer"]):
            raise RuntimeError("ChatService returned an empty answer.")
        if not contexts:
            raise RuntimeError("ChatService returned no retrieved evidence text.")
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def _fresh_score_row(saved: dict[str, Any]) -> dict[str, Any]:
    """Copy generation fields without carrying scores from another run."""
    fields = (
        "id", "category", "question", "reference", "decision", "answer",
        "retrieved_contexts", "retrieved_sources", "scope_match",
    )
    row = {field: saved[field] for field in fields}
    if saved.get("error"):
        row["error"] = str(saved["error"])
    return row


async def _score_row(
    row: dict[str, Any], metrics: dict[str, Any]
) -> dict[str, Any]:
    if row.get("error"):
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
    scores, metric_errors = await score_one(
        metrics,
        question=str(row["question"]),
        answer=answer,
        contexts=contexts,
        reference=str(row["reference"]),
    )
    row.update(scores)
    if metric_errors:
        row["metric_errors"] = metric_errors
    return row


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
        return 0

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
    return 0


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
