from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from evals.evaluate_ragas import (
    _clean_answer,
    load_generated_answers,
    run,
    score_one,
)
from src.schemas import Decision, Scope


class _Metric:
    def __init__(self, *, fail_once: bool = False, error: str = "") -> None:
        self.calls = 0
        self.fail_once = fail_once
        self.error = error
        self.last_kwargs = {}

    async def ascore(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        if self.error:
            raise RuntimeError(self.error)
        if self.fail_once and self.calls == 1:
            raise RuntimeError("Error code: 429 - try again in 5ms")
        return SimpleNamespace(value=0.75)


def _gold_record() -> dict:
    return {
        "id": 1,
        "category": "A - Narrative facts",
        "question": "What was reported?",
        "reference": "The reference answer.",
        "scope": {
            "tickers": ["MSFT"],
            "years": ["2025"],
            "doc_types": ["10K"],
        },
    }


def _generated_row() -> dict:
    record = _gold_record()
    return {
        "id": record["id"],
        "category": record["category"],
        "question": record["question"],
        "reference": record["reference"],
        "decision": "answered",
        "answer": "The generated answer. [S1]",
        "retrieved_contexts": ["Supporting evidence."],
        "retrieved_sources": [{"id": "S1"}],
        "scope_match": True,
    }


def _args(**overrides) -> SimpleNamespace:
    values = {
        "dataset": "gold.json",
        "ids": None,
        "limit": None,
        "validate_only": False,
        "generate_only": False,
        "score_input": None,
        "output": "output.json",
        "index_version": "test-index",
        "judge_model": "test-judge",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class EvaluateRagasTests(unittest.TestCase):
    def test_clean_answer_removes_compact_citations_and_source_appendix(self) -> None:
        answer = (
            "Revenue was $245,122 million. [S1][S2]\n\n"
            "**Sources**\n\n"
            "- [S1] MSFT 2024 10-K — Item 8; p. 56\n"
            "- [S2] MSFT 2024 10-K — Item 7; p. 39"
        )

        self.assertEqual(
            _clean_answer(answer),
            "Revenue was $245,122 million.",
        )

    def test_clean_answer_removes_legacy_expanded_citation(self) -> None:
        self.assertEqual(
            _clean_answer("Revenue was $245,122 million [S1: Item 8, page 56]."),
            "Revenue was $245,122 million.",
        )

    def test_clean_answer_preserves_financial_values_lists_and_tables(self) -> None:
        answer = (
            "Values:\n\n"
            "1. Cash was $30,242 million. [S1]\n"
            "2. Investments were $64,323 million. [S1]\n\n"
            "| Metric | 2025 |\n"
            "|---|---:|\n"
            "| Total | $94,565 million |"
        )
        expected = (
            "Values:\n\n"
            "1. Cash was $30,242 million.\n"
            "2. Investments were $64,323 million.\n\n"
            "| Metric | 2025 |\n"
            "|---|---:|\n"
            "| Total | $94,565 million |"
        )

        self.assertEqual(_clean_answer(answer), expected)

    def test_clean_answer_leaves_plain_answer_unchanged(self) -> None:
        answer = "Microsoft reported net income of $101,832 million."
        self.assertEqual(_clean_answer(answer), answer)

    def test_rate_limit_is_retried_without_erasing_other_metrics(self) -> None:
        names = (
            "faithfulness", "context_precision", "context_recall",
            "response_relevancy", "factual_correctness",
        )
        metrics = {name: _Metric(fail_once=name == "context_precision") for name in names}
        with patch("evals.evaluate_ragas.asyncio.sleep", new_callable=AsyncMock) as sleep:
            scores, errors = asyncio.run(score_one(
                metrics, question="question", answer="answer",
                contexts=["context"], reference="reference",
            ))
        self.assertEqual(scores, {name: 0.75 for name in names})
        self.assertEqual(errors, {})
        self.assertEqual(metrics["context_precision"].calls, 2)
        sleep.assert_awaited_once()

    def test_non_rate_limit_failure_stays_a_metric_error(self) -> None:
        names = (
            "faithfulness", "context_precision", "context_recall",
            "response_relevancy", "factual_correctness",
        )
        metrics = {name: _Metric(error="invalid response" if name == "context_precision" else "") for name in names}
        scores, errors = asyncio.run(score_one(
            metrics, question="question", answer="answer",
            contexts=["context"], reference="reference",
        ))
        self.assertIsNone(scores["context_precision"])
        self.assertIn("invalid response", errors["context_precision"])
        self.assertEqual(metrics["context_precision"].calls, 1)

    def test_generated_answer_file_is_matched_to_gold(self) -> None:
        payload = {
            "run": {"mode": "generation"},
            "results": [_generated_row()],
        }
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "read_text", return_value=json.dumps(payload)),
        ):
            rows = load_generated_answers(Path("fixed.json"), [_gold_record()])

        self.assertEqual(rows, payload["results"])

    def test_score_input_rejects_incomplete_or_scored_files(self) -> None:
        row = _generated_row()
        row["retrieved_contexts"] = []
        payload = {"run": {"mode": "live"}, "results": [row]}
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "read_text", return_value=json.dumps(payload)),
        ):
            with self.assertRaisesRegex(ValueError, "created by --generate-only"):
                load_generated_answers(Path("scored.json"), [_gold_record()])

        payload["run"]["mode"] = "generation"
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "read_text", return_value=json.dumps(payload)),
        ):
            with self.assertRaisesRegex(ValueError, "non-empty answer and context"):
                load_generated_answers(Path("fixed.json"), [_gold_record()])

    def test_generate_only_saves_answers_without_building_metrics(self) -> None:
        source = {
            "id": "S1",
            "ticker": "MSFT",
            "fiscal_year": "2025",
            "doc_type": "10K",
            "evidence_text": "Supporting evidence.",
        }
        result = SimpleNamespace(
            decision=Decision.ANSWERED,
            answer="The generated answer. [S1]",
            sources=[source],
            scope=Scope(("MSFT",), ("2025",), "10K"),
        )
        service = Mock()
        service.ask.return_value = result

        with (
            patch("evals.evaluate_ragas.load_gold", return_value=[_gold_record()]),
            patch("src.app.chat_service.ChatService", return_value=service),
            patch("evals.evaluate_ragas.build_metrics") as build_metrics,
            patch("evals.evaluate_ragas.save_checkpoint") as save_checkpoint,
        ):
            status = asyncio.run(run(_args(generate_only=True)))

        self.assertEqual(status, 0)
        build_metrics.assert_not_called()
        service.ask.assert_called_once_with("What was reported?")
        service.close.assert_called_once_with()
        saved = save_checkpoint.call_args.kwargs
        self.assertEqual(saved["mode"], "generation")
        self.assertEqual(saved["rows"][0]["retrieved_contexts"], ["Supporting evidence."])

    def test_score_input_does_not_import_or_call_chat_service(self) -> None:
        names = (
            "faithfulness", "context_precision", "context_recall",
            "response_relevancy", "factual_correctness",
        )
        metrics = {name: _Metric() for name in names}
        judge_client = SimpleNamespace(close=AsyncMock())

        with (
            patch("evals.evaluate_ragas.load_gold", return_value=[_gold_record()]),
            patch(
                "evals.evaluate_ragas.load_generated_answers",
                return_value=[_generated_row()],
            ),
            patch(
                "evals.evaluate_ragas.build_metrics",
                return_value=(metrics, judge_client),
            ),
            patch("evals.evaluate_ragas.save_checkpoint") as save_checkpoint,
            patch.dict(sys.modules, {"src.app.chat_service": None}),
        ):
            status = asyncio.run(run(_args(score_input="fixed.json")))

        self.assertEqual(status, 0)
        self.assertTrue(all(metric.calls == 1 for metric in metrics.values()))
        judge_client.close.assert_awaited_once()
        saved = save_checkpoint.call_args.kwargs
        self.assertEqual(saved["mode"], "score_saved")
        self.assertEqual(
            metrics["factual_correctness"].last_kwargs["response"],
            "The generated answer.",
        )

    def test_validate_only_cannot_be_combined_with_an_execution_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "--validate-only cannot be combined"):
            asyncio.run(run(_args(validate_only=True, generate_only=True)))


if __name__ == "__main__":
    unittest.main()
