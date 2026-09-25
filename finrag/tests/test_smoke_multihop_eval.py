from __future__ import annotations

import unittest

from evals.smoke_multihop import evaluate_result, validate_cases
from src.schemas import ChatResult, Decision, Scope


def _case(expected: dict) -> dict:
    return {
        "id": "CASE_1",
        "category": "test",
        "question": "Compare the requested financial values.",
        "expected": expected,
    }


def _result(*, decision: Decision = Decision.ANSWERED) -> ChatResult:
    facts = [
        {
            "fact_id": "F1",
            "ticker": "AAA",
            "metric": "Total revenue",
            "period": "2025",
            "raw_value": "1,250",
        },
        {
            "fact_id": "F2",
            "ticker": "AAA",
            "metric": "Total revenue",
            "period": "2024",
            "raw_value": "1,000",
        },
    ]
    trace = {
        "reranker": {"configured": True},
        "retrieval": {
            "candidate_count": 12,
            "reranker_applied": True,
            "sources": [{"rerank_score": 0.9}],
        },
        "generation": {"mode": "verified_calculations_and_exact_quotes"},
        "multihop": {
            "selected": True,
            "issues": [],
            "execution": {
                "requirements": [{"facts": facts}],
                "calculations": [
                    {
                        "calculation_id": "aaa_revenue_change",
                        "input_fact_ids": ["F2", "F1"],
                        "operation": "percentage_change",
                        "result": "25.0001",
                        "result_unit": "percent",
                        "status": "completed",
                    }
                ],
            },
        },
    }
    return ChatResult(
        decision=decision,
        answer="AAA revenue was 1,250 and the percentage change was 25% [S1].",
        scope=Scope(("AAA",), ("2025",), "10K"),
        sources=[{"id": "S1", "ticker": "AAA", "fiscal_year": "2025", "doc_type": "10K"}],
        trace=trace,
    )


class RegressionCaseValidationTests(unittest.TestCase):
    def test_valid_case_schema_is_normalized(self) -> None:
        cases = validate_cases(
            [{"id": " A1 ", "question": "  Find   revenue. ", "expected": {"decision": "answered"}}]
        )

        self.assertEqual(cases[0]["id"], "A1")
        self.assertEqual(cases[0]["question"], "Find revenue.")

    def test_duplicate_ids_are_rejected(self) -> None:
        payload = [
            {"id": "A1", "question": "Find revenue.", "expected": {"decision": "answered"}},
            {"id": "A1", "question": "Find income.", "expected": {"decision": "answered"}},
        ]

        with self.assertRaisesRegex(ValueError, "duplicate id"):
            validate_cases(payload)

    def test_unknown_decision_is_rejected(self) -> None:
        payload = [{"id": "A1", "question": "Find revenue.", "expected": {"decision": "maybe"}}]

        with self.assertRaisesRegex(ValueError, "unsupported decision"):
            validate_cases(payload)


class RegressionResultEvaluationTests(unittest.TestCase):
    def test_complete_matching_result_passes(self) -> None:
        case = _case(
            {
                "decision": "answered",
                "multihop": True,
                "scope_groups": [["AAA", "2025"]],
                "scope_doc_type": "10-K",
                "source_groups": [["AAA", "2025", "10-K"]],
                "answer_contains": ["1,250", "25%"],
                "facts": [
                    {"ticker": "AAA", "period": "2025", "metric_contains": "revenue", "raw_value": "$1,250"}
                ],
                "calculations": [
                    {
                        "ticker": "AAA",
                        "operation": "percentage_change",
                        "result": "25",
                        "tolerance": "0.001",
                        "status": "completed",
                    }
                ],
                "no_issues": True,
                "generation_mode": "verified_calculations_and_exact_quotes",
                "max_elapsed_seconds": 2,
            }
        )

        evaluated = evaluate_result(case, _result(), 1.25)

        self.assertTrue(evaluated["passed"])
        self.assertEqual(evaluated["failures"], [])
        self.assertTrue(evaluated["actual"]["reranker"]["applied"])

    def test_mismatches_are_reported_independently(self) -> None:
        case = _case(
            {
                "decision": "insufficient_evidence",
                "multihop": False,
                "source_groups": [["BBB", "2025", "10K"]],
                "answer_contains": ["missing phrase"],
                "calculations": [{"operation": "difference", "result": "99"}],
            }
        )

        evaluated = evaluate_result(case, _result(), 0.5)

        self.assertFalse(evaluated["passed"])
        self.assertIn("decision", evaluated["failures"])
        self.assertIn("multihop", evaluated["failures"])
        self.assertIn("source_groups", evaluated["failures"])
        self.assertIn("answer_contains:missing phrase", evaluated["failures"])
        self.assertIn("calculation", evaluated["failures"])

    def test_one_of_multiple_safe_decisions_can_pass(self) -> None:
        case = _case({"decision": ["answered", "insufficient_evidence"]})

        evaluated = evaluate_result(case, _result(decision=Decision.INSUFFICIENT_EVIDENCE), 0.1)

        self.assertTrue(evaluated["passed"])

    def test_missing_multihop_trace_is_treated_as_not_selected(self) -> None:
        result = ChatResult(
            decision=Decision.OUT_OF_SCOPE,
            answer="This request is outside the corpus.",
            trace={},
        )

        evaluated = evaluate_result(
            _case({"decision": "out_of_scope", "multihop": False}), result, 0.01
        )

        self.assertTrue(evaluated["passed"])
        self.assertFalse(evaluated["actual"]["multihop_selected"])


if __name__ == "__main__":
    unittest.main()
