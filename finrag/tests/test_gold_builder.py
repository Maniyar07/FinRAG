from __future__ import annotations

import unittest
from pathlib import Path

from evals.gold_builder.builder import (
    CalculationRecord,
    EvidenceItem,
    GoldDraft,
    QuestionSpec,
    _focused_excerpt,
    _query_match_scores,
    _readable_source_text,
    _normalize_related_calculation_operands,
    _repair_evidence_citations,
    _select_extraction_evidence,
    _recompute_calculations,
    calculate,
    load_specs,
    validate_draft,
)
from src.retrieval.query_understanding import understand_query


class GoldBuilderTests(unittest.TestCase):
    def test_checked_in_question_set_contains_30_unique_specs(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "evals"
            / "gold_builder"
            / "questions_30.json"
        )
        specs = load_specs(path)

        self.assertEqual(len(specs), 30)
        self.assertEqual(len({spec.id for spec in specs}), 30)
        self.assertTrue(any("TRANSCRIPT" in spec.scope["doc_types"] for spec in specs))
        self.assertTrue(any(len(spec.scope["doc_types"]) > 1 for spec in specs))
        self.assertTrue(any(spec.calculation_operations for spec in specs))

    def test_every_question_resolves_to_its_declared_runtime_scope(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "evals"
            / "gold_builder"
            / "questions_30.json"
        )
        for spec in load_specs(path):
            with self.subTest(question_id=spec.id):
                parsed = understand_query(spec.question)
                parsed_types = set(
                    parsed.requested_doc_types
                    or ((parsed.doc_type,) if parsed.doc_type else ())
                )
                self.assertEqual(set(parsed.tickers), set(spec.scope["tickers"]))
                self.assertEqual(set(parsed.years), set(spec.scope["years"]))
                self.assertEqual(parsed_types, set(spec.scope["doc_types"]))

    def test_calculation_uses_declared_operation_and_rounding(self) -> None:
        percentage = CalculationRecord(
            operation="percentage_of",
            operands=["32488", "281724"],
            result="11.5",
            unit="percent",
            decimal_places=1,
            explanation="R&D divided by revenue times 100.",
        )
        change = CalculationRecord(
            operation="percentage_change",
            operands=["10086", "6035"],
            result="67.1",
            unit="percent",
            decimal_places=1,
            explanation="2024 minus 2023, divided by 2023, times 100.",
        )

        self.assertEqual(str(calculate(percentage)), "11.5")
        self.assertEqual(str(calculate(change)), "67.1")

    def test_numeric_targets_outweigh_generic_financial_words(self) -> None:
        query = "Research and development 2025 32,488 consolidated statements"
        wrong = "Research tax credit in the 2025 consolidated statements was 1.5%."
        right = "Research and development in 2025 was $32,488 million."

        wrong_numeric, wrong_lexical = _query_match_scores(query, wrong)
        right_numeric, right_lexical = _query_match_scores(query, right)

        self.assertGreater(right_numeric, wrong_numeric)
        self.assertGreater(4 * right_numeric + right_lexical, 4 * wrong_numeric + wrong_lexical)

    def test_focused_excerpt_keeps_distant_target_rows(self) -> None:
        parent = (
            "[SPEAKER: AMY HOOD]\n"
            + ("unrelated filing text " * 500)
            + "Research and development was $32,488 million.\n"
            + ("more unrelated filing text " * 500)
            + "Total revenue was $281,724 million."
        )
        focused = _focused_excerpt(
            parent,
            [
                "Research and development 2025 32,488",
                "Revenue 2025 281,724",
            ],
        )

        self.assertLess(len(focused), len(parent))
        self.assertIn("[SPEAKER: AMY HOOD]", focused)
        self.assertIn("$32,488", focused)
        self.assertIn("$281,724", focused)

    def test_readable_source_text_preserves_table_values(self) -> None:
        raw = (
            "<tr><td>Research and development</td>"
            "<td><strong>$32,488</strong></td></tr>"
        )

        readable = _readable_source_text(raw)

        self.assertNotIn("<td>", readable)
        self.assertIn("Research and development", readable)
        self.assertIn("$32,488", readable)

    def test_model_arithmetic_is_recomputed_without_changing_operands(self) -> None:
        draft = GoldDraft.model_validate(
            {
                "supported": True,
                "reference": "The percentage was 0%.",
                "claims": [],
                "numeric_facts": [],
                "calculations": [
                    {
                        "operation": "percentage_of",
                        "operands": ["32488", "281724"],
                        "result": "0",
                        "unit": "percent",
                        "decimal_places": 1,
                        "explanation": "R&D divided by revenue.",
                    }
                ],
                "limitations": [],
            }
        )

        corrected = _recompute_calculations(draft)

        self.assertEqual(corrected.calculations[0].operands, ["32488", "281724"])
        self.assertEqual(corrected.calculations[0].result, "11.5")

    def test_paired_change_uses_same_new_and_old_operands_as_difference(self) -> None:
        spec = self._spec().model_copy(
            update={"calculation_operations": ["difference", "percentage_change"]}
        )
        draft = GoldDraft.model_validate(
            {
                "supported": True,
                "reference": "Liabilities increased.",
                "claims": [],
                "numeric_facts": [],
                "calculations": [
                    {
                        "operation": "difference",
                        "operands": ["141218", "125286"],
                        "result": "15932",
                        "unit": "million",
                        "decimal_places": 0,
                        "explanation": "2025 minus 2024.",
                    },
                    {
                        "operation": "percentage_change",
                        "operands": ["15932", "125286"],
                        "result": "0",
                        "unit": "percent",
                        "decimal_places": 1,
                        "explanation": "Incorrect model operands.",
                    },
                ],
                "limitations": [],
            }
        )

        normalized = _normalize_related_calculation_operands(spec, draft)
        corrected = _recompute_calculations(normalized)

        self.assertEqual(
            corrected.calculations[1].operands, ["141218", "125286"]
        )
        self.assertEqual(corrected.calculations[1].result, "12.7")

    def test_quote_citation_is_attached_to_matching_eligible_evidence(self) -> None:
        evidence = self._evidence() + [
            self._evidence()[0].model_copy(
                update={
                    "id": "E2",
                    "text": "Management said revenue was exactly $46.7 billion.",
                    "lexical_score": 2.0,
                }
            )
        ]
        draft = GoldDraft.model_validate(
            {
                "supported": True,
                "reference": "Revenue was USD 46.7 billion.",
                "claims": [
                    {
                        "claim": "Revenue was USD 46.7 billion.",
                        "requirement_ids": ["reported_revenue"],
                        "evidence_ids": ["E1"],
                        "evidence_quotes": [
                            "Management said revenue was exactly $46.7 billion."
                        ],
                    }
                ],
                "numeric_facts": [],
                "calculations": [],
                "limitations": [],
            }
        )

        repaired = _repair_evidence_citations(draft, evidence)

        self.assertEqual(repaired.claims[0].evidence_ids, ["E1", "E2"])

    def test_model_added_terminal_quote_punctuation_is_removed(self) -> None:
        evidence = [
            self._evidence()[0].model_copy(
                update={
                    "text": "Both businesses continue to be supply constrained, and production is ramping."
                }
            )
        ]
        draft = GoldDraft.model_validate(
            {
                "supported": True,
                "reference": "Both businesses were supply constrained.",
                "claims": [
                    {
                        "claim": "Both businesses were supply constrained.",
                        "requirement_ids": ["reported_revenue"],
                        "evidence_ids": ["E1"],
                        "evidence_quotes": [
                            "Both businesses continue to be supply constrained."
                        ],
                    }
                ],
                "numeric_facts": [],
                "calculations": [],
                "limitations": [],
            }
        )

        repaired = _repair_evidence_citations(draft, evidence)

        self.assertEqual(
            repaired.claims[0].evidence_quotes,
            ["Both businesses continue to be supply constrained"],
        )

    def test_model_added_leading_attribution_is_removed(self) -> None:
        evidence = [
            self._evidence()[0].model_copy(
                update={"text": "The company ended the year with strong revenue."}
            )
        ]
        draft = GoldDraft.model_validate(
            {
                "supported": True,
                "reference": "The company ended the year strongly.",
                "claims": [
                    {
                        "claim": "The company ended the year strongly.",
                        "requirement_ids": ["reported_revenue"],
                        "evidence_ids": ["E1"],
                        "evidence_quotes": ["we ended the year with strong revenue."],
                    }
                ],
                "numeric_facts": [],
                "calculations": [],
                "limitations": [],
            }
        )

        repaired = _repair_evidence_citations(draft, evidence)

        self.assertEqual(
            repaired.claims[0].evidence_quotes,
            ["ended the year with strong revenue"],
        )

    def test_extraction_evidence_is_limited_per_requirement(self) -> None:
        evidence = [
            self._evidence()[0].model_copy(
                update={"id": f"E{index}", "lexical_score": float(index)}
            )
            for index in range(1, 5)
        ]

        selected = _select_extraction_evidence(self._spec(), evidence)

        self.assertEqual(len(selected), 2)

    @staticmethod
    def _spec() -> QuestionSpec:
        return QuestionSpec.model_validate(
            {
                "id": 1,
                "category": "test",
                "question": "According to Microsoft's 2025 transcript, what was revenue?",
                "scope": {
                    "tickers": ["MSFT"],
                    "years": ["2025"],
                    "doc_types": ["TRANSCRIPT"],
                },
                "fact_years": ["2025"],
                "answer_requirements": ["State revenue"],
                "evidence_requests": [
                    {
                        "id": "reported_revenue",
                        "ticker": "MSFT",
                        "year": "2025",
                        "doc_type": "TRANSCRIPT",
                        "source": "MSFT_2025_TRANSCRIPT.html",
                        "search_queries": ["revenue 46.7 billion"],
                    }
                ],
                "calculation_operations": [],
            }
        )

    @staticmethod
    def _evidence() -> list[EvidenceItem]:
        return [
            EvidenceItem(
                id="E1",
                request_ids=["reported_revenue"],
                source="MSFT_2025_TRANSCRIPT.html",
                ticker="MSFT",
                fiscal_year="2025",
                doc_type="TRANSCRIPT",
                section="Q4 Earnings Call",
                parent_id="abc",
                lexical_score=1.0,
                lexical_coverage=1.0,
                text="Microsoft Cloud revenue was $46.7 billion in fiscal 2025.",
            )
        ]

    def test_valid_draft_requires_exact_grounding_and_source_coverage(self) -> None:
        draft = GoldDraft.model_validate(
            {
                "supported": True,
                "reference": "Microsoft Cloud revenue was USD 46.7 billion in fiscal 2025.",
                "claims": [
                    {
                        "claim": "Microsoft Cloud revenue was USD 46.7 billion.",
                        "requirement_ids": ["reported_revenue"],
                        "evidence_ids": ["E1"],
                        "evidence_quotes": ["Microsoft Cloud revenue was $46.7 billion"],
                    }
                ],
                "numeric_facts": [
                    {
                        "label": "Microsoft Cloud revenue",
                        "value": "46.7",
                        "unit": "USD billions",
                        "period": "fiscal 2025",
                        "evidence_ids": ["E1"],
                    }
                ],
                "calculations": [],
                "limitations": [],
            }
        )

        result = validate_draft(self._spec(), self._evidence(), draft)

        self.assertTrue(result.passed, result.errors)

    def test_validator_rejects_invented_quote_and_number(self) -> None:
        draft = GoldDraft.model_validate(
            {
                "supported": True,
                "reference": "Microsoft Cloud revenue was USD 99 billion in fiscal 2025.",
                "claims": [
                    {
                        "claim": "Revenue was USD 99 billion.",
                        "requirement_ids": ["reported_revenue"],
                        "evidence_ids": ["E1"],
                        "evidence_quotes": ["Revenue was exactly $99 billion."],
                    }
                ],
                "numeric_facts": [
                    {
                        "label": "Revenue",
                        "value": "99",
                        "unit": "USD billions",
                        "period": "fiscal 2025",
                        "evidence_ids": ["E1"],
                    }
                ],
                "calculations": [],
                "limitations": [],
            }
        )

        result = validate_draft(self._spec(), self._evidence(), draft)

        self.assertFalse(result.passed)
        self.assertTrue(any("quote_1:not_found" in error for error in result.errors))
        self.assertTrue(any("value_not_found:99" in error for error in result.errors))
        self.assertIn("reference_number_not_supported:99", result.errors)

    def test_validator_rejects_calculation_operand_not_backed_by_fact(self) -> None:
        spec = self._spec().model_copy(
            update={"calculation_operations": ["percentage_of"]}
        )
        draft = GoldDraft.model_validate(
            {
                "supported": True,
                "reference": "Microsoft Cloud revenue represented 23.4% of USD 200 billion.",
                "claims": [
                    {
                        "claim": "Microsoft Cloud revenue was USD 46.7 billion.",
                        "requirement_ids": ["reported_revenue"],
                        "evidence_ids": ["E1"],
                        "evidence_quotes": ["Microsoft Cloud revenue was $46.7 billion"],
                    }
                ],
                "numeric_facts": [
                    {
                        "label": "Microsoft Cloud revenue",
                        "value": "46.7",
                        "unit": "USD billions",
                        "period": "fiscal 2025",
                        "evidence_ids": ["E1"],
                    }
                ],
                "calculations": [
                    {
                        "operation": "percentage_of",
                        "operands": ["46.7", "200"],
                        "result": "23.4",
                        "unit": "percent",
                        "decimal_places": 1,
                        "explanation": "Revenue divided by an invented denominator.",
                    }
                ],
                "limitations": [],
            }
        )

        result = validate_draft(spec, self._evidence(), draft)

        self.assertFalse(result.passed)
        self.assertTrue(
            any("operand_2:not_evidence_backed:200" in error for error in result.errors)
        )

    def test_validator_allows_percentage_formula_constant_in_reference(self) -> None:
        spec = self._spec().model_copy(
            update={"calculation_operations": ["percentage_of"]}
        )
        draft = GoldDraft.model_validate(
            {
                "supported": True,
                "reference": "USD 46.7 billion divided by USD 46.7 billion times 100 equals 100.0%.",
                "claims": [
                    {
                        "claim": "Microsoft Cloud revenue was USD 46.7 billion.",
                        "requirement_ids": ["reported_revenue"],
                        "evidence_ids": ["E1"],
                        "evidence_quotes": ["Microsoft Cloud revenue was $46.7 billion"],
                    }
                ],
                "numeric_facts": [
                    {
                        "label": "Microsoft Cloud revenue",
                        "value": "46.7",
                        "unit": "USD billions",
                        "period": "fiscal 2025",
                        "evidence_ids": ["E1"],
                    }
                ],
                "calculations": [
                    {
                        "operation": "percentage_of",
                        "operands": ["46.7", "46.7"],
                        "result": "100.0",
                        "unit": "percent",
                        "decimal_places": 1,
                        "explanation": "Revenue divided by itself times 100.",
                    }
                ],
                "limitations": [],
            }
        )

        result = validate_draft(spec, self._evidence(), draft)

        self.assertTrue(result.passed, result.errors)


if __name__ == "__main__":
    unittest.main()
