from __future__ import annotations

import unittest

from src.generation.answer_generator import AnswerGenerator
from src.schemas import RetrievalBundle, Scope


class FakeChain:
    def __init__(self, outputs: list[dict | Exception]):
        self.outputs = list(outputs)
        self.calls = 0

    def invoke(self, values: dict) -> dict:
        del values
        output = self.outputs[self.calls]
        self.calls += 1
        if isinstance(output, Exception):
            raise output
        return output


def bundle() -> RetrievalBundle:
    return RetrievalBundle(
        context=(
            "[SOURCE S1 | TICKER: MSFT | FISCAL YEAR: 2024 | TYPE: TRANSCRIPT]\n"
            "Capital expenditures included cloud and AI infrastructure investment."
        ),
        sources=[
            {
                "id": "S1",
                "ticker": "MSFT",
                "fiscal_year": "2024",
                "doc_type": "TRANSCRIPT",
                "section": "Prepared Remarks",
                "pdf_page": "not available",
                "call_date": "2024-07-30",
                "source": "MSFT_2024_TRANSCRIPT.html",
            }
        ],
        scope=Scope(
            tickers=("MSFT",),
            years=("2024",),
            doc_type="TRANSCRIPT",
        ),
        candidate_count=1,
    )


class AnswerGeneratorTests(unittest.TestCase):
    def test_wrong_table_value_is_repaired_before_answering(self) -> None:
        evidence = (
            "<table><tr><th></th><th>2024</th><th>2023</th></tr>"
            "<tr><td>Total automotive revenues</td><td>77,070</td>"
            "<td>82,419</td></tr></table>"
        )
        source = {
            "id": "S1", "ticker": "TSLA", "fiscal_year": "2024",
            "doc_type": "10K", "source": "TSLA_2024_10K.pdf",
            "pdf_page": "52", "section": "Item 8", "evidence_text": evidence,
        }
        test_bundle = RetrievalBundle(
            context=f"[SOURCE S1 | TICKER: TSLA | FISCAL YEAR: 2024]\n{evidence}",
            sources=[source], scope=Scope(("TSLA",), ("2024",), "10K"),
            candidate_count=1,
        )
        chain = FakeChain([
            {"answer": "Total automotive revenues were $87,604 million [S1].", "source_ids": ["S1"]},
            {"answer": "Total automotive revenues were $77,070 million [S1].", "source_ids": ["S1"]},
        ])
        result = AnswerGenerator(chain=chain).generate_with_trace(
            "What were total automotive revenues for 2024?", test_bundle
        )
        self.assertEqual(result.attempts, 2)
        self.assertIn("77,070", result.answer)
        self.assertNotIn("87,604", result.answer)

    def test_structured_id_prevents_false_rejection(self) -> None:
        chain = FakeChain(
            [
                {
                    "answer": "Management discussed AI infrastructure investment.",
                    "source_ids": ["S1"],
                }
            ]
        )
        result = AnswerGenerator(chain=chain).generate_with_trace(
            "What did Microsoft management say about AI infrastructure?", bundle()
        )
        self.assertEqual(result.attempts, 1)
        self.assertIn("MSFT 2024 Transcript", result.answer)
        self.assertEqual(result.validation_reason, "valid_structured_ids_appended")

    def test_compound_answer_requires_inline_citations(self) -> None:
        uncited = (
            "### Results\n\n- 2024 value was 100.\n- 2025 value was 110.\n\n"
            "### Explanation\n\nManagement discussed infrastructure investment and "
            "capacity constraints across several business units in detail. " * 4
        )
        chain = FakeChain(
            [
                {"answer": uncited, "source_ids": ["S1"]},
                {
                    "answer": (
                        "### Results\n\n- 2024 value was 100 [S1].\n"
                        "- 2025 value was 110 [S1].\n\n"
                        "### Explanation\n\nManagement discussed infrastructure "
                        "investment [S1]."
                    ),
                    "source_ids": ["S1"],
                },
            ]
        )

        result = AnswerGenerator(chain=chain).generate_with_trace(
            "Compare the values and explain the change.", bundle()
        )

        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.validation_reason, "valid_inline_citations")

    def test_transcript_commentary_rejects_company_10k_citation(self) -> None:
        sources = [
            {"id": "S1", "ticker": "TSLA", "doc_type": "10K"},
            {"id": "S2", "ticker": "TSLA", "doc_type": "TRANSCRIPT"},
        ]

        wrong = AnswerGenerator._requested_transcript_citation_error(
            "Explain TSLA management commentary from the transcript.",
            "- TSLA management highlighted AI spending [S1].",
            sources,
        )
        correct = AnswerGenerator._requested_transcript_citation_error(
            "Explain TSLA management commentary from the transcript.",
            "- TSLA management highlighted AI spending [S2].",
            sources,
        )

        self.assertEqual(wrong, "requested_transcript_citation_missing:TSLA")
        self.assertIsNone(correct)

    def test_narrative_coverage_requires_each_company_group(self) -> None:
        role = [
            {
                "requirement_id": "drivers",
                "evidence_type": "narrative",
                "document_type": "TRANSCRIPT",
            }
        ]
        sources = [
            {
                "id": "S1",
                "ticker": "MSFT",
                "fiscal_year": "2025",
                "evidence_requirements": role,
            },
            {
                "id": "S2",
                "ticker": "TSLA",
                "fiscal_year": "2025",
                "evidence_requirements": role,
            },
        ]

        error = AnswerGenerator._narrative_coverage_error(("S1",), sources)
        complete = AnswerGenerator._narrative_coverage_error(("S1", "S2"), sources)

        self.assertEqual(error, "missing_narrative_citation_groups:TSLA-2025")
        self.assertIsNone(complete)

    def test_invalid_first_answer_is_repaired_once(self) -> None:
        chain = FakeChain(
            [
                {"answer": "Draft without citations.", "source_ids": []},
                {"answer": "Capacity investment was discussed [S1].", "source_ids": ["S1"]},
            ]
        )
        result = AnswerGenerator(chain=chain).generate_with_trace(
            "What did Microsoft management say about AI infrastructure?", bundle()
        )
        self.assertEqual(result.attempts, 2)
        self.assertIn("MSFT 2024 Transcript", result.answer)
        self.assertEqual(chain.calls, 2)

    def test_two_invalid_answers_do_not_show_unrelated_evidence(self) -> None:
        chain = FakeChain(
            [
                {"answer": "Draft one.", "source_ids": []},
                {"answer": "Draft two.", "source_ids": []},
            ]
        )
        result = AnswerGenerator(chain=chain).generate_with_trace(
            "What did Microsoft management say about AI infrastructure?", bundle()
        )
        self.assertIn("validation failed", result.answer)
        self.assertNotIn("MSFT 2024 Transcript", result.answer)
        self.assertTrue(result.validation_reason.startswith("generation_validation_failed:"))

    def test_multihop_third_attempt_repairs_missing_narrative_group(self) -> None:
        role = [{"requirement_id": "explanation", "evidence_type": "narrative"}]
        test_bundle = RetrievalBundle(
            context="[SOURCE S1] Microsoft explanation.\n\n[SOURCE S2] Tesla explanation.",
            sources=[
                {"id": "S1", "ticker": "MSFT", "fiscal_year": "2025", "doc_type": "10K", "evidence_requirements": role},
                {"id": "S2", "ticker": "TSLA", "fiscal_year": "2025", "doc_type": "10K", "evidence_requirements": role},
            ],
            scope=Scope(("MSFT", "TSLA"), ("2025",), "10K"),
            candidate_count=2,
        )
        chain = FakeChain([
            {"answer": "Microsoft explanation [S1].", "source_ids": ["S1"]},
            {"answer": "Microsoft explanation [S1].", "source_ids": ["S1"]},
            {"answer": "Microsoft explanation [S1]. Tesla explanation [S2].", "source_ids": ["S1", "S2"]},
        ])
        result = AnswerGenerator(chain=chain).generate_with_trace(
            "Explain both companies' tax factors.",
            test_bundle,
            fail_closed_on_invalid=True,
        )
        self.assertEqual(result.attempts, 3)
        self.assertEqual(result.validation_reason, "valid_inline_citations")
        self.assertEqual(chain.calls, 3)

    def test_multihop_three_invalid_answers_fail_closed(self) -> None:
        chain = FakeChain([
            {"answer": "Uncited draft.", "source_ids": []},
            {"answer": "Uncited draft.", "source_ids": []},
            {"answer": "Uncited draft.", "source_ids": []},
        ])
        result = AnswerGenerator(chain=chain).generate_with_trace(
            "Explain the results.", bundle(), fail_closed_on_invalid=True
        )
        self.assertEqual(result.attempts, 3)
        self.assertTrue(result.validation_reason.startswith("generation_validation_failed:"))
        self.assertNotIn("most relevant retrieved evidence", result.answer)

    def test_multihop_rejects_end_only_citations_on_every_attempt(self) -> None:
        chain = FakeChain([
            {"answer": "Microsoft value was 100. Tesla value was 90.", "source_ids": ["S1"]},
            {"answer": "Microsoft value was 100. Tesla value was 90.", "source_ids": ["S1"]},
            {"answer": "Microsoft value was 100. Tesla value was 90.", "source_ids": ["S1"]},
        ])
        result = AnswerGenerator(chain=chain).generate_with_trace(
            "Compare Microsoft and Tesla values.", bundle(), fail_closed_on_invalid=True
        )
        self.assertEqual(result.attempts, 3)
        self.assertEqual(
            result.validation_reason,
            "generation_validation_failed:compound_answer_requires_inline_citations",
        )

    def test_future_outlook_cannot_explain_historical_decline(self) -> None:
        error = AnswerGenerator._forward_looking_as_historical_cause_error(
            "Tesla's 2025 decline was attributed to a change that is expected "
            "to impact margins next year [S2]."
        )
        commentary = AnswerGenerator._forward_looking_as_historical_cause_error(
            "Tesla discussed a change expected to impact future margins [S2]."
        )
        self.assertEqual(error, "forward_looking_as_historical_cause")
        self.assertIsNone(commentary)

    def test_structured_output_error_is_retried(self) -> None:
        chain = FakeChain(
            [
                ValueError("invalid structured response"),
                {"answer": "AI capacity investment was discussed [S1].", "source_ids": ["S1"]},
            ]
        )
        result = AnswerGenerator(chain=chain).generate_with_trace(
            "What did Microsoft management say about AI infrastructure?", bundle()
        )
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.validation_reason, "valid_inline_citations")
        self.assertIn("MSFT 2024 Transcript", result.answer)

    def test_table_wider_than_eight_columns_is_repaired(self) -> None:
        chain = FakeChain(
            [
                {
                    "answer": (
                        "| A | B | C | D | E | F | G | H | I |\n"
                        "|---|---|---|---|---|---|---|---|---|\n"
                        "| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |\n\n[S1]"
                    ),
                    "source_ids": ["S1"],
                },
                {
                    "answer": (
                        "| Issuance | Maturity | Stated rate | Effective rate |\n"
                        "|---|---|---:|---:|\n"
                        "| 2009 | 2039 | 5.20% | 5.24% |\n\n"
                        "| Issuance | 2024 balance | 2023 balance |\n"
                        "|---|---:|---:|\n"
                        "| 2009 | 520 | 520 |\n\n"
                        "Unit: USD millions. [S1]"
                    ),
                    "source_ids": ["S1"],
                },
            ]
        )
        result = AnswerGenerator(chain=chain).generate_with_trace(
            "Show all columns in the debt table.", bundle()
        )
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.validation_reason, "valid_inline_citations")
        self.assertIn("2024 balance", result.answer)
        self.assertEqual(chain.calls, 2)


if __name__ == "__main__":
    unittest.main()
