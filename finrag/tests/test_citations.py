from __future__ import annotations

import unittest

from src.generation.answer_guardrails import (
    UNVERIFIABLE_RESPONSE,
    normalize_table_currency,
    validate_answer_payload,
    validate_markdown_tables,
)
from src.generation.citations import expand_citations, strip_citations


class CitationTests(unittest.TestCase):
    def test_expands_source_id_with_required_metadata(self) -> None:
        sources = [
            {
                "id": "S1",
                "ticker": "MSFT",
                "fiscal_year": "2024",
                "doc_type": "10K",
                "section": "Item 8. Financial Statements",
                "pdf_page": "56",
                "source": "MSFT_2024_10K.pdf",
                "call_date": None,
            }
        ]
        expanded = expand_citations("Revenue was reported [S1].", sources)
        self.assertIn("MSFT 2024 10-K", expanded)
        self.assertIn("Item 8", expanded)
        self.assertIn("p. 56", expanded)
        self.assertIn("MSFT_2024_10K.pdf", expanded)

    def test_transcript_without_page_uses_call_date(self) -> None:
        sources = [
            {
                "id": "S2",
                "ticker": "TSLA",
                "fiscal_year": "2024",
                "doc_type": "TRANSCRIPT",
                "section": "Question and Answer",
                "pdf_page": "not available",
                "source": "TSLA_2024_TRANSCRIPT.html",
                "call_date": "2025-01-29",
            }
        ]
        expanded = expand_citations("Management discussed demand [S2].", sources)
        self.assertIn("call 2025-01-29", expanded)
        self.assertNotIn("p. not available", expanded)

    def test_history_citations_are_removed(self) -> None:
        value = strip_citations(
            "Revenue [S1] and income [S2: MSFT 2024 10-K | Item 8 | p. 56]."
        )
        self.assertNotIn("[S1]", value)
        self.assertNotIn("[S2:", value)

    def test_accepts_and_normalizes_common_citation_variations(self) -> None:
        sources = [{"id": "S1"}]
        result = validate_answer_payload("Capacity was constrained (S1).", sources)
        self.assertTrue(result.valid)
        self.assertIn("[S1]", result.answer)

    def test_appends_valid_structured_ids_when_inline_id_is_missing(self) -> None:
        sources = [{"id": "S1"}, {"id": "S2"}]
        result = validate_answer_payload(
            "Management discussed infrastructure investment.",
            sources,
            ["S1", "S2"],
        )
        self.assertTrue(result.valid)
        self.assertTrue(result.answer.endswith("[S1][S2]"))

    def test_rejects_unknown_structured_source_id(self) -> None:
        result = validate_answer_payload("A supported claim.", [{"id": "S1"}], ["S7"])
        self.assertFalse(result.valid)
        self.assertEqual(result.answer, UNVERIFIABLE_RESPONSE)
        self.assertIn("unknown_declared_source_ids:S7", result.reason)

    def test_accepts_compact_financial_table_and_escapes_currency(self) -> None:
        answer = (
            "| Metric | 2024 | 2023 |\n"
            "|---|---:|---:|\n"
            "| Long-term debt | $42,688 | $41,990 |\n\n"
            "Unit: USD millions. [S1]"
        )
        result = validate_answer_payload(answer, [{"id": "S1"}], ["S1"])
        self.assertTrue(result.valid)
        self.assertIn(r"\$42,688", result.answer)
        self.assertEqual(result.reason, "valid_inline_citations")

    def test_accepts_exact_six_column_financial_table(self) -> None:
        answer = (
            "| Issuance | Maturity | Stated rate | Effective rate | 2024 | 2023 |\n"
            "|---|---|---|---|---:|---:|\n"
            "| 2009 | 2039 | 5.20% | 5.24% | 520 | 520 |\n\n[S1]"
        )
        result = validate_answer_payload(answer, [{"id": "S1"}], ["S1"])
        self.assertTrue(result.valid)
        self.assertEqual(result.reason, "valid_inline_citations")

    def test_rejects_markdown_table_with_more_than_eight_columns(self) -> None:
        answer = (
            "| A | B | C | D | E | F | G | H | I |\n"
            "|---|---|---|---|---|---|---|---|---|\n"
            "| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |\n\n[S1]"
        )
        result = validate_answer_payload(answer, [{"id": "S1"}], ["S1"])
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "wide_markdown_table:9_columns")

    def test_rejects_inconsistent_table_columns(self) -> None:
        answer = (
            "| Metric | 2024 | 2023 |\n"
            "|---|---:|---:|\n"
            "| Total debt | 44,937 |\n\n[S1]"
        )
        self.assertEqual(
            validate_markdown_tables(answer),
            "inconsistent_table_columns:3:expected_3:found_2",
        )

    def test_rejects_corrupted_financial_markdown(self) -> None:
        answer = "Balance was `****520 million [S1]."
        result = validate_answer_payload(answer, [{"id": "S1"}], ["S1"])
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "malformed_empty_bold_marker")

    def test_rejects_code_backticks_inside_table(self) -> None:
        answer = "| Metric | Value |\n|---|---:|\n| Debt | `$520` |\n\n[S1]"
        result = validate_answer_payload(answer, [{"id": "S1"}], ["S1"])
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "code_backtick_in_table:3")

    def test_currency_normalization_does_not_change_prose(self) -> None:
        answer = "$520 million.\n\n| Metric | Value |\n|---|---:|\n| Debt | $520 |"
        normalized = normalize_table_currency(answer)
        self.assertTrue(normalized.startswith("$520 million."))
        self.assertIn(r"| Debt | \$520 |", normalized)


if __name__ == "__main__":
    unittest.main()
