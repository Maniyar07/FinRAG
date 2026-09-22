from __future__ import annotations

import unittest

from src.retrieval.query_understanding import understand_query
from src.retrieval.scope_policy import resolve_scope
from src.schemas import Decision, Scope


class QueryUnderstandingTests(unittest.TestCase):
    def test_exposes_normalized_query_for_retrieval(self) -> None:
        result = understand_query("Show microsft's balace statment from the 2024 10 k")
        self.assertEqual(
            result.normalized_query,
            "Show microsoft's balance statement from the 2024 10-K",
        )

    def test_extracts_company_year_and_risk_document(self) -> None:
        result = understand_query("Summarize Microsoft's 2024 risk factors")
        self.assertEqual(result.tickers, ("MSFT",))
        self.assertEqual(result.years, ("2024",))
        self.assertEqual(result.doc_type, "10K")
        self.assertEqual(result.topic, "risk")

    def test_extracts_multi_company_comparison(self) -> None:
        result = understand_query("Compare JPM and TSLA revenue in 2025")
        self.assertEqual(result.tickers, ("JPM", "TSLA"))
        self.assertTrue(result.comparison)

    def test_each_requested_company_does_not_expand_to_all_companies(self) -> None:
        result = understand_query(
            "Compare MSFT and TSLA revenue and preserve each company's units."
        )
        self.assertEqual(result.tickers, ("MSFT", "TSLA"))
        self.assertFalse(result.all_tickers)

    def test_both_table_years_do_not_expand_source_scope(self) -> None:
        result = understand_query(
            "Using the MSFT 2024 10-K, report 2024 and 2023 and label both years."
        )
        self.assertEqual(result.years, ("2024",))
        self.assertFalse(result.all_years)

    def test_extracts_asymmetric_company_year_pairs(self) -> None:
        result = understand_query("Compare MSFT 2024 with TSLA 2025 revenue")
        self.assertEqual(result.requested_groups, (("MSFT", "2024"), ("TSLA", "2025")))

    def test_marks_live_price_request_out_of_scope(self) -> None:
        self.assertTrue(
            understand_query("What is TSLA's current stock price?").explicit_out_of_scope
        )

    def test_marks_clearly_unrelated_request_out_of_scope(self) -> None:
        result = resolve_scope(understand_query("Explain photosynthesis"))
        self.assertEqual(result.decision, Decision.OUT_OF_SCOPE)

    def test_marks_unknown_nonfinancial_request_out_of_scope(self) -> None:
        result = resolve_scope(understand_query("What is two plus two?"))
        self.assertEqual(result.decision, Decision.OUT_OF_SCOPE)

    def test_recognizes_unsupported_year_and_company(self) -> None:
        result = understand_query("Compare Apple 2023 revenue with MSFT")
        self.assertEqual(result.unsupported_companies, ("AAPL",))
        self.assertEqual(result.unsupported_years, ("2023",))

    def test_table_column_year_is_not_treated_as_a_source_year(self) -> None:
        result = understand_query(
            "Using the MSFT 2024 10-K, calculate revenue growth from 2023 to 2024."
        )
        self.assertEqual(result.years, ("2024",))
        self.assertEqual(result.reference_years, ("2023",))
        self.assertEqual(result.unsupported_years, ())

    def test_unsupported_filing_year_is_still_rejected(self) -> None:
        result = understand_query("Compare the TSLA 2023 10-K with its 2024 10-K.")
        self.assertEqual(result.unsupported_years, ("2023",))

    def test_plural_ten_ks_are_recognized(self) -> None:
        result = understand_query("Compare revenue in the MSFT and TSLA 2024 10-Ks.")
        self.assertEqual(result.doc_type, "10K")
        self.assertEqual(result.requested_doc_types, ("10K",))

    def test_explicit_filing_and_transcript_request_is_preserved(self) -> None:
        result = understand_query(
            "Compare JPM's 2024 10-K risk disclosure with its earnings transcript."
        )
        self.assertIsNone(result.doc_type)
        self.assertEqual(result.requested_doc_types, ("10K", "TRANSCRIPT"))

    def test_risk_word_does_not_force_filing_when_transcript_is_explicit(self) -> None:
        result = understand_query(
            "What risk factors did JPM management discuss in its Q4 2024 transcript?"
        )
        self.assertEqual(result.requested_doc_types, ("TRANSCRIPT",))
        self.assertEqual(result.doc_type, "TRANSCRIPT")


class ScopePolicyTests(unittest.TestCase):
    def test_first_ambiguous_question_requests_clarification(self) -> None:
        resolution = resolve_scope(understand_query("What were the main risks?"))
        self.assertEqual(resolution.decision, Decision.CLARIFY)
        self.assertIn("company", resolution.message)
        self.assertIn("year", resolution.message)

    def test_follow_up_inherits_previous_scope(self) -> None:
        resolution = resolve_scope(
            understand_query("What about the risk factors?"),
            previous_scope=Scope(("JPM",), ("2024",), "10K"),
        )
        self.assertEqual(resolution.decision, Decision.SEARCH)
        self.assertEqual(resolution.scope, Scope(("JPM",), ("2024",), "10K"))
        self.assertEqual(set(resolution.inherited_fields), {"company", "year"})

    def test_query_and_ui_conflict_is_not_silently_overridden(self) -> None:
        resolution = resolve_scope(
            understand_query("What were TSLA's 2024 risks?"),
            ui_scope=Scope(("JPM",), ("2024",), "10K"),
        )
        self.assertEqual(resolution.decision, Decision.SCOPE_CONFLICT)

    def test_query_can_override_previous_scope(self) -> None:
        resolution = resolve_scope(
            understand_query("What about Tesla in 2025?"),
            previous_scope=Scope(("JPM",), ("2024",), "10K"),
        )
        self.assertEqual(resolution.decision, Decision.SEARCH)
        self.assertEqual(resolution.scope.tickers, ("TSLA",))
        self.assertEqual(resolution.scope.years, ("2025",))

    def test_missing_requested_document_is_reported_before_search(self) -> None:
        resolution = resolve_scope(
            understand_query("What did TSLA say in its 2025 earnings call?"),
            available_keys={("TSLA", "2025", "10K")},
        )
        self.assertEqual(resolution.decision, Decision.DATA_UNAVAILABLE)
        self.assertIn("TSLA 2025 TRANSCRIPT", resolution.message)

    def test_all_companies_expands_intentionally(self) -> None:
        resolution = resolve_scope(
            understand_query("Compare all companies in 2024"),
        )
        self.assertEqual(resolution.decision, Decision.SEARCH)
        self.assertEqual(resolution.scope.tickers, ("JPM", "MSFT", "TSLA"))

    def test_asymmetric_pairs_do_not_become_cross_product(self) -> None:
        resolution = resolve_scope(
            understand_query("Compare MSFT 2024 with TSLA 2025 revenue")
        )
        self.assertEqual(resolution.decision, Decision.CLARIFY)
        self.assertEqual(
            resolution.scope.groups, (("MSFT", "2024"), ("TSLA", "2025"))
        )

    def test_partial_follow_up_after_asymmetric_comparison_clarifies(self) -> None:
        previous = Scope(
            ("MSFT", "TSLA"),
            ("2024", "2025"),
            None,
            (("MSFT", "2024"), ("TSLA", "2025")),
        )
        resolution = resolve_scope(understand_query("What about TSLA?"), previous_scope=previous)
        self.assertEqual(resolution.decision, Decision.CLARIFY)
        self.assertIn("year", resolution.message)

    def test_new_question_without_reference_does_not_silently_inherit(self) -> None:
        resolution = resolve_scope(
            understand_query("Summarize the main risk factors."),
            previous_scope=Scope(("JPM",), ("2024",), "10K"),
        )
        self.assertEqual(resolution.decision, Decision.CLARIFY)

    def test_explicit_both_document_types_override_previous_transcript(self) -> None:
        resolution = resolve_scope(
            understand_query(
                "Compare JPM's 2024 10-K credit risks with its earnings transcript."
            ),
            previous_scope=Scope(("JPM",), ("2024",), "TRANSCRIPT"),
        )
        self.assertEqual(resolution.decision, Decision.SEARCH)
        self.assertIsNone(resolution.scope.doc_type)
        self.assertEqual(
            resolution.scope.required_doc_types, ("10K", "TRANSCRIPT")
        )

    def test_pronoun_inside_explicit_multi_company_query_is_not_a_followup(self) -> None:
        resolution = resolve_scope(
            understand_query("Compare JPM and TSLA 2024 revenue and their margins."),
            previous_scope=Scope(("MSFT",), ("2024",), "TRANSCRIPT"),
        )
        self.assertEqual(resolution.decision, Decision.CLARIFY)
        self.assertIsNone(resolution.scope.doc_type)
        self.assertNotIn("document type", resolution.inherited_fields)

    def test_explicit_both_document_types_require_both_sources(self) -> None:
        resolution = resolve_scope(
            understand_query(
                "Compare JPM's 2024 10-K credit risks with its earnings transcript."
            ),
            available_keys={("JPM", "2024", "10K")},
        )
        self.assertEqual(resolution.decision, Decision.DATA_UNAVAILABLE)
        self.assertIn("JPM 2024 TRANSCRIPT", resolution.message)

    def test_latest_available_year_resolves_from_manifest(self) -> None:
        resolution = resolve_scope(
            understand_query("What is Tesla's most recent filing revenue?"),
            available_keys={
                ("TSLA", "2024", "10K"),
                ("TSLA", "2025", "10K"),
            },
        )
        self.assertEqual(resolution.decision, Decision.SEARCH)
        self.assertEqual(resolution.scope.years, ("2025",))

    def test_bare_revenue_question_clarifies_document_basis(self) -> None:
        resolution = resolve_scope(
            understand_query("What was MSFT revenue in 2024?"),
            available_keys={
                ("MSFT", "2024", "10K"),
                ("MSFT", "2024", "TRANSCRIPT"),
            },
        )
        self.assertEqual(resolution.decision, Decision.CLARIFY)
        self.assertIn("evidence basis", resolution.message)

    def test_bare_revenue_uses_only_available_document_type(self) -> None:
        resolution = resolve_scope(
            understand_query("What was TSLA revenue in 2025?"),
            available_keys={("TSLA", "2025", "10K")},
        )
        self.assertEqual(resolution.decision, Decision.SEARCH)

    def test_short_document_choice_inherits_pending_company_and_year(self) -> None:
        resolution = resolve_scope(
            understand_query("both"),
            previous_scope=Scope(("MSFT",), ("2024",)),
        )
        self.assertEqual(resolution.decision, Decision.SEARCH)
        self.assertEqual(
            resolution.scope.required_doc_types, ("10K", "TRANSCRIPT")
        )
        self.assertEqual(set(resolution.inherited_fields), {"company", "year"})


if __name__ == "__main__":
    unittest.main()
