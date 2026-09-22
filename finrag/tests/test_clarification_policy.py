from __future__ import annotations

import unittest

from src.retrieval.clarification_policy import (
    merge_clarification_reply,
    pending_from_resolution,
)
from src.schemas import (
    Decision,
    PendingClarification,
    QueryUnderstanding,
    Scope,
    ScopeResolution,
)


class ClarificationPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pending = PendingClarification(
            original_question=(
                "How did the automaker's automotive gross margin change in 2025?"
            ),
            scope=Scope(years=("2025",)),
            candidate_tickers=("TSLA",),
            missing_fields=("company",),
        )

    def test_affirmation_confirms_candidate_and_preserves_year(self) -> None:
        turn = merge_clarification_reply(self.pending, "yes Tesla")

        self.assertIsNotNone(turn)
        self.assertEqual(turn.understanding.tickers, ("TSLA",))
        self.assertEqual(turn.understanding.years, ("2025",))
        self.assertEqual(turn.candidate_tickers, ())

    def test_plain_yes_confirms_only_the_pending_candidate(self) -> None:
        turn = merge_clarification_reply(self.pending, "yes")

        self.assertEqual(turn.understanding.tickers, ("TSLA",))
        self.assertEqual(turn.understanding.years, ("2025",))

    def test_document_reply_preserves_company_and_year(self) -> None:
        pending = PendingClarification(
            self.pending.original_question,
            Scope(("TSLA",), ("2025",)),
            missing_fields=("document type",),
        )

        turn = merge_clarification_reply(pending, "10-K")

        self.assertEqual(turn.understanding.tickers, ("TSLA",))
        self.assertEqual(turn.understanding.years, ("2025",))
        self.assertEqual(turn.understanding.doc_type, "10K")

    def test_explicit_year_overrides_pending_year_without_losing_company(self) -> None:
        pending = PendingClarification(
            self.pending.original_question,
            Scope(("TSLA",), ("2025",), "10K"),
            missing_fields=("year",),
        )

        turn = merge_clarification_reply(pending, "2024")

        self.assertEqual(turn.understanding.tickers, ("TSLA",))
        self.assertEqual(turn.understanding.years, ("2024",))
        self.assertEqual(turn.understanding.doc_type, "10K")

    def test_new_substantive_question_does_not_reuse_pending_state(self) -> None:
        turn = merge_clarification_reply(
            self.pending,
            "What was Microsoft revenue in 2024?",
        )

        self.assertIsNone(turn)

    def test_document_reply_preserves_asymmetric_comparison_groups(self) -> None:
        pending = PendingClarification(
            original_question="Compare MSFT 2024 with TSLA 2025 revenue.",
            scope=Scope(
                ("MSFT", "TSLA"),
                ("2024", "2025"),
                requested_groups=(("MSFT", "2024"), ("TSLA", "2025")),
            ),
            missing_fields=("document type",),
        )

        turn = merge_clarification_reply(pending, "10-K")

        self.assertEqual(
            turn.understanding.requested_groups,
            (("MSFT", "2024"), ("TSLA", "2025")),
        )

    def test_rejection_does_not_silently_confirm_candidate(self) -> None:
        turn = merge_clarification_reply(self.pending, "no")

        self.assertEqual(turn.understanding.tickers, ())
        self.assertEqual(turn.understanding.years, ("2025",))
        self.assertEqual(turn.candidate_tickers, ())

    def test_candidate_does_not_coexist_with_conflicting_inherited_company(self) -> None:
        understanding = QueryUnderstanding(
            tickers=(),
            years=("2025",),
            doc_type=None,
            domain_relevant=True,
            ambiguous_fields=("company",),
        )
        resolution = ScopeResolution(
            Decision.CLARIFY,
            Scope(("MSFT",), ("2025",)),
            "Confirm the company.",
            ("company",),
        )

        pending = pending_from_resolution(
            original_question=self.pending.original_question,
            understanding=understanding,
            resolution=resolution,
            candidate_tickers=("TSLA",),
        )

        self.assertEqual(pending.scope.tickers, ())
        self.assertEqual(pending.scope.years, ("2025",))
        self.assertEqual(pending.candidate_tickers, ("TSLA",))


if __name__ == "__main__":
    unittest.main()
