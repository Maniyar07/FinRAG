from __future__ import annotations

import unittest

from src.retrieval.query_understanding import understand_query
from src.retrieval.scope_policy import resolve_scope
from src.retrieval.semantic_query_parser import (
    SemanticQueryParser,
    SemanticQueryResult,
    apply_semantic_result,
    semantic_clarification_message,
    should_use_semantic_fallback,
)
from src.schemas import Decision, Scope


class SemanticFallbackPolicyTests(unittest.TestCase):
    def test_parser_receives_only_active_index_catalogue(self) -> None:
        class FakeChain:
            def __init__(self) -> None:
                self.payload = None

            def invoke(self, payload):
                self.payload = payload
                return SemanticQueryResult()

        chain = FakeChain()
        parser = SemanticQueryParser(chain=chain)
        parser.parse(
            question="Question",
            ui_scope=None,
            previous_scope=None,
            available_keys={("MSFT", "2025", "10K")},
        )

        self.assertIn("MSFT = Microsoft Corporation", chain.payload["companies"])
        self.assertNotIn("JPM", chain.payload["companies"])
        self.assertEqual(chain.payload["years"], "2025")
        self.assertEqual(chain.payload["document_types"], "10K")

    def test_missing_financial_scope_triggers_fallback(self) -> None:
        question = (
            "How did the software maker's contracted but unrecognized revenue "
            "develop in the later fiscal period?"
        )
        understanding = understand_query(question)
        resolution = resolve_scope(understanding)

        trigger = should_use_semantic_fallback(
            question, understanding, resolution
        )

        self.assertIsNotNone(trigger)
        self.assertEqual(trigger.reason, "unresolved_scope")

    def test_document_type_only_clarification_does_not_trigger(self) -> None:
        question = "What was MSFT revenue in 2024?"
        understanding = understand_query(question)
        resolution = resolve_scope(
            understanding,
            available_keys={
                ("MSFT", "2024", "10K"),
                ("MSFT", "2024", "TRANSCRIPT"),
            },
        )

        self.assertEqual(resolution.decision, Decision.CLARIFY)
        self.assertIsNone(
            should_use_semantic_fallback(question, understanding, resolution)
        )

    def test_relative_followup_can_trigger_after_deterministic_out_of_scope(self) -> None:
        question = "Now do the earlier period."
        understanding = understand_query(question)
        previous = Scope(("MSFT",), ("2024", "2025"), "10K")
        resolution = resolve_scope(understanding, previous_scope=previous)

        trigger = should_use_semantic_fallback(
            question,
            understanding,
            resolution,
            previous_scope=previous,
        )

        self.assertEqual(resolution.decision, Decision.OUT_OF_SCOPE)
        self.assertEqual(trigger.reason, "unrecognized_relative_followup")

    def test_candidate_company_stays_ambiguous(self) -> None:
        question = "How did the software maker's revenue develop?"
        application = apply_semantic_result(
            question,
            understand_query(question),
            SemanticQueryResult(
                candidate_tickers=["MSFT"],
                comparison_intent=True,
                query_expansions=["remaining performance obligations", "RPO"],
                needs_clarification=True,
                clarification_fields=["company", "comparison years"],
            ),
        )

        self.assertEqual(application.understanding.tickers, ())
        self.assertEqual(application.candidate_tickers, ("MSFT",))
        self.assertEqual(
            application.understanding.query_expansions,
            ("remaining performance obligations", "RPO"),
        )
        self.assertTrue(application.needs_clarification)
        self.assertIn("Microsoft Corporation (MSFT)", semantic_clarification_message(application, ""))

    def test_relative_period_resolves_only_from_two_previous_years(self) -> None:
        question = "Now do the later fiscal period."
        previous = Scope(("MSFT",), ("2024", "2025"), "10K")
        application = apply_semantic_result(
            question,
            understand_query(question),
            SemanticQueryResult(relative_period="later", comparison_intent=True),
            previous_scope=previous,
        )
        resolution = resolve_scope(
            application.understanding,
            previous_scope=previous,
            available_keys={("MSFT", "2025", "10K")},
        )

        self.assertEqual(application.understanding.years, ("2025",))
        self.assertEqual(application.understanding.reference_years, ("2024",))
        self.assertEqual(resolution.decision, Decision.SEARCH)
        self.assertEqual(resolution.scope, Scope(("MSFT",), ("2025",), "10K"))

    def test_unanchored_relative_period_remains_ambiguous(self) -> None:
        question = "Use the later fiscal period."
        application = apply_semantic_result(
            question,
            understand_query(question),
            SemanticQueryResult(relative_period="later"),
            previous_scope=Scope(("MSFT",), ("2025",), "10K"),
        )

        self.assertEqual(application.understanding.years, ())
        self.assertIn("comparison years", application.ambiguous_fields)

    def test_literal_unsupported_company_is_validated_against_question(self) -> None:
        question = "Compare IBM with MSFT in 2025."
        application = apply_semantic_result(
            question,
            understand_query(question),
            SemanticQueryResult(
                explicit_company_mentions=["IBM"],
                comparison_intent=True,
            ),
        )
        resolution = resolve_scope(application.understanding)

        self.assertEqual(application.unsupported_companies, ("IBM",))
        self.assertEqual(resolution.decision, Decision.OUT_OF_SCOPE)

    def test_hallucinated_company_not_present_in_question_is_rejected(self) -> None:
        question = "Compare MSFT revenue with expenses in 2025."
        application = apply_semantic_result(
            question,
            understand_query(question),
            SemanticQueryResult(explicit_company_mentions=["IBM"]),
        )

        self.assertEqual(application.unsupported_companies, ())

    def test_candidate_not_in_active_index_is_not_proposed(self) -> None:
        question = "How did the bank perform?"
        application = apply_semantic_result(
            question,
            understand_query(question),
            SemanticQueryResult(
                candidate_tickers=["JPM"],
                clarification_fields=["company"],
            ),
            available_keys={("MSFT", "2025", "10K")},
        )

        self.assertEqual(application.candidate_tickers, ())
        self.assertIn("company", application.ambiguous_fields)


if __name__ == "__main__":
    unittest.main()
