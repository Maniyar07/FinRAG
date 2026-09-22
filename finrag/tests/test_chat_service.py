from __future__ import annotations

import unittest
from decimal import Decimal
from types import SimpleNamespace

from src.app.chat_service import ChatService
from src.generation.narrative_quotes import VerifiedQuote
from src.financial.models import ValueType
from src.orchestration.models import EvidenceType
from src.retrieval.semantic_query_parser import SemanticQueryResult
from src.schemas import Decision, RetrievalBundle, Scope
from src.tools.document_search import DocumentSearchTool


class FakeLogger:
    def info(self, message) -> None:
        pass


class FakeRetriever:
    reranker = None

    def __init__(self) -> None:
        self.calls = []

    def retrieve(self, query, scope, **kwargs):
        self.calls.append((query, scope, kwargs))
        ticker, year = scope.groups[0]
        doc_type = scope.doc_type or (
            scope.required_doc_types[0] if scope.required_doc_types else "10K"
        )
        source = {
            "id": "S1",
            "ticker": ticker,
            "fiscal_year": year,
            "doc_type": doc_type,
            "evidence_text": "| Metric | 2024 |\n|---|---|\n| Revenue | 1 |",
        }
        return RetrievalBundle(
            "context",
            [source],
            scope,
            1,
            covered_groups=scope.groups,
        )


class FakeGenerator:
    def __init__(self) -> None:
        self.calls = []

    def generate_with_trace(self, question, bundle, **kwargs):
        self.calls.append((question, bundle, kwargs))
        return SimpleNamespace(
            answer="Completed.",
            attempts=1,
            validation_reason="valid",
            raw_output_previews=(),
        )


class FakeSemanticParser:
    def __init__(self, result=None, error=None) -> None:
        self.result = result
        self.error = error
        self.calls = 0

    def parse(self, **kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


def _service(*, semantic_parser=None) -> ChatService:
    service = ChatService.__new__(ChatService)
    service.available_keys = {
        ("MSFT", "2024", "10K"),
        ("MSFT", "2025", "10K"),
        ("TSLA", "2025", "10K"),
        ("TSLA", "2025", "TRANSCRIPT"),
    }
    service.retriever = FakeRetriever()
    service.document_search_tool = DocumentSearchTool(service.retriever)
    service.generator = FakeGenerator()
    service.semantic_parser = semantic_parser
    service.logger = FakeLogger()
    return service


class ChatServiceOrchestrationTests(unittest.TestCase):
    def test_verified_calculation_displays_currency_and_scale(self) -> None:
        facts = (
            SimpleNamespace(
                fact_id="F1", ticker="MSFT", period="2024", raw_value="$245,122",
                source_id="S1", value_type=ValueType.CURRENCY,
                currency="USD", scale="millions",
            ),
            SimpleNamespace(
                fact_id="F2", ticker="MSFT", period="2025", raw_value="281,724",
                source_id="S2", value_type=ValueType.CURRENCY,
                currency="USD", scale="millions",
            ),
        )
        calculation = SimpleNamespace(
            label="MSFT revenue change",
            result=SimpleNamespace(
                input_fact_ids=("F1", "F2"), result=Decimal("14.93"),
                result_unit="percent", currency=None, scale=None,
                source_ids=("S1", "S2"),
                operation=SimpleNamespace(value="percentage_change"),
            ),
        )

        answer = ChatService._verified_calculation_text(
            SimpleNamespace(facts=facts, calculations=(calculation,))
        )

        self.assertIn("$245,122 million", answer)
        self.assertIn("USD 281,724 million", answer)

    def test_calculation_and_exact_quote_are_composed_without_freeform_generation(self) -> None:
        service = _service()
        scope = Scope(("MSFT",), ("2024", "2025"), "10K")
        bundle = RetrievalBundle(
            "context",
            [
                {"id": "S1", "ticker": "MSFT", "fiscal_year": "2024", "doc_type": "10K"},
                {"id": "S2", "ticker": "MSFT", "fiscal_year": "2025", "doc_type": "10K"},
                {"id": "S3", "ticker": "MSFT", "fiscal_year": "2025", "doc_type": "10K"},
            ],
            scope,
            3,
        )
        facts = (
            SimpleNamespace(fact_id="F1", ticker="MSFT", period="2024", raw_value="100", source_id="S1"),
            SimpleNamespace(fact_id="F2", ticker="MSFT", period="2025", raw_value="120", source_id="S2"),
        )
        calculation = SimpleNamespace(
            label="MSFT cash change",
            result=SimpleNamespace(
                input_fact_ids=("F1", "F2"), result=Decimal("20"),
                result_unit="currency", currency="USD", scale="millions",
                source_ids=("S1", "S2"), operation=SimpleNamespace(value="absolute_change"),
            ),
        )
        requirement = SimpleNamespace(
            requirement_id="risk", evidence_type=EvidenceType.NARRATIVE,
            document_type="10K", question="summarize liquidity risks",
            groups=(SimpleNamespace(key=("MSFT", "2025"), ticker="MSFT", fiscal_year="2025"),),
        )
        plan = SimpleNamespace(requirements=(requirement,), model_dump=lambda **_: {})
        service.multihop_planner = SimpleNamespace(plan=lambda **_: plan)
        service.multihop_executor = SimpleNamespace(execute=lambda *_, **__: SimpleNamespace(
            bundle=bundle, complete=True, trace={}, issues=(),
            calculations=(calculation,), facts=facts,
        ))
        service.narrative_quote_selector = SimpleNamespace(select=lambda **_: (
            VerifiedQuote("MSFT", "2025", "10K", "S3", "Cash commitments may exceed available funding under adverse conditions."),
        ))

        result = service._run_multihop(
            question="Compare cash and summarize liquidity risks.", scope=scope,
            inherited_fields=(), history=None, query_expansions=(),
            wants_table=False, trace={"multihop": {}},
        )

        self.assertEqual(result.decision, Decision.ANSWERED)
        self.assertIn("20 USD millions", result.answer)
        self.assertIn("Cash commitments may exceed", result.answer)
        self.assertEqual(service.generator.calls, [])

    def test_verified_calculation_partial_keeps_numbers_but_omits_unchecked_prose(self) -> None:
        facts = (
            SimpleNamespace(fact_id="F1", period="2024", raw_value="100", source_id="S1"),
            SimpleNamespace(fact_id="F2", period="Year Ended December 31,", column_label="2025", raw_value="120", source_id="S2"),
        )
        calculation = SimpleNamespace(
            label="MSFT revenue change",
            result=SimpleNamespace(
                input_fact_ids=("F1", "F2"),
                result=Decimal("20"),
                result_unit="percent",
                currency=None,
                scale=None,
                source_ids=("S1", "S2"),
                operation=SimpleNamespace(value="percentage_change"),
            ),
        )
        execution = SimpleNamespace(
            facts=facts,
            calculations=(calculation,),
            bundle=RetrievalBundle(
                "context",
                [
                    {"id": "S1", "ticker": "MSFT", "fiscal_year": "2024", "doc_type": "10K"},
                    {"id": "S2", "ticker": "MSFT", "fiscal_year": "2025", "doc_type": "10K"},
                ],
                Scope(("MSFT",), ("2024", "2025"), "10K"),
                2,
            ),
        )
        answer = ChatService._verified_calculation_partial(execution)
        self.assertIn("20 percent", answer)
        self.assertIn("2024: 100", answer)
        self.assertIn("Year Ended December 31, 2025: 120", answer)
        self.assertIn("could not validate", answer)

    def test_compound_comparison_uses_enabled_multihop_path(self) -> None:
        service = _service()
        scope_bundle = RetrievalBundle(
            "merged context",
            [
                {
                    "id": "S1",
                    "ticker": "MSFT",
                    "fiscal_year": "2025",
                    "doc_type": "10K",
                    "evidence_text": "MSFT revenue",
                },
                {
                    "id": "S2",
                    "ticker": "TSLA",
                    "fiscal_year": "2025",
                    "doc_type": "10K",
                    "evidence_text": "TSLA revenue",
                },
            ],
            scope=Scope(
                ("MSFT", "TSLA"),
                ("2025",),
                "10K",
                requested_groups=(("MSFT", "2025"), ("TSLA", "2025")),
            ),
            candidate_count=2,
            covered_groups=(("MSFT", "2025"), ("TSLA", "2025")),
        )

        class FakePlan:
            def model_dump(self, **kwargs):
                del kwargs
                return {"requirements": ["revenue"]}

        class FakePlanner:
            def __init__(self):
                self.calls = []

            def plan(self, **kwargs):
                self.calls.append(kwargs)
                return FakePlan()

        class FakeExecutor:
            def __init__(self):
                self.calls = []

            def execute(self, plan, **kwargs):
                self.calls.append((plan, kwargs))
                return SimpleNamespace(
                    bundle=scope_bundle,
                    complete=True,
                    trace={"requirements": []},
                    issues=(),
                    calculations=(),
                    facts=(),
                )

        service.multihop_enabled = True
        service.multihop_planner = FakePlanner()
        service.multihop_executor = FakeExecutor()

        result = service.ask("Compare MSFT and TSLA revenue in their 2025 10-Ks.")

        self.assertEqual(result.decision, Decision.ANSWERED)
        self.assertTrue(result.trace["multihop"]["selected"])
        self.assertEqual(service.retriever.calls, [])
        self.assertEqual(len(service.multihop_planner.calls), 1)
        self.assertEqual(len(service.multihop_executor.calls), 1)

    def test_clarification_accumulates_company_year_and_document_type(self) -> None:
        parser = FakeSemanticParser(
            SemanticQueryResult(
                candidate_tickers=["TSLA"],
                needs_clarification=True,
                clarification_fields=["company"],
            )
        )
        service = _service(semantic_parser=parser)
        original = "How did the automaker's automotive gross margin change in 2025?"

        first = service.ask(original)
        self.assertEqual(first.decision, Decision.CLARIFY)
        self.assertEqual(first.pending_clarification.candidate_tickers, ("TSLA",))
        self.assertEqual(first.pending_clarification.scope.years, ("2025",))

        second = service.ask(
            "yes Tesla",
            pending_clarification=first.pending_clarification,
        )
        self.assertEqual(second.decision, Decision.CLARIFY)
        self.assertEqual(second.pending_clarification.scope.tickers, ("TSLA",))
        self.assertEqual(second.pending_clarification.scope.years, ("2025",))
        self.assertIn("document type", second.pending_clarification.missing_fields)

        final = service.ask(
            "10-K",
            pending_clarification=second.pending_clarification,
        )
        self.assertEqual(final.decision, Decision.ANSWERED)
        self.assertIsNone(final.pending_clarification)
        self.assertEqual(final.scope.tickers, ("TSLA",))
        self.assertEqual(final.scope.years, ("2025",))
        self.assertEqual(final.scope.doc_type, "10K")
        self.assertEqual(service.retriever.calls[0][0], original)
        self.assertEqual(service.generator.calls[0][0], original)
        self.assertEqual(parser.calls, 1)

    def test_new_substantive_question_discards_pending_clarification(self) -> None:
        parser = FakeSemanticParser(
            SemanticQueryResult(
                candidate_tickers=["TSLA"],
                clarification_fields=["company"],
            )
        )
        service = _service(semantic_parser=parser)
        first = service.ask(
            "How did the automaker's automotive gross margin change in 2025?"
        )
        new_question = "What was Microsoft's revenue in the 2024 10-K?"

        result = service.ask(
            new_question,
            pending_clarification=first.pending_clarification,
        )

        self.assertEqual(result.decision, Decision.ANSWERED)
        self.assertEqual(result.scope.tickers, ("MSFT",))
        self.assertEqual(result.scope.years, ("2024",))
        self.assertIsNone(result.pending_clarification)
        self.assertEqual(service.retriever.calls[0][0], new_question)

    def test_complete_table_intent_reaches_retrieval_and_generation(self) -> None:
        service = _service()
        question = "Show Microsoft's 2024 10-K balance sheet as a complete table."

        result = service.ask(question)

        self.assertEqual(result.decision, Decision.ANSWERED)
        retrieval_call = service.retriever.calls[0]
        self.assertTrue(retrieval_call[2]["preserve_all"])
        self.assertEqual(retrieval_call[2]["query_expansions"], ())
        generation_call = service.generator.calls[0]
        self.assertEqual(generation_call[0], question)
        self.assertTrue(generation_call[2]["wants_table"])
        self.assertTrue(generation_call[2]["wants_complete_table"])

    def test_explicit_unsupported_company_from_fallback_stops_retrieval(self) -> None:
        parser = FakeSemanticParser(
            SemanticQueryResult(
                explicit_company_mentions=["IBM"],
                comparison_intent=True,
            )
        )
        service = _service(semantic_parser=parser)

        result = service.ask("Compare IBM with MSFT in 2025.")

        self.assertEqual(result.decision, Decision.OUT_OF_SCOPE)
        self.assertEqual(parser.calls, 1)
        self.assertEqual(service.retriever.calls, [])

    def test_suspicious_comparison_fails_closed_when_parser_fails(self) -> None:
        parser = FakeSemanticParser(error=TimeoutError("semantic timeout"))
        service = _service(semantic_parser=parser)

        result = service.ask("Compare IBM with MSFT in 2025.")

        self.assertEqual(result.decision, Decision.CLARIFY)
        self.assertIn("confirm every company", result.answer)
        self.assertEqual(service.retriever.calls, [])


if __name__ == "__main__":
    unittest.main()
