from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from openai import APIConnectionError

from src.app.chat_service import ChatService
from src.schemas import Decision, Scope


class FakeLogger:
    def __init__(self) -> None:
        self.records: list[str] = []

    def info(self, message: str) -> None:
        self.records.append(message)


class FakeGraph:
    def __init__(
        self,
        *,
        error: Exception | None = None,
        answer: str = "Microsoft reported revenue of $281,724 million [S1].",
    ) -> None:
        self.error = error
        self.answer = answer
        self.calls: list[tuple[dict, dict]] = []

    def invoke(self, inputs: dict, *, config: dict) -> dict:
        self.calls.append((inputs, config))
        if self.error is not None:
            raise self.error
        source = {
            "id": "S1",
            "ticker": "MSFT",
            "fiscal_year": "2025",
            "doc_type": "10K",
            "source": "MSFT_2025_10K.pdf",
        }
        tool_result = {
            "context": "[SOURCE S1 | TICKER: MSFT | FISCAL YEAR: 2025 | TYPE: 10K]\nTotal revenue: 281,724 million",
            "sources": [source],
            "validated_facts": [],
        }
        return {
            "messages": [
                *inputs["messages"],
                {
                    "type": "tool",
                    "name": "search_financial_documents",
                    "tool_call_id": "call-1",
                    "content": json.dumps(tool_result),
                },
                {
                    "type": "ai",
                    "content": self.answer,
                },
            ]
        }


class PlanCoverageGraph:
    def invoke(self, inputs: dict, *, config: dict) -> dict:
        del config
        return {
            "messages": [
                *inputs["messages"],
                {"type": "ai", "tool_calls": [{
                    "id": "plan-1", "name": "plan_financial_request", "args": {},
                }]},
                {"type": "tool", "name": "plan_financial_request",
                 "tool_call_id": "plan-1", "content": json.dumps({
                    "plan_validated": True,
                    "plan": {"requirements": [
                        {"requirement_id": "msft_rev", "ticker": "MSFT",
                         "source_year": "2025", "doc_type": "10K",
                         "reported_period": "FY2025", "metric": "revenue"},
                        {"requirement_id": "tsla_rev", "ticker": "TSLA",
                         "source_year": "2025", "doc_type": "10K",
                         "reported_period": "FY2025", "metric": "revenue"},
                    ], "calculations": []},
                 })},
                {"type": "tool", "name": "search_financial_documents",
                 "tool_call_id": "search-1", "content": json.dumps({
                    "context": "[SOURCE S1 | TICKER: MSFT | FISCAL YEAR: 2025 | TYPE: 10K]\nRevenue: 100",
                    "sources": [{"id": "S1", "ticker": "MSFT", "fiscal_year": "2025", "doc_type": "10K"}],
                    "validated_facts": [],
                 })},
                {"type": "ai", "content": "Microsoft revenue was 100 [S1]."},
            ]
        }


class PlanClarificationGraph:
    def invoke(self, inputs: dict, *, config: dict) -> dict:
        del config
        return {"messages": [
            *inputs["messages"],
            {"type": "tool", "name": "plan_financial_request",
             "tool_call_id": "plan-1", "content": json.dumps({
                "plan_validated": True,
                "plan": {"requirements": [], "calculations": [],
                         "clarification": "Which revenue measure do you mean?"},
             })},
            {"type": "ai", "content": "Which revenue measure do you mean?"},
        ]}


class PartialStreamGraph:
    def stream(self, inputs: dict, *, config: dict, stream_mode: str):
        del config
        assert stream_mode == "values"
        yield {
            "messages": [
                *inputs["messages"],
                {
                    "type": "ai",
                    "tool_calls": [{
                        "id": "search-1",
                        "name": "search_financial_documents",
                        "args": {"ticker": "MSFT", "filing_year": "2025",
                                 "doc_type": "10K", "query": "revenue"},
                    }],
                },
                {
                    "type": "tool",
                    "name": "search_financial_documents",
                    "tool_call_id": "search-1",
                    "content": json.dumps({
                        "ticker": "MSFT", "filing_year": "2025", "doc_type": "10K",
                        "context": "[SOURCE S1 | TICKER: MSFT | FISCAL YEAR: 2025 | TYPE: 10K]\nRevenue: 100",
                        "sources": [{
                            "id": "S1", "parent_id": "P1", "ticker": "MSFT",
                            "fiscal_year": "2025", "doc_type": "10K",
                        }],
                        "validated_facts": [],
                    }),
                },
            ]
        }
        raise GraphRecursionError("test recursion exhaustion")


class BudgetLimitedStreamGraph:
    def stream(self, inputs: dict, *, config: dict, stream_mode: str):
        del config
        assert stream_mode == "values"
        yield {
            "messages": [
                *inputs["messages"],
                {
                    "type": "tool",
                    "name": "search_financial_documents",
                    "tool_call_id": "search-1",
                    "content": json.dumps({
                        "ticker": "MSFT", "filing_year": "2025", "doc_type": "10K",
                        "context": "[SOURCE S1 | TICKER: MSFT | FISCAL YEAR: 2025 | TYPE: 10K]\nRevenue: 100",
                        "sources": [{
                            "id": "S1", "parent_id": "P1", "ticker": "MSFT",
                            "fiscal_year": "2025", "doc_type": "10K",
                        }],
                        "validated_facts": [],
                    }),
                },
                {
                    "type": "tool",
                    "name": "search_financial_documents",
                    "tool_call_id": "search-2",
                    "content": json.dumps({
                        "action_budget_exhausted": True,
                        "message": "Request budget reached.",
                    }),
                },
            ]
        }


class GraphRecursionError(Exception):
    pass


class ChatServiceLangGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ChatService.__new__(ChatService)
        self.service.available_keys = {("MSFT", "2025", "10K")}
        self.service.retriever = SimpleNamespace(reranker=None)
        self.service.semantic_parser = None
        self.service.agent_graph = FakeGraph()
        self.service.logger = FakeLogger()

    def test_process_query_invokes_graph_and_extracts_cited_sources(self) -> None:
        result = self.service.process_query(
            {"messages": [{"role": "user", "content": "Revenue?"}]},
            thread_id="conversation-1",
            scope=Scope(("MSFT",), ("2025",), "10K"),
        )

        self.assertEqual(result.decision, Decision.ANSWERED)
        self.assertIn("281,724 million [S1]", result.answer)
        self.assertEqual(result.sources[0]["id"], "S1")
        _, config = self.service.agent_graph.calls[0]
        self.assertNotEqual(config["configurable"]["thread_id"], "conversation-1")
        self.assertGreater(config["recursion_limit"], 1)
        self.assertEqual(result.trace["agent"]["backend"], "langgraph")
        self.assertTrue(result.trace["answer_validation"]["valid"])

    def test_process_query_rejects_unknown_citation_ids(self) -> None:
        self.service.agent_graph = FakeGraph(
            answer="Microsoft reported revenue of $281,724 million [S9]."
        )

        result = self.service.process_query(
            {"messages": [{"role": "user", "content": "Revenue?"}]},
            thread_id="conversation-1",
            scope=Scope(("MSFT",), ("2025",), "10K"),
        )

        self.assertEqual(result.decision, Decision.VALIDATION_FAILED)
        self.assertIn("could not be verified", result.answer)
        self.assertEqual(
            result.trace["answer_validation"]["reason"], "unknown_inline_source_ids:S9"
        )

    def test_process_query_repairs_missing_citations_from_validated_evidence(self) -> None:
        self.service.agent_graph = FakeGraph(
            answer="Microsoft reported revenue of $281,724 million."
        )
        repair_model = SimpleNamespace(invoke=lambda messages: SimpleNamespace(
            content="Microsoft reported revenue of $281,724 million [S1]."
        ))

        with patch("src.app.chat_service.get_llm_engine", return_value=repair_model):
            result = self.service.process_query(
                {"messages": [{"role": "user", "content": "Revenue?"}]},
                thread_id="conversation-1",
                scope=Scope(("MSFT",), ("2025",), "10K"),
            )

        self.assertEqual(result.decision, Decision.ANSWERED)
        self.assertIn("[S1]", result.answer)
        self.assertEqual(
            result.trace["citation_repair"], {"attempted": True, "succeeded": True}
        )

    def test_streaming_retains_tool_evidence_when_graph_recursion_fails(self) -> None:
        self.service.agent_graph = PartialStreamGraph()
        repair_model = SimpleNamespace(invoke=lambda messages: SimpleNamespace(
            content="Microsoft reported revenue of $100 million [S1]."
        ))

        with patch("src.app.chat_service.get_llm_engine", return_value=repair_model):
            result = self.service.process_query(
                {"messages": [{"role": "user", "content": "Revenue?"}]},
                thread_id="conversation-1",
                scope=Scope(("MSFT",), ("2025",), "10K"),
            )

        self.assertEqual(result.decision, Decision.ANSWERED)
        self.assertEqual(result.trace["error_type"], "GraphRecursionError")
        self.assertEqual(result.trace["answer_recovery"]["reason"], "graph_recursion")
        self.assertEqual(
            result.trace["agent"]["tool_calls"], ["search_financial_documents"]
        )
        self.assertEqual(result.trace["retrieval"]["source_ids"], ["S1"])

    def test_tool_budget_finalizes_from_partial_evidence_without_more_graph_turns(self) -> None:
        self.service.agent_graph = BudgetLimitedStreamGraph()
        repair_model = SimpleNamespace(invoke=lambda messages: SimpleNamespace(
            content="Microsoft reported revenue of $100 million [S1]."
        ))

        with patch("src.app.chat_service.get_llm_engine", return_value=repair_model):
            result = self.service.process_query(
                {"messages": [{"role": "user", "content": "Revenue?"}]},
                thread_id="conversation-1",
                scope=Scope(("MSFT",), ("2025",), "10K"),
            )

        self.assertEqual(result.decision, Decision.ANSWERED)
        self.assertEqual(
            result.trace["answer_recovery"]["reason"], "tool_action_budget"
        )

    def test_process_query_requires_citations_for_every_comparison_group(self) -> None:
        result = self.service.process_query(
            {"messages": [{"role": "user", "content": "Compare both companies."}]},
            thread_id="conversation-1",
            scope=Scope(
                tickers=("MSFT", "TSLA"),
                years=("2025",),
                doc_type="10K",
                requested_groups=(("MSFT", "2025"), ("TSLA", "2025")),
            ),
        )

        self.assertEqual(result.decision, Decision.VALIDATION_FAILED)
        self.assertEqual(
            result.trace["answer_validation"]["reason"],
            "missing_scope_group_citation",
        )
        self.assertEqual(
            result.trace["answer_validation"]["missing_groups"],
            [["TSLA", "2025", "10K"]],
        )

    def test_process_query_checks_citations_against_planned_source_groups(self) -> None:
        self.service.agent_graph = PlanCoverageGraph()

        result = self.service.process_query(
            {"messages": [{"role": "user", "content": "Compare revenue."}]},
            thread_id="conversation-3",
            scope=Scope(
                tickers=("MSFT", "TSLA"), years=("2025",), doc_type="10K",
                requested_groups=(("MSFT", "2025"), ("TSLA", "2025")),
            ),
        )

        self.assertEqual(result.decision, Decision.VALIDATION_FAILED)
        self.assertEqual(
            result.trace["answer_validation"]["reason"],
            "missing_planned_source_group_citation",
        )
        self.assertEqual(
            result.trace["answer_validation"]["missing_planned_source_groups"],
            [["TSLA", "2025", "10K"]],
        )

    def test_process_query_returns_plan_requested_clarification(self) -> None:
        self.service.agent_graph = PlanClarificationGraph()

        result = self.service.process_query(
            {"messages": [{"role": "user", "content": "Compare revenue."}]},
            thread_id="conversation-4",
            scope=Scope(("MSFT",), ("2025",), "10K"),
        )

        self.assertEqual(result.decision, Decision.CLARIFY)
        self.assertEqual(result.answer, "Which revenue measure do you mean?")
        self.assertIsNotNone(result.pending_clarification)

    def test_ask_routes_resolved_scope_to_langgraph(self) -> None:
        result = self.service.ask(
            "What was Microsoft's total revenue in fiscal 2025 annual 10-K?",
            thread_id="ui-conversation-7",
        )

        self.assertEqual(result.decision, Decision.ANSWERED)
        self.assertEqual(len(self.service.agent_graph.calls), 1)
        inputs, config = self.service.agent_graph.calls[0]
        self.assertNotEqual(config["configurable"]["thread_id"], "ui-conversation-7")
        user_content = inputs["messages"][-1]["content"]
        self.assertIn("Application-resolved scope (locked)", user_content)
        self.assertIn('"MSFT"', user_content)
        self.assertIn('"2025"', user_content)

    def test_ask_passes_bounded_citation_free_history_even_with_conversation_id(self) -> None:
        result = self.service.ask(
            "What about its operating income?",
            history=[
                {"role": "user", "content": "What was Microsoft's 2025 revenue?"},
                {"role": "assistant", "content": "Revenue was $281,724 million [S999]."},
            ],
            previous_scope=Scope(("MSFT",), ("2025",), "10K"),
            thread_id="ui-conversation-7",
        )

        self.assertEqual(result.decision, Decision.ANSWERED)
        inputs, _ = self.service.agent_graph.calls[0]
        self.assertEqual(len(inputs["messages"]), 3)
        self.assertNotIn("[S999]", inputs["messages"][1]["content"])

    def test_incomplete_scope_is_clarified_before_graph_invocation(self) -> None:
        result = self.service.ask("What was total revenue?")

        self.assertEqual(result.decision, Decision.CLARIFY)
        self.assertEqual(self.service.agent_graph.calls, [])

    def test_graph_failure_returns_a_structured_error(self) -> None:
        self.service.agent_graph = FakeGraph(error=TimeoutError("model timed out"))

        result = self.service.process_query(
            {"messages": [{"role": "user", "content": "Revenue?"}]},
            thread_id="conversation-2",
            scope=Scope(("MSFT",), ("2025",), "10K"),
        )

        self.assertEqual(result.decision, Decision.ERROR)
        self.assertEqual(result.trace["error_type"], "TimeoutError")
        self.assertTrue(self.service.logger.records)

    def test_provider_connection_failure_explains_the_problem_in_both_graph_paths(self) -> None:
        error = APIConnectionError(
            message="private provider diagnostic",
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )

        class FailedStream:
            def stream(self, inputs, *, config, stream_mode):
                yield {"messages": inputs["messages"]}
                raise error

        for graph in (FakeGraph(error=error), FailedStream()):
            with self.subTest(graph=type(graph).__name__):
                self.service.agent_graph = graph
                self.service.require_validated_answer = True

                result = self.service.process_query(
                    {"messages": [{"role": "user", "content": "Revenue?"}]},
                    thread_id="connection-failure",
                    scope=Scope(("MSFT",), ("2025",), "10K"),
                )

                self.assertEqual(result.decision, Decision.ERROR)
                self.assertEqual(result.trace["error_code"], "provider_unavailable")
                self.assertIn("could not connect", result.answer)
                self.assertNotIn("private provider diagnostic", result.answer)


    def test_tool_quota_failure_is_not_reported_as_a_bad_question_or_missing_evidence(self):
        payload = {"plan_validated": False, "plan_attempt_exhausted": True,
            "error": "Request planning failed.", "error_code": "provider_quota_exhausted",
            "error_type": "RateLimitError", "retryable": False}

        class QuotaGraph:
            def invoke(self, inputs, *, config):
                return {"messages": [*inputs["messages"], {"type": "tool", "name": "plan_financial_request",
                    "tool_call_id": "plan-1", "content": json.dumps(payload)},
                    {"type": "ai", "content": "Please rephrase."}]}

        class QuotaStream(QuotaGraph):
            def stream(self, inputs, *, config, stream_mode):
                yield self.invoke(inputs, config=config)

        for graph in (QuotaGraph(), QuotaStream()):
            with self.subTest(graph=type(graph).__name__):
                self.service.agent_graph = graph
                self.service.require_validated_answer = True
                result = self.service.process_query({"messages": [{"role": "user", "content": "Revenue?"}]},
                    thread_id="quota-failure", scope=Scope(("MSFT",), ("2025",), "10K"))
                self.assertEqual(result.decision, Decision.ERROR)
                self.assertEqual(result.trace["error_code"], "provider_quota_exhausted")
                self.assertFalse(result.trace["retryable"])
                self.assertIn("billing", result.answer)
                self.assertNotIn("rephrase", result.answer)


if __name__ == "__main__":
    unittest.main()
