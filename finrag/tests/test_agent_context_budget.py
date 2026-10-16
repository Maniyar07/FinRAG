from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from src.agent.tools import create_financial_tools
from src.config import (
    AGENT_MAX_ACTIONS,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_PARENTS,
    MAX_CONTEXT_TOKENS,
)
from src.financial.calculator import FinancialCalculator
from src.financial.models import FactValidationResult
from src.schemas import RetrievalBundle, Scope
from src.retrieval.token_budget import count_tokens


class CapturingRetriever:
    def __init__(self) -> None:
        self.limits: dict[str, int] = {}
        self.calls = 0

    def retrieve(self, query, scope, **kwargs):
        del query
        self.calls += 1
        self.limits = kwargs
        return RetrievalBundle(
            "evidence text",
            [{"id": f"source-{self.calls}", "parent_id": f"parent-{self.calls}"}],
            scope,
            1,
        )


class EmptyFactPipeline:
    def run(self, **kwargs):
        del kwargs
        return FactValidationResult()


def validate_plan(tools, runtime, groups):
    plan_tool = next(tool for tool in tools if tool.name == "plan_financial_request")
    requirements = [
        {
            "requirement_id": f"r{index}",
            "ticker": ticker,
            "source_year": year,
            "doc_type": doc_type or "10K",
            "reported_period": f"FY{year}",
            "metric": "total revenue",
        }
        for index, (ticker, year, doc_type) in enumerate(groups)
    ]
    call_id = "plan-validated"
    content = plan_tool.func(
        requirements=requirements,
        calculations=[],
        runtime=runtime,
    )
    runtime.state["messages"].extend([
        {"type": "ai", "tool_calls": [{
            "name": "plan_financial_request", "args": {}, "id": call_id,
        }]},
        {"type": "tool", "name": "plan_financial_request",
         "tool_call_id": call_id, "content": content},
    ])


class RequestContextBudgetTests(unittest.TestCase):
    def test_two_resolved_groups_share_the_request_evidence_budget(self) -> None:
        retriever = CapturingRetriever()
        permitted = Scope(
            tickers=("MSFT", "TSLA"),
            years=("2025",),
            doc_type="10K",
            requested_groups=(("MSFT", "2025"), ("TSLA", "2025")),
        )
        locked = {
            "tickers": ["MSFT", "TSLA"],
            "years": ["2025"],
            "doc_type": "10K",
            "required_doc_types": [],
            "requested_groups": [["MSFT", "2025"], ["TSLA", "2025"]],
        }
        runtime = SimpleNamespace(
            state={
                "messages": [
                    {
                        "type": "human",
                        "content": "Application-resolved scope (locked): "
                        + json.dumps(locked)
                        + "\nUser question: compare revenue",
                    }
                ]
            }
        )
        tools = create_financial_tools(
            retriever,
            FinancialCalculator(),
            permitted_scope=permitted,
            fact_pipeline=EmptyFactPipeline(),
        )
        validate_plan(tools, runtime, [("MSFT", "2025", "10K"), ("TSLA", "2025", "10K")])
        search_tool = next(tool for tool in tools if tool.name == "search_financial_documents")

        search_tool.func(
            ticker="MSFT",
            filing_year="2025",
            doc_type="10K",
            query="total revenue",
            runtime=runtime,
        )

        self.assertEqual(
            retriever.limits["max_parents"], (max(MAX_CONTEXT_PARENTS, 2) + 1) // 2
        )
        self.assertEqual(
            retriever.limits["max_chars"], (MAX_CONTEXT_CHARS + 1) // 2
        )
        self.assertEqual(
            retriever.limits["max_tokens"], (MAX_CONTEXT_TOKENS + 1) // 2
        )

    def test_refinement_consumes_only_remaining_group_budget(self) -> None:
        retriever = CapturingRetriever()
        permitted = Scope(
            tickers=("MSFT", "TSLA"), years=("2025",), doc_type="10K",
            requested_groups=(("MSFT", "2025"), ("TSLA", "2025")),
        )
        locked = {
            "tickers": ["MSFT", "TSLA"], "years": ["2025"],
            "doc_type": "10K", "required_doc_types": [],
            "requested_groups": [["MSFT", "2025"], ["TSLA", "2025"]],
        }
        runtime = SimpleNamespace(state={"messages": [{
            "type": "human",
            "content": "Application-resolved scope (locked): "
            + json.dumps(locked) + "\nUser question: compare revenue",
        }]})
        tools = create_financial_tools(
            retriever, FinancialCalculator(), permitted_scope=permitted,
            fact_pipeline=EmptyFactPipeline(),
        )
        validate_plan(tools, runtime, [("MSFT", "2025", "10K"), ("TSLA", "2025", "10K")])
        search = next(tool for tool in tools if tool.name == "search_financial_documents")
        first = json.loads(search.func(
            ticker="MSFT", filing_year="2025", doc_type="10K",
            query="total revenue", unresolved_requirement_ids=["r0"], runtime=runtime,
        ))
        first_limits = dict(retriever.limits)
        runtime.state["messages"].extend([
            {"type": "ai", "tool_calls": [{
                "name": "search_financial_documents",
                "args": {"ticker": "MSFT", "filing_year": "2025",
                         "doc_type": "10K", "query": "total revenue", "unresolved_requirement_ids": ["r0"]},
                "id": "search-1",
            }]},
            {"type": "tool", "name": "search_financial_documents",
             "tool_call_id": "search-1", "content": json.dumps(first)},
        ])

        search.func(
            ticker="MSFT", filing_year="2025", doc_type="10K",
            query="total revenue",
            unresolved_requirement_ids=["r0"],
            runtime=runtime,
        )

        self.assertEqual(retriever.calls, 2)
        self.assertEqual(retriever.limits["max_parents"], first_limits["max_parents"] - 1)
        self.assertEqual(
            retriever.limits["max_chars"],
            first_limits["max_chars"] - len("evidence text"),
        )
        self.assertEqual(
            retriever.limits["max_tokens"],
            first_limits["max_tokens"] - count_tokens("evidence text"),
        )

    def test_identical_search_reuses_prior_result_without_retrieving_again(self) -> None:
        retriever = CapturingRetriever()
        permitted = Scope(("MSFT",), ("2025",), "10K")
        locked = {"tickers": ["MSFT"], "years": ["2025"], "doc_type": "10K",
                  "required_doc_types": [], "requested_groups": [["MSFT", "2025"]]}
        runtime = SimpleNamespace(state={"messages": [{
            "type": "human", "content": "Application-resolved scope (locked): "
            + json.dumps(locked) + "\nUser question: revenue",
        }]})
        tools = create_financial_tools(
            retriever, FinancialCalculator(), permitted_scope=permitted,
            fact_pipeline=EmptyFactPipeline(),
        )
        validate_plan(tools, runtime, [("MSFT", "2025", "10K")])
        search = next(tool for tool in tools if tool.name == "search_financial_documents")
        first = json.loads(search.func(
            ticker="MSFT", filing_year="2025", doc_type="10K",
            query="total revenue", runtime=runtime,
        ))
        runtime.state["messages"].extend([
            {"type": "ai", "tool_calls": [{
                "name": "search_financial_documents",
                "args": {"ticker": "MSFT", "filing_year": "2025",
                         "doc_type": "10K", "query": "total revenue"},
                "id": "search-1",
            }]},
            {"type": "tool", "name": "search_financial_documents",
             "tool_call_id": "search-1", "content": json.dumps(first)},
        ])

        cached = json.loads(search.func(
            ticker="MSFT", filing_year="2025", doc_type="10K",
            query="total revenue", runtime=runtime,
        ))

        self.assertTrue(cached["already_retrieved"])
        self.assertTrue(cached["context_budget_exhausted"])
        self.assertEqual(retriever.calls, 1)

    def test_tool_action_limit_stops_an_agent_search_loop(self) -> None:
        retriever = CapturingRetriever()
        permitted = Scope(("MSFT",), ("2025",), "10K")
        locked = {"tickers": ["MSFT"], "years": ["2025"], "doc_type": "10K",
                  "required_doc_types": [], "requested_groups": [["MSFT", "2025"]]}
        messages = [{
            "type": "human", "content": "Application-resolved scope (locked): "
            + json.dumps(locked) + "\nUser question: revenue",
        }]
        for index in range(AGENT_MAX_ACTIONS):
            call_id = f"call-{index}"
            messages.extend([
                {"type": "ai", "tool_calls": [{
                    "name": "financial_calculator", "args": {}, "id": call_id,
                }]},
                {"type": "tool", "name": "financial_calculator",
                 "tool_call_id": call_id, "content": "{}"},
            ])
        runtime = SimpleNamespace(state={"messages": messages})
        tools = create_financial_tools(
            retriever, FinancialCalculator(), permitted_scope=permitted,
            fact_pipeline=EmptyFactPipeline(),
        )
        validate_plan(tools, runtime, [("MSFT", "2025", "10K")])
        search = next(tool for tool in tools if tool.name == "search_financial_documents")

        response = json.loads(search.func(
            ticker="MSFT", filing_year="2025", doc_type="10K",
            query="total revenue", runtime=runtime,
        ))

        self.assertTrue(response["action_budget_exhausted"])
        self.assertEqual(retriever.calls, 0)

    def test_same_turn_searches_share_their_group_budget(self) -> None:
        retriever = CapturingRetriever()
        permitted = Scope(
            ("MSFT", "TSLA"), ("2025",), "10K",
            requested_groups=(("MSFT", "2025"), ("TSLA", "2025")),
        )
        locked = {
            "tickers": ["MSFT", "TSLA"], "years": ["2025"], "doc_type": "10K",
            "required_doc_types": [],
            "requested_groups": [["MSFT", "2025"], ["TSLA", "2025"]],
        }
        current_calls = [{
            "name": "search_financial_documents",
            "args": {"ticker": ticker, "filing_year": "2025",
                     "doc_type": "10K", "query": query},
            "id": f"search-{index}",
        } for index, (ticker, query) in enumerate((
            ("MSFT", "revenue amount"), ("TSLA", "revenue growth")
        ))]
        runtime = SimpleNamespace(
            tool_call_id="search-0",
            state={"messages": [
                {"type": "human", "content": "Application-resolved scope (locked): "
                 + json.dumps(locked) + "\nUser question: compare revenue"},
                {"type": "ai", "tool_calls": current_calls},
            ]},
        )
        tools = create_financial_tools(
            retriever, FinancialCalculator(), permitted_scope=permitted,
            fact_pipeline=EmptyFactPipeline(),
        )
        validate_plan(tools, runtime, [("MSFT", "2025", "10K"), ("TSLA", "2025", "10K")])
        search = next(tool for tool in tools if tool.name == "search_financial_documents")

        search.func(
            ticker="MSFT", filing_year="2025", doc_type="10K",
            query="revenue amount", runtime=runtime,
        )

        self.assertEqual(retriever.limits["max_parents"], MAX_CONTEXT_PARENTS // 2)
        self.assertEqual(retriever.limits["max_chars"], MAX_CONTEXT_CHARS // 2)
        self.assertEqual(retriever.limits["max_tokens"], MAX_CONTEXT_TOKENS // 2)

    def test_parallel_searches_for_one_group_share_one_retrieval(self) -> None:
        retriever = CapturingRetriever()
        permitted = Scope(("MSFT",), ("2025",), "10K")
        locked = {
            "tickers": ["MSFT"], "years": ["2025"], "doc_type": "10K",
            "required_doc_types": [], "requested_groups": [["MSFT", "2025"]],
        }
        calls = [{
            "name": "search_financial_documents",
            "args": {"ticker": "MSFT", "filing_year": "2025", "doc_type": "10K", "query": query},
            "id": f"search-{index}",
        } for index, query in enumerate(("revenue amount", "operating income"))]
        runtime = SimpleNamespace(
            tool_call_id="search-0",
            state={"messages": [
                {"type": "human", "content": "Application-resolved scope (locked): "
                 + json.dumps(locked) + "\nUser question: compare metrics"},
                {"type": "ai", "tool_calls": calls},
            ]},
        )
        tools = create_financial_tools(
            retriever, FinancialCalculator(), permitted_scope=permitted,
            fact_pipeline=EmptyFactPipeline(),
        )
        validate_plan(tools, runtime, [("MSFT", "2025", "10K")])

        search = next(tool for tool in tools if tool.name == "search_financial_documents")
        response = json.loads(search.func(
            ticker="MSFT", filing_year="2025", doc_type="10K",
            query="revenue amount", runtime=runtime,
        ))

        self.assertIn("context", response)
        self.assertEqual(retriever.calls, 1)
        runtime.tool_call_id = "search-1"
        sibling = json.loads(search.func(ticker="MSFT", filing_year="2025", doc_type="10K",
            query="operating income", runtime=runtime))
        self.assertTrue(sibling["shared_source_group_search"])
        self.assertEqual(retriever.calls, 1)

    def test_locked_scope_search_is_rejected_until_plan_is_validated(self) -> None:
        retriever = CapturingRetriever()
        permitted = Scope(("MSFT",), ("2025",), "10K")
        locked = {
            "tickers": ["MSFT"], "years": ["2025"], "doc_type": "10K",
            "required_doc_types": [], "requested_groups": [["MSFT", "2025"]],
        }
        runtime = SimpleNamespace(state={"messages": [{
            "type": "human", "content": "Application-resolved scope (locked): "
            + json.dumps(locked) + "\nUser question: revenue",
        }]})
        tools = create_financial_tools(
            retriever, FinancialCalculator(), permitted_scope=permitted,
            fact_pipeline=EmptyFactPipeline(),
        )
        search = next(tool for tool in tools if tool.name == "search_financial_documents")

        response = json.loads(search.func(
            ticker="MSFT", filing_year="2025", doc_type="10K",
            query="total revenue", runtime=runtime,
        ))

        self.assertTrue(response["request_plan_required"])
        self.assertEqual(retriever.calls, 0)

    def test_out_of_scope_period_year_uses_only_the_locked_source_year(self) -> None:
        retriever = CapturingRetriever()
        permitted = Scope(("TSLA",), ("2024",), "10K")
        locked = {"tickers": ["TSLA"], "years": ["2024"], "doc_type": "10K",
                  "required_doc_types": [], "requested_groups": [["TSLA", "2024"]]}
        runtime = SimpleNamespace(state={"messages": [{
            "type": "human", "content": "Application-resolved scope (locked): "
            + json.dumps(locked) + "\nUser question: compare FY2023 with FY2024",
        }]})
        tools = create_financial_tools(
            retriever, FinancialCalculator(), permitted_scope=permitted,
            fact_pipeline=EmptyFactPipeline(),
        )
        validate_plan(tools, runtime, [("TSLA", "2024", "10K")])
        search = next(tool for tool in tools if tool.name == "search_financial_documents")

        response = json.loads(search.func(
            ticker="TSLA", filing_year="2023", doc_type="10K",
            query="FY2023 revenue from the 2024 filing", runtime=runtime,
        ))

        self.assertEqual(response.get("filing_year"), "2024", response)
        self.assertEqual(response["scope_adjustment"], {
            "requested_source_year": "2023", "locked_source_year": "2024",
        })


if __name__ == "__main__":
    unittest.main()
