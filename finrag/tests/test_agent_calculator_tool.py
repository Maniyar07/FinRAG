from __future__ import annotations

import json
import unittest
from decimal import Decimal
from types import SimpleNamespace

from langchain_core.messages import ToolMessage

from src.agent.tools import create_financial_tools
from src.financial.calculator import FinancialCalculator
from src.financial.models import ValidatedFinancialFact, ValueType


def _fact(fact_id: str, value: str, source_id: str) -> ValidatedFinancialFact:
    amount = Decimal(value)
    return ValidatedFinancialFact(
        fact_id=fact_id,
        ticker="MSFT",
        metric="Financial amount",
        period="2025",
        raw_value=value,
        numeric_value=amount,
        base_value=amount * Decimal("1000000"),
        value_type=ValueType.CURRENCY,
        normalized_unit="USD",
        currency="USD",
        scale="millions",
        source_id=source_id,
        evidence_excerpt=f"Financial amount {value}",
        validation_checks=("test",),
    )


class AgentCalculatorToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.facts = {
            "F1": _fact("F1", "32488", "S1"),
            "F2": _fact("F2", "281724", "S2"),
            "F3": _fact("F3", "10", "S3"),
        }
        search_result = ToolMessage(
            name="search_financial_documents",
            tool_call_id="search-1",
            content=json.dumps(
                {"validated_facts": [fact.model_dump(mode="json") for fact in self.facts.values()]}
            ),
        )
        self.runtime = SimpleNamespace(state={"messages": [search_result]})
        tools = create_financial_tools(
            retriever=None,
            calculator=FinancialCalculator(),
            fact_pipeline=object(),
        )
        self.calculator_tool = next(tool for tool in tools if tool.name == "financial_calculator")

    def test_tool_forwards_rounding_to_percentage_calculation(self) -> None:
        response = json.loads(
            self.calculator_tool.func(
                fact_id_1="F1",
                fact_id_2="F2",
                operation="percentage_of",
                decimal_places=1,
                runtime=self.runtime,
            )
        )

        self.assertEqual(response["result"], "11.5")
        self.assertEqual(response["result_unit"], "percent")
        self.assertEqual(response["input_fact_ids"], ["F1", "F2"])

    def test_tool_accepts_three_ids_for_sum(self) -> None:
        response = json.loads(
            self.calculator_tool.func(
                fact_ids=["F1", "F2", "F3"],
                operation="sum",
                runtime=self.runtime,
            )
        )

        self.assertEqual(response["result"], "314222")
        self.assertEqual(response["input_fact_ids"], ["F1", "F2", "F3"])

    def test_duplicate_period_fact_is_reported_as_skipped_not_a_calculation_error(self) -> None:
        response = json.loads(self.calculator_tool.func(
            fact_id_1="F1",
            fact_id_2="F1",
            operation="percentage_change",
            runtime=self.runtime,
        ))

        self.assertTrue(response["calculation_skipped"])
        self.assertNotIn("result", response)
        self.assertNotIn("error", response)

    def test_calculator_requires_facts_to_match_the_planned_metric_and_period(self) -> None:
        start = _fact("F_START", "100", "S1").model_copy(
            update={"metric": "Total revenue", "period": "2024"}
        )
        end = _fact("F_END", "110", "S2").model_copy(
            update={"metric": "Total revenue", "period": "2025"}
        )
        locked = {
            "tickers": ["MSFT"], "years": ["2025"], "doc_type": "10K",
            "required_doc_types": [], "requested_groups": [["MSFT", "2025"]],
        }
        self.runtime = SimpleNamespace(state={"messages": [
            {"type": "human", "content": "Application-resolved scope (locked): "
             + json.dumps(locked) + "\nUser question: revenue change"},
            ToolMessage(
                name="search_financial_documents", tool_call_id="search-1",
                content=json.dumps({
                    "ticker": "MSFT", "filing_year": "2025", "doc_type": "10K",
                    "sources": [
                        {"id": "S1", "ticker": "MSFT", "fiscal_year": "2025", "doc_type": "10K"},
                        {"id": "S2", "ticker": "MSFT", "fiscal_year": "2025", "doc_type": "10K"},
                    ],
                    "validated_facts": [
                        start.model_dump(mode="json"), end.model_dump(mode="json"),
                    ],
                }),
            ),
        ]})
        tools = create_financial_tools(
            retriever=None, calculator=FinancialCalculator(), fact_pipeline=object(),
        )
        planner = next(tool for tool in tools if tool.name == "plan_financial_request")
        plan = planner.func(
            requirements=[
                {"requirement_id": "rev24", "ticker": "MSFT", "source_year": "2025",
                 "doc_type": "10K", "reported_period": "FY2024", "metric": "revenue"},
                {"requirement_id": "rev25", "ticker": "MSFT", "source_year": "2025",
                 "doc_type": "10K", "reported_period": "FY2025", "metric": "revenue"},
            ],
            calculations=[{
                "operation": "percentage_change",
                "input_requirement_ids": ["rev24", "rev25"],
            }],
            runtime=self.runtime,
        )
        self.runtime.state["messages"].extend([
            {"type": "ai", "tool_calls": [{
                "name": "plan_financial_request", "args": {}, "id": "plan-1",
            }]},
            {"type": "tool", "name": "plan_financial_request",
             "tool_call_id": "plan-1", "content": plan},
        ])
        calculator = next(tool for tool in tools if tool.name == "financial_calculator")

        response = json.loads(calculator.func(
            fact_id_1="F_START", fact_id_2="F_END",
            operation="percentage_change", runtime=self.runtime,
        ))

        self.assertEqual(response["result"], "10.0")
        self.assertEqual(response["operation"], "percentage_change")


if __name__ == "__main__":
    unittest.main()
