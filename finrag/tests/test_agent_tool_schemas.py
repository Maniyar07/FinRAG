from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.agent.tools import (
    CalculatorInput,
    RequestPlanInput,
    SearchFilingsInput,
    StatementLookupInput,
)


class AgentToolSchemaTests(unittest.TestCase):
    def test_search_schema_separates_filing_year_from_reported_period(self) -> None:
        request = SearchFilingsInput(
            ticker="msft",
            filing_year="2025",
            doc_type="10K",
            query="FY2024 total revenue",
        )

        self.assertEqual(request.ticker, "MSFT")
        self.assertEqual(request.filing_year, "2025")
        description = SearchFilingsInput.model_json_schema()["properties"]["filing_year"]["description"]
        self.assertIn("source filing", description)
        self.assertIn("different reported period", description)

    def test_legacy_year_argument_is_rejected_as_ambiguous(self) -> None:
        with self.assertRaises(ValidationError):
            SearchFilingsInput(
                ticker="MSFT",
                year="2025",
                doc_type="10K",
                query="FY2024 total revenue",
            )

    def test_statement_lookup_uses_source_filing_year(self) -> None:
        request = StatementLookupInput(
            ticker="MSFT",
            filing_year="2025",
            statement_type="income_statement",
        )

        self.assertEqual(request.filing_year, "2025")

    def test_request_plan_separates_source_and_reported_periods(self) -> None:
        plan = RequestPlanInput.model_validate({
            "requirements": [{
                "requirement_id": "revenue_2024",
                "ticker": "msft",
                "source_year": "2025",
                "doc_type": "10K",
                "reported_period": "FY2024",
                "metric": "total revenue",
                "qualifiers": ["GAAP"],
            }],
            "calculations": [],
        })

        requirement = plan.requirements[0]
        self.assertEqual(requirement.ticker, "MSFT")
        self.assertEqual(requirement.source_year, "2025")
        self.assertEqual(requirement.reported_period, "FY2024")

    def test_request_plan_rejects_calculations_with_unknown_requirements(self) -> None:
        with self.assertRaises(ValidationError):
            RequestPlanInput.model_validate({
                "requirements": [],
                "calculations": [{
                    "operation": "cagr",
                    "input_requirement_ids": ["start", "end"],
                    "periods": 2,
                }],
            })

    def test_calculator_schema_supports_all_rounding_and_multi_sum_inputs(self) -> None:
        request = CalculatorInput(
            operation="sum",
            fact_ids=["F1", "F2", "F3"],
            decimal_places=1,
        )

        self.assertEqual(request.fact_ids, ["F1", "F2", "F3"])
        self.assertEqual(request.decimal_places, 1)

    def test_calculator_schema_rejects_mixed_fact_id_forms(self) -> None:
        with self.assertRaises(ValidationError):
            CalculatorInput(
                operation="sum",
                fact_id_1="F1",
                fact_ids=["F2", "F3"],
            )

    def test_duplicate_binary_fact_ids_reach_tool_for_safe_skip_response(self) -> None:
        request = CalculatorInput(
            operation="percentage_change",
            fact_id_1="F1",
            fact_id_2="F1",
        )

        self.assertEqual(request.fact_id_1, request.fact_id_2)


if __name__ == "__main__":
    unittest.main()
