from __future__ import annotations

import unittest
from decimal import Decimal

from pydantic import ValidationError

from src.financial.calculator import CalculationOperation, CalculationRequest
from src.financial.models import ValidatedFinancialFact, ValueType
from src.tools.financial_calculator import FinancialCalculatorTool


def fact(
    fact_id: str,
    value: str,
    *,
    base_value: str | None = None,
    value_type: ValueType = ValueType.CURRENCY,
    unit: str = "USD",
    currency: str | None = "USD",
    scale: str | None = "millions",
    source_id: str = "S1",
) -> ValidatedFinancialFact:
    return ValidatedFinancialFact(
        fact_id=fact_id,
        ticker="MSFT",
        metric="Revenue",
        period="2024",
        raw_value=value,
        numeric_value=Decimal(value),
        base_value=Decimal(base_value or value) * (
            Decimal("1000000") if base_value is None and scale == "millions" else Decimal("1")
        ),
        value_type=value_type,
        normalized_unit=unit,
        currency=currency,
        scale=scale,
        source_id=source_id,
        evidence_excerpt=f"Revenue {value}",
        validation_checks=("test",),
    )


class FinancialCalculatorToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = FinancialCalculatorTool()

    def test_percentage_and_absolute_change_handle_different_scales(self) -> None:
        old = fact("F1", "3.3", base_value="3300000000", scale="billions")
        new = fact("F2", "3500", base_value="3500000000", scale="millions", source_id="S2")
        facts = {old.fact_id: old, new.fact_id: new}

        percentage = self.tool.execute(
            CalculationRequest(
                operation=CalculationOperation.PERCENTAGE_CHANGE,
                input_fact_ids=("F1", "F2"),
            ),
            available_facts=facts,
        )
        absolute = self.tool.execute(
            CalculationRequest(
                operation=CalculationOperation.ABSOLUTE_CHANGE,
                input_fact_ids=("F1", "F2"),
            ),
            available_facts=facts,
        )

        self.assertAlmostEqual(float(percentage.result), 6.060606, places=5)
        self.assertEqual(percentage.result_unit, "percent")
        self.assertEqual(percentage.source_ids, ("S1", "S2"))
        self.assertEqual(absolute.result, Decimal("200"))
        self.assertEqual(absolute.scale, "millions")
        self.assertEqual(absolute.substituted_formula, "3500 - 3300")

    def test_percentage_point_and_basis_point_changes(self) -> None:
        old = fact(
            "F1",
            "19.0",
            base_value="19.0",
            value_type=ValueType.PERCENT,
            unit="percent",
            currency=None,
            scale=None,
        )
        new = fact(
            "F2",
            "18.2",
            base_value="18.2",
            value_type=ValueType.PERCENT,
            unit="percent",
            currency=None,
            scale=None,
        )
        facts = {"F1": old, "F2": new}

        points = self.tool.execute(
            CalculationRequest(
                operation="percentage_point_change",
                input_fact_ids=("F1", "F2"),
            ),
            available_facts=facts,
        )
        bps = self.tool.execute(
            CalculationRequest(
                operation="basis_point_change",
                input_fact_ids=("F1", "F2"),
            ),
            available_facts=facts,
        )

        self.assertEqual(points.result, Decimal("-0.8"))
        self.assertEqual(points.result_unit, "percentage_points")
        self.assertEqual(bps.result, Decimal("-80.0"))
        self.assertEqual(bps.result_unit, "basis_points")

    def test_sum_ratio_and_cagr(self) -> None:
        first = fact("F1", "100")
        second = fact("F2", "121", source_id="S2")
        facts = {"F1": first, "F2": second}

        summed = self.tool.execute(
            CalculationRequest(operation="sum", input_fact_ids=("F1", "F2")),
            available_facts=facts,
        )
        ratio = self.tool.execute(
            CalculationRequest(operation="ratio", input_fact_ids=("F2", "F1")),
            available_facts=facts,
        )
        cagr = self.tool.execute(
            CalculationRequest(
                operation="cagr",
                input_fact_ids=("F1", "F2"),
                periods=2,
            ),
            available_facts=facts,
        )

        self.assertEqual(summed.result, Decimal("221"))
        self.assertEqual(ratio.result, Decimal("1.21"))
        self.assertAlmostEqual(float(cagr.result), 10.0, places=8)

    def test_rejects_unknown_facts_zero_denominators_and_unit_mismatches(self) -> None:
        usd = fact("F1", "100")
        zero = fact("F0", "0")
        eur = fact("F2", "100", unit="EUR", currency="EUR")

        with self.assertRaisesRegex(ValueError, "Unknown or unvalidated"):
            self.tool.execute(
                CalculationRequest(
                    operation="ratio",
                    input_fact_ids=("F1", "missing"),
                ),
                available_facts={"F1": usd},
            )
        with self.assertRaisesRegex(ValueError, "denominator is zero"):
            self.tool.execute(
                CalculationRequest(
                    operation="ratio",
                    input_fact_ids=("F1", "F0"),
                ),
                available_facts={"F1": usd, "F0": zero},
            )
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.tool.execute(
                CalculationRequest(
                    operation="sum",
                    input_fact_ids=("F1", "F2"),
                ),
                available_facts={"F1": usd, "F2": eur},
            )

    def test_temporal_change_rejects_cross_company_or_different_metrics(self) -> None:
        old = fact("F1", "100")
        other_company = fact("F2", "110")
        other_company = other_company.model_copy(update={"ticker": "TSLA"})
        other_metric = fact("F3", "110")
        other_metric = other_metric.model_copy(update={"metric": "Net income"})

        for newer, message in (
            (other_company, "same company"),
            (other_metric, "same financial metric"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.tool.execute(
                        CalculationRequest(
                            operation="percentage_change",
                            input_fact_ids=("F1", newer.fact_id),
                        ),
                        available_facts={"F1": old, newer.fact_id: newer},
                    )

    def test_temporal_change_accepts_equivalent_filing_row_names(self) -> None:
        for earlier, later in (
            ("Revenue", "Total revenue"),
            ("Operating income", "Income from operations"),
        ):
            with self.subTest(earlier=earlier, later=later):
                old = fact("F1", "100").model_copy(update={"metric": earlier})
                new = fact("F2", "110").model_copy(update={"metric": later})
                result = self.tool.execute(
                    CalculationRequest(
                        operation="percentage_change",
                        input_fact_ids=("F1", "F2"),
                    ),
                    available_facts={"F1": old, "F2": new},
                )
                self.assertEqual(result.result, Decimal("10.0"))

    def test_request_rejects_wrong_arity_duplicate_ids_and_irrelevant_periods(self) -> None:
        invalid_payloads = (
            {"operation": "ratio", "input_fact_ids": ("F1",)},
            {"operation": "sum", "input_fact_ids": ("F1", "F1")},
            {"operation": "sum", "input_fact_ids": ("F1",), "periods": 2},
            {"operation": "cagr", "input_fact_ids": ("F1", "F2")},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValidationError):
                    CalculationRequest(**payload)


if __name__ == "__main__":
    unittest.main()
