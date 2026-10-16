from __future__ import annotations

import unittest
from decimal import Decimal

from pydantic import ValidationError

from src.financial.calculator import (
    CalculationOperation,
    CalculationRequest,
    FinancialCalculator,
)
from src.financial.models import ValidatedFinancialFact, ValueType


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
    accounting_basis: str | None = None,
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
        accounting_basis=accounting_basis,
        source_id=source_id,
        evidence_excerpt=f"Revenue {value}",
        validation_checks=("test",),
    )


class FinancialCalculatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calculator = FinancialCalculator()

    def test_percentage_and_absolute_change_handle_different_scales(self) -> None:
        old = fact("F1", "3.3", base_value="3300000000", scale="billions")
        new = fact("F2", "3500", base_value="3500000000", scale="millions", source_id="S2")
        facts = {old.fact_id: old, new.fact_id: new}

        percentage = self.calculator.calculate(
            CalculationRequest(
                operation=CalculationOperation.PERCENTAGE_CHANGE,
                input_fact_ids=("F1", "F2"),
            ),
            available_facts=facts,
        )
        absolute = self.calculator.calculate(
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

    def test_missing_accounting_basis_does_not_block_compatible_calculation(self) -> None:
        first = fact("F1", "100", accounting_basis="null")
        second = fact("F2", "25", accounting_basis="US GAAP", source_id="S2")

        result = self.calculator.calculate(
            CalculationRequest(operation="sum", input_fact_ids=("F1", "F2")),
            available_facts={"F1": first, "F2": second},
        )

        self.assertEqual(result.result, Decimal("125"))
        self.assertIsNone(first.accounting_basis)

    def test_distinct_accounting_bases_remain_incompatible(self) -> None:
        gaap = fact("F1", "100", accounting_basis="US GAAP")
        non_gaap = fact("F2", "25", accounting_basis="Non-GAAP", source_id="S2")

        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.calculator.calculate(
                CalculationRequest(operation="sum", input_fact_ids=("F1", "F2")),
                available_facts={"F1": gaap, "F2": non_gaap},
            )

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

        points = self.calculator.calculate(
            CalculationRequest(
                operation="percentage_point_change",
                input_fact_ids=("F1", "F2"),
            ),
            available_facts=facts,
        )
        bps = self.calculator.calculate(
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

        summed = self.calculator.calculate(
            CalculationRequest(operation="sum", input_fact_ids=("F1", "F2")),
            available_facts=facts,
        )
        ratio = self.calculator.calculate(
            CalculationRequest(operation="ratio", input_fact_ids=("F2", "F1")),
            available_facts=facts,
        )
        cagr = self.calculator.calculate(
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

    def test_percentage_of_rounds_after_unit_normalization(self) -> None:
        expense = fact("F1", "32488", scale="millions")
        revenue = fact("F2", "281724", scale="millions", source_id="S2")

        result = self.calculator.calculate(
            CalculationRequest(
                operation="percentage_of",
                input_fact_ids=("F1", "F2"),
                decimal_places=1,
            ),
            available_facts={"F1": expense, "F2": revenue},
        )

        self.assertEqual(result.result, Decimal("11.5"))
        self.assertEqual(result.result_unit, "percent")
        self.assertEqual(result.input_fact_ids, ("F1", "F2"))
        self.assertEqual(result.source_ids, ("S1", "S2"))

    def test_sum_accepts_multiple_validated_fact_ids(self) -> None:
        facts = {
            f"F{index}": fact(f"F{index}", str(value), source_id=f"S{index}")
            for index, value in enumerate((10, 20, 30), start=1)
        }

        result = self.calculator.calculate(
            CalculationRequest(operation="sum", input_fact_ids=tuple(facts)),
            available_facts=facts,
        )

        self.assertEqual(result.result, Decimal("60"))
        self.assertEqual(result.input_fact_ids, ("F1", "F2", "F3"))

    def test_product_absolute_value_and_percent_complement(self) -> None:
        income = fact("F1", "100")
        tax_rate = fact(
            "F2",
            "21",
            base_value="21",
            value_type=ValueType.PERCENT,
            unit="percent",
            currency=None,
            scale=None,
            source_id="S2",
        )
        capex = fact("F3", "-25", source_id="S3")
        facts = {item.fact_id: item for item in (income, tax_rate, capex)}

        complement = self.calculator.calculate(
            CalculationRequest(
                operation="percent_complement", input_fact_ids=("F2",)
            ),
            available_facts=facts,
        )
        absolute = self.calculator.calculate(
            CalculationRequest(operation="absolute_value", input_fact_ids=("F3",)),
            available_facts=facts,
        )
        complement_fact = tax_rate.model_copy(
            update={
                "fact_id": "F4",
                "numeric_value": complement.result,
                "base_value": complement.result,
            }
        )
        facts["F4"] = complement_fact
        product = self.calculator.calculate(
            CalculationRequest(operation="product", input_fact_ids=("F1", "F4")),
            available_facts=facts,
        )

        self.assertEqual(complement.result, Decimal("79"))
        self.assertEqual(absolute.result, Decimal("25"))
        self.assertEqual(product.result, Decimal("79.00"))
        self.assertEqual(product.result_unit, "currency")

    def test_currency_divided_by_share_count_returns_per_share(self) -> None:
        revenue = fact("F1", "100", scale="millions")
        shares = fact(
            "F2",
            "20",
            value_type=ValueType.COUNT,
            unit="shares",
            currency=None,
            scale="millions",
            source_id="S2",
        )
        result = self.calculator.calculate(
            CalculationRequest(operation="ratio", input_fact_ids=("F1", "F2")),
            available_facts={"F1": revenue, "F2": shares},
        )

        self.assertEqual(result.result, Decimal("5"))
        self.assertEqual(result.result_unit, "per_share")
        self.assertEqual(result.currency, "USD")

    def test_rejects_unknown_facts_zero_denominators_and_unit_mismatches(self) -> None:
        usd = fact("F1", "100")
        zero = fact("F0", "0")
        eur = fact("F2", "100", unit="EUR", currency="EUR")

        with self.assertRaisesRegex(ValueError, "Unknown or unvalidated"):
            self.calculator.calculate(
                CalculationRequest(
                    operation="ratio",
                    input_fact_ids=("F1", "missing"),
                ),
                available_facts={"F1": usd},
            )
        with self.assertRaisesRegex(ValueError, "denominator is zero"):
            self.calculator.calculate(
                CalculationRequest(
                    operation="ratio",
                    input_fact_ids=("F1", "F0"),
                ),
                available_facts={"F1": usd, "F0": zero},
            )
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.calculator.calculate(
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
                    self.calculator.calculate(
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
                result = self.calculator.calculate(
                    CalculationRequest(
                        operation="percentage_change",
                        input_fact_ids=("F1", "F2"),
                    ),
                    available_facts={"F1": old, "F2": new},
                )
                self.assertEqual(result.result, Decimal("10.0"))

    def test_bounds_cannot_be_used_as_exact_calculation_inputs(self) -> None:
        request = CalculationRequest(operation="absolute_value", input_fact_ids=("F1",))
        for qualifier in ("over", "exceed", "exceeds", "exceeded", "below", "at least", "up to"):
            with self.subTest(qualifier=qualifier):
                bounded = fact("F1", "12").model_copy(update={"value_qualifier": qualifier})
                with self.assertRaisesRegex(ValueError, "bounded value"):
                    self.calculator.calculate(request, {"F1": bounded})

    def test_request_rejects_wrong_arity_duplicate_ids_and_irrelevant_periods(self) -> None:
        invalid_payloads = (
            {"operation": "ratio", "input_fact_ids": ("F1",)},
            {"operation": "sum", "input_fact_ids": ("F1", "F1")},
            {"operation": "sum", "input_fact_ids": ("F1",), "periods": 2},
            {"operation": "cagr", "input_fact_ids": ("F1", "F2")},
            {"operation": "absolute_value", "input_fact_ids": ("F1", "F2")},
            {"operation": "percent_complement", "input_fact_ids": ("F1", "F2")},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValidationError):
                    CalculationRequest(**payload)


if __name__ == "__main__":
    unittest.main()
