from __future__ import annotations

import unittest
from decimal import Decimal

from src.financial.models import CandidateFinancialFact, ValueType
from src.financial.value_normalizer import (
    FinancialValueNormalizer,
    ValueNormalizationError,
)


def candidate(raw_value: str, **overrides) -> CandidateFinancialFact:
    values = {
        "ticker": "MSFT",
        "metric": "Revenue",
        "period": "2024",
        "raw_value": raw_value,
        "unit": None,
        "scale": None,
        "currency": None,
        "source_id": "S1",
        "evidence_excerpt": f"Revenue {raw_value}",
    }
    values.update(overrides)
    return CandidateFinancialFact(**values)


class FinancialValueNormalizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.normalizer = FinancialValueNormalizer()

    def test_currency_parentheses_and_scale_are_preserved(self) -> None:
        result = self.normalizer.normalize(
            candidate("$(4,550)", scale="millions")
        )

        self.assertEqual(result.numeric_value, Decimal("-4550"))
        self.assertEqual(result.base_value, Decimal("-4550000000"))
        self.assertEqual(result.value_type, ValueType.CURRENCY)
        self.assertEqual(result.currency, "USD")
        self.assertEqual(result.scale, "millions")

    def test_declared_currency_symbol_is_normalized_to_iso_code(self) -> None:
        result = self.normalizer.normalize(
            candidate("$94,827", currency="$", scale="millions")
        )

        self.assertEqual(result.currency, "USD")
        self.assertEqual(result.base_value, Decimal("94827000000"))

    def test_percent_and_basis_points_are_not_scaled(self) -> None:
        percent = self.normalizer.normalize(candidate("(1.4)%"))
        basis_points = self.normalizer.normalize(
            candidate("80", unit="basis points")
        )

        self.assertEqual(percent.numeric_value, Decimal("-1.4"))
        self.assertEqual(percent.base_value, Decimal("-1.4"))
        self.assertEqual(percent.value_type, ValueType.PERCENT)
        self.assertEqual(basis_points.numeric_value, Decimal("80"))
        self.assertEqual(basis_points.value_type, ValueType.BASIS_POINTS)

    def test_per_share_value_is_not_treated_as_total_currency(self) -> None:
        result = self.normalizer.normalize(
            candidate("$2.50", unit="USD per share", scale="ones")
        )

        self.assertEqual(result.numeric_value, Decimal("2.50"))
        self.assertEqual(result.base_value, Decimal("2.50"))
        self.assertEqual(result.value_type, ValueType.PER_SHARE)

    def test_rejects_blank_or_ambiguous_values(self) -> None:
        for raw_value in ("—", "N/A", "$24 to $30"):
            with self.subTest(raw_value=raw_value):
                with self.assertRaises(ValueNormalizationError):
                    self.normalizer.normalize(candidate(raw_value))

    def test_rejects_conflicting_currency_and_scale_metadata(self) -> None:
        invalid = (
            candidate("$3.3 billion", scale="millions"),
            candidate("EUR 20", currency="USD"),
            candidate("18.2%", currency="USD"),
        )
        for value in invalid:
            with self.subTest(raw_value=value.raw_value):
                with self.assertRaises(ValueNormalizationError):
                    self.normalizer.normalize(value)


if __name__ == "__main__":
    unittest.main()
