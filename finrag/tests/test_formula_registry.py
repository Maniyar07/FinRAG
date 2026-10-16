from __future__ import annotations

import unittest

from src.financial.formula_registry import load_formula_registry


class FormulaRegistryTests(unittest.TestCase):
    def test_catalog_loads_and_matches_financial_terms(self) -> None:
        registry = load_formula_registry()

        self.assertGreaterEqual(len(registry.formulas), 20)
        self.assertEqual(
            registry.match("Calculate free cash flow margin").formula_id,
            "free_cash_flow_margin",
        )
        self.assertEqual(
            registry.match("What is the current ratio?").formula_id,
            "current_ratio",
        )


if __name__ == "__main__":
    unittest.main()
