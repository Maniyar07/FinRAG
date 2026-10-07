from __future__ import annotations

import unittest

from src.financial.formula_registry import load_formula_registry
from src.orchestration.formula_planner import formula_plan
from src.orchestration.planner import MultiHopPlanner, should_use_multihop
from src.schemas import Scope


class FormulaRegistryTests(unittest.TestCase):
    def test_catalog_loads_and_prefers_the_longest_alias(self) -> None:
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

    def test_gross_margin_compiles_to_a_generic_dependency_graph(self) -> None:
        plan = formula_plan(
            question="Calculate Microsoft's gross margin from its 2025 10-K.",
            permitted_scope=Scope(("MSFT",), ("2025",), "10K"),
        )

        self.assertEqual(len(plan.requirements), 2)
        self.assertEqual(
            [item.operation.value for item in plan.calculations],
            ["difference", "ratio"],
        )
        self.assertEqual(
            plan.calculations[-1].inputs[0].calculation_id,
            plan.calculations[0].calculation_id,
        )

    def test_quick_ratio_reuses_a_sum_as_the_ratio_numerator(self) -> None:
        plan = formula_plan(
            question="Calculate Tesla's quick ratio for 2025.",
            permitted_scope=Scope(("TSLA",), ("2025",), "10K"),
        )

        self.assertEqual(len(plan.requirements), 4)
        self.assertEqual(
            [item.operation.value for item in plan.calculations],
            ["sum", "ratio"],
        )

    def test_revenue_growth_compiles_across_two_source_years(self) -> None:
        plan = formula_plan(
            question="Calculate Microsoft's revenue growth from 2024 to 2025.",
            permitted_scope=Scope(
                ("MSFT",),
                ("2024", "2025"),
                "10K",
                requested_groups=(("MSFT", "2024"), ("MSFT", "2025")),
            ),
        )

        self.assertEqual(len(plan.requirements), 1)
        self.assertEqual(len(plan.calculations), 1)
        calculation = plan.calculations[0]
        self.assertEqual(calculation.operation.value, "percentage_change")
        self.assertEqual(
            [item.fiscal_year for item in calculation.inputs],
            ["2024", "2025"],
        )

    def test_main_planner_uses_catalog_without_calling_the_model(self) -> None:
        class ModelMustNotRun:
            def invoke(self, _: object) -> object:
                raise AssertionError("formula planning should not call the model")

        scope = Scope(("MSFT",), ("2025",), "10K")
        question = "Calculate Microsoft's current ratio for 2025."

        self.assertTrue(should_use_multihop(question, scope))
        plan = MultiHopPlanner(chain=ModelMustNotRun()).plan(
            question=question,
            permitted_scope=scope,
        )

        self.assertEqual(len(plan.requirements), 2)
        self.assertEqual(plan.calculations[0].operation.value, "ratio")

    def test_roic_catalog_formula_fits_the_bounded_single_group_budget(self) -> None:
        plan = formula_plan(
            question="Calculate Microsoft's ROIC for 2025.",
            permitted_scope=Scope(("MSFT",), ("2025",), "10K"),
        )

        self.assertEqual(len(plan.requirements), 4)
        self.assertEqual(
            [item.operation.value for item in plan.calculations],
            ["percent_complement", "product", "sum", "difference", "ratio"],
        )

    def test_narrative_formula_question_uses_the_general_planner(self) -> None:
        with self.assertRaisesRegex(ValueError, "general planner"):
            formula_plan(
                question="Calculate gross margin and explain why it changed.",
                permitted_scope=Scope(("MSFT",), ("2025",), "10K"),
            )


if __name__ == "__main__":
    unittest.main()

