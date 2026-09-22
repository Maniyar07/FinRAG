from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.orchestration.planner import MultiHopPlanner, should_use_multihop
from src.schemas import Scope


class FakeChain:
    def __init__(self, payload) -> None:
        self.payload = payload
        self.calls = []

    def invoke(self, values):
        self.calls.append(values)
        return self.payload


def requirement(*, ticker="MSFT", year="2024", document_type="10K") -> dict:
    return {
        "requirement_id": "revenue",
        "question": "Find reported revenue.",
        "evidence_type": "numeric",
        "document_type": document_type,
        "groups": [{"ticker": ticker, "fiscal_year": year}],
        "requires_complete_coverage": True,
    }


class MultiHopPlannerTests(unittest.TestCase):
    def test_common_comparison_has_a_scope_safe_deterministic_fallback(self) -> None:
        planner = MultiHopPlanner(chain=FakeChain({}))
        scope = Scope(
            ("MSFT", "TSLA"),
            ("2024", "2025"),
            requested_groups=(
                ("MSFT", "2024"),
                ("MSFT", "2025"),
                ("TSLA", "2024"),
                ("TSLA", "2025"),
            ),
            required_doc_types=("10K", "TRANSCRIPT"),
        )

        plan = planner.plan(
            question=(
                "Compare MSFT and TSLA revenue for 2024 and 2025. Calculate each "
                "company's percentage change and explain the reasons in the 2025 "
                "earnings transcripts."
            ),
            permitted_scope=scope,
        )

        self.assertEqual(len(plan.requirements), 2)
        self.assertEqual(
            {requirement.document_type for requirement in plan.requirements},
            {"10K", "TRANSCRIPT"},
        )
        self.assertEqual(len(plan.calculations), 2)
        self.assertEqual(
            {calculation.inputs[0].ticker for calculation in plan.calculations},
            {"MSFT", "TSLA"},
        )
        narrative = next(
            item for item in plan.requirements if item.evidence_type.value == "narrative"
        )
        self.assertIn("revenue", narrative.question.casefold())
        self.assertEqual(
            {group.key for group in narrative.groups},
            {("MSFT", "2025"), ("TSLA", "2025")},
        )

    def test_liquidity_risk_search_does_not_repeat_balance_sheet_metric(self) -> None:
        planner = MultiHopPlanner(chain=FakeChain({}))
        scope = Scope(
            ("MSFT", "TSLA"), ("2024", "2025"),
            requested_groups=(("MSFT", "2024"), ("MSFT", "2025"),
                              ("TSLA", "2024"), ("TSLA", "2025")),
            doc_type="10K",
        )
        plan = planner.plan(
            question=("Compare MSFT and TSLA cash and cash equivalents for 2024 "
                      "and 2025. Calculate the absolute change and summarize "
                      "principal liquidity risks in the 2025 10-Ks."),
            permitted_scope=scope,
        )
        narrative = next(item for item in plan.requirements if item.evidence_type.value == "narrative")
        self.assertIn("liquidity risks", narrative.question)
        self.assertNotIn("cash and cash equivalents", narrative.question)

    def test_planner_drops_a_derived_comparison_search_over_the_bound(self) -> None:
        requirements = []
        for index in range(4):
            item = requirement()
            item["requirement_id"] = f"evidence_{index}"
            item["question"] = f"Find reported revenue evidence part {index}."
            requirements.append(item)
        derived = requirement()
        derived["requirement_id"] = "which_improved"
        derived["question"] = "Identify which company improved more from the calculated results."
        requirements.append(derived)
        calculation = {
            "calculation_id": "revenue_change",
            "label": "Revenue percentage change",
            "operation": "percentage_change",
            "inputs": [
                {
                    "requirement_id": "evidence_0",
                    "ticker": "MSFT",
                    "fiscal_year": "2024",
                    "period": "2023",
                },
                {
                    "requirement_id": "evidence_0",
                    "ticker": "MSFT",
                    "fiscal_year": "2024",
                    "period": "2024",
                },
            ],
        }
        raw = SimpleNamespace(
            tool_calls=[
                {"args": {"requirements": requirements, "calculations": [calculation]}}
            ]
        )
        planner = MultiHopPlanner(
            chain=FakeChain({"raw": raw, "parsed": None, "parsing_error": ValueError()})
        )

        plan = planner.plan(
            question="Calculate revenue change and identify which improved more.",
            permitted_scope=Scope(("MSFT",), ("2024",), "10K"),
        )

        self.assertEqual(len(plan.requirements), 4)
        self.assertNotIn(
            "which_improved",
            {item.requirement_id for item in plan.requirements},
        )

    def test_planner_tolerates_raw_tool_output_with_narrative_calculation_noise(self) -> None:
        valid_calculation = {
            "calculation_id": "revenue_change",
            "label": "Revenue percentage change",
            "operation": "percentage_change",
            "inputs": [
                {
                    "calculation_id": "revenue_change",
                    "ticker": "MSFT",
                    "fiscal_year": "2024",
                    "metric_hint": "Revenue",
                },
                {
                    "requirement_id": "revenue",
                    "ticker": "MSFT",
                    "fiscal_year": "2024",
                    "period": "2023",
                    "metric_hint": "Revenue",
                },
            ],
            "requires_complete_coverage": True,
        }
        args = {
            "requirements": [requirement()],
            "calculations": [
                valid_calculation,
                {
                    "calculation_id": "explain",
                    "label": "Narrative explanation",
                    "operation": "narrative",
                    "inputs": [],
                },
            ],
        }
        raw = SimpleNamespace(tool_calls=[{"args": args}])
        planner = MultiHopPlanner(
            chain=FakeChain(
                {"raw": raw, "parsed": None, "parsing_error": ValueError("bad")}
            )
        )

        plan = planner.plan(
            question="Calculate MSFT revenue percentage change.",
            permitted_scope=Scope(("MSFT",), ("2024",), "10K"),
        )

        self.assertEqual(len(plan.calculations), 1)
        self.assertEqual(plan.calculations[0].calculation_id, "revenue_change")

    def test_planner_accepts_only_a_plan_inside_the_permitted_scope(self) -> None:
        chain = FakeChain({"requirements": [requirement()], "calculations": []})
        planner = MultiHopPlanner(chain=chain)
        scope = Scope(("MSFT",), ("2024",), "10K")

        plan = planner.plan(
            question="What was MSFT revenue and why did it change?",
            permitted_scope=scope,
        )

        self.assertEqual(plan.original_question, "What was MSFT revenue and why did it change?")
        self.assertEqual(plan.requirements[0].groups[0].key, ("MSFT", "2024"))
        self.assertIn('"ticker": "MSFT"', chain.calls[0]["scope"])

    def test_planner_rejects_scope_expansion(self) -> None:
        planner = MultiHopPlanner(
            chain=FakeChain(
                {
                    "requirements": [requirement(ticker="TSLA")],
                    "calculations": [],
                }
            )
        )

        with self.assertRaisesRegex(ValueError, "expands the permitted groups"):
            planner.plan(
                question="What was MSFT revenue?",
                permitted_scope=Scope(("MSFT",), ("2024",), "10K"),
            )

    def test_planner_cannot_omit_an_explicitly_required_document_type(self) -> None:
        planner = MultiHopPlanner(
            chain=FakeChain({"requirements": [requirement()], "calculations": []})
        )

        with self.assertRaisesRegex(ValueError, "omits required document types"):
            planner.plan(
                question="Use the 10-K and transcript to explain revenue.",
                permitted_scope=Scope(
                    ("MSFT",),
                    ("2024",),
                    required_doc_types=("10K", "TRANSCRIPT"),
                ),
            )

    def test_router_selects_comparisons_but_not_simple_questions(self) -> None:
        comparison = Scope(
            ("MSFT", "TSLA"),
            ("2025",),
            "10K",
            requested_groups=(("MSFT", "2025"), ("TSLA", "2025")),
        )
        simple = Scope(("MSFT",), ("2025",), "10K")

        self.assertFalse(should_use_multihop("Compare their revenue.", comparison))
        self.assertTrue(
            should_use_multihop("Compare their revenue and explain the difference.", comparison)
        )
        self.assertFalse(should_use_multihop("What was revenue?", simple))
        self.assertTrue(
            should_use_multihop("What was revenue; and why did it change?", simple)
        )


if __name__ == "__main__":
    unittest.main()
