from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.orchestration.models import (
    EvidenceGroup,
    EvidenceRequirement,
    EvidenceType,
    FactReference,
    MultiHopPlan,
    PlannedCalculation,
    RequirementResult,
)


class OrchestrationModelTests(unittest.TestCase):
    def test_requirement_builds_exact_non_cartesian_scope(self) -> None:
        requirement = EvidenceRequirement(
            requirement_id="revenue_values",
            question="Find reported revenue for the requested filings.",
            evidence_type=EvidenceType.NUMERIC,
            document_type="10-K",
            groups=(
                EvidenceGroup(ticker="msft", fiscal_year="2024"),
                EvidenceGroup(ticker="tsla", fiscal_year="2025"),
            ),
        )

        scope = requirement.to_scope()

        self.assertEqual(requirement.document_type, "10K")
        self.assertEqual(scope.tickers, ("MSFT", "TSLA"))
        self.assertEqual(scope.years, ("2024", "2025"))
        self.assertEqual(
            scope.requested_groups,
            (("MSFT", "2024"), ("TSLA", "2025")),
        )
        self.assertEqual(
            scope.retrieval_groups,
            (("MSFT", "2024", "10K"), ("TSLA", "2025", "10K")),
        )

    def test_requirement_rejects_duplicate_groups(self) -> None:
        group = EvidenceGroup(ticker="MSFT", fiscal_year="2024")

        with self.assertRaisesRegex(ValidationError, "duplicate"):
            EvidenceRequirement(
                requirement_id="revenue",
                question="Find reported revenue.",
                evidence_type="numeric",
                document_type="10K",
                groups=(group, group),
            )

    def test_models_reject_invalid_boundary_values(self) -> None:
        invalid_groups = (
            {"ticker": "not a ticker!", "fiscal_year": "2024"},
            {"ticker": "MSFT", "fiscal_year": "24"},
        )
        for values in invalid_groups:
            with self.subTest(values=values):
                with self.assertRaises(ValidationError):
                    EvidenceGroup(**values)

        group = EvidenceGroup(ticker="MSFT", fiscal_year="2024")
        invalid_requirements = (
            {"requirement_id": "Revenue Values", "document_type": "10K"},
            {"requirement_id": "revenue", "document_type": "8K"},
        )
        for values in invalid_requirements:
            with self.subTest(values=values):
                with self.assertRaises(ValidationError):
                    EvidenceRequirement(
                        question="Find reported revenue.",
                        evidence_type="numeric",
                        groups=(group,),
                        **values,
                    )

    def test_result_rejects_duplicate_provenance_ids(self) -> None:
        with self.assertRaisesRegex(ValidationError, "duplicates"):
            RequirementResult(
                requirement_id="revenue",
                source_ids=("S1", "S1"),
            )

    def test_models_are_immutable(self) -> None:
        group = EvidenceGroup(ticker="MSFT", fiscal_year="2024")

        with self.assertRaises(ValidationError):
            group.ticker = "TSLA"

    def test_plan_validates_calculation_references_and_ordered_inputs(self) -> None:
        groups = (
            EvidenceGroup(ticker="MSFT", fiscal_year="2024"),
            EvidenceGroup(ticker="MSFT", fiscal_year="2025"),
        )
        requirement = EvidenceRequirement(
            requirement_id="revenue",
            question="Find revenue for both fiscal years.",
            evidence_type="numeric",
            document_type="10K",
            groups=groups,
        )
        calculation = PlannedCalculation(
            calculation_id="revenue_growth",
            label="MSFT revenue percentage change",
            operation="percentage_change",
            inputs=(
                FactReference(
                    requirement_id="revenue",
                    ticker="MSFT",
                    fiscal_year="2024",
                    metric_hint="Revenue",
                ),
                FactReference(
                    requirement_id="revenue",
                    ticker="MSFT",
                    fiscal_year="2025",
                    metric_hint="Revenue",
                ),
            ),
        )

        plan = MultiHopPlan(
            original_question="How did MSFT revenue change?",
            requirements=(requirement,),
            calculations=(calculation,),
        )

        self.assertEqual(plan.calculations[0].operation.value, "percentage_change")

    def test_plan_rejects_unknown_or_narrative_calculation_references(self) -> None:
        group = EvidenceGroup(ticker="MSFT", fiscal_year="2025")
        narrative = EvidenceRequirement(
            requirement_id="drivers",
            question="Explain the revenue drivers.",
            evidence_type="narrative",
            document_type="TRANSCRIPT",
            groups=(group,),
        )
        calculation = PlannedCalculation(
            calculation_id="invalid_change",
            label="Invalid narrative calculation",
            operation="difference",
            inputs=(
                FactReference(
                    requirement_id="drivers",
                    ticker="MSFT",
                    fiscal_year="2025",
                ),
                FactReference(
                    requirement_id="drivers",
                    ticker="MSFT",
                    fiscal_year="2025",
                    metric_hint="Revenue",
                ),
            ),
        )

        with self.assertRaisesRegex(ValidationError, "numeric requirements"):
            MultiHopPlan(
                original_question="Explain and calculate the change.",
                requirements=(narrative,),
                calculations=(calculation,),
            )

    def test_fact_references_distinguish_source_year_from_table_period(self) -> None:
        group = EvidenceGroup(ticker="MSFT", fiscal_year="2024")
        requirement = EvidenceRequirement(
            requirement_id="revenue",
            question="Find the 2023 and 2024 revenue columns in the 2024 filing.",
            evidence_type="numeric",
            document_type="10K",
            groups=(group,),
        )
        calculation = PlannedCalculation(
            calculation_id="revenue_change",
            label="Revenue change between comparative columns",
            operation="percentage_change",
            inputs=(
                FactReference(
                    requirement_id="revenue",
                    ticker="MSFT",
                    fiscal_year="2024",
                    period="2023",
                    metric_hint="Revenue",
                ),
                FactReference(
                    requirement_id="revenue",
                    ticker="MSFT",
                    fiscal_year="2024",
                    period="2024",
                    metric_hint="Revenue",
                ),
            ),
        )

        plan = MultiHopPlan(
            original_question="How did revenue change from 2023 to 2024?",
            requirements=(requirement,),
            calculations=(calculation,),
        )

        self.assertEqual(
            tuple(reference.period for reference in plan.calculations[0].inputs),
            ("2023", "2024"),
        )


if __name__ == "__main__":
    unittest.main()
