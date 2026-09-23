from __future__ import annotations

import unittest
import json
from decimal import Decimal

from src.financial.models import FactValidationResult, ValidatedFinancialFact, ValueType
from src.financial.fact_pipeline import FinancialFactPipeline
from src.orchestration.executor import MultiHopExecutor, _focused_numeric_sources, _select_fact
from src.orchestration.models import (
    EvidenceGroup,
    EvidenceRequirement,
    FactReference,
    MultiHopPlan,
    PlannedCalculation,
)
from src.schemas import RetrievalBundle, Scope
from src.retrieval.structured_lookup import StructuredDocumentLookup
from src.tools.document_search import DocumentSearchResult


class FakeDocumentSearch:
    def __init__(self, *, omit_year: str | None = None) -> None:
        self.calls = []
        self.omit_year = omit_year

    def execute(self, request, *, permitted_scope):
        self.calls.append((request, permitted_scope))
        sources = []
        for ticker, year in request.scope.groups:
            if year == self.omit_year:
                continue
            local_id = f"S{len(sources) + 1}"
            sources.append(
                {
                    "id": local_id,
                    "ticker": ticker,
                    "fiscal_year": year,
                    "doc_type": request.scope.doc_type,
                    "parent_id": f"{request.scope.doc_type}-{ticker}-{year}",
                    "source_hash": f"hash-{request.scope.doc_type}-{ticker}-{year}",
                    "source": f"{ticker}-{year}.pdf",
                    "section": "Results",
                    "pdf_page": "10",
                    "evidence_text": f"Revenue {year} {100 if year == '2024' else 110}",
                    "full_evidence_text": f"Revenue {year} {100 if year == '2024' else 110}",
                }
            )
        bundle = RetrievalBundle(
            context="local context",
            sources=sources,
            scope=request.scope,
            candidate_count=len(sources),
            covered_groups=tuple(
                (source["ticker"], source["fiscal_year"]) for source in sources
            ),
        )
        return DocumentSearchResult(request.query, request.purpose, bundle)


class EmptyExtractor:
    def extract(self, *, question, sources):
        del question, sources
        return ()


def test_indexed_statement_recovers_fact_missed_by_ranked_search(tmp_path) -> None:
    (tmp_path / "statement.json").write_text(json.dumps({
        "page_content": (
            "## Consolidated Statements of Operations\n(in millions)\n"
            "<table><tr><th>Metric</th><th>2024</th><th>2023</th></tr>"
            "<tr><td>Automotive sales</td><td>$72,480</td><td>$78,509</td></tr>"
            "<tr><td>Total revenues</td><td>97,690</td><td>96,773</td></tr></table>"
        ),
        "metadata": {"ticker": "TSLA", "fiscal_year": "2024", "doc_type": "10K",
                     "item": "Item 8", "section": "Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
                     "source": "TSLA_2024_10K.pdf", "pdf_page_start": 50},
    }), encoding="utf-8")
    requirement = EvidenceRequirement(
        requirement_id="revenue", question="What is Tesla revenue in 2024?",
        evidence_type="numeric", document_type="10K",
        groups=(EvidenceGroup(ticker="TSLA", fiscal_year="2024"),),
    )
    executor = MultiHopExecutor(
        document_search=FakeDocumentSearch(),
        fact_pipeline=FinancialFactPipeline(extractor=EmptyExtractor()),
        statement_lookup=StructuredDocumentLookup(tmp_path),
    )

    result = executor.execute(
        MultiHopPlan(original_question=requirement.question,
                     requirements=(requirement,), calculations=()),
        permitted_scope=Scope(("TSLA",), ("2024",), "10K"),
    )

    assert result.complete
    assert any(fact.raw_value == "97,690" for fact in result.facts)


class MissingInitialGroupSearch(FakeDocumentSearch):
    def execute(self, request, *, permitted_scope):
        self.omit_year = "2025" if not self.calls else None
        return super().execute(request, permitted_scope=permitted_scope)


class FakeFactPipeline:
    def run(self, *, question, sources, permitted_scope):
        del question, permitted_scope
        facts = []
        for source in sources:
            year = str(source["fiscal_year"])
            value = "100" if year == "2024" else "110"
            facts.append(
                ValidatedFinancialFact(
                    fact_id=f"F-{source['ticker']}-{year}",
                    ticker=source["ticker"],
                    metric="Revenue",
                    period=year,
                    raw_value=value,
                    numeric_value=Decimal(value),
                    base_value=Decimal(value) * Decimal("1000000"),
                    value_type=ValueType.CURRENCY,
                    normalized_unit="USD",
                    currency="USD",
                    scale="millions",
                    source_id=source["id"],
                    evidence_excerpt=f"Revenue {year} {value}",
                    row_label="Revenue",
                    column_label=year,
                    validation_checks=("test",),
                )
            )
        return FactValidationResult(valid_facts=tuple(facts))


class WrongFactWithExactRecoveryPipeline(FakeFactPipeline):
    def recover_exact_table_row(
        self, *, metric_hint, period_year, sources, permitted_scope
    ):
        return FinancialFactPipeline(extractor=EmptyExtractor()).recover_exact_table_row(
            metric_hint=metric_hint,
            period_year=period_year,
            sources=sources,
            permitted_scope=permitted_scope,
        )


class RecoveringFactPipeline(FakeFactPipeline):
    def __init__(self) -> None:
        self.calls = 0

    def run(self, *, question, sources, permitted_scope):
        self.calls += 1
        result = super().run(
            question=question,
            sources=sources,
            permitted_scope=permitted_scope,
        )
        if self.calls == 1:
            return FactValidationResult(valid_facts=result.valid_facts[1:])
        return result


class ExactRowRecoveringPipeline(FakeFactPipeline):
    def run(self, *, question, sources, permitted_scope):
        result = super().run(
            question=question, sources=sources, permitted_scope=permitted_scope
        )
        return FactValidationResult(
            valid_facts=tuple(
                fact for fact in result.valid_facts if fact.period != "2025"
            )
        )

    def recover_exact_table_row(self, *, metric_hint, period_year, sources, permitted_scope):
        del metric_hint
        return FakeFactPipeline.run(
            self,
            question=period_year,
            sources=sources,
            permitted_scope=permitted_scope,
        )


class WrongPeriodThenRecoveringFactPipeline(FakeFactPipeline):
    def __init__(self) -> None:
        self.calls = 0

    def run(self, *, question, sources, permitted_scope):
        self.calls += 1
        result = super().run(
            question=question,
            sources=sources,
            permitted_scope=permitted_scope,
        )
        if self.calls != 1:
            return result
        facts = tuple(
            fact.model_copy(update={"period": "2024", "column_label": "2024"})
            if fact.period == "2025"
            else fact
            for fact in result.valid_facts
        )
        return FactValidationResult(valid_facts=facts)


def plan() -> MultiHopPlan:
    groups = (
        EvidenceGroup(ticker="MSFT", fiscal_year="2024"),
        EvidenceGroup(ticker="MSFT", fiscal_year="2025"),
    )
    revenue = EvidenceRequirement(
        requirement_id="revenue",
        question="Find MSFT revenue for 2024 and 2025.",
        evidence_type="numeric",
        document_type="10K",
        groups=groups,
    )
    explanation = EvidenceRequirement(
        requirement_id="drivers",
        question="Explain management's revenue drivers.",
        evidence_type="narrative",
        document_type="TRANSCRIPT",
        groups=(groups[1],),
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
    return MultiHopPlan(
        original_question="How did MSFT revenue change and why?",
        requirements=(revenue, explanation),
        calculations=(calculation,),
    )


class MultiHopExecutorTests(unittest.TestCase):
    def test_indexed_statement_row_replaces_model_extracted_variant(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "statement.json").write_text(json.dumps({
                "page_content": (
                    "## Consolidated Statements of Operations\n(in millions)\n"
                    "<table><tr><th>Metric</th><th>2024</th></tr>"
                    "<tr><td>Total revenues</td><td>$97,690</td></tr></table>"
                ),
                "metadata": {
                    "ticker": "TSLA", "fiscal_year": "2024", "doc_type": "10K",
                    "item": "Item 8",
                    "section": "Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
                    "source": "TSLA_2024_10K.pdf", "pdf_page_start": 50,
                },
            }), encoding="utf-8")
            requirement = EvidenceRequirement(
                requirement_id="revenue",
                question="Find Tesla total revenue in 2024.",
                evidence_type="numeric",
                document_type="10K",
                groups=(EvidenceGroup(ticker="TSLA", fiscal_year="2024"),),
            )
            executor = MultiHopExecutor(
                document_search=FakeDocumentSearch(),
                fact_pipeline=WrongFactWithExactRecoveryPipeline(),
                statement_lookup=StructuredDocumentLookup(root),
            )

            result = executor.execute(
                MultiHopPlan(
                    original_question=requirement.question,
                    requirements=(requirement,),
                    calculations=(),
                ),
                permitted_scope=Scope(("TSLA",), ("2024",), "10K"),
            )

        self.assertTrue(result.complete)
        self.assertEqual([fact.raw_value for fact in result.facts], ["$97,690"])
        self.assertEqual(result.facts[0].currency, "USD")
        self.assertEqual(result.facts[0].scale, "millions")

    def test_temporal_calculation_orders_reversed_planner_inputs(self) -> None:
        normal = plan()
        reversed_calculation = normal.calculations[0].model_copy(
            update={"inputs": tuple(reversed(normal.calculations[0].inputs))}
        )
        reversed_plan = normal.model_copy(
            update={"calculations": (reversed_calculation,)}
        )
        executor = MultiHopExecutor(
            document_search=FakeDocumentSearch(),
            fact_pipeline=FakeFactPipeline(),
        )

        result = executor.execute(
            reversed_plan,
            permitted_scope=Scope(
                ("MSFT",), ("2024", "2025"),
                required_doc_types=("10K", "TRANSCRIPT"),
            ),
        )

        calculation = result.calculations[0].result
        self.assertEqual(calculation.result, Decimal("10.0"))
        fact_by_id = {fact.fact_id: fact for fact in result.facts}
        self.assertEqual(
            [fact_by_id[fact_id].period for fact_id in calculation.input_fact_ids],
            ["2024", "2025"],
        )

    def test_exact_row_recovery_completes_a_missing_calculation_input(self) -> None:
        executor = MultiHopExecutor(
            document_search=FakeDocumentSearch(),
            fact_pipeline=ExactRowRecoveringPipeline(),
        )
        scope = Scope(
            ("MSFT",), ("2024", "2025"),
            required_doc_types=("10K", "TRANSCRIPT"),
        )

        result = executor.execute(plan(), permitted_scope=scope)

        self.assertTrue(result.complete)
        self.assertEqual(len(result.calculations), 1)
        self.assertEqual(result.calculations[0].result.result, Decimal("10"))

    def test_focused_recovery_prefers_exact_table_row(self) -> None:
        sources = [
            {"id": "S1", "full_evidence_text": "<td>Other research and development</td>"},
            {"id": "S2", "full_evidence_text": "<td>Research and development</td><td>32,488</td>"},
        ]
        focused = _focused_numeric_sources(
            sources,
            "Find research and development expense.",
            ("research and development expense",),
        )
        self.assertEqual([source["id"] for source in focused], ["S2"])

    def test_fact_binding_prefers_consolidated_revenue_over_segments(self) -> None:
        base = FakeFactPipeline().run(
            question="revenue",
            sources=[{"id": "S1", "ticker": "MSFT", "fiscal_year": "2025"}],
            permitted_scope=Scope(("MSFT",), ("2025",), "10K"),
        ).valid_facts[0]
        segment = base.model_copy(update={
            "fact_id": "F-segment", "numeric_value": Decimal("120810"),
            "base_value": Decimal("120810000000"),
        })
        consolidated = base.model_copy(update={
            "fact_id": "F-total", "source_id": "S2", "metric": "Total revenue",
            "row_label": "Total revenue", "numeric_value": Decimal("281724"),
            "base_value": Decimal("281724000000"),
        })
        sources = {
            "S1": {"fiscal_year": "2025", "section": "Item 7"},
            "S2": {
                "fiscal_year": "2025",
                "section": "Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
                "full_evidence_text": "## INCOME STATEMENTS",
            },
        }

        selected = _select_fact(
            FactReference(
                requirement_id="revenue", ticker="MSFT",
                fiscal_year="2025", metric_hint="Revenue",
            ),
            {"revenue": (segment, consolidated)},
            sources,
        )

        self.assertEqual(selected.fact_id, "F-total")

    def test_fact_binding_prefers_validated_usd_row_over_untyped_number(self) -> None:
        base = FakeFactPipeline().run(
            question="operating income",
            sources=[{"id": "S1", "ticker": "TSLA", "fiscal_year": "2025"}],
            permitted_scope=Scope(("TSLA",), ("2025",), "10K"),
        ).valid_facts[0]
        untyped = base.model_copy(update={
            "fact_id": "F-untyped", "metric": "Income from operations",
            "row_label": "Income from operations", "numeric_value": Decimal("4355"),
            "base_value": Decimal("4355000000"), "value_type": ValueType.NUMBER,
            "normalized_unit": "number", "currency": None,
        })
        typed = untyped.model_copy(update={
            "fact_id": "F-typed", "value_type": ValueType.CURRENCY,
            "normalized_unit": "USD", "currency": "USD",
        })
        source = {
            "S1": {
                "fiscal_year": "2025",
                "section": "Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
                "full_evidence_text": "## Consolidated Statements of Operations",
            },
        }

        selected = _select_fact(
            FactReference(
                requirement_id="income", ticker="TSLA",
                fiscal_year="2025", metric_hint="Operating income",
            ),
            {"income": (untyped, typed)}, source,
        )

        self.assertEqual(selected.fact_id, "F-typed")

    def test_fact_binding_prefers_exact_row_over_model_variant(self) -> None:
        base = FakeFactPipeline().run(
            question="revenue",
            sources=[{"id": "S1", "ticker": "TSLA", "fiscal_year": "2025"}],
            permitted_scope=Scope(("TSLA",), ("2025",), "10K"),
        ).valid_facts[0]
        model_fact = base.model_copy(update={"accounting_basis": "non-GAAP"})
        exact = base.model_copy(update={
            "fact_id": "F-exact", "metric": "Total revenues",
            "row_label": "Total revenues", "accounting_basis": None,
            "validation_checks": (*base.validation_checks, "exact_table_row"),
        })

        selected = _select_fact(
            FactReference(
                requirement_id="revenue", ticker="TSLA",
                fiscal_year="2025", metric_hint="Revenue",
            ),
            {"revenue": (model_fact, exact)},
            {"S1": {"fiscal_year": "2025"}},
        )

        self.assertEqual(selected.fact_id, "F-exact")

    def test_missing_source_group_triggers_focused_search(self) -> None:
        search = MissingInitialGroupSearch()
        executor = MultiHopExecutor(
            document_search=search,
            fact_pipeline=FakeFactPipeline(),
        )
        scope = Scope(
            ("MSFT",), ("2024", "2025"),
            required_doc_types=("10K", "TRANSCRIPT"),
        )

        result = executor.execute(plan(), permitted_scope=scope)

        self.assertTrue(result.complete)
        self.assertTrue(any(call[0].scope.groups == (("MSFT", "2025"),) for call in search.calls))
        self.assertFalse(any("missing_evidence_groups" in issue for issue in result.issues))

    def test_executor_recovers_a_fact_omitted_from_a_multi_group_extraction(self) -> None:
        pipeline = RecoveringFactPipeline()
        executor = MultiHopExecutor(
            document_search=FakeDocumentSearch(),
            fact_pipeline=pipeline,
        )
        scope = Scope(
            ("MSFT",),
            ("2024", "2025"),
            required_doc_types=("10K", "TRANSCRIPT"),
        )

        result = executor.execute(plan(), permitted_scope=scope)

        self.assertTrue(result.complete)
        self.assertGreaterEqual(pipeline.calls, 2)
        self.assertEqual(
            result.trace["requirements"][0]["recovered_fact_groups"],
            [["MSFT", "2024"]],
        )

    def test_executor_recovers_when_filing_contains_wrong_comparative_period(self) -> None:
        pipeline = WrongPeriodThenRecoveringFactPipeline()
        executor = MultiHopExecutor(
            document_search=FakeDocumentSearch(),
            fact_pipeline=pipeline,
        )
        scope = Scope(
            ("MSFT",),
            ("2024", "2025"),
            required_doc_types=("10K", "TRANSCRIPT"),
        )

        result = executor.execute(plan(), permitted_scope=scope)

        self.assertTrue(result.complete)
        self.assertGreaterEqual(pipeline.calls, 2)
        self.assertEqual(
            result.trace["requirements"][0]["recovered_fact_groups"],
            [["MSFT", "2025"]],
        )
        self.assertEqual(result.calculations[0].result.result, Decimal("10.0"))

    def test_fact_binding_uses_table_period_within_one_source_filing(self) -> None:
        source = {
            "id": "S1",
            "ticker": "MSFT",
            "fiscal_year": "2024",
        }
        facts = FakeFactPipeline().run(
            question="revenue",
            sources=[source],
            permitted_scope=Scope(("MSFT",), ("2024",), "10K"),
        ).valid_facts
        current = facts[0]
        prior = current.model_copy(
            update={
                "fact_id": "F-MSFT-2023",
                "period": "2023",
                "column_label": "2023",
                "raw_value": "90",
                "numeric_value": Decimal("90"),
                "base_value": Decimal("90000000"),
            }
        )
        facts_by_requirement = {"revenue": (prior, current)}
        sources = {"S1": source}

        selected = _select_fact(
            FactReference(
                requirement_id="revenue",
                ticker="MSFT",
                fiscal_year="2024",
                period="2023",
                metric_hint="Revenue",
            ),
            facts_by_requirement,
            sources,
        )

        self.assertEqual(selected.fact_id, "F-MSFT-2023")

    def test_fact_binding_ignores_generic_expense_suffix(self) -> None:
        source = {"id": "S1", "ticker": "MSFT", "fiscal_year": "2024"}
        fact = FakeFactPipeline().run(
            question="research and development",
            sources=[source],
            permitted_scope=Scope(("MSFT",), ("2024",), "10K"),
        ).valid_facts[0].model_copy(
            update={
                "metric": "Research and development",
                "row_label": "Research and development",
            }
        )

        selected = _select_fact(
            FactReference(
                requirement_id="rd",
                ticker="MSFT",
                fiscal_year="2024",
                metric_hint="Research and development expense",
            ),
            {"rd": (fact,)},
            {"S1": source},
        )

        self.assertEqual(selected.fact_id, fact.fact_id)

    def test_fact_binding_matches_operating_income_to_income_from_operations(self) -> None:
        source = {"id": "S1", "ticker": "TSLA", "fiscal_year": "2025"}
        fact = FakeFactPipeline().run(
            question="operating income",
            sources=[source],
            permitted_scope=Scope(("TSLA",), ("2025",), "10K"),
        ).valid_facts[0].model_copy(
            update={
                "metric": "Income (loss) from operations",
                "row_label": "Income (loss) from operations",
            }
        )

        selected = _select_fact(
            FactReference(
                requirement_id="operating_income",
                ticker="TSLA",
                fiscal_year="2025",
                metric_hint="Operating income",
            ),
            {"operating_income": (fact,)},
            {"S1": source},
        )

        self.assertEqual(selected.fact_id, fact.fact_id)

    def test_fact_binding_rejects_cash_row_that_includes_restricted_cash(self) -> None:
        source = {"id": "S1", "ticker": "TSLA", "fiscal_year": "2025"}
        fact = FakeFactPipeline().run(
            question="cash and cash equivalents",
            sources=[source],
            permitted_scope=Scope(("TSLA",), ("2025",), "10K"),
        ).valid_facts[0].model_copy(
            update={
                "metric": "Cash, cash equivalents and restricted cash",
                "row_label": "Cash, cash equivalents and restricted cash",
            }
        )

        with self.assertRaisesRegex(ValueError, "matched 0 validated facts"):
            _select_fact(
                FactReference(
                    requirement_id="cash",
                    ticker="TSLA",
                    fiscal_year="2025",
                    metric_hint="Cash and cash equivalents",
                ),
                {"cash": (fact,)},
                {"S1": source},
            )

    def test_fact_binding_accepts_effective_rate_tax_table_alias(self) -> None:
        source = {"id": "S1", "ticker": "MSFT", "fiscal_year": "2025"}
        fact = FakeFactPipeline().run(
            question="effective tax rate",
            sources=[source],
            permitted_scope=Scope(("MSFT",), ("2025",), "10K"),
        ).valid_facts[0].model_copy(
            update={
                "metric": "Effective rate",
                "row_label": "Effective rate",
                "raw_value": "17.6%",
                "numeric_value": Decimal("17.6"),
                "base_value": Decimal("17.6"),
                "value_type": ValueType.PERCENT,
                "normalized_unit": "percent",
                "currency": None,
                "scale": None,
            }
        )

        selected = _select_fact(
            FactReference(
                requirement_id="tax",
                ticker="MSFT",
                fiscal_year="2025",
                metric_hint="Effective tax rate",
            ),
            {"tax": (fact,)},
            {"S1": source},
        )

        self.assertEqual(selected.fact_id, fact.fact_id)

    def test_fact_binding_rejects_amount_cell_for_rate_metric(self) -> None:
        source = {"id": "S1", "ticker": "TSLA", "fiscal_year": "2025"}
        fact = FakeFactPipeline().run(
            question="effective tax rate",
            sources=[source],
            permitted_scope=Scope(("TSLA",), ("2025",), "10K"),
        ).valid_facts[0].model_copy(
            update={
                "metric": "Effective tax rate",
                "row_label": "Effective tax rate",
                "raw_value": "$1,423",
                "numeric_value": Decimal("1423"),
                "base_value": Decimal("1423000000"),
                "value_type": ValueType.CURRENCY,
                "normalized_unit": "USD",
                "currency": "USD",
                "scale": "millions",
            }
        )

        with self.assertRaisesRegex(ValueError, "matched 0 validated facts"):
            _select_fact(
                FactReference(
                    requirement_id="tax",
                    ticker="TSLA",
                    fiscal_year="2025",
                    metric_hint="Effective tax rate",
                ),
                {"tax": (fact,)},
                {"S1": source},
            )

    def test_fact_binding_prefers_more_precise_percent_fact(self) -> None:
        source_a = {"id": "S1", "ticker": "MSFT", "fiscal_year": "2024"}
        source_b = {"id": "S2", "ticker": "MSFT", "fiscal_year": "2024"}
        rounded = FakeFactPipeline().run(
            question="effective tax rate",
            sources=[source_a],
            permitted_scope=Scope(("MSFT",), ("2024",), "10K"),
        ).valid_facts[0].model_copy(
            update={
                "metric": "Effective tax rate",
                "row_label": "Effective tax rate",
                "raw_value": "18%",
                "numeric_value": Decimal("18"),
                "base_value": Decimal("18"),
                "value_type": ValueType.PERCENT,
                "normalized_unit": "percent",
                "currency": None,
                "scale": None,
            }
        )
        precise = rounded.model_copy(
            update={
                "fact_id": "F-precise",
                "source_id": "S2",
                "metric": "Effective rate",
                "row_label": "Effective rate",
                "raw_value": "18.2%",
                "numeric_value": Decimal("18.2"),
                "base_value": Decimal("18.2"),
            }
        )

        selected = _select_fact(
            FactReference(
                requirement_id="tax",
                ticker="MSFT",
                fiscal_year="2024",
                metric_hint="Effective tax rate",
            ),
            {"tax": (rounded, precise)},
            {"S1": source_a, "S2": source_b},
        )

        self.assertEqual(selected.fact_id, "F-precise")

    def test_executor_merges_source_ids_validates_facts_and_calculates(self) -> None:
        search = FakeDocumentSearch()
        executor = MultiHopExecutor(
            document_search=search,
            fact_pipeline=FakeFactPipeline(),
        )
        scope = Scope(
            ("MSFT",),
            ("2024", "2025"),
            required_doc_types=("10K", "TRANSCRIPT"),
        )

        result = executor.execute(plan(), permitted_scope=scope)

        self.assertTrue(result.complete)
        self.assertEqual([source["id"] for source in result.bundle.sources], ["S1", "S2", "S3"])
        self.assertEqual(result.calculations[0].result.result, Decimal("10.0"))
        self.assertIn("[APPLICATION-VALIDATED FINANCIAL FACTS]", result.bundle.context)
        self.assertIn("[APPLICATION-VERIFIED CALCULATIONS]", result.bundle.context)
        self.assertEqual(len(search.calls), 2)

    def test_executor_fails_closed_when_a_required_group_is_missing(self) -> None:
        executor = MultiHopExecutor(
            document_search=FakeDocumentSearch(omit_year="2024"),
            fact_pipeline=FakeFactPipeline(),
        )
        scope = Scope(
            ("MSFT",),
            ("2024", "2025"),
            required_doc_types=("10K", "TRANSCRIPT"),
        )

        result = executor.execute(plan(), permitted_scope=scope)

        self.assertFalse(result.complete)
        self.assertTrue(
            any("missing_evidence_groups:MSFT-2024" in issue for issue in result.issues)
        )
        self.assertTrue(
            any("calculation_unavailable" in issue for issue in result.issues)
        )


if __name__ == "__main__":
    unittest.main()
