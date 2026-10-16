from __future__ import annotations

import unittest

from src.financial.fact_extractor import FinancialFactExtractor
from src.financial.fact_pipeline import FinancialFactPipeline
from src.financial.extracted_fact_validator import ExtractedFactValidator
from src.financial.models import CandidateFinancialFact, FactExtractionPayload
from src.schemas import Scope


class FakeChain:
    def __init__(self, payload) -> None:
        self.payload = payload
        self.calls = []

    def invoke(self, values):
        self.calls.append(values)
        return self.payload


def source() -> dict:
    return {
        "id": "S1",
        "ticker": "MSFT",
        "fiscal_year": "2024",
        "doc_type": "10K",
        "section": "Effective Tax Rate",
        "pdf_page": "83",
        "parent_id": "parent-1",
        "source_hash": "hash-1",
        "evidence_text": (
            "<table><tr><th>Year Ended June 30,</th><th>2024</th>"
            "<th>2023</th></tr><tr><td>Effective rate</td>"
            "<td>18.2%</td><td>19.0%</td></tr></table>"
        ),
        "full_evidence_text": (
            "Effective Tax Rate. Year Ended June 30, 2024 2023. "
            "Effective rate 18.2% 19.0%."
        ),
    }


def candidate(**overrides) -> CandidateFinancialFact:
    values = {
        "ticker": "MSFT",
        "metric": "Effective rate",
        "period": "FY2024",
        "raw_value": "18.2%",
        "unit": "percent",
        "source_id": "S1",
        "evidence_excerpt": "Effective rate 18.2% 19.0%.",
        "row_label": "Effective rate",
        "column_label": "2024",
    }
    values.update(overrides)
    return CandidateFinancialFact(**values)


class FinancialFactPipelineTests(unittest.TestCase):
    def test_extractor_returns_structured_candidates_without_validating(self) -> None:
        expected = candidate()
        chain = FakeChain(FactExtractionPayload(facts=[expected]))
        extractor = FinancialFactExtractor(chain=chain)

        result = extractor.extract(
            question="What was the effective tax rate?",
            sources=[source()],
        )

        self.assertEqual(result, (expected,))
        self.assertIn("[SOURCE S1", chain.calls[0]["context"])
        self.assertIn("18.2%", chain.calls[0]["context"])

    def test_candidate_normalizes_harmless_source_header_prefix(self) -> None:
        fact = candidate(source_id="SOURCE S1")

        self.assertEqual(fact.source_id, "S1")

    def test_accounting_basis_placeholders_normalize_to_missing(self) -> None:
        for placeholder in (" null ", "N/A", "not applicable", "—", "unknown"):
            with self.subTest(placeholder=placeholder):
                self.assertIsNone(candidate(accounting_basis=placeholder).accounting_basis)

    def test_validator_accepts_a_fact_tied_to_source_and_scope(self) -> None:
        validator = ExtractedFactValidator()
        result = validator.validate(
            [candidate()],
            [source()],
            permitted_scope=Scope(("MSFT",), ("2024",), "10K"),
        )

        self.assertEqual(len(result.valid_facts), 1)
        self.assertEqual(result.valid_facts[0].numeric_value.as_tuple().exponent, -1)
        self.assertEqual(str(result.valid_facts[0].numeric_value), "18.2")
        self.assertEqual(result.valid_facts[0].source_id, "S1")
        self.assertEqual(result.rejected_facts, ())

    def test_fact_id_is_stable_when_retrieval_rank_changes_source_id(self) -> None:
        validator = ExtractedFactValidator()
        first = validator.validate([candidate()], [source()])
        reranked_source = {**source(), "id": "S2"}
        reranked = validator.validate(
            [candidate(source_id="S2")],
            [reranked_source],
        )

        self.assertEqual(first.valid_facts[0].fact_id, reranked.valid_facts[0].fact_id)

    def test_validator_accepts_supported_table_tokens_when_cells_are_noncontiguous(self) -> None:
        result = ExtractedFactValidator().validate(
            [candidate(evidence_excerpt="Effective rate 2024 18.2%")],
            [source()],
        )

        self.assertEqual(len(result.valid_facts), 1)

    def test_validator_rejects_excerpt_words_absent_from_source(self) -> None:
        result = ExtractedFactValidator().validate(
            [candidate(evidence_excerpt="Effective rate 2024 18.2% estimated")],
            [source()],
        )

        self.assertEqual(result.valid_facts, ())
        self.assertEqual(
            result.rejected_facts[0].reason,
            "evidence_excerpt_not_in_source",
        )

    def test_validator_rejects_unknown_source_and_fabricated_value(self) -> None:
        validator = ExtractedFactValidator()
        result = validator.validate(
            [candidate(source_id="S9"), candidate(raw_value="17.4%")],
            [source()],
        )

        self.assertEqual(result.valid_facts, ())
        self.assertEqual(
            [item.reason for item in result.rejected_facts],
            ["unknown_source_id", "raw_value_not_in_source"],
        )

    def test_missing_extractor_source_id_is_rejected_without_aborting_search(self) -> None:
        unlinked = candidate(source_id="null")

        result = ExtractedFactValidator().validate([unlinked], [source()])

        self.assertEqual(result.valid_facts, ())
        self.assertEqual(result.rejected_facts[0].source_id, None)
        self.assertEqual(result.rejected_facts[0].reason, "unknown_source_id")

    def test_validator_rejects_source_outside_permitted_scope(self) -> None:
        result = ExtractedFactValidator().validate(
            [candidate()],
            [source()],
            permitted_scope=Scope(("TSLA",), ("2024",), "10K"),
        )

        self.assertEqual(result.valid_facts, ())
        self.assertEqual(
            result.rejected_facts[0].reason,
            "source_outside_permitted_scope",
        )

    def test_validator_accepts_transcript_period_from_locked_source_metadata(self) -> None:
        transcript = {
            **source(),
            "fiscal_year": "2024",
            "doc_type": "TRANSCRIPT",
            "evidence_text": (
                "Commercial remaining performance obligation increased to $269 billion."
            ),
            "full_evidence_text": (
                "Commercial remaining performance obligation increased to $269 billion."
            ),
        }
        fact = candidate(
            metric="Commercial remaining performance obligation",
            period="Fiscal 2024",
            raw_value="$269 billion",
            unit="USD",
            source_id="S1",
            evidence_excerpt=(
                "Commercial remaining performance obligation $269 billion"
            ),
            row_label="Commercial remaining performance obligation",
            column_label="Fiscal 2024",
        )

        result = ExtractedFactValidator().validate(
            [fact], [transcript],
            permitted_scope=Scope(("MSFT",), ("2024",), "TRANSCRIPT"),
        )

        self.assertEqual(len(result.valid_facts), 1)
        self.assertEqual(result.rejected_facts, ())

    def test_validator_still_rejects_comparative_period_absent_from_evidence(self) -> None:
        transcript = {
            **source(),
            "fiscal_year": "2024",
            "doc_type": "TRANSCRIPT",
            "evidence_text": (
                "Commercial remaining performance obligation increased to $269 billion."
            ),
            "full_evidence_text": (
                "Commercial remaining performance obligation increased to $269 billion."
            ),
        }
        fact = candidate(
            metric="Commercial remaining performance obligation",
            period="Fiscal 2023",
            raw_value="$269 billion",
            unit="USD",
            source_id="S1",
            evidence_excerpt=(
                "Commercial remaining performance obligation $269 billion"
            ),
            row_label="Commercial remaining performance obligation",
            column_label="Fiscal 2023",
        )

        result = ExtractedFactValidator().validate([fact], [transcript])

        self.assertEqual(result.valid_facts, ())
        self.assertEqual(result.rejected_facts[0].reason, "period_not_in_source")

    def test_empty_evidence_returns_no_extracted_facts(self) -> None:
        chain = FakeChain({"facts": []})
        extractor = FinancialFactExtractor(chain=chain)

        self.assertEqual(
            extractor.extract(question="What was revenue?", sources=[]),
            (),
        )
        self.assertEqual(chain.calls, [])

    def test_pipeline_composes_extraction_and_validation(self) -> None:
        chain = FakeChain({"facts": [candidate().model_dump()]})
        pipeline = FinancialFactPipeline(
            extractor=FinancialFactExtractor(chain=chain),
            validator=ExtractedFactValidator(),
        )

        result = pipeline.run(
            question="What was the effective rate in 2024?",
            sources=[source()],
            permitted_scope=Scope(("MSFT",), ("2024",), "10K"),
        )

        self.assertEqual(len(result.valid_facts), 1)
        self.assertEqual(result.rejected_facts, ())

    def test_pipeline_recovers_rejected_fact_from_statement_table(self) -> None:
        table = (
            "## INCOME STATEMENTS\n(In millions)\n"
            "<table><tr><th>2025</th><th>2024</th><th>2023</th></tr>"
            "<tr><td>Research and development</td><td>32,488</td>"
            "<td>29,510</td><td>27,195</td></tr>"
            "<tr><td>Total revenue</td><td>281,724</td>"
            "<td>245,122</td><td>211,915</td></tr></table>"
        )
        evidence = {
            **source(),
            "fiscal_year": "2025",
            "section": "Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
            "evidence_text": table,
            "full_evidence_text": table,
        }
        extracted = FactExtractionPayload(
            facts=[
                candidate(
                    metric="Research and development expense",
                    period="FY2025",
                    raw_value="32,488",
                    unit="USD",
                    scale="millions",
                    currency="USD",
                    evidence_excerpt="Research and development 2025 32,488",
                    row_label="Research and development",
                    column_label="2025",
                ),
                candidate(
                    metric="Total revenue",
                    period="FY2025",
                    raw_value="281,724",
                    unit="USD",
                    scale="millions",
                    currency="USD",
                    evidence_excerpt="Total revenue 2025 281,724 reported estimate",
                    row_label="Total revenue",
                    column_label="2025",
                ),
            ]
        )
        pipeline = FinancialFactPipeline(
            extractor=FinancialFactExtractor(chain=FakeChain(extracted)),
        )

        result = pipeline.run(
            question="R&D expense and total revenue for FY2025",
            sources=[evidence],
            permitted_scope=Scope(("MSFT",), ("2025",), "10K"),
        )

        self.assertEqual(
            {fact.row_label: fact.numeric_value for fact in result.valid_facts},
            {
                "Research and development": 32488,
                "Total revenue": 281724,
            },
        )
        self.assertTrue(
            all("exact_table_row" in fact.validation_checks for fact in result.valid_facts if fact.row_label == "Total revenue")
        )
        self.assertEqual(result.rejected_facts, ())

    def test_pipeline_recovers_grouped_per_share_row_before_model_extraction(self) -> None:
        table = (
            "## INCOME STATEMENTS\n(In millions, except per share data)\n"
            "<table><tr><th>Metric</th><th>2025</th><th>2024</th></tr>"
            "<tr><td colspan='3'>Net income per share of common stock attributable to common stockholders</td></tr>"
            "<tr><td>Basic</td><td>$14.10</td><td>$12.00</td></tr>"
            "<tr><td>Diluted</td><td>$13.64</td><td>$11.80</td></tr>"
            "<tr><td colspan='3'>Weighted average shares used in computing net income per share</td></tr>"
            "<tr><td>Basic</td><td>3,225</td><td>3,197</td></tr>"
            "<tr><td>Diluted</td><td>3,528</td><td>3,498</td></tr></table>"
        )
        evidence = {
            **source(),
            "fiscal_year": "2025",
            "section": "Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
            "evidence_text": table,
            "full_evidence_text": table,
        }

        class ExtractorMustNotRun:
            def extract(self, **kwargs):
                raise AssertionError("An exact statement cell must bypass model extraction.")

        requirement = {
            "requirement_id": "eps_2025",
            "ticker": "MSFT",
            "source_year": "2025",
            "doc_type": "10K",
            "kind": "fact",
            "reported_period": "FY2025",
            "metric": "diluted earnings per share",
            "qualifiers": [],
            "value_type": None,
        }
        result = FinancialFactPipeline(extractor=ExtractorMustNotRun()).run(
            question="What was diluted EPS in 2025?",
            sources=[evidence],
            permitted_scope=Scope(("MSFT",), ("2025",), "10K"),
            requirements=(requirement,),
        )

        self.assertEqual(len(result.valid_facts), 1)
        fact = result.valid_facts[0]
        self.assertEqual(str(fact.numeric_value), "13.64")
        self.assertEqual(fact.value_type, "per_share")
        self.assertEqual(fact.normalized_unit, "per_share")
        self.assertIsNone(fact.scale)
        self.assertEqual(fact.requirement_id, "eps_2025")
        self.assertIn("exact_table_row", fact.validation_checks)

    def test_grouped_basic_row_cannot_satisfy_diluted_eps_requirement(self) -> None:
        table = (
            "## INCOME STATEMENTS\n(In millions, except per share data)\n"
            "<table><tr><th>Metric</th><th>2025</th></tr>"
            "<tr><td colspan='2'>Earnings per share</td></tr>"
            "<tr><td>Basic</td><td>$14.10</td></tr></table>"
        )
        evidence = {
            **source(),
            "fiscal_year": "2025",
            "section": "Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
            "evidence_text": table,
            "full_evidence_text": table,
        }
        requirement = {
            "requirement_id": "eps_2025",
            "ticker": "MSFT",
            "source_year": "2025",
            "doc_type": "10K",
            "kind": "fact",
            "reported_period": "FY2025",
            "metric": "diluted earnings per share",
            "qualifiers": [],
            "value_type": None,
        }

        result = FinancialFactPipeline().recover_exact_table_row(
            metric_hint=requirement["metric"],
            period_year="2025",
            sources=[evidence],
            permitted_scope=Scope(("MSFT",), ("2025",), "10K"),
            requirements=(requirement,),
        )

        self.assertEqual(result.valid_facts, ())

    def test_per_share_caption_does_not_change_a_currency_row_unit(self) -> None:
        table = (
            "## INCOME STATEMENTS\n(In millions, except per share data)\n"
            "<table><tr><th>Metric</th><th>2025</th></tr>"
            "<tr><td>Total revenue</td><td>281,724</td></tr>"
            "<tr><td colspan='2'>Earnings per share</td></tr>"
            "<tr><td>Diluted</td><td>$13.64</td></tr></table>"
        )
        evidence = {
            **source(),
            "fiscal_year": "2025",
            "section": "Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
            "evidence_text": table,
            "full_evidence_text": table,
        }
        requirement = {
            "requirement_id": "revenue_2025",
            "ticker": "MSFT",
            "source_year": "2025",
            "doc_type": "10K",
            "kind": "fact",
            "reported_period": "FY2025",
            "metric": "total revenue",
            "qualifiers": [],
            "value_type": "currency",
        }

        result = FinancialFactPipeline().recover_exact_table_row(
            metric_hint=requirement["metric"], period_year="2025",
            sources=[evidence],
            permitted_scope=Scope(("MSFT",), ("2025",), "10K"),
            requirements=(requirement,),
        )

        self.assertEqual(len(result.valid_facts), 1)
        self.assertEqual(result.valid_facts[0].value_type, "currency")
        self.assertEqual(result.valid_facts[0].scale, "millions")

    def test_exact_table_row_recovery_validates_requested_year_cell(self) -> None:
        table = (
            "## INCOME STATEMENTS\n(In millions)\n"
            "<table><tr><th>Year</th><th>2025</th><th>2024</th></tr>"
            "<tr><td>Product</td><td>$63,946</td><td>$64,773</td></tr>"
            "<tr><td>Research and development</td><td>32,488</td>"
            "<td>29,510</td></tr></table>"
        )
        evidence = {
            **source(),
            "fiscal_year": "2025",
            "section": "Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
            "evidence_text": table,
            "full_evidence_text": table,
        }
        pipeline = FinancialFactPipeline(
            extractor=FinancialFactExtractor(chain=FakeChain({"facts": []})),
        )

        result = pipeline.recover_exact_table_row(
            metric_hint="research and development expense",
            period_year="2025",
            sources=[evidence],
            permitted_scope=Scope(("MSFT",), ("2025",), "10K"),
        )

        self.assertEqual(len(result.valid_facts), 1)
        fact = result.valid_facts[0]
        self.assertEqual(str(fact.numeric_value), "32488")
        self.assertEqual(fact.column_label, "2025")
        self.assertEqual(fact.scale, "millions")
        self.assertEqual(fact.currency, "USD")
        self.assertIn("exact_table_row", fact.validation_checks)
        self.assertEqual(result.rejected_facts, ())

    def test_exact_row_recovery_uses_consolidated_total_not_segment_rows(self) -> None:
        table = (
            "## INCOME STATEMENTS\n(In millions)\n"
            "<table><tr><th>Year</th><th>2025</th></tr>"
            "<tr><td>Product</td><td>$63,946</td></tr>"
            "<tr><td>Total revenue</td><td>281,724</td></tr></table>"
        )
        evidence = {
            **source(),
            "fiscal_year": "2025",
            "section": "Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
            "evidence_text": table,
            "full_evidence_text": table,
        }

        pipeline = FinancialFactPipeline(
            extractor=FinancialFactExtractor(chain=FakeChain({"facts": []}))
        )
        result = pipeline.recover_exact_table_row(
            metric_hint="Revenue",
            period_year="2025",
            sources=[evidence],
            permitted_scope=Scope(("MSFT",), ("2025",), "10K"),
        )

        self.assertEqual(len(result.valid_facts), 1)
        self.assertEqual(result.valid_facts[0].numeric_value, 281724)
        self.assertEqual(result.valid_facts[0].row_label, "Total revenue")

    def test_exact_row_recovery_accepts_plural_label_and_statement_currency(self) -> None:
        table = (
            "## CASH FLOWS STATEMENTS\n(In millions)\n"
            "<table><tr><th>Year</th><th>2025</th></tr>"
            "<tr><td>Total revenues</td><td>94,827</td></tr></table>"
        )
        evidence = {
            **source(),
            "ticker": "TSLA",
            "fiscal_year": "2025",
            "section": "Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
            "evidence_text": table,
            "full_evidence_text": table,
        }

        pipeline = FinancialFactPipeline(
            extractor=FinancialFactExtractor(chain=FakeChain({"facts": []}))
        )
        result = pipeline.recover_exact_table_row(
            metric_hint="total revenue",
            period_year="2025",
            sources=[evidence],
            permitted_scope=Scope(("TSLA",), ("2025",), "10K"),
        )

        self.assertEqual(len(result.valid_facts), 1)
        self.assertEqual(result.valid_facts[0].row_label, "Total revenues")
        self.assertEqual(result.valid_facts[0].currency, "USD")

    def test_exact_row_recovery_accepts_date_qualified_year_header(self) -> None:
        table = (
            "## CONSOLIDATED BALANCE SHEETS\n(In millions)\n"
            "<table><tr><th></th><th>December 31, 2025</th>"
            "<th>December 31, 2024</th></tr>"
            "<tr><td>Short-term investments</td><td>$27,546</td>"
            "<td>$20,424</td></tr></table>"
        )
        evidence = {
            **source(),
            "ticker": "TSLA",
            "fiscal_year": "2025",
            "section": "Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
            "evidence_text": table,
            "full_evidence_text": table,
        }

        result = FinancialFactPipeline().recover_exact_table_row(
            metric_hint="short-term investments",
            period_year="2025",
            sources=[evidence],
            permitted_scope=Scope(("TSLA",), ("2025",), "10K"),
        )

        self.assertEqual(len(result.valid_facts), 1)
        self.assertEqual(result.valid_facts[0].numeric_value, 27546)
        self.assertEqual(result.valid_facts[0].row_label, "Short-term investments")


if __name__ == "__main__":
    unittest.main()
