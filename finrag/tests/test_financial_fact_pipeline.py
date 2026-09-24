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

        result = FinancialFactPipeline().recover_exact_table_row(
            metric_hint="Revenue",
            period_year="2025",
            sources=[evidence],
            permitted_scope=Scope(("MSFT",), ("2025",), "10K"),
        )

        self.assertEqual(len(result.valid_facts), 1)
        self.assertEqual(result.valid_facts[0].numeric_value, 281724)
        self.assertEqual(result.valid_facts[0].row_label, "Total revenue")


if __name__ == "__main__":
    unittest.main()
