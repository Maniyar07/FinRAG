from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError

import pytest

from src.financial.coverage import fact_matches_requirement, normalize_reported_period
from src.financial.extracted_fact_validator import ExtractedFactValidator
from src.financial.fact_pipeline import FinancialFactPipeline
from src.financial.models import CandidateFinancialFact
from src.generation.evidence_answer import (
    EvidenceQuote, EvidenceSynthesis, NarrativeAudit, RequirementAnswer,
    evidence_passages, render_evidence_answer, resolve_evidence_answers,
)
from src.retrieval.structured_lookup import StructuredDocumentLookup
from src.retrieval.reranker import CohereRerankSettings, CohereReranker
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.lexical_index import LexicalHit
from src.schemas import RankedChild, Scope
from src.agent.tools import RequestPlanInput, PlannedRequirement, create_financial_tools
from src.financial.calculator import FinancialCalculator


SOURCE = dict(id="S1", ticker="ACME", fiscal_year="2025", doc_type="10K")
REQUIREMENT = dict(requirement_id="r1", ticker="ACME", source_year="2025", doc_type="10K",
                   reported_period="FY2025", metric="development expense", kind="fact",
                   qualifiers=[], value_type="currency")


@pytest.mark.parametrize("unit_heading", ["Earnings per share", "Earnings per-share", "Net income per share"])
def test_nested_per_share_units_do_not_inherit_the_amount_scale(unit_heading):
    source = {**SOURCE, "section": "Financial statements", "evidence_text":
        "## Income Statements\n(In millions, except per-share amounts)\n"
        "<table><tr><th>Metric</th><th>2025</th></tr>"
        "<tr><td>Net income</td><td>100</td></tr>"
        f"<tr><td colspan='2'>{unit_heading}</td></tr>"
        "<tr><td>Diluted</td><td>$1.25</td></tr></table>"}
    requirement = {**REQUIREMENT, "metric": f"diluted {unit_heading}", "value_type": None}
    candidate = CandidateFinancialFact(ticker="ACME", metric=requirement["metric"], period="2025",
        raw_value="$1.25", unit="USD", currency="USD", scale="millions", source_id="S1", row_label="Diluted",
        column_label="2025", requirement_id="r1", evidence_excerpt="Diluted 2025 $1.25")
    result = FinancialFactPipeline(extractor=SimpleNamespace(extract=lambda **kwargs: (candidate,))).run(
        question="Report the diluted per-share value", sources=[source], permitted_scope=Scope(("ACME",), ("2025",), "10K"),
        requirements=(requirement,))
    assert len(result.valid_facts) == 1, result
    fact = result.valid_facts[0]
    assert fact.value_type == "per_share" and fact.scale is None
    assert fact.numeric_value == fact.base_value == 1.25


def test_caption_cannot_promote_a_component_to_an_aggregate():
    source = {**SOURCE, "evidence_text": 'Cash and investments (in millions) <table>'
        '<tr><th></th><th>2025</th></tr>'
        '<tr><td>Cash</td><td>10</td></tr><tr><td>Investments</td><td>20</td></tr>'
        '<tr><td>Total cash and investments</td><td>30</td></tr></table>'}
    requirement = {**REQUIREMENT, "metric": "cash and investments"}
    candidate = CandidateFinancialFact(ticker="ACME", metric="cash and investments", period="2025",
        raw_value="10", unit="USD", scale="millions", source_id="S1", row_label="Cash",
        column_label="2025", requirement_id="r1", evidence_excerpt="Cash 2025 10")
    assert not ExtractedFactValidator().validate([candidate], [source], requirements=(requirement,)).valid_facts
    recovered = FinancialFactPipeline(extractor=SimpleNamespace(extract=lambda **kwargs: ())).run(
        question="Cash and investments", sources=[source], permitted_scope=Scope(("ACME",), ("2025",), "10K"),
        requirements=(requirement,))
    assert [fact.numeric_value for fact in recovered.valid_facts] == [30]


def test_leaf_column_cannot_narrow_an_aggregate_row():
    source = {**SOURCE, "evidence_text": 'Cash and investments (in millions) <table>'
        '<tr><th></th><th colspan="3">2025</th></tr>'
        '<tr><th></th><th>Fair Value</th><th>Cash</th><th>Investments</th></tr>'
        '<tr><td>Total cash and investments</td><td>30</td><td>10</td><td>20</td></tr></table>'}
    requirement = {**REQUIREMENT, "metric": "cash and investments"}
    candidate = CandidateFinancialFact(ticker="ACME", metric="cash and investments", period="2025",
        raw_value="10", unit="USD", scale="millions", source_id="S1", row_label="Total cash and investments",
        column_label="2025 Cash", requirement_id="r1", evidence_excerpt="Total cash and investments 2025 10")
    assert not ExtractedFactValidator().validate([candidate], [source], requirements=(requirement,)).valid_facts
    candidate = candidate.model_copy(update={"raw_value": "30", "column_label": "2025 Fair Value",
                                             "evidence_excerpt": "Total cash and investments 2025 30"})
    assert len(ExtractedFactValidator().validate([candidate], [source], requirements=(requirement,)).valid_facts) == 1


def test_multiple_possible_columns_cannot_supply_a_component_with_sibling_labels():
    source = {**SOURCE, "evidence_text": 'Cash and investments (in millions) <table>'
        '<tr><th></th><th colspan="3">2025</th></tr>'
        '<tr><th></th><th>Fair Value</th><th>Cash</th><th>Investments</th></tr>'
        '<tr><td>Cash</td><td>10</td><td>10</td><td>—</td></tr>'
        '<tr><td>Total cash and investments</td><td>30</td><td>10</td><td>20</td></tr></table>'}
    candidate = CandidateFinancialFact(ticker="ACME", metric="cash and investments", period="2025",
        raw_value="10", unit="USD", scale="millions", source_id="S1", row_label="Cash",
        column_label="2025 Cash Investments", requirement_id="r1", evidence_excerpt="Cash 2025 10")
    result = ExtractedFactValidator().validate([candidate], [source],
        requirements=({**REQUIREMENT, "metric": "cash and investments"},))
    assert not result.valid_facts


def test_misplaced_source_reference_requires_independent_cell_binding():
    incorrect = {**SOURCE, "evidence_text": 'Unrelated balance in 2025 was $10 million.'}
    correct = {**SOURCE, "id": "S2", "evidence_text": 'Asset balance (in millions) <table>'
        '<tr><th></th><th colspan="2">2025</th></tr>'
        '<tr><th></th><th>Cost</th><th>Fair Value</th></tr>'
        '<tr><td>Asset balance</td><td>20</td><td>25</td></tr></table>'}
    candidate = CandidateFinancialFact(ticker="ACME", metric="Asset balance", period="2025",
        raw_value="25", unit="USD", scale="millions", source_id="S1", row_label="Asset balance",
        column_label="2025 Fair Value", requirement_id="r1", evidence_excerpt="Asset balance 2025 25")
    pipeline = FinancialFactPipeline(extractor=SimpleNamespace(extract=lambda **kwargs: (candidate,)))
    result = pipeline.run(question="Asset balance", sources=[incorrect, correct],
        permitted_scope=Scope(("ACME",), ("2025",), "10K"), requirements=({**REQUIREMENT, "metric": "asset balance"},))
    assert [(fact.source_id, fact.numeric_value) for fact in result.valid_facts] == [("S2", 25)]
    assert not result.rejected_facts
    result = pipeline.run(question="Asset balance", sources=[incorrect],
        permitted_scope=Scope(("ACME",), ("2025",), "10K"), requirements=({**REQUIREMENT, "metric": "asset balance"},))
    assert not result.valid_facts


def test_narrative_subject_is_shared_without_altering_numeric_requirements():
    numeric = PlannedRequirement.model_validate(REQUIREMENT)
    narrative = PlannedRequirement.model_validate({**REQUIREMENT, "requirement_id": "r2", "kind": "narrative",
                                                   "metric": "deployment outlook", "value_type": None})
    plan = RequestPlanInput(narrative_subject="storage systems", requirements=[numeric, narrative])
    assert plan.requirements[0] == numeric
    assert plan.requirements[1] == narrative
    assert plan.narrative_subject == "storage systems"
    # Revalidation must not repeatedly expand the same evidence question.
    assert RequestPlanInput.model_validate(plan.model_dump()).model_dump() == plan.model_dump()


def test_internal_planner_failure_terminates_instead_of_repeating():
    builder = Mock(side_effect=TypeError("internal planning failure"))
    tools = {tool.name: tool for tool in create_financial_tools(SimpleNamespace(), FinancialCalculator(),
        permitted_scope=Scope(("ACME",), ("2025",), "10K"), plan_builder=builder)}
    import json
    payload = json.loads(tools["plan_financial_request"].func(runtime=SimpleNamespace(state={"messages": []})))
    assert payload["error_code"] == "internal_error"
    assert payload["plan_attempt_exhausted"]
    assert not payload["retryable"]
    assert builder.call_count == 1


def test_rejected_conversion_gets_one_literal_copy_repair_from_same_evidence():
    source = {**SOURCE, "evidence_text": "Service obligations in 2025 were $3 billion."}
    wrong = CandidateFinancialFact(ticker="ACME", metric="service obligations", period="2025",
        raw_value="$3,000 million", unit="USD", source_id="S1", requirement_id="r1",
        evidence_excerpt=source["evidence_text"])
    correct = wrong.model_copy(update={"raw_value": "$3 billion"})
    extractor = SimpleNamespace(extract=Mock(side_effect=[(wrong,), (correct,)]))
    result = FinancialFactPipeline(extractor=extractor).run(question="Service obligations",
        sources=[source], permitted_scope=Scope(("ACME",), ("2025",), "10K"),
        requirements=({**REQUIREMENT, "metric": "service obligations"},))
    assert extractor.extract.call_count == 2
    assert extractor.extract.call_args.kwargs["sources"] == [source]
    assert len(result.valid_facts) == 1
    assert result.valid_facts[0].base_value == 3_000_000_000
    assert "repaired_extraction" in result.valid_facts[0].validation_checks
    assert not result.rejected_facts


def test_prose_cannot_assemble_a_fact_from_unrelated_sentences():
    source = {**SOURCE, "evidence_text": "Revenue in 2025 was $10 million. Debt in 2025 was $20 million."}
    candidate = CandidateFinancialFact(ticker="ACME", metric="Revenue", period="2025",
        raw_value="$20 million", unit="USD", source_id="S1", requirement_id="r1",
        evidence_excerpt="Revenue in 2025 was $20 million")
    result = ExtractedFactValidator().validate([candidate], [source], requirements=({**REQUIREMENT, "metric": "revenue"},))
    assert not result.valid_facts


@pytest.mark.parametrize("metric,row", [("total gross profit", "Gross profit"), ("operating income", "Income from operations")])
def test_primary_statement_subject_uses_shared_matching_and_scope(metric, row):
    source = {**SOURCE, "section": "Financial Statements and Supplementary Data",
        "evidence_text": 'CONSOLIDATED STATEMENTS OF OPERATIONS (in millions) <table>'
        '<tr><th>Metric</th><th>2025</th></tr><tr><td colspan="2">Cost of revenue</td></tr>'
        f'<tr><td>{row}</td><td>20</td></tr></table>'}
    candidate = CandidateFinancialFact(ticker="ACME", metric=metric, period="2025", raw_value="20",
        unit="USD", scale="millions", source_id="S1", row_label=row, column_label="2025", requirement_id="r1",
        evidence_excerpt=f"{row} 2025 20")
    result = ExtractedFactValidator().validate([candidate], [source], requirements=({**REQUIREMENT, "metric": metric},))
    assert len(result.valid_facts) == 1


def test_monetary_statement_cell_cannot_become_a_percent_from_a_unit_guess():
    source = {**SOURCE, "evidence_text": 'STATEMENTS OF OPERATIONS (in millions) <table>'
        '<tr><th>Metric</th><th>2025</th></tr><tr><td>Gross margin</td><td>20</td></tr></table>'}
    candidate = CandidateFinancialFact(ticker="ACME", metric="gross margin", period="2025", raw_value="20",
        unit="percent", source_id="S1", row_label="Gross margin", column_label="2025", requirement_id="r1",
        evidence_excerpt="Gross margin 2025 20")
    result = ExtractedFactValidator().validate([candidate], [source],
        requirements=({**REQUIREMENT, "metric": "gross margin", "value_type": None},))
    assert not result.valid_facts
    assert result.rejected_facts[0].reason == "table_unit_type_mismatch"


def test_fiscal_year_shorthand_is_resolved_against_the_source_year():
    source = {**SOURCE, "doc_type": "TRANSCRIPT", "fiscal_period": "Q4",
        "evidence_text": "Our FY26 gross margin outlook for Q1 is roughly 20%."}
    candidate = CandidateFinancialFact(ticker="ACME", metric="gross margin", period="Q1 2026", raw_value="roughly 20%",
        unit="percent", source_id="S1", requirement_id="r1", evidence_excerpt=source["evidence_text"])
    result = ExtractedFactValidator().validate([candidate], [source], requirements=({**REQUIREMENT,
        "doc_type": "TRANSCRIPT", "metric": "gross margin", "reported_period": "Q1 2026", "value_type": "percent"},))
    assert len(result.valid_facts) == 1


@pytest.mark.parametrize("quote,proof,supported", [
    ("Device production will increase.", "Device production", False),
    ("Server infrastructure capacity is constrained.", "Server infrastructure", True),
    ("Data centers remain capacity constrained. Our modular compute racks will expand next year.", "modular compute racks", True),
])
def test_narrative_scope_requires_a_subject_phrase_from_its_own_quotes(quote, proof, supported):
    source = {**SOURCE, "evidence_text": quote}
    requirement = {**REQUIREMENT, "kind": "narrative", "metric": "capacity outlook", "value_type": None}
    plan = {"narrative_subject": "server infrastructure", "requirements": [requirement]}
    answers = [RequirementAnswer(requirement_id="r1", quotes=[EvidenceQuote(source_id="S1", excerpt=quote)])]
    auditor = SimpleNamespace(invoke=Mock(return_value={"assessments": [{"requirement_id": "r1",
        "quote_assessments": {"0": {"reason": "This quote establishes the requested subject." if supported else "The quote concerns another product.", "answers_question": supported}},
        "reason": "The evidence supports the requested detail.", "supported": supported}]}))
    model = SimpleNamespace(with_structured_output=Mock(return_value=auditor))
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=model):
        if supported:
            assert render_evidence_answer(plan, answers, [], {}, {"S1": source}, {}, {})["complete"]
        else:
            with pytest.raises(ValueError, match="Unsupported narrative"):
                render_evidence_answer(plan, answers, [], {}, {"S1": source}, {}, {})


def test_semantic_request_binding_preserves_source_wording():
    source = {**SOURCE, "evidence_text": "Development expenditures in 2025 were $20 million."}
    candidate = CandidateFinancialFact(ticker="ACME", metric="Development expenditures", period="2025",
        raw_value="$20 million", unit="USD", source_id="S1", requirement_id="r1",
        evidence_excerpt=source["evidence_text"])
    result = ExtractedFactValidator().validate([candidate], [source], requirements=(REQUIREMENT,))
    assert len(result.valid_facts) == 1
    assert fact_matches_requirement(result.valid_facts[0], REQUIREMENT, source)
    assert result.valid_facts[0].metric == "Development expenditures"
    assert not fact_matches_requirement(result.valid_facts[0], {**REQUIREMENT, "requirement_id": "r2"}, source)


@pytest.mark.parametrize("update, reason", [
    ({"requirement_id": "unknown"}, "unknown_requirement_id"),
    ({"raw_value": "20%", "unit": "percent"}, "requirement_source_or_type_mismatch"),
])
def test_request_binding_cannot_bypass_type_or_identity(update, reason):
    source = {**SOURCE, "evidence_text": "Development expenditures in 2025 were $20 million and grew 20%."}
    candidate = CandidateFinancialFact(ticker="ACME", metric="Development expenditures", period="2025",
        raw_value="$20 million", unit="USD", source_id="S1", requirement_id="r1",
        evidence_excerpt=source["evidence_text"]).model_copy(update=update)
    result = ExtractedFactValidator().validate([candidate], [source], requirements=(REQUIREMENT,))
    assert not result.valid_facts
    assert result.rejected_facts[0].reason == reason


def test_exact_dates_are_preserved_without_fiscal_quarter_inference():
    assert normalize_reported_period("Q2 2025", "Assets as of June 30, 2025") == "2025-06-30"
    assert normalize_reported_period("June 30, 2025") == "2025-06-30"
    assert normalize_reported_period("Q4 2025", "Q4 2025 revenue") == "Q4 2025"


def test_spanned_year_headers_resolve_currency_and_numeric_cells_safely():
    source = {**SOURCE, "section": "Financial Statements and Supplementary Data",
        "evidence_text": 'CONSOLIDATED BALANCE SHEETS (in millions) <table>'
        '<tr><th>Metric</th><th colspan="2">2025</th><th colspan="2">2024</th></tr>'
        '<tr><td>Assets</td><td>$</td><td>120</td><td>$</td><td>100</td></tr>'
        '<tr><td>Empty row</td><td></td><td></td><td></td><td></td></tr></table>'}
    result = FinancialFactPipeline(extractor=SimpleNamespace(extract=lambda **kwargs: ())).run(
        question="Assets", sources=[source], permitted_scope=Scope(("ACME",), ("2025",), "10K"),
        requirements=({**REQUIREMENT, "metric": "Assets"},))
    assert len(result.valid_facts) == 1
    assert result.valid_facts[0].numeric_value == 120
    candidate = CandidateFinancialFact(ticker="ACME", metric="Assets", period="2025", raw_value="100",
        unit="USD", scale="millions", source_id="S1", row_label="Assets", column_label="2025",
        evidence_excerpt="Assets 2025 100")
    invalid = ExtractedFactValidator().validate([candidate], [source])
    assert not invalid.valid_facts


def test_forward_period_requires_explicit_request_and_forecast_evidence():
    source = {**SOURCE, "doc_type": "TRANSCRIPT", "evidence_text": "We expect Q1 capital spending to be over $30 billion."}
    requirement = {**REQUIREMENT, "doc_type": "TRANSCRIPT", "metric": "capital spending", "reported_period": "Q1 2026"}
    candidate = CandidateFinancialFact(ticker="ACME", metric="capital spending", period="Q1 2026",
        raw_value="over $30 billion", source_id="S1", requirement_id="r1", evidence_excerpt=source["evidence_text"])
    result = ExtractedFactValidator().validate([candidate], [source], requirements=(requirement,))
    assert len(result.valid_facts) == 1
    wrong = candidate.model_copy(update={"period": "Q2 2026"})
    assert not ExtractedFactValidator().validate([wrong], [source], requirements=(requirement,)).valid_facts


def test_narrative_ids_merge_repeated_selections_without_rewriting_evidence():
    source = {**SOURCE, "evidence_text": "Demand is strong.\n\nCapacity remains constrained."}
    requirement = {**REQUIREMENT, "kind": "narrative", "metric": "demand and capacity"}
    selector = Mock(return_value={"answers": [
        {"requirement_id": "r1", "passage_ids": ["P1"]},
        {"requirement_id": "r1", "passage_ids": ["P2", "P1"]}]})
    model = SimpleNamespace(with_structured_output=lambda *a, **k: SimpleNamespace(invoke=selector))
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=model):
        answers = resolve_evidence_answers({"requirements": [requirement]}, [], {}, {"S1": source})
    assert [quote.excerpt for quote in answers[0].quotes] == ["Demand is strong.", "Capacity remains constrained."]
    assert len(evidence_passages([source])) == 2


def test_rate_limit_opens_reranker_cooldown_and_preserves_fallback():
    settings = CohereRerankSettings("test-key", "model", 20, 5, 2, 12, 1, 6000)
    reranker = CohereReranker(settings)
    child = RankedChild("c", "p", "Evidence", dict(ticker="ACME", fiscal_year="2025", doc_type="10K"))
    error = HTTPError("https://example.invalid", 429, "quota", {"Retry-After": "60"}, None)
    with patch("src.retrieval.reranker.urlopen", side_effect=error) as request:
        assert reranker.rerank("Revenue", [child], Scope(("ACME",), ("2025",), "10K")) == [child]
        assert reranker.rerank("Income", [child], Scope(("ACME",), ("2025",), "10K")) == [child]
    assert request.call_count == 1


def test_request_binding_does_not_turn_total_revenue_into_a_region_value():
    source = {**SOURCE, "evidence_text": 'Revenue (in millions) <table>'
        '<tr><th>Region</th><th>2025</th></tr>'
        '<tr><td>Total revenue</td><td>100</td></tr>'
        '<tr><td>Region A</td><td>20</td></tr></table>'}
    requirement = {**REQUIREMENT, "metric": "revenue", "qualifiers": ["Region A"]}
    candidate = CandidateFinancialFact(ticker="ACME", metric="Total revenue", period="2025",
        raw_value="100", unit="USD", scale="millions", source_id="S1", requirement_id="r1",
        row_label="Total revenue", column_label="2025", evidence_excerpt="Total revenue 100")
    result = ExtractedFactValidator().validate([candidate], [source], requirements=(requirement,))
    assert all(not fact_matches_requirement(fact, requirement, source) for fact in result.valid_facts)


def test_indexed_rows_use_caption_and_qualifiers_across_filing_notes(tmp_path):
    import json
    parent = {"metadata": {"ticker": "ACME", "fiscal_year": "2025", "doc_type": "10K", "item": "Note 3"},
        "page_content": 'Revenue by region (in millions):\n\n<table>'
        '<tr><th>Region</th><th colspan="2">2025</th></tr>'
        '<tr><td>Region A</td><td>$</td><td>20</td></tr></table>\n\n'
        'Assets by region (in millions):\n\n<table>'
        '<tr><th>Region</th><th colspan="2">2025</th></tr>'
        '<tr><td>Region A</td><td>$</td><td>30</td></tr></table>'}
    (tmp_path / "parent.json").write_text(json.dumps(parent), encoding="utf-8")
    rows = StructuredDocumentLookup(tmp_path).statement_rows("revenue Region A",
        Scope(("ACME",), ("2025",), "10K"), qualifiers=("Region A",))
    assert [(row.label, row.value) for row in rows] == [("Region A", "20")]


def test_numeric_requirements_need_typed_facts_including_calculator_operands():
    source = {**SOURCE, "evidence_text": "Development expense was $20 million in 2025."}
    selector = Mock(return_value={"answers": [{"requirement_id": "r1", "passage_ids": ["P1"]}]})
    model = SimpleNamespace(with_structured_output=lambda *a, **k: SimpleNamespace(invoke=selector))
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=model):
        answer = resolve_evidence_answers({"requirements": [REQUIREMENT]}, [], {}, {"S1": source})[0]
        assert answer.unavailable_reason and not answer.quotes
        selector.assert_not_called()
        selector.reset_mock()
        answer = resolve_evidence_answers({"requirements": [REQUIREMENT],
            "calculations": [{"input_requirement_ids": ["r1"]}]}, [], {}, {"S1": source})[0]
        assert answer.unavailable_reason and not answer.quotes
        selector.assert_not_called()


@pytest.mark.parametrize("interpretation, valid", [
    ("The planned investment responds to demand while resource dependencies constrain expansion.", True),
    ("Investment will produce 999 additional facilities.", False),
])
def test_synthesis_has_provenance_and_cannot_introduce_computed_numbers(interpretation, valid):
    source = {**SOURCE, "evidence_text": "Investment responds to demand. Expansion depends on land and energy."}
    requirement = {**REQUIREMENT, "kind": "narrative", "metric": "investment and dependencies"}
    answers = [RequirementAnswer(requirement_id="r1", quotes=[EvidenceQuote(source_id="S1", excerpt=source["evidence_text"])])]
    def structured(schema, **kwargs):
        response = {"interpretation": interpretation, "source_ids": ["S1"]} if schema is EvidenceSynthesis else {"assessments": [
            {"requirement_id": key, "reason": "The evidence supports the interpretation.", "supported": True} for key in ("r1", "synthesis")]}
        return SimpleNamespace(invoke=lambda messages: response)
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=SimpleNamespace(with_structured_output=structured)):
        plan = {"requirements": [requirement], "synthesis": "How do investment and dependencies frame expansion?"}
        if valid:
            result = render_evidence_answer(plan, answers, [], {}, {"S1": source}, {}, {})
            assert result["complete"] and "Interpretation:" in result["answer"]
        else:
            with pytest.raises(ValueError, match="introduces numbers"):
                render_evidence_answer(plan, answers, [], {}, {"S1": source}, {}, {})


def test_audit_schema_uses_typed_rejection_entries():
    schema = NarrativeAudit.model_json_schema()
    assert schema["properties"]["assessments"]["type"] == "array"
    assert schema["$defs"]["EvidenceAssessment"]["additionalProperties"] is False


def test_nested_date_headers_preserve_monetary_scale_and_unscaled_counts():
    source = {**SOURCE, "evidence_text": 'Holdings (in millions except units). <table>'
        '<tr><th></th><th colspan="2">December 31, 2025</th><th colspan="2">December 31, 2024</th></tr>'
        '<tr><th></th><th>Units</th><th>Fair Value</th><th>Units</th><th>Fair Value</th></tr>'
        '<tr><td>Asset A</td><td>12</td><td>$20</td><td>10</td><td>$15</td></tr></table>'}
    requirement = {**REQUIREMENT, "metric": "fair value", "qualifiers": ["Asset A"], "reported_period": "December 31, 2025"}
    candidate = CandidateFinancialFact(ticker="ACME", metric="fair value", period="December 31, 2025",
        raw_value="$20", unit="USD", source_id="S1", row_label="Asset A", column_label="Fair Value",
        requirement_id="r1", evidence_excerpt="Asset A Fair Value December 31, 2025 $20")
    validator = ExtractedFactValidator()
    result = validator.validate([candidate], [source], requirements=(requirement,))
    assert len(result.valid_facts) == 1
    assert result.valid_facts[0].base_value == 20_000_000 and result.valid_facts[0].scale == "millions"
    assert not validator.validate([candidate.model_copy(update={"raw_value": "$15"})], [source], requirements=(requirement,)).valid_facts
    count_requirement = {**requirement, "metric": "units", "value_type": "count"}
    count = candidate.model_copy(update={"metric": "units", "column_label": "Units", "raw_value": "12", "unit": "count",
                                         "evidence_excerpt": "Asset A Units December 31, 2025 12"})
    count_result = validator.validate([count], [source], requirements=(count_requirement,))
    assert count_result.valid_facts[0].base_value == 12 and count_result.valid_facts[0].scale is None


def test_explicit_numeric_plan_recovers_an_omitted_row_from_a_note():
    source = {**SOURCE, "section": "Note 3", "evidence_text": 'Revenue (in millions) <table>'
        '<tr><th>Metric</th><th>2025</th></tr><tr><td>Total revenue</td><td>120</td></tr></table>'}
    result = FinancialFactPipeline(extractor=SimpleNamespace(extract=lambda **kwargs: ())).run(
        question="Revenue", sources=[source], permitted_scope=Scope(("ACME",), ("2025",), "10K"),
        requirements=({**REQUIREMENT, "metric": "total revenue"},))
    assert len(result.valid_facts) == 1 and result.valid_facts[0].numeric_value == 120


def test_period_transitions_are_not_table_row_qualifiers():
    requirement = PlannedRequirement.model_validate({**REQUIREMENT,
        "qualifiers": ["from fiscal 2024 to fiscal 2025", "Region A", "constant currency"]})
    assert requirement.qualifiers == ["Region A", "constant currency"]


def test_temporal_calculations_require_distinct_reported_periods():
    with pytest.raises(ValueError, match="separate earlier and later"):
        RequestPlanInput.model_validate({"requirements": [REQUIREMENT,
            {**REQUIREMENT, "requirement_id": "r2"}], "calculations": [
            {"operation": "percentage_change", "input_requirement_ids": ["r1", "r2"]}]})


def test_total_metric_cannot_use_an_individual_segment_row():
    source = {**SOURCE, "evidence_text": 'Revenue (in millions) <table>'
        '<tr><th>Metric</th><th>2025</th></tr>'
        '<tr><td colspan="2">Segment A</td></tr><tr><td>Revenue</td><td>20</td></tr>'
        '<tr><td colspan="2">Segment B</td></tr><tr><td>Revenue</td><td>30</td></tr>'
        '<tr><td colspan="2">Total</td></tr><tr><td>Revenue</td><td>50</td></tr></table>'}
    requirement = {**REQUIREMENT, "metric": "total revenue"}
    result = FinancialFactPipeline(extractor=SimpleNamespace(extract=lambda **kwargs: ())).run(
        question="Total revenue", sources=[source], permitted_scope=Scope(("ACME",), ("2025",), "10K"),
        requirements=(requirement,))
    assert [fact.numeric_value for fact in result.valid_facts] == [50]


@pytest.mark.parametrize("values, supported", [("20</td><td>25", True), ("25</td><td>25", False)])
def test_missing_leaf_label_binds_only_a_unique_value_in_its_year(values, supported):
    source = {**SOURCE, "evidence_text": 'Asset balance (in millions) <table>'
        '<tr><th></th><th colspan="2">2025</th></tr>'
        '<tr><th></th><th>Cost</th><th>Fair Value</th></tr>'
        f'<tr><td>Asset balance</td><td>{values}</td></tr></table>'}
    requirement = {**REQUIREMENT, "metric": "asset balance"}
    candidate = CandidateFinancialFact(ticker="ACME", metric="asset balance", period="FY2025",
        raw_value="25", unit="USD", source_id="S1", row_label="Asset balance", column_label="2025",
        requirement_id="r1", evidence_excerpt="Asset balance 2025 25")
    result = ExtractedFactValidator().validate([candidate], [source], requirements=(requirement,))
    assert bool(result.valid_facts) is supported
    if supported:
        assert result.valid_facts[0].column_label == "2025 | Fair Value"
        assert result.valid_facts[0].base_value == 25_000_000


def test_planned_subject_search_reuses_the_parent_budget_and_locked_scope():
    scope = Scope(("ACME",), ("2025",), "10K")
    metadata = dict(ticker="ACME", fiscal_year="2025", doc_type="10K", parent_id="p2")
    lexical = SimpleNamespace(search=Mock(return_value=[LexicalHit(
        record={"metadata": metadata, "text": "Capacity remains constrained."}, score=3.0, coverage=1.0)]))
    store = SimpleNamespace(get_many=Mock(return_value=[SimpleNamespace(
        page_content="Capacity remains constrained. Spending is planned.", metadata=metadata)]))
    builder = SimpleNamespace(parent_store=store, build=Mock(return_value="bundle"))
    retriever = HybridRetriever(SimpleNamespace(), lexical_index=lexical, context_builder=builder)
    baseline = RankedChild("c1", "p1", "Other evidence", metadata, fused_score=1.0)
    with patch.object(retriever, "_fuse", return_value=[baseline]):
        assert retriever.retrieve("Combined question", scope,
            requirement_queries=("capacity constraints", "capacity constraints"),
            max_parents=2, max_tokens=3000, max_chars=12000) == "bundle"
    assert lexical.search.call_count == 1
    assert lexical.search.call_args.args[1].groups == scope.groups
    assert store.get_many.call_args.args[0] == ["p2"]
    ranked = builder.build.call_args.args[0]
    assert next(child for child in ranked if child.parent_id == "p2").fused_score > baseline.fused_score
    assert builder.build.call_args.kwargs["max_tokens"] == 3000
    assert builder.build.call_args.kwargs["max_parents"] == 2


@pytest.mark.parametrize("metric, row_label", [
    ("cash and cash equivalents", "Cash and cash equivalents and restricted cash"),
    ("revenue from Region A", "Automotive sales"),
])
def test_semantic_requirement_id_cannot_change_the_quantity_in_a_row(metric, row_label):
    source = {**SOURCE, "evidence_text": f'Revenue and balances (in millions) <table>'
        f'<tr><th>Metric</th><th>2025</th></tr><tr><td>{row_label}</td><td>$20</td></tr></table>'}
    requirement = {**REQUIREMENT, "metric": metric}
    candidate = CandidateFinancialFact(ticker="ACME", metric=metric, period="2025", raw_value="$20",
        row_label=row_label, column_label="2025", source_id="S1", requirement_id="r1",
        evidence_excerpt=f"{row_label} 2025 $20")
    result = ExtractedFactValidator().validate([candidate], [source], requirements=(requirement,))
    assert not result.valid_facts
    assert result.rejected_facts[0].reason == "table_metric_label_mismatch"


def test_omitted_nested_measure_cells_are_recovered_from_the_actual_column():
    source = {**SOURCE, "evidence_text": 'Holdings (in millions) <table>'
        '<tr><th></th><th colspan="2">2025</th></tr>'
        '<tr><th></th><th>Cost Basis</th><th>Fair Value</th></tr>'
        '<tr><td>Asset A</td><td>$20</td><td>$25</td></tr></table>'}
    requirements = tuple({**REQUIREMENT, "requirement_id": f"r{index}", "metric": f"Asset A {measure}"}
                         for index, measure in enumerate(("Cost Basis", "Fair Value"), 1))
    result = FinancialFactPipeline(extractor=SimpleNamespace(extract=lambda **kwargs: ())).run(
        question="Cost basis and fair value", sources=[source], permitted_scope=Scope(("ACME",), ("2025",), "10K"),
        requirements=requirements)
    assert {fact.requirement_id: fact.numeric_value for fact in result.valid_facts} == {"r1": 20, "r2": 25}


def test_nonfinancial_tables_and_lists_preserve_the_requested_output_kind():
    for kind in ("statement", "narrative"):
        plan = RequestPlanInput.model_validate({"requirements": [
            {**REQUIREMENT, "kind": kind, "metric": "operational milestones"}]})
        assert plan.requirements[0].kind == kind


@pytest.mark.parametrize("raw, excerpt, reason", [
    ("$30", "Net income was $20 million in 2025.", "raw_value_not_in_excerpt"),
    ("$20 million", "Net income was $20 million in 2025.", "excerpt_metric_mismatch"),
])
def test_prose_binding_requires_its_own_value_and_subject(raw, excerpt, reason):
    source = {**SOURCE, "evidence_text": "Other income was $30 million in 2025. Net income was $20 million in 2025."}
    requirement = {**REQUIREMENT, "metric": "operating income"}
    candidate = CandidateFinancialFact(ticker="ACME", metric="operating income", period="FY2025",
        raw_value=raw, unit="USD", scale="millions", source_id="S1", requirement_id="r1", evidence_excerpt=excerpt)
    # Include a distant topic mention: it cannot establish this quoted amount.
    source["evidence_text"] += " Operating income was discussed separately."
    result = ExtractedFactValidator().validate([candidate], [source], requirements=(requirement,))
    assert not result.valid_facts
    assert result.rejected_facts[0].reason == reason


def test_narrative_passages_preserve_inherited_subject_without_duplicate_context():
    text = "Storage demand remains strong. It will benefit from the new modular product."
    passages = evidence_passages([{**SOURCE, "evidence_text": text}])
    assert " ".join(item["excerpt"] for item in passages.values()) == text
    assert len({item["paragraph_id"] for item in passages.values()}) == 1


def test_scoped_narrative_renders_only_individually_supported_quotes():
    sources = {"S1": {**SOURCE, "evidence_text": "Vehicle demand is growing."},
               "S2": {**SOURCE, "id": "S2", "evidence_text": "Storage demand is strong and modular battery deployments will increase."}}
    requirement = {**REQUIREMENT, "kind": "narrative", "metric": "storage backlog outlook", "value_type": None}
    plan = {"narrative_subject": "storage", "requirements": [requirement]}
    answer = RequirementAnswer(requirement_id="r1", quotes=[EvidenceQuote(source_id=key, excerpt=value["evidence_text"])
                                                           for key, value in sources.items()])
    audit = {"assessments": [{"requirement_id": "r1", "reason": "Only quote 1 discusses storage; quote 0 concerns vehicles.",
        "quote_assessments": {"0": {"reason": "Vehicles are another business.", "answers_question": False}, "1": {"reason": "Storage is the requested business.", "answers_question": True}}, "supported": True}]}
    model = SimpleNamespace(with_structured_output=lambda *a, **kw: SimpleNamespace(invoke=lambda messages: audit))
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=model):
        result = render_evidence_answer(plan, [answer], [], {}, sources, {}, {})
    assert "Vehicle" not in result["answer"]
    assert "Storage demand" in result["answer"]
    assert result["cited_source_ids"] == ["S2"]


def test_scope_links_product_names_from_evidence_without_accepting_unrelated_backlog():
    texts = ["Backlog is larger due to vehicle deliveries in several markets.",
             "Storage systems will grow with the launch of CellStack 3 and Cell Block.",
             "CellStack 3 and Cell Block will support increased deployments next year."]
    source = {**SOURCE, "evidence_text": " ".join(texts)}
    requirement = {**REQUIREMENT, "kind": "narrative", "metric": "storage backlog and product outlook", "value_type": None}
    answer = RequirementAnswer(requirement_id="r1", quotes=[EvidenceQuote(source_id="S1", excerpt=text) for text in texts])
    audit = {"assessments": [{"requirement_id": "r1", "reason": "Outlook is supported.", "quote_assessments": {str(index): {"reason": "Storage evidence." if index else "Vehicle evidence.", "answers_question": bool(index)} for index in range(3)}, "supported": True}]}
    model = SimpleNamespace(with_structured_output=lambda *a, **kw: SimpleNamespace(invoke=lambda messages: audit))
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=model):
        result = render_evidence_answer({"narrative_subject": "storage systems", "requirements": [requirement]},
                                        [answer], [], {}, {"S1": source}, {}, {})
    assert "vehicle deliveries" not in result["answer"]
    assert "next year" in result["answer"]


def test_stale_section_heading_cannot_change_a_later_rows_quantity(tmp_path):
    import json
    body = ('## INCOME STATEMENTS\n(in millions)\n<table>'
        '<tr><th></th><th>2025</th></tr>'
        '<tr><td colspan="2">Operating expenses</td></tr>'
        '<tr><td>Income from operations</td><td>30</td></tr>'
        '<tr><td>Net income</td><td>20</td></tr></table>')
    source = {**SOURCE, "section": "Financial statements", "evidence_text": body}
    requirement = {**REQUIREMENT, "metric": "operating income"}
    result = FinancialFactPipeline(extractor=SimpleNamespace(extract=lambda **kwargs: ())).run(
        question="operating income", sources=[source], permitted_scope=Scope(("ACME",), ("2025",), "10K"),
        requirements=(requirement,))
    assert [(fact.row_label, fact.numeric_value) for fact in result.valid_facts] == [("Income from operations", 30)]
    (tmp_path / "primary.json").write_text(json.dumps({"metadata": source, "page_content": body}), encoding="utf-8")
    rows = StructuredDocumentLookup(tmp_path).statement_rows("operating income",
        Scope(("ACME",), ("2025",), "10K"), metric="operating income", include_ambiguous=True)
    assert any("Income from operations" in row.label for row in rows)
