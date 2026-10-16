"""Shared request contracts; no gold question IDs or real-company answers."""

import json
import pytest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.runnables import RunnableLambda

from src.agent.tools import RequestPlanInput
from src.agent.graph import build_request_plan
from src.financial.fact_extractor import FinancialFactExtractor
from src.financial.fact_pipeline import FinancialFactPipeline
from src.financial.extracted_fact_validator import ExtractedFactValidator
from src.financial.models import CandidateFinancialFact
from src.generation.evidence_answer import EvidenceQuote, RequirementAnswer, render_evidence_answer
from src.schemas import Scope
from src.generation.answer_guardrails import validate_answer_payload


SOURCE = dict(id="S1", ticker="ACME", fiscal_year="2025", doc_type="10K")
REQUIREMENT = dict(requirement_id="r1", ticker="ACME", source_year="2025", doc_type="10K",
                   metric="asset balance", reported_period="FY2025", kind="fact", qualifiers=[])


def test_duplicate_scalar_slots_are_merged_and_calculation_references_follow_them():
    plan = RequestPlanInput.model_validate({"requirements": [REQUIREMENT,
        {**REQUIREMENT, "requirement_id": "duplicate"},
        {**REQUIREMENT, "requirement_id": "earlier", "reported_period": "FY2024"}],
        "calculations": [{"operation": "percentage_change", "input_requirement_ids": ["earlier", "duplicate"]}]})
    assert [item.requirement_id for item in plan.requirements] == ["r1", "earlier"]
    assert plan.calculations[0].input_requirement_ids == ["earlier", "r1"]
    assert RequestPlanInput.model_validate(plan.model_dump()) == plan


def test_one_clause_does_not_become_a_qualifier_on_independent_topics():
    narrative = {**REQUIREMENT, "kind": "narrative", "metric": "datacenter dependencies", "reported_period": None}
    plan = RequestPlanInput(narrative_subject="AI spending", requirements=[narrative])
    assert plan.requirements[0].metric == "datacenter dependencies"
    assert plan.requirements[0].qualifiers == []


def test_audits_have_fixed_requirement_and_quote_keys_and_accept_a_negative_answer():
    source = {**SOURCE, "evidence_text": "Device X is not design-locked. Storage demand is strong."}
    requirements = [{**REQUIREMENT, "kind": "narrative", "metric": "Device X design status", "reported_period": None},
                    {**REQUIREMENT, "requirement_id": "r2", "kind": "narrative", "metric": "storage demand", "reported_period": None}]
    quotes = ["Device X is not design-locked.", "Storage demand is strong."]
    response = {"assessments": {item["requirement_id"]: {"reason": "This question can be answered from the quote.",
        "answers_question": True, "quote_assessments": {"0": {"reason": "Responsive evidence, even when the answer is no.",
        "answers_question": True}}} for item in requirements}}
    auditor = SimpleNamespace(invoke=Mock(return_value=response))
    model = SimpleNamespace(with_structured_output=Mock(return_value=auditor))
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=model):
        result = render_evidence_answer({"requirements": requirements}, [RequirementAnswer(requirement_id=item["requirement_id"],
            quotes=[EvidenceQuote(source_id="S1", excerpt=quote)]) for item, quote in zip(requirements, quotes)],
            [], {}, {"S1": source}, {}, {})
    assert result["complete"] and "not design-locked" in result["answer"]
    schema = model.with_structured_output.call_args.args[0]
    assert schema["properties"]["assessments"]["required"] == ["r1", "r2"]
    assert schema["properties"]["assessments"]["properties"]["r1"]["properties"]["quote_assessments"]["required"] == ["0"]


def test_extraction_slots_bind_ids_and_missing_values_are_null():
    source = {**SOURCE, "evidence_text": "Asset balance was $20 million in 2025."}
    candidate = CandidateFinancialFact(ticker="ACME", source_id="S1", metric="Asset balance", period="FY2025",
        raw_value="$20 million", unit="USD", evidence_excerpt=source["evidence_text"]).model_dump()
    candidate.pop("requirement_id")
    model = SimpleNamespace(with_structured_output=Mock(return_value=RunnableLambda(
        lambda messages: {"facts": {"r1": candidate, "missing": None}})))
    with patch("src.generation.llm_engine.get_llm_engine", return_value=model):
        facts = FinancialFactExtractor().extract(question=json.dumps({"requirements": [REQUIREMENT,
            {**REQUIREMENT, "requirement_id": "missing", "metric": "another balance"}]}), sources=[source])
    assert len(facts) == 1 and facts[0].requirement_id == "r1"
    schema = model.with_structured_output.call_args.args[0]
    assert schema["properties"]["facts"]["required"] == ["r1", "missing"]
    assert "requirement_id" not in schema["$defs"]["CandidateFinancialFact"]["properties"]


def test_table_group_heading_does_not_disqualify_actual_measure_columns():
    source = {**SOURCE, "evidence_text": 'Consolidated balance sheets (in millions except units) <table>'
        '<tr><th></th><th colspan="3">2025</th></tr><tr><th></th><th>Units</th><th>Cost Basis</th><th>Fair Value</th></tr>'
        '<tr><td colspan="4">Assets held</td></tr><tr><td>Instrument X</td><td>12</td><td>$20</td><td>$25</td></tr></table>'}
    measures = [("units", "12", "units"), ("cost basis", "$20", "USD"), ("fair value", "$25", "USD")]
    requirements = tuple({**REQUIREMENT, "requirement_id": f"r{index}", "metric": f"Instrument X {measure}"}
                         for index, (measure, _, _) in enumerate(measures, 1))
    candidates = tuple(CandidateFinancialFact(ticker="ACME", source_id="S1", requirement_id=item["requirement_id"],
        metric=item["metric"], period="FY2025", raw_value=raw, unit=unit, row_label="Instrument X",
        column_label=measure, evidence_excerpt=f"Instrument X 2025 {measure} {raw}")
        for item, (measure, raw, unit) in zip(requirements, measures))
    result = FinancialFactPipeline(extractor=SimpleNamespace(extract=lambda **kw: candidates)).run(
        question="Units, cost basis and fair value", sources=[source], permitted_scope=Scope(("ACME",), ("2025",), "10K"),
        requirements=requirements)
    assert {fact.requirement_id: fact.numeric_value for fact in result.valid_facts} == {"r1": 12, "r2": 20, "r3": 25}
    assert not result.rejected_facts


def test_exact_measurement_and_period_correct_a_swapped_routing_id():
    source = {**SOURCE, "evidence_text": "Total revenue was $20 million in 2025. Regional revenue was $12 million in 2025."}
    requirements = ({**REQUIREMENT, "metric": "total revenue"},
                    {**REQUIREMENT, "requirement_id": "regional", "metric": "regional revenue"})
    candidate = CandidateFinancialFact(ticker="ACME", source_id="S1", requirement_id="regional", metric="Total revenue",
        period="FY2025", raw_value="$20 million", unit="USD", evidence_excerpt="Total revenue was $20 million in 2025.")
    result = ExtractedFactValidator().validate([candidate], [source], requirements=requirements)
    assert [(fact.requirement_id, fact.numeric_value) for fact in result.valid_facts] == [("r1", 20)]
    unknown = candidate.model_copy(update={"requirement_id": "unplanned"})
    assert not ExtractedFactValidator().validate([unknown], [source], requirements=requirements).valid_facts


def test_planning_constraints_use_paired_scope_when_axis_fields_are_empty():
    model = SimpleNamespace(with_structured_output=Mock(return_value=SimpleNamespace(
        invoke=lambda messages: {"requirements": [REQUIREMENT]})))
    scope = Scope(requested_groups=(("ACME", "2025"),), doc_type="10K")
    with patch("src.agent.graph.get_llm_engine", return_value=model):
        plan = build_request_plan("Asset balance FY2025 from the annual filing", scope)
    assert plan.requirements[0].ticker == "ACME"
    properties = model.with_structured_output.call_args.args[0]["$defs"]["PlannedRequirement"]["properties"]
    assert properties["ticker"]["enum"] == ["ACME"]
    assert properties["source_year"]["enum"] == ["2025"]
    assert properties["kind"]["enum"] == ["fact", "narrative"]


def test_source_units_header_overrides_model_currency_guess():
    source = {**SOURCE, "evidence_text": 'Consolidated balance sheets (in millions except units) <table>'
        '<tr><th></th><th>2025</th></tr><tr><th></th><th>Units</th></tr>'
        '<tr><td>Instrument X</td><td>12</td></tr></table>'}
    requirement = {**REQUIREMENT, "metric": "Instrument X units", "value_type": "count"}
    candidate = CandidateFinancialFact(ticker="ACME", source_id="S1", requirement_id="r1",
        metric="Instrument X units", period="FY2025", raw_value="12", unit="USD", currency="USD",
        scale="millions", row_label="Instrument X", column_label="Units",
        evidence_excerpt="Instrument X 2025 Units 12")
    result = ExtractedFactValidator().validate([candidate], [source], requirements=(requirement,))
    assert not result.rejected_facts
    fact = result.valid_facts[0]
    assert fact.value_type == "count" and fact.base_value == 12
    assert fact.currency is None and fact.scale is None


def test_prose_uses_literal_measurement_label_and_its_own_value():
    source = {**SOURCE, "evidence_text": "Cash paid for P&E was $13 million in 2025. Other spending was $7 million."}
    requirement = {**REQUIREMENT, "metric": "cash paid for property and equipment"}
    candidate = CandidateFinancialFact(ticker="ACME", source_id="S1", requirement_id="r1",
        metric=requirement["metric"], row_label="Cash paid for P&E", period="FY2025",
        raw_value="$13 million", unit="USD", evidence_excerpt="Cash paid for P&E was $13 million in 2025.")
    result = ExtractedFactValidator().validate([candidate], [source], requirements=(requirement,))
    assert len(result.valid_facts) == 1
    wrong = candidate.model_copy(update={"raw_value": "$7 million"})
    rejected = ExtractedFactValidator().validate([wrong], [source], requirements=(requirement,))
    assert rejected.rejected_facts[0].reason == "raw_value_not_in_excerpt"


def test_table_quote_keeps_source_year_headers_for_audit_and_display():
    source = {**SOURCE, "evidence_text": '<table><tr><th colspan="3">Year ended December 31</th></tr>'
        '<tr><th></th><th>2025</th><th>2024</th></tr><tr><th></th><th>Current</th><th>Prior</th></tr>'
        '<tr><td>Revenue</td><td>20</td><td>18</td></tr></table>'}
    requirement = {**REQUIREMENT, "kind": "narrative", "metric": "reported revenue"}
    auditor = SimpleNamespace(invoke=Mock(return_value={"assessments": {"r1": {
        "reason": "Headers establish the requested period", "answers_question": True,
        "quote_assessments": {"0": {"reason": "Source row", "answers_question": True}}}}}))
    model = SimpleNamespace(with_structured_output=Mock(return_value=auditor))
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=model):
        result = render_evidence_answer({"requirements": [requirement]}, [RequirementAnswer(requirement_id="r1",
            quotes=[EvidenceQuote(source_id="S1", excerpt="Revenue 20 18")])], [], {}, {"S1": source}, {}, {})
    assert "2025 2024" in result["answer"]
    assert "Table columns in source order:" in auditor.invoke.call_args.args[0][1].content
    assert validate_answer_payload(result["answer"], [source]).valid


def test_planner_preserves_requested_asset_as_a_measurement_qualifier():
    requirement = {**REQUIREMENT, "metric": "cost basis", "subject_id": "b1"}
    model = SimpleNamespace(with_structured_output=Mock(return_value=SimpleNamespace(
        invoke=lambda messages: {"subjects": [{"subject_id": "b1", "name": "Instrument X", "entity_type": "asset_or_product"}], "requirements": [dict(requirement)]})))
    with patch("src.agent.graph.get_llm_engine", return_value=model):
        plan = build_request_plan("What is the cost basis of Instrument X in ACME's 2025 filing?",
                                  Scope(("ACME",), ("2025",), "10K"))
    assert plan.requirements[0].qualifiers == ["Instrument X"]
    assert plan.requirements[0].subject == "Instrument X"


def test_original_request_coverage_catches_a_detail_omitted_by_the_plan():
    source = {**SOURCE, "evidence_text": "Storage deployment increased."}
    requirement = {**REQUIREMENT, "kind": "narrative", "metric": "storage deployment"}
    response = {"assessments": {"r1": {"reason": "The planned observation is answered", "answers_question": True,
        "quote_assessments": {"0": {"reason": "Deployment is supported", "answers_question": True}}}},
        "coverage_reason": "The original request also asks for supply constraints; that detail is missing.",
        "request_complete": False}
    model = SimpleNamespace(with_structured_output=Mock(return_value=SimpleNamespace(invoke=lambda messages: response)))
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=model), pytest.raises(ValueError, match="Original request coverage"):
        render_evidence_answer({"question": "Report storage deployment and supply constraints", "requirements": [requirement]},
            [RequirementAnswer(requirement_id="r1", quotes=[EvidenceQuote(source_id="S1", excerpt=source["evidence_text"])])],
            [], {}, {"S1": source}, {}, {})


def test_comparison_clauses_reuse_one_subject_reference_across_documents():
    requirements = [{**REQUIREMENT, "subject_id": "b1", "kind": "narrative", "metric": "supply",
                     "evidence_question": "What was the supply constraint?", "retrieval_query": "supply constraints"},
                    {**REQUIREMENT, "requirement_id": "r2", "source_year": "2024", "subject_id": "b1",
                     "kind": "narrative", "metric": "new-product outlook", "evidence_question": "What was the product outlook?"}]
    model = SimpleNamespace(with_structured_output=Mock(return_value=SimpleNamespace(invoke=lambda messages: {
        "subjects": [{"subject_id": "b1", "name": "Widget division", "entity_type": "business_area"}], "requirements": [dict(item) for item in requirements]})))
    with patch("src.agent.graph.get_llm_engine", return_value=model):
        plan = build_request_plan("Compare the Widget division supply and product outlook across ACME filings",
                                  Scope(("ACME",), ("2024", "2025"), "10K"))
    assert all(item.subject == "Widget division" and "Widget division" in item.evidence_question
               for item in plan.requirements)
    assert plan.narrative_subject == "Widget division"


def test_prose_measurement_is_not_bound_to_an_unrelated_table():
    source = {**SOURCE, "evidence_text": 'Cloud revenue reached $20 billion in 2025.\n\n'
        '<table><tr><th></th><th>2025</th></tr><tr><td>Other revenue</td><td>12</td></tr></table>'}
    requirement = {**REQUIREMENT, "metric": "Cloud revenue"}
    candidate = CandidateFinancialFact(ticker="ACME", source_id="S1", requirement_id="r1", metric="Cloud revenue",
        row_label="Cloud revenue", column_label="2025", period="FY2025", raw_value="$20 billion", unit="USD",
        evidence_excerpt="Cloud revenue reached $20 billion in 2025.")
    result = ExtractedFactValidator().validate([candidate], [source], requirements=(requirement,))
    assert len(result.valid_facts) == 1 and not result.rejected_facts


def test_prose_qualifiers_use_the_own_paragraph_antecedent():
    paragraph = "Commercial backlog was $20 billion. Roughly 40% will be recognized in revenue in the next year."
    source = {**SOURCE, "evidence_text": paragraph}
    requirement = {**REQUIREMENT, "metric": "portion recognized next year", "qualifiers": ["Commercial backlog"]}
    candidate = CandidateFinancialFact(ticker="ACME", source_id="S1", requirement_id="r1", metric=requirement["metric"],
        row_label="recognized in revenue", period="FY2025", raw_value="40%", unit="percent",
        evidence_excerpt="Roughly 40% will be recognized in revenue in the next year.")
    result = ExtractedFactValidator().validate([candidate], [source], requirements=(requirement,))
    from src.financial.coverage import fact_matches_requirement
    assert fact_matches_requirement(result.valid_facts[0], requirement, source)


def test_unrequested_missing_level_does_not_block_a_complete_reported_change():
    source = {**SOURCE, "evidence_text": "Expenses increased by $3 million because hiring increased."}
    narrative = {**REQUIREMENT, "kind": "narrative", "metric": "reported expense change"}
    extra = {**REQUIREMENT, "requirement_id": "extra", "metric": "expense level"}
    response = {"assessments": {"r1": {"reason": "The reported increase and cause are quoted", "answers_question": True,
        "quote_assessments": {"0": {"reason": "Responsive", "answers_question": True}}}},
        "request_complete": True, "coverage_reason": "The change and cause answer the original question.",
        "unresolved_requirements": {"extra": {"reason": "The user did not ask for a separate expense level.", "required_by_user": False}}}
    model = SimpleNamespace(with_structured_output=Mock(return_value=SimpleNamespace(invoke=lambda messages: response)))
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=model):
        result = render_evidence_answer({"question": "By how much did expenses increase, and why?", "requirements": [narrative, extra]},
            [RequirementAnswer(requirement_id="r1", quotes=[EvidenceQuote(source_id="S1", excerpt=source["evidence_text"])]),
             RequirementAnswer(requirement_id="extra", unavailable_reason="Not extracted")], [], {}, {"S1": source}, {}, {})
    assert result["complete"] and "unavailable" not in result["answer"]


def test_per_share_exception_does_not_remove_the_share_count_scale():
    source = {**SOURCE, "evidence_text": 'Consolidated balance sheets (in millions except per-share amounts) <table>'
        '<tr><th></th><th>2025</th></tr><tr><th></th><th>Shares</th></tr>'
        '<tr><td>Instrument X</td><td>12</td></tr></table>'}
    requirement = {**REQUIREMENT, "metric": "Instrument X shares"}
    candidate = CandidateFinancialFact(ticker="ACME", source_id="S1", requirement_id="r1", metric=requirement["metric"],
        period="FY2025", raw_value="12", unit="shares", row_label="Instrument X", column_label="Shares",
        evidence_excerpt="Instrument X 2025 Shares 12")
    result = ExtractedFactValidator().validate([candidate], [source], requirements=(requirement,))
    assert result.valid_facts[0].base_value == 12000000 and result.valid_facts[0].scale == "millions"


def test_explicit_component_inherits_the_table_measure_caption():
    source = {**SOURCE, "evidence_text": 'Revenue (in millions) <table><tr><th></th><th>2025</th></tr>'
        '<tr><td>Widget division</td><td>$12</td></tr><tr><td>Total revenue</td><td>$20</td></tr></table>'}
    requirement = {**REQUIREMENT, "metric": "revenue", "qualifiers": ["Widget division"]}
    candidate = CandidateFinancialFact(ticker="ACME", source_id="S1", requirement_id="r1", metric="Widget division revenue",
        period="FY2025", raw_value="$12", unit="USD", scale="millions", row_label="Widget division", column_label="2025",
        evidence_excerpt="Widget division 2025 $12")
    result = ExtractedFactValidator().validate([candidate], [source], requirements=(requirement,))
    assert len(result.valid_facts) == 1 and not result.rejected_facts


def test_quote_responsiveness_cannot_override_a_wrong_source_subject():
    source = {**SOURCE, "evidence_text": "The vehicle division expects higher margins next year."}
    requirement = {**REQUIREMENT, "kind": "narrative", "metric": "margin outlook", "subject": "Widget division"}
    response = {"assessments": {"r1": {"reason": "An outlook was selected", "answers_question": True,
        "quote_assessments": {"0": {"source_subject": "vehicle division", "source_observation": source["evidence_text"],
            "reason": "The source concerns another division", "scope_matches": False, "answers_question": True}}}}}
    model = SimpleNamespace(with_structured_output=Mock(return_value=SimpleNamespace(invoke=lambda messages: response)))
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=model), pytest.raises(ValueError, match="responsive stored quotes"):
        render_evidence_answer({"question": "What is the Widget division margin outlook?", "requirements": [requirement]},
            [RequirementAnswer(requirement_id="r1", quotes=[EvidenceQuote(source_id="S1", excerpt=source["evidence_text"])])],
            [], {}, {"S1": source}, {}, {})


def test_one_stored_passage_can_cover_multiple_narrative_requirements():
    from src.generation.evidence_answer import resolve_evidence_answers
    source = {**SOURCE, "evidence_text": "Widget backlog is strong and deployments will grow with product Alpha."}
    requirements = [{**REQUIREMENT, "kind": "narrative", "metric": "backlog"},
                    {**REQUIREMENT, "requirement_id": "r2", "kind": "narrative", "metric": "new-product outlook"}]
    response = {"answers": {key: {"passage_ids": ["P1"]} for key in ("r1", "r2")}}
    model = SimpleNamespace(with_structured_output=Mock(return_value=SimpleNamespace(invoke=lambda messages: response)))
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=model):
        answers = resolve_evidence_answers({"requirements": requirements}, [], {}, {"S1": source}, {})
    assert len(answers) == 2 and answers[0].quotes == answers[1].quotes


def test_audit_stores_shared_quotes_once_with_separate_requirement_verdicts():
    import json
    source = {**SOURCE, "evidence_text": "Widget backlog is strong and deployments will grow next year."}
    requirements = [{**REQUIREMENT, "kind": "narrative", "metric": "backlog"},
                    {**REQUIREMENT, "requirement_id": "r2", "kind": "narrative", "metric": "deployment outlook"}]
    response = {"assessments": {key: {"reason": "This observation is stated", "answers_question": True,
        "quote_assessments": {"0": {"reason": "Responsive", "scope_matches": True, "answers_question": True}}}
        for key in ("r1", "r2")}}
    auditor = SimpleNamespace(invoke=Mock(return_value=response))
    model = SimpleNamespace(with_structured_output=Mock(return_value=auditor))
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=model):
        result = render_evidence_answer({"requirements": requirements}, [RequirementAnswer(requirement_id=key,
            quotes=[EvidenceQuote(source_id="S1", excerpt=source["evidence_text"])]) for key in ("r1", "r2")],
            [], {}, {"S1": source}, {}, {})
    payload = json.loads(auditor.invoke.call_args.args[0][1].content)
    assert len(payload["stored_quotes"]) == 1
    assert payload["claims"][0]["quotes"] == payload["claims"][1]["quotes"]
    assert len(result["request_coverage"]["assessments"]) == 2


@pytest.mark.parametrize("missing_details", [[], ["requested deployment rate"]])
def test_narrative_coverage_uses_retained_quotes_and_actual_missing_details(missing_details):
    source = {**SOURCE, "evidence_text": "Widget deployments will grow next year. The vehicle division had record revenue."}
    requirement = {**REQUIREMENT, "kind": "narrative", "metric": "deployment outlook", "subject": "Widget division"}
    response = {"assessments": {"r1": {"reason": "The first quote states the requested outlook",
        "missing_details": missing_details, "quote_assessments": {
            "0": {"reason": "Responsive outlook", "scope_matches": True, "answers_question": True},
            "1": {"reason": "Another business and observation", "scope_matches": False, "answers_question": False}}}}}
    model = SimpleNamespace(with_structured_output=Mock(return_value=SimpleNamespace(invoke=lambda messages: response)))
    answers = [RequirementAnswer(requirement_id="r1", quotes=[EvidenceQuote(source_id="S1", excerpt=excerpt)
        for excerpt in ("Widget deployments will grow next year.", "The vehicle division had record revenue.")])]
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=model):
        if missing_details:
            with pytest.raises(ValueError, match="Unsupported narrative"):
                render_evidence_answer({"requirements": [requirement]}, answers, [], {}, {"S1": source}, {}, {})
        else:
            result = render_evidence_answer({"requirements": [requirement]}, answers, [], {}, {"S1": source}, {}, {})
            assert result["complete"] and "vehicle division" not in result["answer"]


@pytest.mark.parametrize("excerpt,valid", [("Widget backlog remains strong.", True), ("Widget backlog improved.", False)])
def test_narrative_selection_binds_exact_stored_text(excerpt, valid):
    from src.generation.evidence_answer import resolve_evidence_answers
    source = {**SOURCE, "evidence_text": "Widget backlog remains strong."}
    requirement = {**REQUIREMENT, "kind": "narrative", "metric": "backlog"}
    response = {"answers": {"r1": {"quotes": [{"source_id": "S1", "excerpt": excerpt}]}}}
    model = SimpleNamespace(with_structured_output=Mock(return_value=SimpleNamespace(invoke=lambda messages: response)))
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=model):
        if valid:
            answers = resolve_evidence_answers({"requirements": [requirement]}, [], {}, {"S1": source}, {})
            assert answers[0].quotes[0].excerpt == source["evidence_text"]
        else:
            answers = resolve_evidence_answers({"requirements": [requirement]}, [], {}, {"S1": source}, {})
            assert not answers[0].quotes and answers[0].unavailable_reason


def test_quote_audit_cannot_apply_a_neighboring_observation_to_a_stored_quote():
    source = {**SOURCE, "evidence_text": "The vehicle division had record revenue. Widget deployments will grow."}
    requirement = {**REQUIREMENT, "kind": "narrative", "metric": "deployment outlook"}
    response = {"assessments": {"r1": {"reason": "An outlook exists nearby", "missing_details": [],
        "quote_assessments": {"0": {"source_observation": "Widget deployments will grow.",
            "reason": "Wrong quote was audited", "scope_matches": True, "answers_question": True}}}}}
    model = SimpleNamespace(with_structured_output=Mock(return_value=SimpleNamespace(invoke=lambda messages: response)))
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=model), pytest.raises(ValueError, match="observation does not match"):
        render_evidence_answer({"requirements": [requirement]}, [RequirementAnswer(requirement_id="r1",
            quotes=[EvidenceQuote(source_id="S1", excerpt="The vehicle division had record revenue.")])],
            [], {}, {"S1": source}, {}, {})


def test_quote_rebinding_requires_unique_exact_evidence_in_the_requested_source_group():
    from src.generation.evidence_answer import resolve_evidence_answers
    sources = {"S1": {**SOURCE, "evidence_text": "Widget backlog remains strong."},
               "S2": {**SOURCE, "id": "S2", "evidence_text": "Vehicle demand declined."}}
    requirement = {**REQUIREMENT, "kind": "narrative", "metric": "backlog"}
    response = {"answers": {"r1": {"quotes": [{"source_id": "S2", "excerpt": sources["S1"]["evidence_text"]}]}}}
    model = SimpleNamespace(with_structured_output=Mock(return_value=SimpleNamespace(invoke=lambda messages: response)))
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=model):
        answer = resolve_evidence_answers({"requirements": [requirement]}, [], {}, sources, {})[0]
    assert answer.quotes[0].source_id == "S1"
    sources["S1"]["fiscal_year"] = "2024"
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=model):
        answer = resolve_evidence_answers({"requirements": [requirement]}, [], {}, sources, {})[0]
    assert not answer.quotes and answer.unavailable_reason


@pytest.mark.parametrize("subject", [None, "Widget division"])
def test_unknown_business_attribution_only_blocks_a_business_specific_claim(subject):
    source = {**SOURCE, "evidence_text": "Deployment will grow next year."}
    requirement = {**REQUIREMENT, "kind": "narrative", "metric": "deployment outlook", "subject": subject}
    response = {"assessments": {"r1": {"reason": "The company outlook is reported", "missing_details": [],
        "quote_assessments": {"0": {"source_subject": None, "reason": "No division is named",
            "scope_matches": True, "answers_question": True}}}}}
    model = SimpleNamespace(with_structured_output=Mock(return_value=SimpleNamespace(invoke=lambda messages: response)))
    answers = [RequirementAnswer(requirement_id="r1", quotes=[EvidenceQuote(source_id="S1", excerpt=source["evidence_text"])])]
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=model):
        if subject:
            with pytest.raises(ValueError, match="responsive stored quotes"):
                render_evidence_answer({"requirements": [requirement]}, answers, [], {}, {"S1": source}, {}, {})
        else:
            assert render_evidence_answer({"requirements": [requirement]}, answers, [], {}, {"S1": source}, {}, {})["complete"]


def test_an_observation_cannot_be_the_owner_of_a_business_specific_quote():
    source = {**SOURCE, "evidence_text": "Backlog increased in the vehicle division."}
    requirement = {**REQUIREMENT, "kind": "narrative", "metric": "backlog", "subject": "Widget division"}
    response = {"assessments": {"r1": {"reason": "A backlog is reported", "missing_details": [],
        "quote_assessments": {"0": {"source_subject": "Backlog", "source_subject_type": "metric_or_action",
            "reason": "Backlog is an observation, not a business", "scope_matches": True, "answers_question": True}}}}}
    model = SimpleNamespace(with_structured_output=Mock(return_value=SimpleNamespace(invoke=lambda messages: response)))
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=model), pytest.raises(ValueError, match="responsive stored quotes"):
        render_evidence_answer({"requirements": [requirement]}, [RequirementAnswer(requirement_id="r1",
            quotes=[EvidenceQuote(source_id="S1", excerpt=source["evidence_text"])])], [], {}, {"S1": source}, {}, {})


def test_source_owner_schema_uses_evidence_vocabulary_and_excludes_requested_observations():
    source = {**SOURCE, "evidence_text": "Widget backlog remains strong."}
    requirement = {**REQUIREMENT, "kind": "narrative", "metric": "backlog", "subject": "Widget division"}
    response = {"assessments": {"r1": {"reason": "The Widget business reports backlog", "missing_details": [],
        "quote_assessments": {"0": {"source_subject": "Widget", "source_subject_type": "business_area",
            "source_observation": source["evidence_text"], "reason": "The requested business is named",
            "scope_matches": True, "answers_question": True}}}}}
    model = SimpleNamespace(with_structured_output=Mock(return_value=SimpleNamespace(invoke=lambda messages: response)))
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=model):
        result = render_evidence_answer({"requirements": [requirement]}, [RequirementAnswer(requirement_id="r1",
            quotes=[EvidenceQuote(source_id="S1", excerpt=source["evidence_text"])])], [], {}, {"S1": source}, {}, {})
    schema = model.with_structured_output.call_args.args[0]
    quote = schema["properties"]["assessments"]["properties"]["r1"]["properties"]["quote_assessments"]["properties"]["0"]
    anchors = quote["properties"]["source_subject"]["anyOf"][0]["enum"]
    assert "Widget" in anchors and "backlog" not in anchors
    assert result["complete"]
