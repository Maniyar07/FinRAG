"""Regression checks for output intent, evidence identity, and shared budgets."""
import re
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from src.agent.graph import build_request_plan, create_finrag_agent
from src.agent.tools import PlannedRequirement, RequestPlanInput
from src.financial.coverage import fact_matches_requirement
from src.financial.metric_terms import is_metric_match
from src.financial.fact_pipeline import FinancialFactPipeline
from src.generation.evidence_answer import EvidenceQuote, RequirementAnswer, evidence_passages, render_evidence_answer, resolve_evidence_answers
from src.retrieval.context_builder import ContextBuilder
from src.retrieval.structured_lookup import StructuredDocumentLookup
from src.schemas import RankedChild, Scope


SCOPE = Scope(("ACME",), ("2024",), "10K")
SOURCE = dict(id="S1", ticker="ACME", fiscal_year="2024", doc_type="10K")


def requirement(key="r1", **values):
    return dict(requirement_id=key, ticker="ACME", source_year="2024", doc_type="10K",
                kind="fact", metric="revenue", reported_period="FY2024", **values)


def test_requested_calculation_cannot_disappear_from_validated_plan():
    output = dict(object_id="o1", quote="calculate growth", subject="growth", kind="calculation")
    with pytest.raises(ValueError, match="needs a formula"):
        RequestPlanInput.model_validate(dict(requested_objects=[output], requirements=[requirement(object_id="o1")]))


def test_derived_output_preserves_operands_instead_of_searching_for_the_ratio():
    formula = dict(operation="percentage_of", input_requirement_ids=["r1", "r2"])
    draft = dict(requested_objects=[dict(object_id="o1", quote="Calculate net margin",
        subject="net margin", kind="calculation", calculation=formula)], requirements=[
        {**requirement(object_id="o1"), "metric": "net income"}, requirement("r2", object_id="o1")])
    planner = SimpleNamespace(invoke=Mock(return_value=draft))
    with patch("src.agent.graph.get_llm_engine", return_value=SimpleNamespace(
            with_structured_output=lambda *a, **k: planner)):
        result = build_request_plan("Calculate net margin for ACME in FY2024 from its 10-K.", SCOPE)
    assert result.calculations[0].operation == "percentage_of"
    assert result.calculations[0].object_id == "o1"
    assert [item.metric for item in result.requirements] == ["net income", "revenue"]
    assert planner.invoke.call_count == 1


def test_table_intent_is_available_without_a_display_keyword():
    draft = dict(requested_objects=[dict(object_id="o1", quote="Reproduce the cash flow statement",
        subject="cash flow statement", kind="statement")], requirements=[
        {**requirement(object_id="o1"), "kind": "statement", "metric": "cash flow statement", "reported_period": None}])
    engine = Mock()
    engine.with_structured_output.return_value.invoke.return_value = draft
    with patch("src.agent.graph.get_llm_engine", return_value=engine):
        result = build_request_plan("Reproduce the cash flow statement for ACME's FY2024 10-K.", SCOPE)
    assert result.requirements[0].kind == "statement"
    assert "source_table" in engine.with_structured_output.call_args.args[0]["$defs"]["RequestedObject"]["properties"]["kind"]["enum"]


def test_table_breakdown_preserves_requested_dimension():
    draft = dict(requested_objects=[dict(object_id="o1", quote="revenue by geography",
        subject="revenue by geography", kind="statement")], requirements=[
        {**requirement(object_id="o1"), "kind": "statement", "metric": "revenue"}])
    with patch("src.agent.graph.get_llm_engine", return_value=SimpleNamespace(
            with_structured_output=lambda *a, **k: SimpleNamespace(invoke=lambda _: draft))):
        result = build_request_plan("Show revenue by geography in ACME's FY2024 10-K.", SCOPE)
    assert result.requirements[0].metric == "revenue by geography"


def test_flat_table_plan_preserves_dimension_from_its_content_query():
    draft = dict(synthesis=False, requirements=[{**requirement(), "kind": "statement",
        "retrieval_query": "ACME FY2024 revenue distribution by geography"}])
    engine = Mock()
    engine.with_structured_output.return_value.invoke.return_value = draft
    with patch("src.agent.graph.get_llm_engine", return_value=engine):
        result = build_request_plan("Show ACME revenue by geography in its FY2024 10-K.", SCOPE)
    assert "geography" in result.requirements[0].metric
    assert result.synthesis is None
    schema = engine.with_structured_output.call_args.args[0]
    assert schema["properties"]["synthesis"]["type"] == "boolean"
    assert "requested_objects" in schema["properties"]
    assert "kind" not in schema["$defs"]["PlannedRequirement"]["properties"]


def test_synthesis_carries_the_original_request_instead_of_an_unresearched_answer():
    question = "Explain ACME revenue trends using its FY2024 10-K."
    draft = dict(synthesis="Invented conclusions before retrieval.", requirements=[requirement()])
    engine = Mock()
    engine.with_structured_output.return_value.invoke.return_value = draft
    with patch("src.agent.graph.get_llm_engine", return_value=engine):
        result = build_request_plan(question, SCOPE)
    assert result.synthesis == question


def test_planning_preserves_requested_basis_and_drops_invented_instructions():
    draft = dict(synthesis=False, requirements=[{**requirement(), "qualifiers": [
        "constant currency", "company-wide; report exactly as stated in the filing"]}])
    engine = Mock()
    engine.with_structured_output.return_value.invoke.return_value = draft
    with patch("src.agent.graph.get_llm_engine", return_value=engine):
        result = build_request_plan("Report ACME's FY2024 revenue in constant currency from its 2024 10-K.", SCOPE)
    assert result.requirements[0].qualifiers == ["constant currency"]


def test_missing_evidence_has_one_recovery_and_finishes_without_a_routing_model():
    import json
    from langchain_core.messages import ToolMessage
    from src.financial.calculator import FinancialCalculator
    from src.financial.models import FactValidationResult
    from src.schemas import RetrievalBundle
    retriever = SimpleNamespace(retrieve=Mock(return_value=RetrievalBundle("", [], SCOPE, 0)))
    plan = RequestPlanInput.model_validate(dict(requirements=[requirement()]))
    pipeline = SimpleNamespace(run=lambda **kwargs: FactValidationResult(valid_facts=()))
    with patch("src.agent.graph.get_llm_engine", side_effect=AssertionError("No routing-model call is needed")), \
            patch("src.agent.tools.FinancialFactPipeline", return_value=pipeline):
        graph = create_finrag_agent(retriever, FinancialCalculator(), permitted_scope=SCOPE,
            plan_builder=lambda *args: plan)
    state = graph.invoke({"messages": [("user", "Report ACME's FY2024 revenue from its 2024 10-K.")]})
    finals = [json.loads(message.content) for message in state["messages"]
              if isinstance(message, ToolMessage) and message.name == "finalize_financial_answer"]
    assert retriever.retrieve.call_count == 2
    assert len(finals) == 2
    assert finals[-1]["answer_validated"] and not finals[-1]["complete"]
    assert "unavailable" in finals[-1]["answer"]


def test_exhausted_provider_quota_does_not_trigger_retrieval_recovery():
    import httpx
    from openai import RateLimitError
    from src.financial.calculator import FinancialCalculator
    error = RateLimitError("quota exhausted", response=httpx.Response(429,
        request=httpx.Request("POST", "https://api.openai.com/v1/embeddings")),
        body={"code": "insufficient_quota"})
    retriever = SimpleNamespace(retrieve=Mock(side_effect=error))
    plan = RequestPlanInput.model_validate(dict(requirements=[requirement()]))
    graph = create_finrag_agent(retriever, FinancialCalculator(), permitted_scope=SCOPE,
        plan_builder=lambda *args: plan)
    graph.invoke({"messages": [("user", "Report ACME's FY2024 revenue from its 2024 10-K.")]})
    assert retriever.retrieve.call_count == 1


def test_a_narrative_evidence_gap_gets_one_focused_recovery_and_then_returns_partial():
    import json
    from langchain_core.messages import ToolMessage
    from src.financial.calculator import FinancialCalculator
    from src.financial.models import FactValidationResult
    from src.schemas import RetrievalBundle
    source = {**SOURCE, "parent_id": "p1", "evidence_text": "The filing describes factory operations."}
    retriever = SimpleNamespace(retrieve=Mock(return_value=RetrievalBundle(source["evidence_text"], [source], SCOPE, 10)))
    plan = RequestPlanInput.model_validate(dict(requirements=[{**requirement(), "kind": "narrative"}]))
    partial = dict(answer_validated=True, complete=False, missing_requirement_ids=["r1"],
                   request_coverage={"reason": "Requested operating drivers remain absent."}, answer="Supported parts; evidence gap.")
    with patch("src.agent.graph.get_llm_engine", side_effect=AssertionError("No routing model needed")), \
         patch("src.agent.tools.resolve_evidence_answers", return_value=[]), \
         patch("src.agent.tools.render_evidence_answer", return_value=partial), \
         patch("src.agent.tools.FinancialFactPipeline", return_value=SimpleNamespace(run=lambda **kwargs: FactValidationResult(valid_facts=()))):
        graph = create_finrag_agent(retriever, FinancialCalculator(), permitted_scope=SCOPE, plan_builder=lambda *args: plan)
        state = graph.invoke({"messages": [("user", "Describe ACME's operating costs in its 2024 10-K.")]})
    finals = [json.loads(message.content) for message in state["messages"]
              if isinstance(message, ToolMessage) and message.name == "finalize_financial_answer"]
    assert retriever.retrieve.call_count == 2 and len(finals) == 2
    assert finals[0]["recovery_required"] and finals[0]["missing_requirement_ids"] == ["r1"]
    assert finals[-1]["answer_validated"] and not finals[-1]["complete"]


def test_gross_profit_synonym_keeps_the_monetary_quantity_contract():
    req = PlannedRequirement.model_validate({**requirement(), "metric": "gross profit"})
    assert req.value_type == "currency"
    assert is_metric_match("Gross margin", req.metric)
    assert not is_metric_match("Operating margin", req.metric)


def test_percentage_row_does_not_inherit_a_table_wide_currency_scale():
    source = {**SOURCE, "evidence_text": 'Income taxes (dollars in millions)\n<table>'
        '<tr><th>Metric</th><th>2024</th><th>2023</th></tr>'
        '<tr><td>Tax provision</td><td>$18</td><td>$20</td></tr>'
        '<tr><td>Effective tax rate</td><td>20%</td><td>(50)%</td></tr></table>'}
    req = {**requirement(), "metric": "effective tax rate", "value_type": "percent"}
    result = FinancialFactPipeline().recover_exact_table_row(metric_hint=req["metric"], period_year="2024",
        sources=[source], permitted_scope=SCOPE, requirements=(req,))
    assert len(result.valid_facts) == 1
    value = result.valid_facts[0]
    assert value.numeric_value == 20 and value.normalized_unit == "percent"
    assert value.currency is None and value.scale is None


def test_company_wide_question_does_not_inherit_an_invented_industry():
    draft = dict(subjects=[dict(subject_id="b1", name="software", entity_type="business_area")],
        requirements=[{**requirement(), "kind": "narrative", "metric": "capital requirements",
                       "subject_id": "b1", "reported_period": None}])
    with patch("src.agent.graph.get_llm_engine", return_value=SimpleNamespace(
            with_structured_output=lambda *a, **k: SimpleNamespace(invoke=lambda _: draft))):
        result = build_request_plan("Describe ACME's capital requirements in its FY2024 10-K.", SCOPE)
    assert result.requirements[0].subject is None


def test_calculation_output_label_is_not_an_operand_evidence_qualifier():
    draft = dict(subjects=[dict(subject_id="b1", name="net margin", entity_type="business_area")],
        requested_objects=[dict(object_id="o1", quote="Calculate net margin", subject="net margin", kind="calculation",
            calculation=dict(operation="percentage_of", input_requirement_ids=["r1", "r2"]))],
        requirements=[{**requirement(object_id="o1"), "subject_id": "b1", "metric": "net income"},
                      {**requirement("r2", object_id="o1"), "subject_id": "b1"}])
    with patch("src.agent.graph.get_llm_engine", return_value=SimpleNamespace(
            with_structured_output=lambda *a, **k: SimpleNamespace(invoke=lambda _: draft))):
        result = build_request_plan("Calculate net margin for ACME's FY2024 10-K.", SCOPE)
    assert all(not item.qualifiers and item.subject is None for item in result.requirements)


def test_nonbusiness_subject_reference_does_not_block_a_company_wide_request():
    draft = dict(subjects=[], requirements=[{**requirement(), "subject_id": "b1"}])
    with patch("src.agent.graph.get_llm_engine", return_value=SimpleNamespace(
            with_structured_output=lambda *a, **k: SimpleNamespace(invoke=lambda _: draft))):
        result = build_request_plan("Report ACME revenue in its 2024 10-K.", SCOPE)
    assert result.requirements[0].subject is None


def test_short_reconciliation_table_remains_one_selectable_piece_of_evidence():
    table = '<table><tr><th>Item</th><th>2024</th></tr><tr><td>Statutory tax</td><td>21%</td></tr><tr><td>Credits</td><td>(1)%</td></tr></table>'
    passages = evidence_passages([{**SOURCE, "evidence_text": table}])
    assert any('Statutory tax 21%' in item['excerpt'] and 'Credits (1)%' in item['excerpt'] for item in passages.values())


def test_growth_observation_retains_its_business_and_capacity_caveat():
    paragraph = ("Widget cloud revenue grew 39%, above expectations. "
                 "AI revenue was in line with expectations. "
                 "Despite new datacenter capacity, demand remains higher than supply.")
    passages = evidence_passages([{**SOURCE, "evidence_text": paragraph}])
    assert any(item["excerpt"] == paragraph for item in passages.values())


def test_table_scale_uses_its_own_caption_not_a_preceding_footnote(tmp_path):
    import json
    body = ('<table><tr><th>Other</th></tr><tr><td>1</td></tr></table>\n\n'
        'Depreciation was $2 billion.\n\nRevenue by geographic area (in millions):\n\n'
        '<table><tr><th>Area</th><th>2024</th></tr><tr><td>Domestic</td><td>100</td></tr></table>')
    (tmp_path / 'report.json').write_text(json.dumps({'page_content': body, 'metadata': SOURCE}), encoding='utf-8')
    result = StructuredDocumentLookup(tmp_path).answer('revenue by geographic area', SCOPE, wants_complete_table=True)
    assert result and 'in millions' in result.answer
    assert 'Depreciation' not in result.answer


def test_selected_passage_and_point_share_the_same_stored_identity():
    source = {**SOURCE, "evidence_text": "Factories require substantial investment.\n\nRevenue comes from product sales."}
    selection = {"answers": {"r1": {"passage_ids": ["P2"], "draft_points": [
        {"heading": "Revenue", "text": "Revenue comes from product sales.", "quote_indices": [0]}]}}}
    engine = Mock()
    engine.with_structured_output.return_value.invoke.return_value = selection
    req = {**requirement(), "kind": "narrative"}
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=engine):
        answers = resolve_evidence_answers({"requirements": [req]}, [], {}, {"S1": source})
    assert answers[0].quotes[0].excerpt == "Revenue comes from product sales."
    assert answers[0].draft_points[0].quote_indices == [0]
    branch = engine.with_structured_output.call_args.args[0]["properties"]["answers"]["properties"]["r1"]
    assert "quotes" not in branch["properties"]


def test_point_citations_resolve_passage_ids_without_guessing_selection_order():
    source = {**SOURCE, "evidence_text": "Factories require substantial investment.\n\nRevenue comes from product sales."}
    selection = {"answers": {"r1": {"passage_ids": ["P2", "P1"], "draft_points": [
        {"heading": "Capital", "text": "Factories require substantial investment.", "passage_ids": ["P1"]}]}}}
    engine = Mock()
    engine.with_structured_output.return_value.invoke.return_value = selection
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=engine):
        answers = resolve_evidence_answers({"requirements": [{**requirement(), "kind": "narrative"}]}, [], {}, {"S1": source})
    point = answers[0].draft_points[0]
    assert point.quote_indices == [1]
    assert answers[0].quotes[point.quote_indices[0]].excerpt == "Factories require substantial investment."


def test_batched_selection_keeps_each_companys_sentence_scope_and_independent_verdict():
    source = {**SOURCE, "evidence_text": "Factories require substantial investment."}
    other = {**SOURCE, "id": "S2", "ticker": "BETA", "evidence_text": "Revenue comes from product sales."}
    requirements = [{**requirement(), "kind": "narrative", "metric": "capital requirements"},
                    {**requirement("r2"), "kind": "narrative", "metric": "cost structure", "ticker": "BETA"}]
    selection = {"answers": {key: {"draft_points": [{"heading": heading, "text": text, "passage_ids": [passage]}]}
        for key, heading, text, passage in (("r1", "Capital", source["evidence_text"], "P1"),
                                           ("r2", "Costs", other["evidence_text"], "P2"))}}
    audit = {"assessments": {key: {"reason": "Investment is responsive; revenue does not establish costs.",
        "supported": supported, "point_support": {"0": {"evidence_spans": [source["evidence_text"]] if supported else [],
            "unsupported_details": [] if supported else ["Revenue does not establish costs."]}}}
        for key, supported in (("r1", True), ("r2", False))}}
    engine = Mock()
    engine.with_structured_output.return_value.invoke.side_effect = [selection, audit]
    plan, sources = {"requirements": requirements}, {"S1": source, "S2": other}
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=engine):
        answers = resolve_evidence_answers(plan, [], {}, sources)
        result = render_evidence_answer(plan, answers, [], {}, sources, {}, {})
    assert engine.with_structured_output.call_count == 2
    schema = engine.with_structured_output.call_args_list[0].args[0]
    branches = schema["properties"]["answers"]["properties"]
    for key, passage in (("r1", "P1"), ("r2", "P2")):
        points = branches[key]["properties"]["draft_points"]["items"]
        pattern = schema['$defs'][points['properties']['sentence_id']['$ref'].rsplit('/', 1)[1]]['pattern']
        assert all(bool(re.fullmatch(pattern, candidate)) == (candidate == f'{passage}:0') for candidate in ['P1:0', 'P2:0'])
    assert source["evidence_text"] in result["answer"]
    assert other["evidence_text"] not in result["answer"]
    assert result["missing_requirement_ids"] == ["r2"]


def test_search_terms_rank_passages_without_expanding_the_answer_contract():
    import json
    source = {**SOURCE, "evidence_text": "Capital requirements are subject to regulation.\n\nFactories require robotics and equipment investment."}
    req = {**requirement(), "kind": "narrative", "metric": "capital requirements",
           "retrieval_query": "factories robotics equipment investment", "evidence_question": "List every factory budget."}
    engine = Mock()
    engine.with_structured_output.return_value.invoke.return_value = {"answers": {"r1": {"passage_ids": ["P2"]}}}
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=engine), \
         patch("src.generation.evidence_answer.MAX_CONTEXT_TOKENS", 100), \
         patch("src.generation.evidence_answer.count_tokens", side_effect=lambda value: len(json.loads(value)["passages"]) * 100):
        answers = resolve_evidence_answers({"requirements": [req]}, [], {}, {"S1": source})
    payload = json.loads(engine.with_structured_output.return_value.invoke.call_args.args[0][-1].content)
    assert payload["requirements"][0]["candidate_passage_ids"] == ["P2"]
    assert "evidence_question" not in payload["requirements"][0] and "retrieval_query" not in payload["requirements"][0]
    assert answers[0].quotes[0].excerpt == "Factories require robotics and equipment investment."


def test_heading_only_evidence_does_not_trigger_a_generation_call():
    engine = Mock()
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=engine):
        answers = resolve_evidence_answers({"requirements": [{**requirement(), "kind": "narrative"}]},
            [], {}, {"S1": {**SOURCE, "evidence_text": "**Capital Requirements**"}})
    assert answers[0].unavailable_reason
    engine.assert_not_called()


def test_passage_budget_preserves_cost_explanation_over_a_related_table():
    import json
    explanation = ("Cost of revenue includes manufacturing costs, distribution costs, "
                   "support costs and datacenter costs for online services.")
    source = {**SOURCE, "evidence_text": explanation + '\n\n<table>'
              '<tr><th>Year</th><th>Cost</th></tr><tr><td>2024</td><td>12</td></tr></table>'
              + ''.join(f'\n\nUnrelated background disclosure {index}.' for index in range(10))}
    req = {**requirement(), "kind": "narrative", "metric": "cost"}
    engine = Mock()
    engine.with_structured_output.return_value.invoke.return_value = {"answers": {"r1": {"passage_ids": ["P1"]}}}
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=engine), \
         patch("src.generation.evidence_answer.MAX_CONTEXT_TOKENS", 100), \
         patch("src.generation.evidence_answer.count_tokens", side_effect=lambda value: len(json.loads(value)["passages"]) * 100):
        answers = resolve_evidence_answers({"requirements": [req]}, [], {}, {"S1": source})
    payload = json.loads(engine.with_structured_output.return_value.invoke.call_args.args[0][-1].content)
    assert [item["excerpt"] for item in payload["passages"]] == [f"[P1:0] {explanation}"]
    assert answers[0].quotes[0].excerpt == explanation


def test_interpretation_is_rejected_when_its_claim_is_unsupported_even_if_quotes_are_relevant():
    import json
    from src.generation.evidence_answer import EvidenceSynthesis, NarrativePoint
    source = {**SOURCE, "evidence_text": "Factories require substantial investment."}
    interpretation = "Factories can expand without requiring investment."
    engine = Mock()
    engine.with_structured_output.return_value.invoke.side_effect = [
        {"interpretation": interpretation, "source_ids": ["S1"]},
        {"assessments": {"r1": {"reason": "The investment is disclosed.", "supported": True,
            "point_support": {"0": {"evidence_spans": [source["evidence_text"]], "unsupported_details": []}}},
            "synthesis": {"reason": "The topic is covered but the interpretation contradicts the evidence.",
                "supported": True, "point_support": {"0": {"evidence_spans": [],
                    "unsupported_details": ["The cited factory investment is required."]}}}}}]
    answer = RequirementAnswer(requirement_id="r1", quotes=[EvidenceQuote(source_id="S1", excerpt=source["evidence_text"])],
        draft_points=[NarrativePoint(heading="Investment", text=source["evidence_text"], quote_indices=[0])])
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=engine):
        result = render_evidence_answer({"requirements": [{**requirement(), "kind": "narrative"}],
            "synthesis": "Interpret the investment requirements."}, [answer], [], {}, {"S1": source}, {}, {})
    synthesis_payload = json.loads(engine.with_structured_output.return_value.invoke.call_args_list[0].args[0][-1].content)
    assert '[S1]' in synthesis_payload['observations_to_be_audited'][0]['observation']
    audit_payload = json.loads(engine.with_structured_output.return_value.invoke.call_args_list[1].args[0][-1].content)
    claim = next(item for item in audit_payload['claims'] if item['requirement']['requirement_id'] == 'synthesis')
    assert claim['draft_points'][0]['text'] == interpretation
    assert claim['draft_points'][0]['evidence'][0]['source']['ticker'] == 'ACME'
    assert not result["complete"] and interpretation not in result["answer"]
    assert source["evidence_text"] in result["answer"]


@pytest.mark.parametrize("owner, supported", [("Widget", True), ("vehicle division", False)])
def test_business_can_be_named_inside_a_source_observation_without_becoming_an_observation_owner(owner, supported):
    source = {**SOURCE, "evidence_text": f"{owner} revenue increased because demand grew."}
    req = {**requirement(), "kind": "narrative", "metric": "revenue drivers", "subject": "Widget division"}
    audit = {"assessments": {"r1": {"reason": "Widget's growth driver is reported.", "supported": True,
        "quote_assessments": {"0": {"source_subject": f"{owner} revenue", "source_subject_type": "metric_or_action",
            "reason": "The observation phrase explicitly names Widget.", "answers_question": True}}}}}
    engine = Mock()
    engine.with_structured_output.return_value.invoke.return_value = audit
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=engine):
        result = render_evidence_answer({"requirements": [req]}, [RequirementAnswer(requirement_id="r1",
            quotes=[EvidenceQuote(source_id="S1", excerpt=source["evidence_text"])])], [], {}, {"S1": source}, {}, {})
    assert result["complete"] is supported
    assert (source["evidence_text"] in result["answer"]) is supported


def test_interpretation_must_be_rebuilt_after_a_cited_base_point_is_discarded():
    from src.generation.evidence_answer import NarrativePoint
    source = {**SOURCE, "evidence_text": "Factories require substantial investment."}
    interpretation = "The company has stable funding for its expansion."
    engine = Mock()
    engine.with_structured_output.return_value.invoke.side_effect = [
        {"interpretation": interpretation, "source_ids": ["S1"]},
        {"assessments": {"r1": {"reason": "Investment is reported; funding stability is not.", "supported": True,
            "point_support": {"0": {"evidence_spans": [source["evidence_text"]], "unsupported_details": []},
                              "1": {"evidence_spans": [], "unsupported_details": ["Funding stability is not disclosed."]}}},
            "synthesis": {"reason": "Topic is covered.", "supported": True,
            "point_support": {"0": {"evidence_spans": [source["evidence_text"]], "unsupported_details": []}}}}}]
    answer = RequirementAnswer(requirement_id="r1", quotes=[EvidenceQuote(source_id="S1", excerpt=source["evidence_text"])],
        draft_points=[NarrativePoint(heading="Investment", text=source["evidence_text"], quote_indices=[0]),
                      NarrativePoint(heading="Funding", text=interpretation, quote_indices=[0])])
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=engine):
        result = render_evidence_answer({"requirements": [{**requirement(), "kind": "narrative"}],
            "synthesis": "Interpret the investment requirements."}, [answer], [], {}, {"S1": source}, {}, {})
    assert result["missing_requirement_ids"] == ["synthesis"]
    assert interpretation not in result["answer"] and source["evidence_text"] in result["answer"]


@pytest.mark.parametrize("owner, complete", [("Widget division", True), ("Vehicle division", False)])
def test_literal_business_attribution_survives_an_auditors_metric_label(owner, complete):
    source = {**SOURCE, "evidence_text": f"Gross margin for the {owner} increased because input costs fell."}
    req = {**requirement(), "kind": "narrative", "metric": "gross margin drivers", "subject": "Widget division"}
    audit = {"assessments": {"r1": {"reason": "The margin driver is disclosed.", "supported": True,
        "quote_assessments": {"0": {"source_subject": "Gross margin", "source_subject_type": "metric_or_action",
            "reason": "The quoted clause names the business and its margin driver.", "answers_question": True}}}}}
    engine = Mock()
    engine.with_structured_output.return_value.invoke.return_value = audit
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=engine):
        result = render_evidence_answer({"requirements": [req]}, [RequirementAnswer(requirement_id="r1",
            quotes=[EvidenceQuote(source_id="S1", excerpt=source["evidence_text"])])], [], {}, {"S1": source}, {}, {})
    assert result["complete"] is complete
    assert (source["evidence_text"] in result["answer"]) is complete


def test_audit_splits_only_at_whole_requirement_boundaries_when_budget_is_exceeded():
    import json
    from src.generation.evidence_answer import NarrativePoint
    source = {**SOURCE, "evidence_text": "Factories require substantial investment. Revenue comes from product sales."}
    requirements = [{**requirement(key), "kind": "narrative", "metric": metric}
                    for key, metric in (("r1", "capital requirements"), ("r2", "revenue sources"))]
    answers = [RequirementAnswer(requirement_id=key, quotes=[EvidenceQuote(source_id="S1", excerpt=text)],
        draft_points=[NarrativePoint(heading="Observation", text=text, quote_indices=[0])])
        for key, text in (("r1", "Factories require substantial investment."), ("r2", "Revenue comes from product sales."))]
    def review(messages):
        payload = json.loads(messages[-1].content)
        return {"assessments": {item["requirement"]["requirement_id"]: {"reason": "Responsive evidence.",
            "supported": True, "point_support": {"0": {"evidence_spans": list(item["draft_points"][0]["evidence"][0]["text"].values()),
                "unsupported_details": []}}} for item in payload["claims"]},
                "request_complete": any(item["requirement"]["requirement_id"] == "r2" for item in payload["claims"])}
    engine = Mock()
    engine.with_structured_output.return_value.invoke.side_effect = review
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=engine), \
         patch("src.generation.evidence_answer.MAX_CONTEXT_TOKENS", 100), \
         patch("src.generation.evidence_answer.count_tokens", side_effect=lambda value: 60 * len(json.loads(value).get("claims", []))):
        result = render_evidence_answer({"requirements": requirements}, answers, [], {}, {"S1": source}, {}, {})
    assert result["complete"] and engine.with_structured_output.call_count == 2
    assert [call.args[0]["properties"]["assessments"]["required"] for call in engine.with_structured_output.call_args_list] == [["r1"], ["r2"]]
    schemas = [call.args[0]["properties"] for call in engine.with_structured_output.call_args_list]
    assert "request_complete" not in schemas[0] and "request_complete" in schemas[1]


def test_unsupported_heading_is_removed_even_when_its_underlying_sentence_is_true():
    from src.generation.evidence_answer import NarrativePoint
    source = {**SOURCE, "evidence_text": "New import restrictions could deter purchases of our products."}
    audit = {"assessments": {"r1": {"reason": "Demand risk does not establish dependence on a few buyers.",
        "retained_evidence_complete": False, "missing_details": ["Dependence on a small number of customers."],
        "quote_assessments": {"0": {"reason": "Regulatory demand risk only.", "answers_question": False}},
        "point_support": {"0": {"evidence_spans": [], "unsupported_details": ["The heading asserts a different relationship."]}}}}}
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=SimpleNamespace(
            with_structured_output=lambda *a, **k: SimpleNamespace(invoke=lambda _: audit))):
        result = render_evidence_answer({"requirements": [{**requirement(), "kind": "narrative", "metric": "customer concentration"}]},
            [RequirementAnswer(requirement_id="r1", quotes=[EvidenceQuote(source_id="S1", excerpt=source["evidence_text"])],
                draft_points=[NarrativePoint(heading="Concentration of buyers", text=source["evidence_text"], quote_indices=[0])])],
            [], {}, {"S1": source}, {}, {})
    assert not result["complete"]
    assert "Concentration of buyers" not in result["answer"]
    assert "Evidence gap:" in result["answer"]


def test_empty_missing_details_cannot_override_an_explicit_negative_verdict():
    source = {**SOURCE, "evidence_text": "Our platform serves customers globally."}
    audit = {"assessments": {"r1": {"reason": "A global presence does not explain investment needs.",
        "missing_details": [], "retained_evidence_complete": False,
        "quote_assessments": {"0": {"reason": "Off topic", "answers_question": False}}}}}
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=SimpleNamespace(
            with_structured_output=lambda *a, **k: SimpleNamespace(invoke=lambda _: audit))):
        result = render_evidence_answer({"requirements": [{**requirement(), "kind": "narrative", "metric": "capital requirements"}]},
            [RequirementAnswer(requirement_id="r1", quotes=[EvidenceQuote(source_id="S1", excerpt=source["evidence_text"])])],
            [], {}, {"S1": source}, {}, {})
        assert not result["complete"]
        assert "Our platform serves customers globally" not in result["answer"]


def test_multi_requirement_context_keeps_separate_evidence_within_budget(tmp_path):
    parents = []
    for index, text in enumerate(("Revenue comes from subscriptions.", "Expenses include labor and materials.",
                                  "Investment includes factories and equipment.")):
        parent_id = str(index + 1)
        metadata = {**SOURCE, "parent_id": parent_id}
        parents.append(Document(page_content=("Unrelated background. " * 300 + "\n\n" + text), metadata=metadata))
    builder = ContextBuilder(tmp_path)
    builder.parent_store.get_many = lambda ids: [parents[int(key) - 1] for key in ids]
    children = [RankedChild(str(i), str(i), p.page_content, p.metadata, fused_score=4-i)
                for i, p in enumerate(parents, 1)]
    result = builder.build(children, SCOPE, max_parents=3, max_chars=2200, max_tokens=1000,
        query="revenue expenses investment", requirement_queries=("revenue", "expenses", "investment"))
    assert len(result.sources) == 3
    assert all(word in result.context for word in ("subscriptions", "materials", "equipment"))
    assert result.context_tokens <= 1000
    assert len(result.context) <= 2200


def test_context_compression_uses_the_observation_that_retrieved_this_parent(tmp_path):
    body = ("Revenue sources include product sales and subscriptions. " * 80
            + "\n\nOperating expenses include factory labor and purchased components.")
    metadata = {**SOURCE, "parent_id": "p1"}
    builder = ContextBuilder(tmp_path)
    builder.parent_store.get_many = lambda ids: [Document(page_content=body, metadata=metadata)]
    child = RankedChild("c1", "p1", body, {**metadata, "evidence_queries": ("operating expenses",)}, fused_score=1)
    result = builder.build([child], SCOPE, max_parents=1, max_chars=1000, max_tokens=400,
        query="revenue sources product sales subscriptions", requirement_queries=("revenue sources", "operating expenses"))
    assert "factory labor and purchased components" in result.context
    assert result.sources[0]["evidence_queries"] == ("operating expenses",)
    assert result.context_tokens <= 400
