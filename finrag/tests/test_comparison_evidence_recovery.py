"""Relationship proofs and coverage must survive selective comparison repair."""
import json
import re
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import ToolMessage

from src.agent.graph import create_finrag_agent
from src.agent.tools import RequestPlanInput
from src.financial.calculator import FinancialCalculator
from src.generation.evidence_answer import EvidenceQuote, NarrativePoint, RequirementAnswer, render_evidence_answer
from src.schemas import RetrievalBundle, Scope


SOURCE = dict(id="S1", ticker="ACME", fiscal_year="2025", doc_type="10K", source="filing.pdf")
REQUIREMENT = dict(requirement_id="r1", ticker="ACME", source_year="2025", doc_type="10K", kind="narrative")


def proof(text, premise=None):
    return dict(source_relationship=text, missing_premise=premise,
        supports_requested_relationship=premise is None, evidence_spans=[text], unsupported_details=[])


def test_source_relationship_cannot_copy_an_unquoted_claim_even_with_a_positive_verdict():
    from src.generation.evidence_answer import PointAssessment, _point_supported
    source = "A loan agreement requires minimum cash reserves."
    claim = "New factories require additional investment."
    review = PointAssessment(**{**proof(source), "source_relationship": claim})
    assert not _point_supported(review, dict(text=claim, heading="Investment", quote_indices=[0]), [source])


@pytest.mark.parametrize("anchor,accepted", [("storage", True), ("unquoted-business", False), ("Original", False)])
def test_business_quote_requires_a_literal_owner_anchor_and_preserves_valid_outlook(anchor, accepted):
    text = "Our storage business expects deployments to increase, with competitive pressure on margins."
    answer = RequirementAnswer(requirement_id="r1", quotes=[EvidenceQuote(source_id="S1", excerpt=text)],
        draft_points=[NarrativePoint(text=text, quote_indices=[0])])
    engine = Mock()
    engine.with_structured_output.return_value.invoke.return_value = {"assessments": {"r1": dict(
        requested_subject="storage business", reason="Check the source business and its outlook.", supported=True,
        quote_assessments={"0": dict(source_subject=anchor, source_subject_type="business_area",
            source_observation=None, reason="The stored observation establishes the outlook.",
            scope_matches=True, answers_question=True)}, point_support={"0": proof(text)})}}
    # The positive quote names its business literally. The negative quote needs
    # context attribution, and an invented anchor must not establish that owner.
    quote = text if accepted else "We expect deployments to increase, with competitive pressure on margins."
    answer.quotes[0].excerpt = quote
    answer.draft_points[0].text = quote
    engine.with_structured_output.return_value.invoke.return_value["assessments"]["r1"]["point_support"]["0"] = proof(quote)
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=engine):
        result = render_evidence_answer({"requirements": [{**REQUIREMENT, "subject": "storage business", "metric": "outlook"}]},
            [answer], [], {}, {"S1": {**SOURCE, "evidence_text": text + " " + quote}}, {}, {})
    assert result["complete"] is accepted
    schema = engine.with_structured_output.call_args.args[0]
    anchor_schema = schema["properties"]["assessments"]["properties"]["r1"]["properties"]["quote_assessments"]["properties"]["0"]["properties"]["source_subject_index"]
    assert anchor_schema["type"] == "integer" and anchor_schema["minimum"] == 0
    payload = json.loads(engine.with_structured_output.return_value.invoke.call_args.args[0][-1].content)
    anchors = payload["claims"][0]["quotes"][0]["source_anchors"].values()
    assert "storage" in anchors and "business" not in anchors
    assert "Original" not in anchors
    detail = schema["properties"]["assessments"]["properties"]["r1"]["properties"]["detail_coverage"]["items"]["properties"]
    assert detail["quote_indices"]["maxItems"] == 0
    quote_fields = schema["properties"]["assessments"]["properties"]["r1"]["properties"]["quote_assessments"]["properties"]["0"]["properties"]
    assert list(quote_fields)[:2] == ["source_subject_index", "source_subject_type"]
    assert payload["claims"][0]["draft_points"][0] == dict(point_index=0, text=quote, quote_indices=[0])


@pytest.mark.parametrize("outside_catalog", [False, True])
def test_source_anchor_indices_bind_an_inherited_outlook_to_its_actual_context(outside_catalog):
    quote = "We expect deployments to rise, with competitive pressure on margins."
    source = "Our storage operations are growing. " + quote
    answer = RequirementAnswer(requirement_id="r1", quotes=[EvidenceQuote(source_id="S1", excerpt=quote)],
        draft_points=[NarrativePoint(text=quote, quote_indices=[0])])
    def respond(messages):
        payload = json.loads(messages[-1].content)
        anchors = payload["claims"][0]["quotes"][0]["source_anchors"]
        index = len(anchors) if outside_catalog else next(int(key) for key, word in anchors.items() if word == "storage")
        return {"assessments": {"r1": dict(reason="The attribution establishes the actual business.", supported=True,
            quote_assessments={"0": dict(source_subject_index=index, source_subject_type="business_area",
                reason="Context establishes storage ownership of this outlook.", scope_matches=True, answers_question=True)},
            point_support={"0": proof(quote)})}}
    engine = Mock()
    engine.with_structured_output.return_value.invoke.side_effect = respond
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=engine):
        arguments = ({"requirements": [{**REQUIREMENT, "subject": "storage business", "metric": "outlook"}]},
            [answer], [], {}, {"S1": {**SOURCE, "evidence_text": source}}, {}, {})
        if outside_catalog:
            with pytest.raises(ValueError, match="outside its own point"):
                render_evidence_answer(*arguments)
        else:
            result = render_evidence_answer(*arguments)
            assert result["complete"] and quote in result["answer"]


@pytest.mark.parametrize("business_scope,aggregate", [(False, True), (True, True), (True, False)])
def test_company_aggregate_cannot_be_attributed_to_one_business_in_its_product_list(business_scope, aggregate):
    quote = ("We expect aggregate spending on robotics and energy storage products to increase." if aggregate
        else "We expect energy storage deployments to increase.")
    source = "Our energy storage business supplies batteries. " + quote
    answer = RequirementAnswer(requirement_id="r1", quotes=[EvidenceQuote(source_id="S1", excerpt=quote)],
        draft_points=[NarrativePoint(text=quote, quote_indices=[0])])
    def respond(messages):
        payload = json.loads(messages[-1].content)
        claim = payload["claims"][0]
        review = dict(reason="This is a company-wide aggregate, not isolated business investment.",
            answers_question=True, scope_matches=True, supports_requested_relationship=True, source_is_aggregate=aggregate)
        if business_scope:
            anchors = claim["quotes"][0]["source_anchors"]
            review.update(source_subject_index=next(int(i) for i, word in anchors.items() if word == "We"), source_subject_type="company")
        return {"assessments": {"r1": dict(reason="Identify the aggregate's actual scope.", supported=True,
            quote_assessments={"0": review})}}
    engine = Mock()
    engine.with_structured_output.return_value.invoke.side_effect = respond
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=engine):
        result = render_evidence_answer({"requirements": [{**REQUIREMENT, "subject": "energy storage business" if business_scope else None,
            "metric": "business outlook" if business_scope else "spending outlook"}]}, [answer], [], {}, {"S1": {**SOURCE, "evidence_text": source}}, {}, {})
    assert result["complete"] is not (business_scope and aggregate)
    assert (quote in result["answer"]) is not (business_scope and aggregate)


def test_shared_long_paragraph_remains_available_without_exhausting_the_audit_budget():
    observations = ["Deployments are expected to grow.", "Demand remains strong.",
        "Competitive pressure may reduce margins.", "Capacity expansion continues."]
    paragraph = "Storage operations serve customers through long-term contracts. " * 75 + " ".join(observations)
    answer = RequirementAnswer(requirement_id="r1", quotes=[EvidenceQuote(source_id="S1", excerpt=text) for text in observations],
        draft_points=[NarrativePoint(text=text, quote_indices=[index]) for index, text in enumerate(observations)])
    payloads = []
    def respond(messages):
        payload = json.loads(messages[-1].content)
        payloads.append(payload)
        claim = payload["claims"][0]
        return {"assessments": {"r1": dict(reason="All outlook observations belong to storage.", supported=True,
            quote_assessments={str(q["index"]): dict(source_subject_index=next(int(i) for i, word in q["source_anchors"].items() if word == "Storage"),
                source_subject_type="business_area", reason="The original paragraph establishes ownership.", scope_matches=True, answers_question=True)
                for q in claim["quotes"]}, point_support={str(i): proof(text) for i, text in enumerate(observations)})}}
    engine = Mock()
    engine.with_structured_output.return_value.invoke.side_effect = respond
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=engine), \
         patch("src.generation.evidence_answer.MAX_CONTEXT_TOKENS", 12000):
        result = render_evidence_answer({"requirements": [{**REQUIREMENT, "subject": "storage business", "metric": "outlook"}]},
            [answer], [], {}, {"S1": {**SOURCE, "evidence_text": paragraph}}, {}, {})
    assert result["complete"] and all(text in result["answer"] for text in observations)
    assert list(payloads[0]["paragraph_contexts"].values()) == [paragraph]
    assert engine.with_structured_output.return_value.invoke.call_count == 1


def test_schema_compaction_preserves_nested_reference_validation():
    from langchain_core.utils.json_schema import dereference_refs
    from src.generation.evidence_answer import _compact_schema
    schema = {"$ref": "#/$defs/Wrapper", "$defs": {
        "Wrapper": {"type": "object", "properties": {"value": {"$ref": "#/$defs/Bound"}}, "required": ["value"]},
        "Bound": {"type": "integer", "minimum": 5}, "Unused": {"type": "boolean"}}}
    compact = _compact_schema(json.loads(json.dumps(schema)))
    assert set(compact["$defs"]) == {"Wrapper", "Bound"}
    expanded = [dereference_refs(current) for current in (schema, compact)]
    for current in expanded:
        current.pop("$defs", None)
    assert expanded[0] == expanded[1]
    assert expanded[1]["properties"]["value"]["minimum"] == 5


@pytest.mark.parametrize("indices", [[-1], [2], [True]])
def test_compact_proofs_cannot_reference_another_points_evidence(indices):
    from src.generation.evidence_answer import _resolve_proof_indices
    with pytest.raises(ValueError, match="outside its own point"):
        _resolve_proof_indices(indices, ["First stored sentence.", "Second stored sentence."])


def test_compact_proofs_resolve_to_original_source_sentences_without_copying_them_into_the_schema():
    text = "Factories require investment in equipment. No expansion is currently planned."
    answer = RequirementAnswer(requirement_id="r1", quotes=[EvidenceQuote(source_id="S1", excerpt=text)],
        draft_points=[NarrativePoint(text="Factories require investment in equipment.", quote_indices=[0])])
    engine = Mock()
    engine.with_structured_output.return_value.invoke.return_value = {"assessments": {"r1": {
        "reason": "The actual investment need is disclosed.", "supported": True,
        "point_support": {"0": {"source_relationship_index": 0, "evidence_indices": [0],
            "unsupported_assertion_indices": [], "missing_premise": None, "supports_requested_relationship": True}},
        "detail_coverage": [{"requested_detail": "investment needs", "point_indices": [0], "quote_indices": []}]}}}
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=engine):
        result = render_evidence_answer({"requirements": [{**REQUIREMENT, "metric": "investment needs"}]},
            [answer], [], {}, {"S1": {**SOURCE, "evidence_text": text}}, {}, {})
    assert result["complete"]
    proof_result = result["request_coverage"]["assessments"][0]["point_support"]["0"]
    assert proof_result["evidence_spans"] == ["Factories require investment in equipment."]
    assert "Factories require investment" not in json.dumps(engine.with_structured_output.call_args.args[0])


def test_interpretation_consolidates_grouped_source_markers_into_application_citations():
    text = "Factories require investment in equipment."
    answer = RequirementAnswer(requirement_id="r1", quotes=[EvidenceQuote(source_id="S1", excerpt=text)],
        draft_points=[NarrativePoint(text=text, quote_indices=[0])])
    engine = Mock()
    engine.with_structured_output.return_value.invoke.side_effect = [
        {"interpretation": text + " [S1, S1] (r1:p0, r1:p0)", "source_ids": ["S1"], "observation_ids": ["r1:p0"]},
        {"assessments": {key: dict(reason="The supplied investment observation is supported.", supported=True,
            point_support={"0": proof(text)}) for key in ("r1", "synthesis")}}]
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=engine):
        result = render_evidence_answer({"requirements": [{**REQUIREMENT, "metric": "investment needs"}],
            "synthesis": "Compare investment needs."}, [answer], [], {}, {"S1": {**SOURCE, "evidence_text": text}}, {}, {})
    assert result["complete"] and "[S1, S1]" not in result["answer"]
    assert result["cited_source_ids"] == ["S1"] and result["answer"].count("[S1]") == 2


@pytest.mark.parametrize("uses_rejected_observation", [False, True])
def test_interpretation_depends_on_retained_observations_not_a_whole_source_id(uses_rejected_observation):
    good, unrelated = "Factories require investment in equipment.", "Cash is available to fund operations."
    answer = RequirementAnswer(requirement_id="r1", quotes=[EvidenceQuote(source_id="S1", excerpt=text) for text in (good, unrelated)],
        draft_points=[NarrativePoint(text=text, quote_indices=[i]) for i, text in enumerate((good, unrelated))])
    interpretation = unrelated if uses_rejected_observation else good
    engine = Mock()
    engine.with_structured_output.return_value.invoke.side_effect = [
        {"interpretation": interpretation, "source_ids": ["S1"], "observation_ids": ["r1:p1" if uses_rejected_observation else "r1:p0"]}, {"assessments": {
            "r1": dict(reason="The investment observation covers the request.", supported=True,
                point_support={"0": proof(good), "1": proof(unrelated, "Funding does not establish investment needs.")}),
            "synthesis": dict(reason="Verify the actual observations used.", supported=True,
                point_support={"0": proof(interpretation)})}}]
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=engine):
        result = render_evidence_answer({"requirements": [{**REQUIREMENT, "metric": "investment needs"}],
            "synthesis": "Compare investment needs."}, [answer], [], {}, {"S1": {**SOURCE, "evidence_text": good + " " + unrelated}}, {}, {})
    assert result["complete"] is (not uses_rejected_observation)
    assert good in result["answer"] and unrelated not in result["answer"]
    assert len(result["retained_answers"][0]["draft_points"]) == 1


def test_provisional_selection_gap_reaches_audit_without_forcing_unrequested_numeric_forecasts():
    text = "Management expects capacity to grow."
    answer = RequirementAnswer(requirement_id="r1", quotes=[EvidenceQuote(source_id="S1", excerpt=text)],
        draft_points=[NarrativePoint(text=text, quote_indices=[0])], coverage_gap="No numeric forecast was selected.")
    engine = Mock()
    engine.with_structured_output.return_value.invoke.return_value = {"assessments": {"r1": dict(
        reason="The qualitative outlook answers the question without unrequested numbers.", supported=True,
        point_support={"0": proof(text)})}}
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=engine):
        result = render_evidence_answer({"requirements": [{**REQUIREMENT, "metric": "capacity outlook"}]}, [answer], [], {},
            {"S1": {**SOURCE, "evidence_text": text}}, {}, {})
    payload = json.loads(engine.with_structured_output.return_value.invoke.call_args.args[0][-1].content)
    assert payload["claims"][0]["selection_gap"] == answer.coverage_gap
    assert result["complete"] and result["verified_answers"][0]["coverage_gap"] is None


@pytest.mark.parametrize("metric,text,premise", [
    ("investment needs", "Cash is available to fund operations.", "Available funding does not establish the investments needed."),
    ("investment needs", "Sales incentive costs are capitalized over the contract term.", "Accounting treatment does not establish infrastructure investment needs."),
    ("funding availability", "Cash is available to fund operations.", None),
    ("contract cost accounting", "Sales incentive costs are capitalized over the contract term.", None),
    ("investment needs", "Facilities require investment in factories and production equipment.", None),
    ("investment needs", "No additional production equipment is needed for the planned expansion.", None),
])
def test_relation_proof_controls_acceptance_without_rejecting_valid_funding_or_negative_answers(metric, text, premise):
    answer = RequirementAnswer(requirement_id="r1", quotes=[EvidenceQuote(source_id="S1", excerpt=text)],
        draft_points=[NarrativePoint(heading=metric, text=text, quote_indices=[0])])
    audit = {"assessments": {"r1": dict(reason="Check the requested relationship.", supported=True,
        point_support={"0": proof(text, premise)},
        detail_coverage=[dict(requested_detail=metric, point_indices=[0], quote_indices=[])])}}
    engine = Mock()
    engine.with_structured_output.return_value.invoke.return_value = audit
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=engine):
        result = render_evidence_answer({"requirements": [{**REQUIREMENT, "metric": metric}]}, [answer], [], {},
            {"S1": {**SOURCE, "evidence_text": text}}, {}, {})
    assert result["complete"] is (premise is None)
    if premise:
        assert text not in result["answer"] and result["missing_requirement_ids"] == ["r1"]
    else:
        assert text in result["answer"] and result["verified_answers"]
    schema = engine.with_structured_output.call_args.args[0]["properties"]["assessments"]["properties"]["r1"]
    assert schema["properties"]["detail_coverage"]["minItems"] == 1
    assert schema["properties"]["point_support"]["properties"]["0"]["properties"]["supports_requested_relationship"]["type"] == "boolean"
    point_schema = schema["properties"]["point_support"]["properties"]["0"]["properties"]
    assert point_schema["source_relationship_index"]["maximum"] == 0
    assert point_schema["unsupported_assertion_indices"]["items"]["maximum"] == 1


@pytest.mark.parametrize("gap_filled", [False, True])
def test_rejected_point_cannot_silently_remove_a_requested_comparison_dimension(gap_filled):
    texts = ["Subscriptions provide recurring revenue.", "Employee compensation drives operating costs.",
             "Factories require investment in production equipment.", "Cash is available to fund operations."]
    points = [NarrativePoint(heading="Source observation", text=text, quote_indices=[index])
              for index, text in enumerate(texts)]
    answer = RequirementAnswer(requirement_id="r1", quotes=[EvidenceQuote(source_id="S1", excerpt=text) for text in texts], draft_points=points)
    audit = {"assessments": {"r1": dict(reason="All three dimensions need support.", supported=True,
        point_support={str(i): proof(text, "Funding is not investment needs." if i == 3 or i == 2 and not gap_filled else None)
                       for i, text in enumerate(texts)}, detail_coverage=[
            dict(requested_detail=name, point_indices=[i], quote_indices=[])
            for i, name in enumerate(("revenue sources", "cost structure", "investment requirements"))])}}
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=SimpleNamespace(
            with_structured_output=lambda *a, **k: SimpleNamespace(invoke=lambda _: audit))):
        result = render_evidence_answer({"requirements": [{**REQUIREMENT, "metric": "business model comparison"}]},
            [answer], [], {}, {"S1": {**SOURCE, "evidence_text": " ".join(texts)}}, {}, {})
    assert result["complete"] is gap_filled
    assert texts[0] in result["answer"] and texts[1] in result["answer"] and texts[3] not in result["answer"]
    assert len(result["retained_answers"][0]["draft_points"]) == (3 if gap_filled else 2)
    if not gap_filled:
        assert "investment requirements" in result["request_coverage"]["assessments"][0]["missing_details"]


def test_focused_recovery_preserves_valid_parts_of_an_incomplete_requirement_and_adds_only_the_gap():
    texts = ["Subscriptions provide recurring revenue.", "Employee compensation drives operating costs.",
             "Factories require investment in production equipment."]
    scope = Scope(("ACME",), ("2025",), "10K")
    plan = RequestPlanInput.model_validate({"requirements": [{**REQUIREMENT,
        "metric": "revenue sources, cost structure and investment requirements"}]})
    retriever = SimpleNamespace(retrieve=Mock(side_effect=[
        RetrievalBundle("[SOURCE S1 | TICKER: ACME | FISCAL YEAR: 2025 | TYPE: 10K]\n" + "\n\n".join(texts[:2]),
            [{**SOURCE, "parent_id": "first"}], scope, 1),
        RetrievalBundle("[SOURCE S2 | TICKER: ACME | FISCAL YEAR: 2025 | TYPE: 10K]\n" + texts[2],
            [{**SOURCE, "id": "S2", "parent_id": "second"}], scope, 1)]))
    payloads = []
    def respond(schema, messages):
        payload = json.loads(messages[-1].content)
        payloads.append(payload)
        if "answers" in schema["properties"]:
            retained = payload["retained_observations"]
            wanted = texts[2:] if retained else texts[:2]
            return {"answers": {"r1": {"draft_points": [dict(heading="Observation", text=text,
                passage_ids=[next(p["passage_id"] for p in payload["passages"] if text in p["excerpt"])]) for text in wanted]}}}
        points = payload["claims"][0]["draft_points"]
        return {"assessments": {"r1": dict(reason="Bind each dimension to displayed observations.", supported=True,
            point_support={str(i): proof(next(iter(point["evidence"][0]["text"].values()))) for i, point in enumerate(points)},
            detail_coverage=[dict(requested_detail=name, point_indices=[i] if i < len(points) else [], quote_indices=[])
                for i, name in enumerate(("revenue sources", "cost structure", "investment requirements"))])}}
    engine = SimpleNamespace(with_structured_output=lambda schema, **kw: SimpleNamespace(invoke=lambda messages: respond(schema, messages)))
    locked = "Application-resolved scope (locked): " + json.dumps({"requested_groups": [["ACME", "2025"]], "doc_type": "10K"})
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=engine):
        state = create_finrag_agent(retriever, FinancialCalculator(), permitted_scope=scope,
            plan_builder=lambda *args: plan).invoke({"messages": [("user", locked + "\nUser question: Compare revenue sources, cost structure and investment requirements.")]})
    finals = [json.loads(message.content) for message in state["messages"]
              if isinstance(message, ToolMessage) and message.name == "finalize_financial_answer"]
    assert len(finals) == 2 and not finals[0]["complete"] and finals[1]["complete"]
    assert all(text in finals[1]["answer"] for text in texts)
    assert len(payloads[2]["retained_observations"][0]["draft_points"]) == 2
    assert len(payloads[3]["claims"][0]["draft_points"]) == 3
    query = retriever.retrieve.call_args_list[1].args[0]
    assert "investment requirements" in query and "revenue sources" not in query and "cost structure" not in query
    assert retriever.retrieve.call_args_list[1].kwargs["requirement_queries"] == (query,)


def test_selection_uses_stored_sentence_ids_without_allowing_generated_observations():
    from src.generation.evidence_answer import resolve_evidence_answers
    text = "We sell batteries. No factory expansion is planned."
    engine = Mock()
    engine.with_structured_output.return_value.invoke.return_value = {
        "answers": {"r1": {"draft_points": [{"sentence_id": "P1:0"}, {"sentence_id": "P1:1"}]}}}
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=engine):
        result = resolve_evidence_answers({"requirements": [{**REQUIREMENT, "metric": "business model"}]}, [], {},
            {"S1": {**SOURCE, "evidence_text": text}})
    assert [point.text for point in result[0].draft_points] == ["We sell batteries.", "No factory expansion is planned."]
    schema = engine.with_structured_output.call_args.args[0]
    point = schema["properties"]["answers"]["properties"]["r1"]["properties"]["draft_points"]["items"]
    assert set(point["properties"]) == {"sentence_id"}
    pattern = schema['$defs'][point['properties']['sentence_id']['$ref'].rsplit('/', 1)[1]]['pattern']
    assert all(re.fullmatch(pattern, value) for value in ['P1:0', 'P1:1'])
    assert not any(re.fullmatch(pattern, value) for value in ['P1:2', 'P2:0'])


def test_passage_budget_keeps_an_investment_sentence_intact_at_a_paragraph_boundary():
    from src.generation.evidence_answer import evidence_passages
    first = "We provide subscription services to " + "customers " * 130 + "worldwide."
    investment = "We expect investment to continue, including new facilities and computer systems for growing operations."
    text = first + " " + investment
    assert len(first) < 1400 < len(text)
    passages = evidence_passages([{**SOURCE, "evidence_text": text}])
    assert investment in [item["excerpt"] for item in passages.values()]
    assert all(item["excerpt"] in text for item in passages.values())


def test_first_narrative_search_keeps_parent_capacity_for_missing_evidence():
    scope = Scope(("ACME",), ("2025",), "10K")
    cost, outlook = "Factories require substantial investment.", "Management expects capacity to expand next year."
    plan = RequestPlanInput.model_validate({"requirements": [{**REQUIREMENT, "requirement_id": key, "metric": metric}
        for key, metric in (("r1", "investment requirements"), ("r2", "capacity outlook"))]})
    limits, queries = [], []
    def retrieve(query, scope, **kwargs):
        limits.append(kwargs["max_parents"])
        queries.append((query, kwargs["requirement_queries"]))
        text = cost if len(limits) == 1 else outlook
        sources = [{**SOURCE, "id": f"S{i}", "parent_id": f"round-{len(limits)}-{i}"}
                   for i in range(1, kwargs["max_parents"] + 1)]
        context = "\n\n".join(f"[SOURCE {source['id']} | TICKER: ACME | FISCAL YEAR: 2025 | TYPE: 10K]\n{text}" for source in sources)
        return RetrievalBundle(context, sources, scope, len(sources))
    def respond(schema, messages):
        payload = json.loads(messages[-1].content)
        if "answers" in schema["properties"]:
            values = {}
            for item in payload["requirements"]:
                key = item["requirement_id"]
                phrase = cost if key == "r1" else outlook
                passage = next((p for p in payload["passages"] if phrase in p["excerpt"]), None)
                values[key] = {"draft_points": [{"sentence_id": passage["excerpt"].split("]", 1)[0][1:]}]} if passage else {
                    "unavailable_reason": "Capacity outlook evidence is missing."}
            return {"answers": values}
        return {"assessments": {item["requirement"]["requirement_id"]: dict(reason="The requested observation is supported.",
            supported=True, quote_assessments={str(quote["index"]): dict(reason="The literal quote establishes this observation.",
                answers_question=True, scope_matches=True, supports_requested_relationship=True) for quote in item["quotes"]})
            for item in payload["claims"]}}
    engine = SimpleNamespace(with_structured_output=lambda schema, **kw: SimpleNamespace(invoke=lambda messages: respond(schema, messages)))
    locked = "Application-resolved scope (locked): " + json.dumps({"requested_groups": [["ACME", "2025"]], "doc_type": "10K"})
    with patch("src.agent.tools.MAX_CONTEXT_PARENTS", 4), patch("src.generation.evidence_answer.get_llm_engine", return_value=engine):
        state = create_finrag_agent(SimpleNamespace(retrieve=retrieve), FinancialCalculator(), permitted_scope=scope,
            plan_builder=lambda *args: plan).invoke({"messages": [("user", locked + "\nUser question: Compare investment requirements and capacity outlook.")]})
    final = next(json.loads(m.content) for m in reversed(state["messages"]) if isinstance(m, ToolMessage) and m.name == "finalize_financial_answer")
    assert limits == [3, 1] and final["complete"]
    assert cost in final["answer"] and outlook in final["answer"]
    assert "Capacity outlook evidence is missing." in queries[1][0]
    assert queries[1][1] == (queries[1][0],)
