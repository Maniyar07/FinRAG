"""Output intent and claim-local proof checks, independent of company or query IDs."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.agent.graph import build_request_plan, create_finrag_agent
from src.agent.tools import RequestPlanInput
from src.financial.calculator import FinancialCalculator
from src.generation.evidence_answer import EvidenceQuote, NarrativePoint, RequirementAnswer, render_evidence_answer
from src.schemas import Scope


SCOPE = Scope(("ACME",), ("2025",), "10K")
REQUIREMENT = dict(requirement_id="r1", ticker="ACME", source_year="2025", doc_type="10K",
                   metric="total revenue", reported_period="FY2025", object_id="o1")
SOURCE = dict(id="S1", ticker="ACME", fiscal_year="2025", doc_type="10K")


def test_scalar_output_is_classified_once_and_reaches_exact_lookup_without_narrative_calls(tmp_path):
    question = "Report ACME total revenue for FY2025, with currency and scale from its 2025 10-K."
    draft = dict(requested_objects=[dict(object_id="o1", quote="total revenue", kind="numeric_value",
        subject="total revenue", reported_period="FY2025")], requirements=[REQUIREMENT,
        {**REQUIREMENT, "requirement_id": "duplicate", "reported_period": None}])
    model = Mock()
    model.with_structured_output.return_value.invoke.return_value = draft
    table = "## INCOME STATEMENTS\n(In millions)\n<table><tr><th>Metric</th><th>2025</th></tr><tr><td>Total revenue</td><td>120</td></tr></table>"
    import json
    (tmp_path / "filing.json").write_text(json.dumps({"page_content": table,
        "metadata": {**SOURCE, "section": "Financial Statements and Supplementary Data"}}))
    retriever = SimpleNamespace(context_builder=SimpleNamespace(parent_store=SimpleNamespace(root=tmp_path)),
        retrieve=Mock(side_effect=AssertionError("The complete exact table should avoid general search.")))
    with patch("src.agent.graph.get_llm_engine", return_value=model), \
         patch("src.generation.evidence_answer.get_llm_engine", side_effect=AssertionError("No explanation was requested.")), \
         patch("src.financial.fact_extractor.FinancialFactExtractor.extract", side_effect=AssertionError("No model extraction needed.")):
        plan = build_request_plan(question, SCOPE)
        locked = "Application-resolved scope (locked): " + json.dumps({"requested_groups": [["ACME", "2025"]], "doc_type": "10K"})
        state = create_finrag_agent(retriever, FinancialCalculator(), permitted_scope=SCOPE,
            plan_builder=lambda *args: plan).invoke({"messages": [("user", locked + "\nUser question: " + question)]})
    from langchain_core.messages import ToolMessage
    final = next(json.loads(message.content) for message in reversed(state["messages"])
        if isinstance(message, ToolMessage) and message.name == "finalize_financial_answer")
    assert len(plan.requirements) == 1 and plan.requirements[0].kind == "fact"
    assert final["complete"] and "120 million USD" in final["answer"], final
    model.with_structured_output.return_value.invoke.assert_called_once()
    retriever.retrieve.assert_not_called()


def test_explicit_explanation_is_preserved_alongside_the_same_numeric_metric():
    draft = dict(requested_objects=[
        dict(object_id="o1", quote="total revenue", subject="total revenue", kind="numeric_value"),
        dict(object_id="o2", quote="explain why it changed", subject="total revenue", kind="explanation")],
        requirements=[REQUIREMENT, {**REQUIREMENT, "requirement_id": "r2", "object_id": "o2"}])
    with patch("src.agent.graph.get_llm_engine", return_value=SimpleNamespace(
            with_structured_output=lambda *a, **k: SimpleNamespace(invoke=lambda _: draft))):
        plan = build_request_plan("Report ACME total revenue for FY2025 and explain why it changed, using its 2025 10-K.", SCOPE)
    assert [(item.object_id, item.kind) for item in plan.requirements] == [("o1", "fact"), ("o2", "narrative")]


@pytest.mark.parametrize("field", ["object_id", "subject"])
def test_requirement_deduplication_preserves_distinct_outputs_and_businesses(field):
    first = {**REQUIREMENT, "kind": "narrative", "subject": "Widget", "evidence_question": "Explain the operating drivers."}
    second = {**first, "requirement_id": "r2", field: "o2" if field == "object_id" else "Gadget"}
    plan = RequestPlanInput.model_validate(dict(requirements=[first, second]))
    assert len(plan.requirements) == 2
    assert RequestPlanInput.model_validate(plan.model_dump()) == plan


def test_paraphrased_shared_clause_does_not_block_an_otherwise_valid_evidence_plan():
    draft = dict(requested_objects=[dict(object_id="o1", quote="ACME's annual total revenue", subject="total revenue",
        kind="numeric_value")], requirements=[REQUIREMENT])
    engine = Mock()
    engine.with_structured_output.return_value.invoke.return_value = draft
    with patch("src.agent.graph.get_llm_engine", return_value=engine):
        plan = build_request_plan("Report ACME total revenue for FY2025 using its 2025 10-K.", SCOPE)
    assert plan.requirements[0].kind == "fact" and plan.requirements[0].reported_period == "FY2025"
    engine.with_structured_output.return_value.invoke.assert_called_once()


@pytest.mark.parametrize("proof,attached,complete", [
    ({"evidence_spans": ["Capital spending will increase."], "unsupported_details": []}, "S1", False),
    ({"evidence_spans": [], "unsupported_details": []}, "S1", False),
    ({"evidence_spans": ["Capital spending will increase."], "unsupported_details": []}, "S2", True),
    ({"evidence_spans": ["Capital spending will increase."],
      "unsupported_details": ["The point promises a guaranteed return absent from the evidence."]}, "S2", False),
])
def test_blank_or_unbound_proof_cannot_be_overridden_by_a_positive_coverage_verdict(proof, attached, complete):
    sources = {"S1": {**SOURCE, "evidence_text": "Minimum liquidity is required under derivative counterparty agreements."},
               "S2": {**SOURCE, "id": "S2", "evidence_text": "Capital spending will increase."}}
    answer = RequirementAnswer(requirement_id="r1", quotes=[EvidenceQuote(source_id=attached,
        excerpt=sources[attached]["evidence_text"])], draft_points=[NarrativePoint(
            heading="Investment plans", text="Capital spending will increase.", quote_indices=[0])])
    audit = {"assessments": {"r1": {"reason": "The requested topic is covered.", "supported": True,
        "point_support": {"0": proof}}}}
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=SimpleNamespace(
            with_structured_output=lambda *a, **k: SimpleNamespace(invoke=lambda _: audit))):
        result = render_evidence_answer({"requirements": [{**REQUIREMENT, "kind": "narrative", "metric": "investment plans"}]},
            [answer], [], {}, sources, {}, {})
    assert result["complete"] is complete
    assert ("Capital spending will increase." in result["answer"]) is complete
    assert result["cited_source_ids"] == (["S2"] if complete else [])


def test_a_point_cannot_borrow_another_points_evidence_within_the_same_requirement():
    texts = ["Capital spending will increase.", "Customer demand remains strong."]
    source = {**SOURCE, "evidence_text": " ".join(texts)}
    answer = RequirementAnswer(requirement_id="r1", quotes=[EvidenceQuote(source_id="S1", excerpt=text) for text in texts],
        draft_points=[NarrativePoint(heading="Investment", text=texts[0], quote_indices=[0]),
                      NarrativePoint(heading="Demand", text=texts[1], quote_indices=[1])])
    audit = {"assessments": {"r1": {"reason": "Both topics are covered.", "supported": True,
        "point_support": {"0": {"evidence_spans": [texts[0]], "unsupported_details": []},
                          "1": {"evidence_spans": [texts[0]], "unsupported_details": []}}}}}
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=SimpleNamespace(
            with_structured_output=lambda *a, **k: SimpleNamespace(invoke=lambda _: audit))):
        result = render_evidence_answer({"requirements": [{**REQUIREMENT, "kind": "narrative"}]},
            [answer], [], {}, {"S1": source}, {}, {})
    assert texts[0] in result["answer"] and texts[1] not in result["answer"]


def test_literal_proof_accepts_whitespace_normalization_without_changing_the_claim():
    text = "Management expects investment\n to grow with demand."
    answer = RequirementAnswer(requirement_id="r1", quotes=[EvidenceQuote(source_id="S1", excerpt=text)],
        draft_points=[NarrativePoint(heading="Investment", text="Management expects investment to grow with demand.", quote_indices=[0])])
    audit = {"assessments": {"r1": {"reason": "The conditional expectation is preserved.", "supported": True,
        "point_support": {"0": {"evidence_spans": [text], "unsupported_details": []}}}}}
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=SimpleNamespace(
            with_structured_output=lambda *a, **k: SimpleNamespace(invoke=lambda _: audit))):
        result = render_evidence_answer({"requirements": [{**REQUIREMENT, "kind": "narrative"}]},
            [answer], [], {}, {"S1": {**SOURCE, "evidence_text": text}}, {}, {})
    assert result["complete"] and "expects investment to grow with demand" in result["answer"]


def test_audit_schema_offers_only_the_points_stored_evidence_and_presents_it_first():
    text = "Minimum liquidity is required. Collateral may be posted."
    invented = "Capital expenditure will expand production."
    answer = RequirementAnswer(requirement_id="r1", quotes=[EvidenceQuote(source_id="S1", excerpt=text)],
        draft_points=[NarrativePoint(heading="Investment plans", text=invented, quote_indices=[0])])
    engine = Mock()
    engine.with_structured_output.return_value.invoke.return_value = {"assessments": {"r1": {
        "reason": "Liquidity provisions do not establish expansion plans.", "supported": False,
        "point_support": {"0": {"evidence_spans": [], "unsupported_details": ["Expansion plans are absent."]}}}}}
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=engine):
        result = render_evidence_answer({"requirements": [{**REQUIREMENT, "kind": "narrative"}]},
            [answer], [], {}, {"S1": {**SOURCE, "evidence_text": text}}, {}, {})
    schema = engine.with_structured_output.call_args.args[0]
    choices = schema["properties"]["assessments"]["properties"]["r1"]["properties"]["point_support"]["properties"]["0"]["properties"]["evidence_indices"]["items"]
    assert choices == {"type": "integer", "minimum": 0, "maximum": 1}
    assert not result["complete"]
    import json
    payload = json.loads(engine.with_structured_output.return_value.invoke.call_args.args[0][-1].content)
    assert next(iter(payload["claims"][0]["draft_points"][0])) == "evidence"
    assert list(payload["claims"][0]["draft_points"][0]["evidence"][0]["text"].values()) == ["Minimum liquidity is required.", "Collateral may be posted."]


def test_numeric_checks_run_before_interpretation_and_coverage_review():
    import json
    from src.generation.evidence_answer import EvidenceSynthesis
    texts = ["Capital spending was $100 million.", "Operations require trained staff."]
    answer = RequirementAnswer(requirement_id="r1", quotes=[EvidenceQuote(source_id="S1", excerpt=text) for text in texts],
        draft_points=[NarrativePoint(heading="Staff", text=texts[1], quote_indices=[1]),
                      NarrativePoint(heading="Investment", text="Capital spending was $999 million.", quote_indices=[0])])
    payloads = []
    def structured(schema, **kwargs):
        def respond(messages):
            payloads.append(json.loads(messages[-1].content))
            if schema is EvidenceSynthesis:
                return {"interpretation": "Operations require trained staff.", "source_ids": ["S1"]}
            return {"assessments": {key: {"reason": "The retained staff observation is supported.", "supported": True,
                "point_support": {"0": {"evidence_spans": [texts[1]], "unsupported_details": []}}}
                for key in ("r1", "synthesis")}}
        return SimpleNamespace(invoke=respond)
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=SimpleNamespace(with_structured_output=structured)):
        result = render_evidence_answer({"requirements": [{**REQUIREMENT, "kind": "narrative"}],
            "synthesis": "Interpret the operational requirements."}, [answer], [], {},
            {"S1": {**SOURCE, "evidence_text": " ".join(texts)}}, {}, {})
    assert result["complete"] and "999" not in result["answer"]
    assert len(payloads) == 2 and all("999" not in json.dumps(payload) for payload in payloads)
