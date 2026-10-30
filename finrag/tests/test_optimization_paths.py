import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from src.agent.graph import create_finrag_agent
from src.agent.tools import RequestPlanInput, create_financial_tools
from src.financial.calculator import FinancialCalculator
from src.financial.fact_pipeline import FinancialFactPipeline
from src.schemas import RetrievalBundle, Scope


SCOPE = Scope(("ACME",), ("2025",), "10K")
SOURCE = {"id": "S1", "ticker": "ACME", "fiscal_year": "2025", "doc_type": "10K",
          "source": "filing.pdf", "section": "Financial Statements and Supplementary Data"}


def request(requirements):
    plan = RequestPlanInput.model_validate({"requirements": requirements})
    locked = {"requested_groups": [["ACME", "2025"]], "doc_type": "10K"}
    return plan, {"messages": [("user", "Application-resolved scope (locked): " + json.dumps(locked)
                              + "\nUser question: Report the requested source observations.")]}


@pytest.mark.parametrize("mode", ["exact", "ambiguous", "missing", "budget"])
def test_exact_indexed_facts_bypass_search_only_when_every_value_is_unambiguous(tmp_path, mode):
    body = "## INCOME STATEMENTS\n(In millions)\n<table><tr><th>Metric</th><th>2024</th></tr><tr><td>Total revenue</td><td>100</td></tr></table>"
    if mode == "ambiguous":
        body += "\n## INCOME STATEMENTS\n(In millions)\n<table><tr><th>Metric</th><th>2024</th></tr><tr><td>Total revenue</td><td>200</td></tr></table>"
    (tmp_path / "filing.json").write_text(json.dumps({"page_content": body, "metadata": SOURCE}), encoding="utf-8")
    fallback = RetrievalBundle("", [], SCOPE, 0)
    retriever = SimpleNamespace(context_builder=SimpleNamespace(parent_store=SimpleNamespace(root=tmp_path)),
        retrieve=Mock(return_value=fallback))
    pipeline = FinancialFactPipeline(extractor=SimpleNamespace(extract=Mock(return_value=())))
    requirements = [{"requirement_id": "r1", "ticker": "ACME", "source_year": "2025",
        "doc_type": "10K", "kind": "fact", "metric": "total revenue", "reported_period": "FY2024"}]
    if mode == "missing":
        requirements.append({**requirements[0], "requirement_id": "r2", "metric": "net income"})
    plan, inputs = request(requirements)
    with patch("src.agent.tools.FinancialFactPipeline", return_value=pipeline), \
         patch("src.agent.tools.MAX_CONTEXT_CHARS", 1 if mode == "budget" else 48000):
        graph = create_finrag_agent(retriever, FinancialCalculator(), permitted_scope=SCOPE, plan_builder=lambda *args: plan)
        state = graph.invoke(inputs)
    final = next(json.loads(message.content) for message in reversed(state["messages"])
        if isinstance(message, ToolMessage) and message.name == "finalize_financial_answer")
    if mode != "exact":
        assert retriever.retrieve.called and not final["complete"]
    else:
        retriever.retrieve.assert_not_called()
        pipeline.extractor.extract.assert_not_called()
        assert final["complete"] and "100 million USD" in final["answer"]


@pytest.mark.parametrize("still_supported", [True, False])
def test_recovery_preserves_unchanged_verified_requirements_and_audits_new_evidence(still_supported):
    cost, outlook = "Factories require substantial investment.", "Management expects capacity to expand next year."
    plan, inputs = request([{"requirement_id": key, "ticker": "ACME", "source_year": "2025",
        "doc_type": "10K", "kind": "narrative", "metric": metric}
        for key, metric in (("r1", "investment requirements"), ("r2", "capacity outlook"))])
    retriever = SimpleNamespace(retrieve=Mock(side_effect=[
        RetrievalBundle("[SOURCE S1 | TICKER: ACME | FISCAL YEAR: 2025 | TYPE: 10K]\n" + cost,
            [{**SOURCE, "parent_id": "first"}], SCOPE, 1),
        RetrievalBundle("[SOURCE S1 | TICKER: ACME | FISCAL YEAR: 2025 | TYPE: 10K]\n" + outlook,
            [{**SOURCE, "parent_id": "second"}], SCOPE, 1)]))
    engine = Mock()
    engine.with_structured_output.return_value.invoke.side_effect = [
        {"answers": {"r1": {"draft_points": [{"heading": "Investment", "text": cost, "passage_ids": ["P1"]}]},
                     "r2": {"unavailable_reason": "The outlook is absent."}}},
        {"assessments": {"r1": {"reason": "Investment is disclosed.", "supported": True,
            "point_support": {"0": {"evidence_spans": [cost], "unsupported_details": []}}}}},
        {"answers": {"r2": {"draft_points": [{"heading": "Outlook", "text": outlook, "passage_ids": ["P2"]}]}}},
        {"assessments": {key: {"reason": "Responsive stored evidence.", "supported": still_supported,
            "point_support": {"0": {"evidence_spans": [text], "unsupported_details": [] if still_supported else ["Scope is unresolved."]}}}
                         for key, text in (("r2", outlook),)}}]
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=engine):
        state = create_finrag_agent(retriever, FinancialCalculator(), permitted_scope=SCOPE,
            plan_builder=lambda *args: plan).invoke(inputs)
    final = next(json.loads(message.content) for message in reversed(state["messages"])
        if isinstance(message, ToolMessage) and message.name == "finalize_financial_answer")
    assert final["complete"] is still_supported, final
    if still_supported:
        assert cost in final["answer"] and outlook in final["answer"]
    else:
        assert final['verified_answers'][0]['requirement_id'] == 'r1'
        assert 'r2' in final['missing_requirement_ids'] and 'r1' not in final['missing_requirement_ids']
    schemas = [call.args[0] for call in engine.with_structured_output.call_args_list]
    assert schemas[2]["properties"]["answers"]["required"] == ["r2"]
    assert schemas[3]["properties"]["assessments"]["required"] == ["r2"]
    assert retriever.retrieve.call_count == 2


def test_evidence_recovery_does_not_consume_the_validation_repair_allowance():
    plan, inputs = request([{"requirement_id": "r1", "ticker": "ACME", "source_year": "2025",
        "doc_type": "10K", "kind": "fact", "metric": "revenue", "reported_period": "FY2024"}])
    tools = {tool.name: tool for tool in create_financial_tools(SimpleNamespace(), FinancialCalculator(),
        permitted_scope=SCOPE, plan_builder=lambda *args: plan)}
    runtime = SimpleNamespace(state={"messages": [AIMessage(content="", tool_calls=[
        {"name": "plan_financial_request", "args": {}, "id": "plan", "type": "tool_call"}])]})
    runtime.state["messages"].append(ToolMessage(name="plan_financial_request", tool_call_id="plan",
        content=tools["plan_financial_request"].func(runtime=runtime)))
    finalize = tools["finalize_financial_answer"].func
    recovery = finalize(runtime=runtime)
    assert json.loads(recovery)["final_answer_attempts"] == 0
    runtime.state["messages"].append(ToolMessage(name="finalize_financial_answer", tool_call_id="first", content=recovery))
    with patch("src.agent.tools.render_evidence_answer", side_effect=ValueError("Invalid candidate shape.")):
        failure = finalize(runtime=runtime)
    assert not json.loads(failure)["final_answer_attempt_exhausted"]
    runtime.state["messages"].append(ToolMessage(name="finalize_financial_answer", tool_call_id="repair", content=failure))
    result = json.loads(finalize(runtime=runtime))
    assert result["answer_validated"] and result["final_answer_attempts"] == 2


def test_interpretation_only_repair_reuses_evidence_without_another_search():
    from src.generation.evidence_answer import EvidenceSynthesis
    cost = "Factories require substantial investment."
    plan, inputs = request([{"requirement_id": "r1", "ticker": "ACME", "source_year": "2025",
        "doc_type": "10K", "kind": "narrative", "metric": "investment requirements"}])
    plan = plan.model_copy(update={"synthesis": "Interpret the investment requirements."})
    retriever = SimpleNamespace(retrieve=Mock(return_value=RetrievalBundle(
        "[SOURCE S1 | TICKER: ACME | FISCAL YEAR: 2025 | TYPE: 10K]\n" + cost,
        [{**SOURCE, "parent_id": "first"}], SCOPE, 1)))
    counts = dict(selection=0, synthesis=0, audit=0)
    audited = []
    def respond(schema, messages):
        payload = json.loads(messages[-1].content)
        if schema is EvidenceSynthesis:
            counts["synthesis"] += 1
            return {"interpretation": "Expansion requires no investment." if counts["synthesis"] == 1 else cost,
                    "source_ids": payload["available_source_ids"]}
        if "answers" in schema["properties"]:
            counts["selection"] += 1
            return {"answers": {"r1": {"draft_points": [{"heading": "Investment", "text": cost, "passage_ids": ["P1"]}]}}}
        counts["audit"] += 1
        keys = [claim['requirement']['requirement_id'] for claim in payload['claims']]
        audited.append(keys)
        return {"assessments": {key: {"reason": "Compare investment with the retained source.",
            "supported": key != "synthesis" or counts["audit"] > 1,
            "point_support": {"0": {"evidence_spans": [cost],
                "unsupported_details": ["Investment is required."] if key == "synthesis" and counts["audit"] == 1 else []}}}
            for key in keys}}
    engine = SimpleNamespace(with_structured_output=lambda schema, **kw:
        SimpleNamespace(invoke=lambda messages: respond(schema, messages)))
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=engine):
        state = create_finrag_agent(retriever, FinancialCalculator(), permitted_scope=SCOPE,
            plan_builder=lambda *args: plan).invoke(inputs)
    finals = [json.loads(message.content) for message in state["messages"]
        if isinstance(message, ToolMessage) and message.name == "finalize_financial_answer"]
    assert len(finals) == 2 and not finals[0].get("recovery_required")
    assert finals[-1]["complete"] and finals[-1]["final_answer_attempts"] == 2
    assert "Expansion requires no investment" not in finals[-1]["answer"]
    assert counts == dict(selection=1, synthesis=2, audit=2)
    assert audited == [['r1', 'synthesis'], ['synthesis']]
    retriever.retrieve.assert_called_once()
