"""Computed outputs reuse source observations without inventing retrieval work."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.agent.graph import build_request_plan
from src.agent.tools import RequestPlanInput
from src.schemas import Scope


@pytest.mark.parametrize("temporal", [False, True])
def test_computed_output_does_not_add_a_lookup_but_keeps_its_operands_and_explanation(temporal):
    scope = Scope(("ACME",) if temporal else ("ACME", "ORBIT"),
                  ("2027", "2028") if temporal else ("2028",), "TRANSCRIPT" if temporal else "10K")
    draft = dict(synthesis=False, subjects=[], requested_objects=[
        dict(object_id="o1", kind="numeric_value", quote="reported balances", subject="contract balance"),
        dict(object_id="o2", kind="calculation", quote="the increase" if temporal else "the difference", subject="balance difference"),
        dict(object_id="o3", kind="explanation", quote="management's explanation", subject="balance drivers")],
        requirements=[
            dict(requirement_id=key, object_id="o1", subject_id=None, ticker=ticker, source_year=year,
                 doc_type=scope.doc_type, metric="contract balance", reported_period=f"FY{year}")
            for key, ticker, year in [("r1", "ACME", "2027" if temporal else "2028"),
                                      ("r2", "ACME" if temporal else "ORBIT", "2028")]],
        calculations=[dict(object_id="o2", operation="absolute_change" if temporal else "difference",
                           input_requirement_ids=["r1", "r2"])])
    draft["requirements"].extend([
        {**draft["requirements"][1], "requirement_id": "r3", "object_id": "o2", "metric": "balance difference"},
        {**draft["requirements"][1], "requirement_id": "r4", "object_id": "o3", "metric": "balance drivers"}])
    planner = SimpleNamespace(invoke=Mock(side_effect=lambda messages: deepcopy(draft)))
    with patch("src.agent.graph.get_llm_engine", return_value=SimpleNamespace(
            with_structured_output=lambda *a, **k: planner)):
        plan = build_request_plan("Compare the reported contract balances, state the numerical difference and management's explanation.", scope)
    assert planner.invoke.call_count == 1
    assert [item.requirement_id for item in plan.requirements] == ["r1", "r2", "r4"]
    assert [item.kind for item in plan.requirements] == ["fact", "fact", "narrative"]
    assert plan.calculations[0].input_requirement_ids == ["r1", "r2"]
    assert RequestPlanInput.model_validate(plan.model_dump()) == plan


def test_operand_aliases_are_resolved_without_losing_the_reported_output():
    common = dict(object_id="o1", ticker="ACME", source_year="2028", doc_type="10K", metric="contract balance", kind="fact")
    plan = RequestPlanInput.model_validate(dict(requested_objects=[
        dict(object_id="o1", kind="fact", quote="both balances", subject="contract balance"),
        dict(object_id="o2", kind="calculation", quote="the increase", subject="balance increase")],
        requirements=[dict(common, requirement_id="first", reported_period="FY2027"),
                      dict(common, requirement_id="duplicate", reported_period="FY2027"),
                      dict(common, requirement_id="last", reported_period="FY2028")],
        calculations=[dict(object_id="o2", operation="absolute_change", input_requirement_ids=["duplicate", "last"])]))
    assert [item.requirement_id for item in plan.requirements] == ["first", "last"]
    assert plan.calculations[0].input_requirement_ids == ["first", "last"]


@pytest.mark.parametrize("explanation", [False, True])
def test_numeric_outputs_cannot_invent_interpretation_but_requested_explanations_survive(explanation):
    draft = dict(synthesis=True, subjects=[], requested_objects=[
        dict(object_id="o1", kind="numeric_value", quote="the balance", subject="contract balance")],
        requirements=[dict(requirement_id="r1", object_id="o1", ticker="ACME", source_year="2028",
                           doc_type="10K", metric="contract balance", reported_period="FY2028")])
    if explanation:
        draft["requested_objects"].append(dict(object_id="o2", kind="explanation", quote="explain the drivers", subject="drivers"))
        draft["requirements"].append({**draft["requirements"][0], "requirement_id": "r2", "object_id": "o2", "metric": "drivers"})
    planner = SimpleNamespace(invoke=Mock(return_value=draft))
    question = "Report the contract balance" + (" and explain the drivers." if explanation else ".")
    with patch("src.agent.graph.get_llm_engine", return_value=SimpleNamespace(
            with_structured_output=lambda *a, **k: planner)):
        plan = build_request_plan(question, Scope(("ACME",), ("2028",), "10K"))
    assert plan.synthesis == (question if explanation else None)
