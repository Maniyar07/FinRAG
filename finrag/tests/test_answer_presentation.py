from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.generation.citations import compact_citations
from src.generation.evidence_answer import (
    EvidenceQuote, NarrativePoint, RequirementAnswer, display_value,
    evidence_text, presentation_text, render_evidence_answer, resolve_evidence_answers,
)


SOURCE = {"id": "S9876543210987654321", "ticker": "ACME", "fiscal_year": "2024", "doc_type": "10K"}
REQUIREMENT = {"requirement_id": "r1", "ticker": "ACME", "source_year": "2024", "reported_period": "FY2024",
               "doc_type": "10K", "metric": "risk factors", "kind": "narrative", "qualifiers": ["risk factors"]}


def test_response_citations_are_short_ordered_collision_safe_and_preserve_identity():
    sources = [{"id": "S1", "evidence_text": "Uncited"}, {"id": SOURCE["id"], "evidence_text": "Cited"}]
    original = deepcopy(sources)
    answer, views = compact_citations(f"First [{SOURCE['id']}]. Second [S1]. Repeat [{SOURCE['id']}].", sources)
    assert answer == "First [S1]. Second [S2]. Repeat [S1]."
    assert [(v["id"], v["internal_id"], v["evidence_text"]) for v in views] == [
        ("S1", SOURCE["id"], "Cited"), ("S2", "S1", "Uncited")]
    assert sources == original
    assert compact_citations(answer, views) == (answer, views)


def test_uncited_evidence_is_retained_after_cited_sources():
    answer, views = compact_citations("A claim [S9].", [{"id": "S8"}, {"id": "S9"}, {"id": "S8"}])
    assert answer == "A claim [S1]."
    assert [(v["id"], v["internal_id"]) for v in views] == [("S1", "S9"), ("S2", "S8")]


def test_an_unknown_citation_cannot_be_renamed_into_a_valid_source():
    with pytest.raises(ValueError, match="absent"):
        compact_citations("Unknown [S2].", [{"id": "S987654321"}])


@pytest.mark.parametrize("value,unit,scale,currency,expected", [
    ("245122", "USD", "millions", "USD", "245,122 million USD"),
    ("-176799.00", "currency", "millions", "USD", "-176,799.00 million USD"),
    ("14.93", "percent", None, None, "14.93 percent"),
    ("1.2500", "ratio", None, None, "1.2500 ratio"),
])
def test_numeric_formatting_preserves_values_and_precision(value, unit, scale, currency, expected):
    assert display_value(value, unit, scale, currency) == expected


def test_source_styling_is_removed_without_changing_math_or_ampersands():
    assert evidence_text("Cash paid for P&E") == "Cash paid for P&E"
    assert presentation_text("### **Competition** _may_ reduce revenue.") == "Competition may reduce revenue."
    assert presentation_text("2*3") == "2\\*3"
    assert presentation_text("[S99]") == "\\[S99\\]"


def test_narrative_points_are_grouped_cited_and_independently_verified():
    quotes = ["## RISK FACTORS", "Competition may reduce sales volumes.", "Taylor.",
              "Acquisitions may disrupt operations."]
    sources = {SOURCE["id"]: {**SOURCE, "evidence_text": "\n\n".join(quotes)}}
    points = [NarrativePoint(heading="Competition", text="Competition may reduce sales volumes.", quote_indices=[1]),
              NarrativePoint(heading="Acquisitions", text="Acquisitions may disrupt operations.", quote_indices=[3])]
    answer = RequirementAnswer(requirement_id="r1", quotes=[EvidenceQuote(source_id=SOURCE["id"], excerpt=q) for q in quotes], draft_points=points)
    audit = {"assessments": {"r1": {"reason": "The two complete statements answer the request", "missing_details": [],
             "point_support": {"0": True, "1": True}, "quote_assessments": {
                 str(index): {"reason": "Statement" if index in (1, 3) else "Heading or fragment",
                              "answers_question": index in (1, 3)} for index in range(4)}}}}
    verifier = Mock(return_value=audit)
    model = SimpleNamespace(with_structured_output=Mock(return_value=SimpleNamespace(invoke=verifier)))
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=model):
        result = render_evidence_answer({"requirements": [REQUIREMENT]}, [answer], [], {}, sources, {}, {})
    assert result["complete"]
    assert "### ACME - FY2024 (10-K)" in result["answer"]
    assert "- **Competition:** Competition may reduce sales volumes." in result["answer"]
    assert "Taylor." not in result["answer"] and "## RISK FACTORS" not in result["answer"]
    assert result["answer"].count("risk factors") == 1
    assert result["answer"].count(f"[{SOURCE['id']}]") == 2
    assert verifier.call_count == 1
    assert "draft_points" in verifier.call_args.args[0][1].content


@pytest.mark.parametrize("supported,number,quote_index", [(False, "100", 0), (True, "999", 0), (True, "100", 1)])
def test_unsupported_points_numbers_and_rejected_quote_references_are_not_displayed(supported, number, quote_index):
    source = {**SOURCE, "evidence_text": "Costs may rise by $100 million. Taylor."}
    answer = RequirementAnswer(requirement_id="r1", quotes=[EvidenceQuote(source_id=SOURCE["id"], excerpt=text)
        for text in ("Costs may rise by $100 million.", "Taylor.")],
        draft_points=[NarrativePoint(heading="Costs", text=f"Costs may rise by ${number} million.", quote_indices=[quote_index])])
    audit = {"assessments": {"r1": {"reason": "Only the actual cost statement is responsive", "missing_details": [],
             "point_support": {"0": supported}, "quote_assessments": {
                 "0": {"reason": "Cost statement", "answers_question": True},
                 "1": {"reason": "Fragment", "answers_question": False}}}}}
    model = SimpleNamespace(with_structured_output=lambda *args, **kwargs: SimpleNamespace(invoke=lambda messages: audit))
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=model):
        result = render_evidence_answer({"requirements": [REQUIREMENT]}, [answer], [], {}, {SOURCE["id"]: source}, {}, {})
    assert result["complete"]
    assert "Costs may rise by $100 million." in result["answer"]
    assert "$999" not in result["answer"] and "Taylor." not in result["answer"]


def test_quote_audit_is_authoritative_for_responsive_company_actor():
    quote = "ACME and its partner jointly deliver the hosted service."
    source = {**SOURCE, "evidence_text": quote}
    requirement = {**REQUIREMENT, "subject": "ACME partnership"}
    answer = RequirementAnswer(requirement_id="r1", quotes=[EvidenceQuote(
        source_id=SOURCE["id"], excerpt=quote)])
    audit = {"assessments": {"r1": {
        "reason": "The company is the correct actor for the requested partnership.",
        "missing_details": [],
        "quote_assessments": {"0": {
            "source_subject": "ACME", "source_subject_type": "company",
            "reason": "The statement directly answers the partnership question.",
            "scope_matches": True, "answers_question": True,
        }},
    }}}
    model = SimpleNamespace(with_structured_output=lambda *args, **kwargs: SimpleNamespace(
        invoke=lambda messages: audit))
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=model):
        result = render_evidence_answer({"requirements": [requirement]}, [answer], [], {},
                                        {SOURCE["id"]: source}, {}, {})
    assert result["complete"]
    assert quote in result["answer"]


def test_selection_remaps_point_references_after_rejecting_an_invented_quote():
    source = {**SOURCE, "evidence_text": "Competition may reduce sales volumes."}
    selection = {"answers": {"r1": {"quotes": [{"source_id": SOURCE["id"], "excerpt": "An invented quote."},
        {"source_id": SOURCE["id"], "excerpt": source["evidence_text"]}], "draft_points": [
        {"heading": "Competition", "text": source["evidence_text"], "quote_indices": [1]}]}}}
    model = SimpleNamespace(with_structured_output=lambda *args, **kwargs: SimpleNamespace(invoke=lambda messages: selection))
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=model):
        answer = resolve_evidence_answers({"requirements": [REQUIREMENT]}, [], {}, {SOURCE["id"]: source}, {})[0]
    assert len(answer.quotes) == 1 and answer.draft_points[0].quote_indices == [0]


def test_selection_and_independent_point_verification_use_the_two_existing_calls():
    source = {**SOURCE, "evidence_text": "Competition may reduce sales volumes."}
    point = {"heading": "Competition", "text": source["evidence_text"], "quote_indices": [0]}
    selector = Mock(return_value={"answers": {"r1": {"quotes": [{"source_id": SOURCE["id"], "excerpt": source["evidence_text"]}],
        "draft_points": [point]}}})
    verifier = Mock(return_value={"assessments": {"r1": {"reason": "The point faithfully summarizes the quote", "missing_details": [],
        "quote_assessments": {"0": {"reason": "Substantive statement", "answers_question": True}}, "point_support": {"0": True}}}})
    model = SimpleNamespace(with_structured_output=Mock(side_effect=[SimpleNamespace(invoke=selector), SimpleNamespace(invoke=verifier)]))
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=model):
        plan = {"requirements": [REQUIREMENT]}
        answers = resolve_evidence_answers(plan, [], {}, {SOURCE["id"]: source}, {})
        result = render_evidence_answer(plan, answers, [], {}, {SOURCE["id"]: source}, {}, {})
    assert result["complete"] and "**Competition:**" in result["answer"]
    assert model.with_structured_output.call_count == 2
    assert selector.call_count == verifier.call_count == 1


def test_multiple_requirements_share_a_company_heading_and_companies_remain_separate():
    source = {**SOURCE, "evidence_text": "Competition may reduce sales volumes."}
    other = {**source, "id": "S1", "ticker": "BETA"}
    requirements = [REQUIREMENT, {**REQUIREMENT, "requirement_id": "r2", "metric": "competition"},
                    {**REQUIREMENT, "requirement_id": "r3", "ticker": "BETA"}]
    answers = [RequirementAnswer(requirement_id=r["requirement_id"], quotes=[EvidenceQuote(
        source_id=other["id"] if r["ticker"] == "BETA" else source["id"], excerpt=source["evidence_text"])],
        draft_points=[NarrativePoint(heading="Competition", text=source["evidence_text"], quote_indices=[0])]) for r in requirements]
    audit = {"assessments": {r["requirement_id"]: {"reason": "The respective filing reports competition", "missing_details": [],
        "quote_assessments": {"0": {"reason": "Responsive statement", "answers_question": True}}, "point_support": {"0": True}} for r in requirements}}
    model = SimpleNamespace(with_structured_output=lambda *args, **kwargs: SimpleNamespace(invoke=lambda messages: audit))
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=model):
        result = render_evidence_answer({"requirements": requirements}, answers, [], {}, {source["id"]: source, other["id"]: other}, {}, {})
    assert result["answer"].count("### ACME") == result["answer"].count("### BETA") == 1
    text, sources = compact_citations(result["answer"], [other, source])
    assert sources[0]["ticker"] == "ACME" and sources[1]["ticker"] == "BETA"
    assert "[S1]" in text.split("### BETA")[0] and "[S2]" in text.split("### BETA")[1]


@pytest.mark.parametrize("new_value", ["100", "120"])
def test_comparison_table_preserves_periods_same_value_rows_and_calculator_details(new_value):
    from src.financial.models import ValidatedFinancialFact
    from src.financial.calculator import CalculationRequest, FinancialCalculator
    from src.generation.answer_guardrails import validate_answer_payload
    source = {**SOURCE, "fiscal_year": "2025", "evidence_text": "Revenue was reported in millions."}
    requirements = [{**REQUIREMENT, "requirement_id": f"r{year}", "kind": "fact", "metric": "revenue",
                     "source_year": "2025", "reported_period": f"FY{year}", "qualifiers": []} for year in ("2024", "2025")]
    facts = {f"F{year}": ValidatedFinancialFact.model_validate({
        "fact_id": f"F{year}", "ticker": "ACME", "metric": "revenue", "period": f"FY{year}",
        "raw_value": value, "numeric_value": value, "base_value": str(int(value) * 1000000),
        "value_type": "currency", "normalized_unit": "USD", "currency": "USD", "scale": "millions",
        "source_id": source["id"], "evidence_excerpt": f"Revenue {year} {value}", "validation_checks": [],
    }) for year, value in (("2024", "100"), ("2025", new_value))}
    request = CalculationRequest(operation="percentage_change", input_fact_ids=["F2024", "F2025"], decimal_places=2)
    calculated = FinancialCalculator().calculate(request, facts).model_dump(mode="json")
    calculated["planned_input_requirement_ids"] = ["r2024", "r2025"]
    calculated["decimal_places"] = request.decimal_places
    plan = {"requirements": requirements, "calculations": [{"operation": "percentage_change",
        "input_requirement_ids": ["r2024", "r2025"], "periods": None, "decimal_places": 2}]}
    answers = [RequirementAnswer(requirement_id=f"r{year}", fact_ids=[f"F{year}"]) for year in ("2024", "2025")]
    result = render_evidence_answer(plan, answers, ["C1"], facts, {source["id"]: source},
                                    {"C1": calculated}, {})
    assert result["complete"]
    assert "### Reported figures" in result["answer"]
    assert "| ACME | FY2024 |" in result["answer"] and "| ACME | FY2025 |" in result["answer"]
    assert result["answer"].count("2025 10-K") == 2
    assert "**Formula:**" in result["answer"] and "**Calculator substitution:**" in result["answer"]
    assert str(calculated["result"]) in result["answer"]
    assert validate_answer_payload(result["answer"], [source]).valid


def test_unspecified_currency_is_not_silently_removed_from_display():
    assert display_value("1000", "currency", None, None) == "1,000 currency"
