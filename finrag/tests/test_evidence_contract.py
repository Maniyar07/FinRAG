from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage

from src.agent.graph import build_request_plan, compact_agent_context, create_finrag_agent
from src.agent.tools import PlannedRequirement, RequestPlanInput, create_financial_tools, _facts_from_tool_messages
from langchain_core.messages import ToolMessage
from src.app.chat_service import ChatService
from src.financial.calculator import FinancialCalculator
from src.financial.coverage import fact_coverage, fact_matches_requirement
from src.financial.fact_pipeline import FinancialFactPipeline
from src.financial.extracted_fact_validator import ExtractedFactValidator
from src.financial.models import CandidateFinancialFact
from src.financial.models import FactValidationResult, ValidatedFinancialFact
from src.generation.evidence_answer import RequirementAnswer, render_evidence_answer, resolve_evidence_answers
from src.generation.evidence_answer import display_value
from src.retrieval.query_understanding import understand_query
from src.retrieval.structured_lookup import StructuredDocumentLookup
from src.schemas import Decision, RetrievalBundle, Scope


def requirement(year="2024", **extra):
    return dict(requirement_id=f"revenue_{year}", ticker="MSFT", source_year="2025",
                doc_type="10K", metric="total revenue", reported_period=f"FY{year}",
                kind="fact", qualifiers=[], **extra)


SOURCE = dict(id="S1", ticker="MSFT", fiscal_year="2025", doc_type="10K")


def fact(year="2024", value="100", **extra):
    return ValidatedFinancialFact.model_validate({
        "fact_id": f"F{year}", "ticker": "MSFT", "metric": "Total revenue",
        "period": f"FY{year}", "raw_value": value, "numeric_value": value,
        "base_value": str(int(value) * 1_000_000), "value_type": "currency",
        "normalized_unit": "USD", "currency": "USD", "scale": "millions",
        "source_id": "S1", "evidence_excerpt": f"Total revenue {year} {value}",
        "row_label": "Total revenue", "column_label": year,
        "validation_checks": [], **extra,
    })


def test_coverage_uses_reported_period_and_source_year_independently():
    assert fact_matches_requirement(fact(), requirement(), SOURCE)
    assert not fact_matches_requirement(fact("2025"), requirement(), SOURCE)
    assert not fact_matches_requirement(fact(), requirement(), {**SOURCE, "fiscal_year": "2024"})
    qualified = {**requirement(), "qualifiers": ["constant currency"]}
    assert not fact_matches_requirement(fact(), qualified, SOURCE)
    assert fact_coverage([requirement()], [], [SOURCE])["missing_requirement_ids"] == ["revenue_2024"]


def test_coverage_cannot_substitute_amount_for_a_growth_rate_of_a_long_metric():
    label = "commercial remaining performance obligation"
    amount = fact().model_copy(update={"metric": label, "row_label": label})
    requested = {**requirement(), "metric": label + " growth rate"}
    assert not fact_matches_requirement(amount, requested, SOURCE)
    mislabelled = amount.model_copy(update={"metric": requested["metric"], "row_label": requested["metric"]})
    assert not fact_matches_requirement(mislabelled, {**requested, "value_type": "percent"}, SOURCE)
    percentage = mislabelled.model_copy(update={"value_type": "percent", "normalized_unit": "percent", "currency": None})
    assert fact_matches_requirement(percentage, {**requested, "value_type": "percent"}, SOURCE)


def test_entirely_omitted_fact_is_recovered_from_existing_table_without_model_retry():
    source = {**SOURCE, "section": "Financial Statements and Supplementary Data",
              "evidence_text": "CONSOLIDATED INCOME STATEMENTS (in millions)\n"
              "<table><tr><th>Metric</th><th>2024</th></tr>"
              "<tr><td>Total revenue</td><td>100</td></tr></table>"}
    extractor = SimpleNamespace(extract=lambda **kwargs: ())
    result = FinancialFactPipeline(extractor=extractor).run(
        question="Revenue", sources=[source], permitted_scope=Scope(("MSFT",), ("2025",), "10K"),
        requirements=(requirement(),),
    )
    assert len(result.valid_facts) == 1
    assert result.valid_facts[0].period == "2024"


@pytest.mark.parametrize("header", [
    "<tr><th>2025</th><th>2024</th></tr>",
    "<tr><th rowspan='2'>Metric</th><th colspan='2'>Fiscal year</th></tr>"
    "<tr><th>2025</th><th>2024</th></tr>",
])
def test_table_recovery_preserves_header_alignment(header):
    html = "<table>" + header + "<tr><td>Total revenue</td><td>120</td><td>100</td></tr></table>"
    source = {**SOURCE, "section": "Financial Statements and Supplementary Data",
              "evidence_text": "CONSOLIDATED INCOME STATEMENTS (in millions)\n" + html}
    result = FinancialFactPipeline().recover_exact_table_row(
        metric_hint="total revenue", period_year="2024", sources=[source],
        permitted_scope=Scope(("MSFT",), ("2025",), "10K"),
    )
    assert len(result.valid_facts) == 1
    assert result.valid_facts[0].numeric_value == 100
    markdown = StructuredDocumentLookup.markdown_table(html)
    assert "| Total revenue | 120 | 100 |" in markdown
    if "rowspan" in header:
        assert "| Metric | 2025 | 2024 |" in markdown


def test_numeric_plan_cannot_omit_reported_period():
    with pytest.raises(ValueError, match="reported period"):
        RequestPlanInput.model_validate({"requirements": [{**requirement(), "reported_period": None}]})


@pytest.mark.parametrize(("metric", "canonical"), [
    ("EPS", "diluted earnings per share"),
    ("earnings per share", "diluted earnings per share"),
    ("basic EPS", "basic earnings per share"),
    ("diluted EPS", "diluted earnings per share"),
    ("adjusted EPS", "adjusted diluted earnings per share"),
    ("total revenue", "total revenue"),
])
def test_numeric_plan_canonicalizes_eps_without_changing_other_metrics(metric, canonical):
    planned = PlannedRequirement.model_validate({**requirement(), "metric": metric})
    assert planned.metric == canonical
    if "earnings per share" in canonical:
        assert planned.value_type == "per_share"


def test_fact_validation_rejects_value_from_another_reported_year_column():
    source = {**SOURCE, "evidence_text": "<table><tr><th>Metric</th><th>2024</th><th>2025</th></tr>"
              "<tr><td>Total revenue</td><td>100</td><td>120</td></tr></table>"}
    candidate = CandidateFinancialFact(ticker="MSFT", metric="Total revenue", period="FY2024",
        raw_value="120", unit="USD", currency="USD", scale="millions", source_id="S1",
        row_label="Total revenue", column_label="2024", evidence_excerpt="Total revenue 2024 120")
    validator = ExtractedFactValidator()
    wrong = validator.validate([candidate], [source])
    assert not wrong.valid_facts
    assert wrong.rejected_facts[0].reason == "table_row_period_value_mismatch"
    correct = validator.validate([candidate.model_copy(update={"raw_value": "100", "evidence_excerpt": "Total revenue 2024 100"})], [source])
    assert len(correct.valid_facts) == 1
    assert "table_row_period_value" in correct.valid_facts[0].validation_checks


def test_numeric_answer_is_rendered_from_facts_and_all_requirements_are_required():
    answer = RequirementAnswer(requirement_id="revenue_2024", fact_ids=["F2024"])
    result = render_evidence_answer({"requirements": [requirement()]}, [answer], [],
                                    {"F2024": fact()}, {"S1": SOURCE}, {}, {})
    assert result["complete"]
    assert "100 million USD [S1]" in result["answer"]
    with pytest.raises(ValueError, match="every planned requirement"):
        render_evidence_answer({"requirements": [requirement(), requirement("2025")]},
                               [answer], [], {"F2024": fact()}, {"S1": SOURCE}, {}, {})
    with pytest.raises(ValueError, match="rendered automatically"):
        render_evidence_answer({"requirements": [requirement()]},
                               [answer.model_copy(update={"summary": "Revenue was 999"})], [],
                               {"F2024": fact()}, {"S1": SOURCE}, {}, {})


def test_unavailable_requirement_produces_explicit_partial_answer():
    result = render_evidence_answer({"requirements": [requirement()]},
        [RequirementAnswer(requirement_id="revenue_2024", unavailable_reason="No matching column")],
        [], {}, {}, {}, {})
    assert result["decision"] == "insufficient_evidence"
    assert not result["complete"]


def test_rendering_preserves_zero_and_per_share_units():
    assert display_value(0, "USD", "millions", "USD") == "0 million USD"
    assert display_value("1.25", "per_share", None, "USD") == "1.25 USD per share"


def test_narrative_quotes_must_exist_and_unsupported_summaries_are_rejected():
    req = {**requirement(), "kind": "narrative", "metric": "outlook"}
    claim = RequirementAnswer(requirement_id=req["requirement_id"], summary="Management expects growth.",
                             quotes=[{"source_id": "S1", "excerpt": "Management expects demand to remain stable."}])
    with pytest.raises(ValueError, match="quote is absent"):
        render_evidence_answer({"requirements": [req]}, [claim], [], {}, {"S1": SOURCE}, {}, {})
    source = {**SOURCE, "evidence_text": claim.quotes[0].excerpt}
    model = SimpleNamespace(with_structured_output=lambda *a, **k: SimpleNamespace(
        invoke=lambda messages: {"assessments": [{"requirement_id": req["requirement_id"], "reason": "Stable demand does not establish growth.", "supported": False}]}))
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=model):
        result = render_evidence_answer({"requirements": [req]}, [claim], [], {}, {"S1": source}, {}, {})
        assert not result["complete"] and "Management expects growth" not in result["answer"]


def test_runtime_source_scope_comes_from_inventory():
    understanding = understand_query("MSFT revenue in its 2026 10-K", available_keys={("MSFT", "2026", "10K")})
    assert understanding.years == ("2026",)
    assert not understanding.unsupported_years


def test_narrative_fields_are_isolated_and_still_require_exact_evidence():
    requirements = [{**requirement(), "requirement_id": key, "kind": "narrative", "metric": metric}
                    for key, metric in (("growth", "reported growth rates"), ("outlook", "management outlook"))]
    source = {**SOURCE, "evidence_text": "Revenue grew 12% and 10% in constant currency. Management expects demand to remain stable."}
    selections = [{"requirement_id": "growth", "passage_ids": ["P1"]},
                  {"requirement_id": "outlook", "passage_ids": ["P1"]}]
    selector = SimpleNamespace(invoke=Mock(return_value={"answers": selections}))
    auditor = SimpleNamespace(invoke=Mock(return_value={"assessments": [
        {"requirement_id": item["requirement_id"], "reason": "The quote supports the requested information.", "supported": True}
        for item in requirements]}))
    model = SimpleNamespace(with_structured_output=Mock(side_effect=lambda schema, **kwargs:
        selector if "answers" in schema["properties"] else auditor))
    plan = {"question": "Report growth and outlook.", "requirements": requirements}
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=model):
        answers = resolve_evidence_answers(plan, [], {}, {"S1": source})
        result = render_evidence_answer(plan, answers, [], {}, {"S1": source}, {}, {})
    assert result["complete"]
    assert selector.invoke.call_count == auditor.invoke.call_count == 1
    assert "constant currency" in result["answer"]
    payload = json.loads(selector.invoke.call_args.args[0][-1].content)
    assert len(payload["requirements"]) == 2 and len(payload["passages"]) == 1
    answers[0].quotes[0].excerpt = "Revenue grew an invented 90 percent."
    with pytest.raises(ValueError, match="quote is absent"):
        render_evidence_answer(plan, answers, [], {}, {"S1": source}, {}, {})


def test_compaction_hides_routing_context_without_mutating_stored_evidence():
    original = json.dumps({"ticker": "MSFT", "filing_year": "2025", "doc_type": "10K",
                           "context": "full evidence", "validated_facts": [fact().model_dump(mode="json")],
                           "coverage": {"covered_requirement_ids": ["revenue_2024"]}})
    messages = [
        ToolMessage(name="plan_financial_request", tool_call_id="p", content=json.dumps({
            "plan_validated": True, "plan": {"requirements": [requirement()]}})),
        ToolMessage(name="search_financial_documents", tool_call_id="s", content=original),
    ]
    compacted = compact_agent_context({"messages": messages})["llm_input_messages"]
    assert "context" not in json.loads(compacted[-1].content)
    assert messages[-1].content == original
    messages.append(ToolMessage(name="search_financial_documents", tool_call_id="refine", content=original))
    incomplete = json.loads(original)
    incomplete["coverage"]["covered_requirement_ids"] = []
    messages[1] = messages[1].model_copy(update={"content": json.dumps(incomplete)})
    compacted = compact_agent_context({"messages": messages})["llm_input_messages"]
    assert "context" not in json.loads(compacted[1].content)
    assert "context" in json.loads(messages[1].content)
    narrative = {**requirement(), "kind": "narrative"}
    messages[0] = ToolMessage(name="plan_financial_request", tool_call_id="p", content=json.dumps({
        "plan_validated": True, "plan": {"requirements": [narrative]}}))
    payload = json.loads(compact_agent_context({"messages": messages})["llm_input_messages"][-1].content)
    assert "context" not in payload and payload["evidence_stored"]
    assert json.loads(messages[-1].content)["context"] == "full evidence"


def test_statement_rows_and_periods_are_rendered_from_indexed_table():
    req = {**requirement(), "kind": "statement", "metric": "income statement"}
    html = "<table><tr><th>Metric</th><th>2024</th></tr><tr><td>Revenue</td><td>100</td></tr></table>"
    statement = {"sources": [SOURCE], "statement_type": "income_statement",
                 "statement_completeness": {"complete": True, "period_columns": ["2024"]}}
    answer = RequirementAnswer(requirement_id="revenue_2024", statement_id="T1")
    result = render_evidence_answer({"requirements": [req]}, [answer], [], {},
                                   {"S1": {**SOURCE, "evidence_text": html}}, {}, {"T1": statement})
    assert "| Revenue | 100 |" in result["answer"]
    assert result["complete"]
    selected = resolve_evidence_answers({"requirements": [req]}, [], {}, {"S1": SOURCE}, {"T1": statement})
    assert selected[0].statement_id == "T1"
    with pytest.raises(ValueError, match="Statement type"):
        render_evidence_answer({"requirements": [{**req, "metric": "balance sheet"}]}, [answer], [], {},
                               {"S1": {**SOURCE, "evidence_text": html}}, {}, {"T1": statement})
    statement["statement_completeness"]["period_columns"] = ["2025"]
    with pytest.raises(ValueError, match="missing a requested"):
        render_evidence_answer({"requirements": [req]}, [answer], [], {},
                               {"S1": {**SOURCE, "evidence_text": html}}, {}, {"T1": statement})


def test_same_table_is_rendered_once_after_every_requested_period_is_validated():
    req = {**requirement(), "kind": "statement", "metric": "income statement"}
    other = {**req, "requirement_id": "r2", "reported_period": "FY2025"}
    html = "<table><tr><th>Metric</th><th>2025</th><th>2024</th></tr><tr><td>Revenue</td><td>120</td><td>100</td></tr></table>"
    statement = {"sources": [SOURCE], "statement_type": "income_statement",
                 "statement_completeness": {"complete": True, "period_columns": ["2024", "2025"]}}
    plan = {"requirements": [req, other]}
    answers = [RequirementAnswer(requirement_id=item["requirement_id"], statement_id="T1") for item in plan["requirements"]]
    result = render_evidence_answer(plan, answers, [], {}, {"S1": {**SOURCE, "evidence_text": html}}, {}, {"T1": statement})
    assert result["complete"] and result["answer"].count("| Revenue | 120 | 100 |") == 1
    statement["statement_completeness"]["period_columns"] = ["2024"]
    with pytest.raises(ValueError, match="missing a requested reported-period"):
        render_evidence_answer(plan, answers, [], {}, {"S1": {**SOURCE, "evidence_text": html}}, {}, {"T1": statement})


def test_active_statement_tool_uses_indexed_table_without_search_or_extraction(tmp_path):
    html = "<table><tr><th>Metric</th><th>2024</th></tr><tr><td>Revenue</td><td>100</td></tr></table>"
    balance = "<table><tr><th>Metric</th><th>2024</th></tr><tr><td>Total assets</td><td>200</td></tr></table>"
    (tmp_path / "statement.json").write_text(json.dumps({
        "page_content": "## INCOME STATEMENTS\n(In millions)\n" + html
                        + "\n## BALANCE SHEETS\n(In millions)\n" + balance,
        "metadata": {**SOURCE, "source": "filing.pdf", "item": "Item 8", "section": "Statements"},
    }), encoding="utf-8")
    retriever = SimpleNamespace(context_builder=SimpleNamespace(parent_store=SimpleNamespace(root=tmp_path)),
                                retrieve=Mock(side_effect=AssertionError("A complete indexed statement needs no search.")))
    pipeline = SimpleNamespace(run=Mock(side_effect=AssertionError("The table needs no fact extractor.")))
    scope = Scope(("MSFT",), ("2025",), "10K")
    tools = {tool.name: tool for tool in create_financial_tools(retriever, FinancialCalculator(),
                permitted_scope=scope, fact_pipeline=pipeline)}
    locked = {"tickers": ["MSFT"], "years": ["2025"], "doc_type": "10K", "requested_groups": [["MSFT", "2025"]]}
    runtime = SimpleNamespace(state={"messages": [{"type": "human", "content":
        "Application-resolved scope (locked): " + json.dumps(locked) + "\nUser question: Show the income statement."}]})
    req = {**requirement(), "kind": "statement", "metric": "income statement"}
    balance_req = {**req, "requirement_id": "balance", "metric": "balance sheet"}
    content = tools["plan_financial_request"].func(requirements=[req, balance_req], runtime=runtime)
    assert json.loads(content)["plan_validated"]
    runtime.state["messages"].extend([
        {"type": "ai", "tool_calls": [{"name": "plan_financial_request", "args": {}, "id": "plan"}]},
        {"type": "tool", "name": "plan_financial_request", "tool_call_id": "plan", "content": content},
    ])
    content = tools["get_financial_statement"].func(ticker="MSFT", filing_year="2025",
                                                  statement_type="income_statement", runtime=runtime)
    payload = json.loads(content)
    assert payload.get("statement_completeness", {}).get("complete"), payload
    runtime.state["messages"].extend([
        {"type": "ai", "tool_calls": [{"name": "get_financial_statement", "args": {
            "ticker": "MSFT", "filing_year": "2025", "statement_type": "income_statement"}, "id": "statement"}]},
        {"type": "tool", "name": "get_financial_statement", "tool_call_id": "statement", "content": content},
    ])
    content = tools["get_financial_statement"].func(ticker="MSFT", filing_year="2025",
        statement_type="balance_sheet", unresolved_requirement_ids=["balance"], runtime=runtime)
    balance_payload = json.loads(content)
    assert balance_payload.get("statement_completeness", {}).get("complete"), balance_payload
    assert balance_payload["statement_id"] != payload["statement_id"]
    runtime.state["messages"].append({"type": "tool", "name": "get_financial_statement", "content": content})
    final = json.loads(tools["finalize_financial_answer"].func(
        answers=[RequirementAnswer(requirement_id=req["requirement_id"], statement_id=payload["statement_id"]),
                 RequirementAnswer(requirement_id="balance", statement_id=balance_payload["statement_id"])], runtime=runtime))
    assert final["complete"], final
    assert "| Revenue | 100 |" in final["answer"]
    assert "| Total assets | 200 |" in final["answer"]
    retriever.retrieve.assert_not_called()
    pipeline.run.assert_not_called()


def test_statement_tool_cannot_treat_a_scalar_fact_as_a_full_table():
    retriever = SimpleNamespace(
        context_builder=None,
        retrieve=Mock(side_effect=AssertionError("A scalar fact must use document search.")),
    )
    scope = Scope(("MSFT",), ("2025",), "10K")
    tools = {tool.name: tool for tool in create_financial_tools(
        retriever, FinancialCalculator(), permitted_scope=scope,
    )}
    locked = {
        "tickers": ["MSFT"], "years": ["2025"], "doc_type": "10K",
        "requested_groups": [["MSFT", "2025"]],
    }
    runtime = SimpleNamespace(state={"messages": [{"type": "human", "content":
        "Application-resolved scope (locked): " + json.dumps(locked)
        + "\nUser question: What was EPS in 2025?"}]})
    req = {**requirement("2025"), "metric": "EPS"}
    plan = tools["plan_financial_request"].func(requirements=[req], runtime=runtime)
    runtime.state["messages"].extend([
        {"type": "ai", "tool_calls": [{"name": "plan_financial_request", "args": {}, "id": "plan"}]},
        {"type": "tool", "name": "plan_financial_request", "tool_call_id": "plan", "content": plan},
    ])

    content = tools["get_financial_statement"].func(
        ticker="MSFT", filing_year="2025",
        statement_type="diluted earnings per share", runtime=runtime,
    )
    payload = json.loads(content)

    assert payload["request_plan_mismatch"]
    assert payload["required_tool"] == "search_financial_documents"
    retriever.retrieve.assert_not_called()


def test_generic_search_executes_a_pure_statement_plan_deterministically(tmp_path):
    html = ("<table><tr><th>Category</th><th>Amount</th></tr>"
            "<tr><td>Federal</td><td>$28</td></tr><tr><td>Total</td><td>$28</td></tr></table>")
    title = "Cash paid for income taxes, net of refunds"
    (tmp_path / "taxes.json").write_text(json.dumps({
        "page_content": f"## {title}\n(In millions)\n{html}",
        "metadata": {**SOURCE, "source": "filing.pdf", "section": "Income Taxes"},
    }), encoding="utf-8")
    retriever = SimpleNamespace(context_builder=SimpleNamespace(parent_store=SimpleNamespace(root=tmp_path)),
                                retrieve=Mock(side_effect=AssertionError("A pure table plan must use indexed structure.")))
    pipeline = SimpleNamespace(run=Mock(side_effect=AssertionError("A pure table plan needs no fact extraction.")))
    scope = Scope(("MSFT",), ("2025",), "10K")
    tools = {tool.name: tool for tool in create_financial_tools(
        retriever, FinancialCalculator(), permitted_scope=scope, fact_pipeline=pipeline)}
    locked = {"tickers": ["MSFT"], "years": ["2025"], "doc_type": "10K",
              "requested_groups": [["MSFT", "2025"]]}
    runtime = SimpleNamespace(state={"messages": [{"type": "human", "content":
        "Application-resolved scope (locked): " + json.dumps(locked)
        + f"\nUser question: Show the {title} table."}]})
    req = {**requirement("2025"), "kind": "statement", "metric": title, "reported_period": None}
    plan = tools["plan_financial_request"].func(requirements=[req], runtime=runtime)
    runtime.state["messages"].extend([
        {"type": "ai", "tool_calls": [{"name": "plan_financial_request", "args": {}, "id": "plan"}]},
        {"type": "tool", "name": "plan_financial_request", "tool_call_id": "plan", "content": plan},
    ])
    search = tools["search_financial_documents"].func(
        ticker="MSFT", filing_year="2025", doc_type="10K", query="income taxes", runtime=runtime)
    payload = json.loads(search)
    assert payload.get("statement_id") and payload["statement_type"] == title, payload
    runtime.state["messages"].extend([
        {"type": "ai", "tool_calls": [{"name": "search_financial_documents", "args": {
            "ticker": "MSFT", "filing_year": "2025", "doc_type": "10K", "query": "income taxes"}, "id": "search"}]},
        {"type": "tool", "name": "search_financial_documents", "tool_call_id": "search", "content": search},
    ])
    final = json.loads(tools["finalize_financial_answer"].func(
        answers=[RequirementAnswer(requirement_id=req["requirement_id"], statement_id=payload["statement_id"])],
        runtime=runtime))
    assert final["complete"] and "| Federal | $28 |" in final["answer"], final
    retriever.retrieve.assert_not_called()
    pipeline.run.assert_not_called()


def test_source_deduplication_keeps_evidence_from_both_searches():
    messages = [ToolMessage(name="search_financial_documents", tool_call_id=str(index), content=json.dumps({
        "context": f"[SOURCE S1 | TYPE: 10K]\n{text}", "sources": [{**SOURCE, "parent_id": "P1"}],
        "validated_facts": [],
    })) for index, text in enumerate(("First distinct evidence", "Second distinct evidence"))]
    _, sources = _facts_from_tool_messages(messages)
    displayed = ChatService._agent_sources(messages)
    assert len(displayed) == 1
    for text in ("First distinct evidence", "Second distinct evidence"):
        assert text in sources["S1"]["evidence_text"]
        assert text in displayed[0]["evidence_text"]


def test_real_graph_contract_runs_plan_search_calculator_and_finalizer_without_a_routing_model():
    scope = Scope(("MSFT",), ("2025",), "10K")
    retriever = SimpleNamespace(retrieve=lambda query, scope, **kwargs: RetrievalBundle(
        "[SOURCE S1 | TICKER: MSFT | FISCAL YEAR: 2025 | TYPE: 10K]\nRevenue 2024 100 2025 120",
        [{**SOURCE, "parent_id": "P1", "parent_complete": True}], scope, 1))
    pipeline = SimpleNamespace(run=lambda **kwargs: FactValidationResult(valid_facts=(fact(), fact("2025", "120"))))
    model = SimpleNamespace(step=0, bindings=[])
    builder = Mock(return_value=RequestPlanInput.model_validate({
        "requirements": [requirement(), requirement("2025")],
        "calculations": [{"operation": "percentage_change", "input_requirement_ids": ["revenue_2024", "revenue_2025"]}],
    }))
    with patch("src.agent.graph.get_llm_engine", return_value=model), patch("src.agent.tools.FinancialFactPipeline", return_value=pipeline):
        graph = create_finrag_agent(retriever, FinancialCalculator(), permitted_scope=scope,
            plan_builder=builder)
    service = ChatService.__new__(ChatService)
    service.agent_graph = graph
    service.require_validated_answer = True
    service.index_version = "offline"
    service.logger = SimpleNamespace(info=lambda value: None)
    locked = {"tickers": ["MSFT"], "years": ["2025"], "doc_type": "10K", "requested_groups": [["MSFT", "2025"]]}
    result = service.process_query({"messages": [{"role": "user", "content":
        "Application-resolved scope (locked): " + json.dumps(locked) + "\nUser question: Compare revenue columns."}]},
        scope=scope, thread_id="offline")
    assert builder.call_args.args[0] == "Compare revenue columns."
    assert model.bindings == [] and model.step == 0
    assert result.decision == Decision.ANSWERED, result.trace
    assert "20.0 percent" in result.answer or "20.00 percent" in result.answer
    assert result.trace["answer_validation"]["reason"] == "validated_evidence_claims"
    assert "999999" not in result.answer and "1000%" not in result.answer
    calculation = result.trace["agent"]["calculator_results"][0]
    assert calculation["calculation_id"].startswith("C")
    assert calculation["planned_input_requirement_ids"] == ["revenue_2024", "revenue_2025"]


def test_two_companies_retrieve_each_source_once_and_never_mix_facts():
    scope = Scope(("ACME", "BETA"), ("2025",), "10K")
    facts = {ticker: fact("2025", str(value), ticker=ticker, fact_id=f"F{index}", source_id=f"S{index}")
             for index, (ticker, value) in enumerate((("ACME", 100), ("BETA", 200)), 1)}
    calls = []
    def retrieve(query, current_scope, **kwargs):
        ticker = current_scope.tickers[0]
        calls.append(ticker)
        item = facts[ticker]
        return RetrievalBundle("Revenue 2025 " + item.raw_value,
            [{"id": item.source_id, "ticker": ticker, "fiscal_year": "2025", "doc_type": "10K",
              "parent_id": ticker, "evidence_text": item.evidence_excerpt}], current_scope, 1)
    pipeline = SimpleNamespace(run=lambda **kwargs: FactValidationResult(
        valid_facts=(facts[kwargs["permitted_scope"].tickers[0]],)))
    plan = RequestPlanInput.model_validate({"requirements": [{**requirement("2025"),
        "ticker": ticker, "requirement_id": ticker} for ticker in facts]})
    with patch("src.agent.tools.FinancialFactPipeline", return_value=pipeline):
        graph = create_finrag_agent(SimpleNamespace(retrieve=retrieve), FinancialCalculator(),
            permitted_scope=scope, plan_builder=lambda *args: plan)
    state = graph.invoke({"messages": [("user", "Compare ACME and BETA revenue for FY2025 from their 2025 10-Ks.")]})
    final = next(json.loads(message.content) for message in state["messages"]
        if isinstance(message, ToolMessage) and message.name == "finalize_financial_answer")
    assert sorted(calls) == ["ACME", "BETA"]
    sources = {source["ticker"]: source["id"] for message in state["messages"]
        if isinstance(message, ToolMessage) and message.name == "search_financial_documents"
        for source in json.loads(message.content)["sources"]}
    assert final["complete"] and set(final["cited_source_ids"]) == set(sources.values())
    for ticker, value in (("ACME", 100), ("BETA", 200)):
        assert any(ticker in line and f"{value} million USD" in line and f"[{sources[ticker]}]" in line
                   for line in final["answer"].splitlines())


@pytest.mark.parametrize(("statement_type", "metric"), [
    ("balance_sheet", "balance sheet"), ("income_statement", "income statement"),
    ("cash_flow_statement", "cash flow statement"),
    ("income_statement", "Consolidated Statement of Operations"),
    ("income_statement", "Consolidated Statements of Operations"),
    ("Future debt maturities", "Future debt maturities"),
    ("Revenue by geographic region", "Revenue by geographic region"),
    ("Operating lease obligations", "Operating lease obligations"),
    ("Consolidated Statements of Comprehensive Income", "Consolidated Statements of Comprehensive Income"),
    ("Changes in stockholders equity", "Changes in stockholders equity"),
    ("Operational milestone schedule", "Operational milestone schedule"),
    ("Supplemental balance sheet information related to leases", "Supplemental balance sheet information related to leases"),
])
@pytest.mark.parametrize("reported_period", [None, "FY2026"])
def test_statement_graph_requires_finalization_and_then_allows_termination(tmp_path, statement_type, metric, reported_period):
    html = ("<table><tr><th>Metric</th><th>2027</th><th>2026</th></tr>"
            "<tr><td>Reported value</td><td>120</td><td>100</td></tr></table>")
    (tmp_path / "statement.json").write_text(json.dumps({
        "page_content": f"## {metric.upper()}\n(In millions)\n{html}",
        "metadata": {"ticker": "ACME", "fiscal_year": "2027", "doc_type": "10K", "source": "filing.pdf"},
    }), encoding="utf-8")
    retriever = SimpleNamespace(context_builder=SimpleNamespace(parent_store=SimpleNamespace(root=tmp_path)),
                               retrieve=Mock(side_effect=AssertionError("Indexed table requires no remote retrieval.")))
    scope = Scope(("ACME",), ("2027",), "10K")
    plan = RequestPlanInput.model_validate({"requirements": [{"requirement_id": "r1", "ticker": "ACME",
        "source_year": "2027", "doc_type": "10K", "metric": metric, "kind": "statement", "qualifiers": [],
        "reported_period": reported_period}]})
    model = SimpleNamespace(step=0, bindings=[])
    with patch("src.agent.graph.get_llm_engine", return_value=model):
        graph = create_finrag_agent(retriever, FinancialCalculator(), permitted_scope=scope,
                                    plan_builder=Mock(return_value=plan))
    locked = {"tickers": ["ACME"], "years": ["2027"], "doc_type": "10K", "requested_groups": [["ACME", "2027"]]}
    state = graph.invoke({"messages": [{"role": "user", "content": "Application-resolved scope (locked): "
        + json.dumps(locked) + f"\nUser question: Show the exact {metric}."}]})
    finals = [json.loads(message.content) for message in state["messages"]
              if isinstance(message, ToolMessage) and message.name == "finalize_financial_answer"]
    assert len(finals) == 1 and finals[0]["answer_validated"] and finals[0]["complete"], finals
    assert "| Reported value | 120 | 100 |" in finals[0]["answer"]
    assert model.step == 0 and model.bindings == []
    retriever.retrieve.assert_not_called()


def test_graph_clarification_can_finish_without_forced_retrieval():
    model = SimpleNamespace(step=0, bindings=[])
    plan = RequestPlanInput(clarification="Which financial year?")
    with patch("src.agent.graph.get_llm_engine", return_value=model):
        graph = create_finrag_agent(SimpleNamespace(), FinancialCalculator(),
                                    plan_builder=Mock(return_value=plan))
    state = graph.invoke({"messages": [{"role": "user", "content": "Show the balance sheet."}]})
    assert model.bindings == [] and model.step == 0
    assert not state["messages"][-1].tool_calls


@pytest.mark.parametrize("columns", [2, 12])
def test_named_note_tables_preserve_text_cells_and_wide_layout_without_year_headers(tmp_path, columns):
    title = "Operational milestone schedule"
    headers = [f"Category {index}" for index in range(columns)]
    values = [f"Status {index}" for index in range(columns)]
    html = "<table><tr>" + "".join(f"<th>{value}</th>" for value in headers) + "</tr><tr>" + \
        "".join(f"<td>{value}</td>" for value in values) + "</tr></table>"
    (tmp_path / "note.json").write_text(json.dumps({"page_content": f"## {title}\n{html}",
        "metadata": {"ticker": "ACME", "fiscal_year": "2027", "doc_type": "10K", "source": "filing.pdf"}}), encoding="utf-8")
    scope = Scope(("ACME",), ("2027",), "10K")
    plan = RequestPlanInput.model_validate({"requirements": [{"requirement_id": "r1", "ticker": "ACME",
        "source_year": "2027", "doc_type": "10K", "kind": "statement", "metric": title}]})
    retriever = SimpleNamespace(context_builder=SimpleNamespace(parent_store=SimpleNamespace(root=tmp_path)),
                               retrieve=Mock(side_effect=AssertionError("No model search for stored tables.")))
    model = SimpleNamespace(step=0, bindings=[])
    with patch("src.agent.graph.get_llm_engine", return_value=model):
        graph = create_finrag_agent(retriever, FinancialCalculator(), permitted_scope=scope,
                                    plan_builder=Mock(return_value=plan))
    locked = {"tickers": ["ACME"], "years": ["2027"], "doc_type": "10K", "requested_groups": [["ACME", "2027"]]}
    state = graph.invoke({"messages": [{"role": "user", "content": "Application-resolved scope (locked): "
        + json.dumps(locked) + f"\nUser question: Show the exact {title} table."}]})
    final = next(json.loads(message.content) for message in state["messages"]
                 if isinstance(message, ToolMessage) and message.name == "finalize_financial_answer")
    assert final["answer_validated"] and final["complete"], final
    assert "| " + " | ".join(headers) + " |" in final["answer"]
    assert "| " + " | ".join(values) + " |" in final["answer"]
    retriever.retrieve.assert_not_called()


def test_auto_coverage_preserves_conflicts_and_rejects_wrong_fact_selection():
    plan = {"requirements": [requirement()]}
    conflicting = fact(value="200").model_copy(update={"fact_id": "Fother"})
    facts = {"F2024": fact(), "Fother": conflicting}
    resolved = resolve_evidence_answers(plan, [], facts, {"S1": SOURCE})
    with pytest.raises(ValueError, match="Conflicting facts"):
        render_evidence_answer(plan, resolved, [], facts, {"S1": SOURCE}, {}, {})
    wrong = fact("2025").model_copy(update={"fact_id": "Fwrong"})
    facts["Fwrong"] = wrong
    resolved = resolve_evidence_answers(plan, [RequirementAnswer(requirement_id="revenue_2024", fact_ids=["Fwrong"])], facts, {"S1": SOURCE})
    with pytest.raises(ValueError, match="does not satisfy"):
        render_evidence_answer(plan, resolved, [], facts, {"S1": SOURCE}, {}, {})


def test_explicit_evidence_scale_reconciles_metadata_without_accepting_unsupported_values():
    source = {**SOURCE, "evidence_text": "Total revenue 2024 was $100 billion."}
    candidate = CandidateFinancialFact(ticker="MSFT", metric="Total revenue", period="2024",
        raw_value="$100 billion", unit="USD", currency="USD", scale="millions", source_id="S1",
        row_label="Total revenue", column_label="2024", evidence_excerpt=source["evidence_text"])
    validator = ExtractedFactValidator()
    result = validator.validate([candidate], [source])
    assert result.valid_facts[0].scale == "billions"
    assert result.valid_facts[0].base_value == 100_000_000_000
    rejected = validator.validate([candidate.model_copy(update={"raw_value": "$200 billion"})], [source])
    assert not rejected.valid_facts


def test_table_metric_inherits_section_and_cannot_relabel_a_cost_as_revenue():
    source = {**SOURCE, "evidence_text": "<table><tr><th>Metric</th><th>2024</th></tr>"
              "<tr><td colspan='2'>Cost of revenues</td></tr>"
              "<tr><td>Example business segment</td><td>100</td></tr></table>"}
    candidate = CandidateFinancialFact(ticker="MSFT", metric="Example business segment revenue", period="2024",
        raw_value="100", unit="USD", currency="USD", scale="millions", source_id="S1",
        row_label="Example business segment", column_label="2024",
        evidence_excerpt="Example business segment 2024 100")
    validator = ExtractedFactValidator()
    rejected = validator.validate([candidate], [source])
    assert not rejected.valid_facts
    assert rejected.rejected_facts[0].reason == "table_metric_label_mismatch"
    result = validator.validate([candidate.model_copy(update={"metric": "Cost of revenues Example business segment"})], [source])
    assert result.valid_facts


def test_short_row_labels_bind_to_the_requested_metric_through_table_sections():
    source = {**SOURCE, "evidence_text": "<table><tr><th>Metric</th><th>2024</th></tr>"
        "<tr><td colspan='2'>Revenues</td></tr><tr><td>Example business</td><td>120</td></tr>"
        "<tr><td colspan='2'>Cost of revenues</td></tr><tr><td>Example business</td><td>100</td></tr></table>"}
    requested = {**requirement(), "metric": "Example business revenue"}
    candidate = CandidateFinancialFact(ticker="MSFT", metric="Example business", period="2024",
        raw_value="120", unit="USD", currency="USD", scale="millions", source_id="S1",
        row_label="Example business", column_label="2024", evidence_excerpt="Example business 2024 120")
    validator = ExtractedFactValidator()
    result = validator.validate([candidate], [source], requirements=(requested,))
    assert fact_matches_requirement(result.valid_facts[0], requested, source)
    cost = candidate.model_copy(update={"raw_value": "100", "evidence_excerpt": "Example business 2024 100"})
    result = validator.validate([cost], [source], requirements=(requested,))
    assert not fact_matches_requirement(result.valid_facts[0], requested, source)
    statement = {**source, "section": "Financial Statements and Supplementary Data",
                 "evidence_text": "Consolidated income statements (in millions)\n" + source["evidence_text"]}
    pipeline = FinancialFactPipeline(extractor=SimpleNamespace(extract=lambda **kwargs: ()))
    result = pipeline.run(question="Report business revenue", sources=[statement],
        permitted_scope=Scope(("MSFT",), ("2025",), "10K"), requirements=(requested,))
    assert len(result.valid_facts) == 1
    assert result.valid_facts[0].numeric_value == 120


def test_plan_review_uses_original_question_and_enforces_scope_after_correction():
    scope = Scope(("MSFT",), ("2025",), "10K")
    runtime = SimpleNamespace(state={"messages": [{"type": "human", "content":
        "User question: State revenue and explain its growth in the 2025 filing."}]})
    corrected = RequestPlanInput.model_validate({"requirements": [requirement(),
        {**requirement(), "requirement_id": "growth", "kind": "narrative", "metric": "reported revenue growth and its bases"}]})
    reviewer = Mock(return_value=corrected)
    tool = create_financial_tools(SimpleNamespace(), FinancialCalculator(), permitted_scope=scope,
                                 plan_builder=reviewer)[0]
    result = json.loads(tool.func(requirements=[requirement()], runtime=runtime))
    assert result["plan_validated"]
    assert len(result["plan"]["requirements"]) == 2
    assert reviewer.call_args.args[0].startswith("State revenue")
    reviewer.return_value = {"requirements": [{**requirement(), "source_year": "2024"}]}
    result = json.loads(tool.func(requirements=[requirement()], runtime=runtime))
    assert not result["plan_validated"]
    assert "locked source scope" in result["error"]


def test_structured_planner_repairs_schema_once_with_original_question_and_feedback():
    valid = {"requirements": [requirement(value_type="currency")]}
    invalid = {"requirements": [{**requirement(value_type="currency"), "reported_period": None}]}
    planner = SimpleNamespace(invoke=Mock(side_effect=[invalid, valid]))
    model = SimpleNamespace(with_structured_output=Mock(return_value=planner))
    with patch("src.agent.graph.get_llm_engine", return_value=model):
        result = build_request_plan("Revenue FY2024 from the 2025 10-K", Scope(("MSFT",), ("2025",), "10K"))
    assert result.requirements[0].reported_period == "FY2024"
    assert planner.invoke.call_count == 2
    schema = model.with_structured_output.call_args.args[0]
    assert isinstance(schema, dict)
    assert "value_type" not in schema["$defs"]["PlannedRequirement"]["properties"]
    repair_messages = planner.invoke.call_args.args[0]
    assert "reported period" in repair_messages[-1].content
    assert "FY2024" in repair_messages[1].content
    planner.invoke = Mock(side_effect=[invalid, invalid])
    with patch("src.agent.graph.get_llm_engine", return_value=model), pytest.raises(ValueError):
        build_request_plan("Revenue FY2024", Scope(("MSFT",), ("2025",), "10K"))
    assert planner.invoke.call_count == 2


def test_structured_planner_binds_source_year_without_changing_reported_year():
    valid = {"requirements": [requirement(value_type="currency")]}
    invalid = {"requirements": [{**requirement(value_type="currency"), "source_year": "2024"}]}
    planner = SimpleNamespace(invoke=Mock(side_effect=[invalid, valid]))
    model = SimpleNamespace(with_structured_output=lambda *args, **kwargs: planner)
    with patch("src.agent.graph.get_llm_engine", return_value=model):
        result = build_request_plan("Revenue FY2024 from the 2025 filing", Scope(("MSFT",), ("2025",), "10K"))
    assert result.requirements[0].source_year == "2025"
    assert result.requirements[0].reported_period == "FY2024"
    assert planner.invoke.call_count == 1


@pytest.mark.parametrize(("title", "canonical"), [
    ("Consolidated Statement of Operations", "income_statement"),
    ("Consolidated Statements of Operations", "income_statement"),
    ("Consolidated Income Statements", "income_statement"),
    ("Consolidated Balance Sheets", "balance_sheet"),
    ("Consolidated Statements of Cash Flows", "cash_flow_statement"),
    ("Future debt maturities", "Future debt maturities"),
    ("Segment operating results", "Segment operating results"),
    ("Operating milestone schedule", "Operating milestone schedule"),
    ("Supplemental balance sheet information related to leases", "Supplemental balance sheet information related to leases"),
])
def test_explicit_statement_table_intent_overrides_narrative_plan_without_extra_model_call(title, canonical):
    from src.agent.tools import StatementLookupInput
    table = {**requirement(), "kind": "narrative", "metric": title, "reported_period": None,
             "evidence_question": f"What is the exact table of {title} in the filing?"}
    narrative = {**table, "requirement_id": "risks", "metric": "operating risks",
                 "evidence_question": "What operating risks are described?"}
    planner = SimpleNamespace(invoke=Mock(return_value={"subjects": [], "requirements": [table, narrative]}))
    model = SimpleNamespace(with_structured_output=lambda *args, **kwargs: planner)
    with patch("src.agent.graph.get_llm_engine", return_value=model):
        plan = build_request_plan(f"Show the exact {title} table and explain operating risks from the 2025 10-K.",
                                  Scope(("MSFT",), ("2025",), "10K"))
    assert [item.kind for item in plan.requirements] == ["statement", "narrative"]
    assert plan.requirements[0].metric == title
    assert planner.invoke.call_count == 1
    assert StatementLookupInput(ticker="MSFT", filing_year="2025", statement_type=title).statement_type == canonical


def test_statement_heading_does_not_turn_a_requested_value_or_summary_into_a_full_table():
    for kind, question in (("fact", "What is revenue in the income statement?"),
                           ("narrative", "Summarize the consolidated statement of operations.")):
        req = {**requirement(), "kind": kind, "metric": "income statement revenue" if kind == "fact"
               else "Consolidated Statement of Operations", "reported_period": "FY2024" if kind == "fact" else None}
        planner = SimpleNamespace(invoke=Mock(return_value={"requirements": [req]}))
        model = SimpleNamespace(with_structured_output=lambda *args, **kwargs: planner)
        with patch("src.agent.graph.get_llm_engine", return_value=model):
            plan = build_request_plan(question, Scope(("MSFT",), ("2025",), "10K"))
        assert plan.requirements[0].kind == kind


def test_source_table_format_is_repaired_when_the_plan_drops_the_table_request():
    from copy import deepcopy
    req = {**requirement(), "kind": "narrative", "metric": "future debt maturities", "reported_period": None,
           "evidence_question": "What are the future debt maturities?"}
    invalid = {"requirements": [req]}
    valid = {"requirements": [{**req, "kind": "statement"}]}
    planner = SimpleNamespace(invoke=Mock(side_effect=[deepcopy(invalid), valid]))
    model = SimpleNamespace(with_structured_output=lambda *args, **kwargs: planner)
    with patch("src.agent.graph.get_llm_engine", return_value=model):
        result = build_request_plan("Show the complete future debt maturities table from the 2025 10-K.",
                                    Scope(("MSFT",), ("2025",), "10K"))
    assert result.requirements[0].kind == "statement" and planner.invoke.call_count == 2
    assert "narrative summaries cannot replace" in planner.invoke.call_args.args[0][-1].content


@pytest.mark.parametrize("kind", ["fact", "statement"])
def test_requested_object_contract_merges_extra_prose_or_subrow_requirements(kind):
    from copy import deepcopy
    question = "Report operating cash flow for FY2024." if kind == "fact" else "Show the complete debt schedule table with all its descriptions."
    subject = "operating cash flow" if kind == "fact" else "debt schedule"
    draft = {"requested_objects": [{"object_id": "o1", "quote": question, "kind": kind,
        "subject": subject, "reported_period": "FY2024" if kind == "fact" else None}],
        "requirements": [{**requirement(), "object_id": "o1", "requirement_id": "r1", "metric": subject,
                          "kind": kind, "reported_period": "FY2024"},
                         {**requirement(), "object_id": "o1", "requirement_id": "r2", "metric": subject + " explanation",
                          "kind": "narrative", "reported_period": None}]}
    planner = SimpleNamespace(invoke=Mock(return_value=deepcopy(draft)))
    model = SimpleNamespace(with_structured_output=lambda *args, **kwargs: planner)
    with patch("src.agent.graph.get_llm_engine", return_value=model):
        result = build_request_plan(question, Scope(("MSFT",), ("2025",), "10K"))
    assert len(result.requirements) == 1
    assert result.requirements[0].kind == kind and result.requirements[0].metric == subject
    assert result.requirements[0].reported_period == ("FY2024" if kind == "fact" else None)
    assert planner.invoke.call_count == 1


def test_requested_object_quote_cannot_replace_the_planned_evidence_question():
    from copy import deepcopy
    draft = {"requested_objects": [{"object_id": "o1", "quote": "Report total revenue", "kind": "narrative",
        "subject": "commentary", "reported_period": None}], "requirements": [{**requirement(), "object_id": "o1",
            "kind": "narrative", "evidence_question": "What was total revenue?"}]}
    planner = SimpleNamespace(invoke=Mock(return_value=deepcopy(draft)))
    model = SimpleNamespace(with_structured_output=lambda *args, **kwargs: planner)
    with patch("src.agent.graph.get_llm_engine", return_value=model):
        plan = build_request_plan("Report total revenue.", Scope(("MSFT",), ("2025",), "10K"))
    assert plan.requirements[0].evidence_question == "What was total revenue?"
    assert plan.requirements[0].metric == "total revenue"


def test_calculation_operands_are_numeric_even_when_table_is_only_output_format():
    question = ("Compare Microsoft's total revenue using its 2024 and 2025 10-K filings. "
                "Present both values in a table and state the dollar increase.")
    requirements = [
        {**requirement("2024"), "object_id": "o1", "source_year": "2024", "kind": "statement"},
        {**requirement("2025"), "object_id": "o1", "source_year": "2025", "kind": "statement"},
    ]
    draft = {
        "requested_objects": [{"object_id": "o1", "quote": "state the dollar increase",
            "kind": "calculation", "subject": "revenue increase", "reported_period": None}],
        "requirements": requirements,
        "calculations": [{"operation": "absolute_change",
            "input_requirement_ids": ["revenue_2024", "revenue_2025"]}],
    }
    planner = SimpleNamespace(invoke=Mock(return_value=draft))
    model = SimpleNamespace(with_structured_output=lambda *args, **kwargs: planner)
    with patch("src.agent.graph.get_llm_engine", return_value=model):
        plan = build_request_plan(question, Scope(("MSFT",), ("2024", "2025"), "10K"))
    assert [item.kind for item in plan.requirements] == ["fact", "fact"]
    assert plan.calculations[0].input_requirement_ids == ["revenue_2024", "revenue_2025"]


def test_requested_calculation_compiles_its_declared_formula():
    question = ("Compare Microsoft's total revenue using its 2024 and 2025 10-K filings. "
                "Present both values in a table and state the dollar increase.")
    draft = {"requested_objects": [{"object_id": "o1", "quote": "state the dollar increase",
        "subject": "revenue increase", "kind": "calculation", "calculation": {
            "operation": "absolute_change", "input_requirement_ids": ["revenue_2024", "revenue_2025"]}}],
        "requirements": [
        {**requirement("2024"), "object_id": "o1", "source_year": "2024"},
        {**requirement("2025"), "object_id": "o1", "source_year": "2025"},
    ], "calculations": []}
    planner = SimpleNamespace(invoke=Mock(return_value=draft))
    model = SimpleNamespace(with_structured_output=lambda *args, **kwargs: planner)
    with patch("src.agent.graph.get_llm_engine", return_value=model):
        plan = build_request_plan(question, Scope(("MSFT",), ("2024", "2025"), "10K"))
    assert [(item.operation, item.input_requirement_ids) for item in plan.calculations] == [
        ("absolute_change", ["revenue_2024", "revenue_2025"])
    ]


def test_calculation_references_normalize_model_misclassified_operands():
    plan = RequestPlanInput.model_validate({"requirements": [
        {**requirement("2024"), "kind": "narrative"},
        {**requirement("2025"), "kind": "statement"},
    ], "calculations": [{"operation": "percentage_change",
        "input_requirement_ids": ["revenue_2024", "revenue_2025"]}]})
    assert [item.kind for item in plan.requirements] == ["fact", "fact"]


def test_presentation_and_duplicate_metric_terms_are_not_evidence_qualifiers():
    planned = PlannedRequirement.model_validate({**requirement("2025"), "metric": "total assets",
        "qualifiers": ["balance-sheet date: June 30, 2025", "currency", "currency: USD", "scale",
                       "scale: millions", "total assets"]})
    assert planned.qualifiers == []


def test_source_caption_period_can_bind_a_single_period_table_without_year_columns(tmp_path):
    (tmp_path / "note.json").write_text(json.dumps({
        "page_content": "Cash paid for taxes during the year ended December 31, 2025 was as follows:\n\n"
                        "<table><tr><th>Category</th><th>Amount</th></tr><tr><td>Domestic</td><td>100</td></tr></table>",
        "metadata": SOURCE}), encoding="utf-8")
    direct = StructuredDocumentLookup(tmp_path).answer("cash paid for taxes", Scope(("MSFT",), ("2025",), "10K"), wants_complete_table=True)
    assert direct and direct.sources[0]["table_completeness"]["period_columns"] == []
    assert direct.sources[0]["table_completeness"]["reported_periods"] == ["2025"]
    req = {**requirement("2025"), "metric": "cash paid for taxes", "kind": "statement"}
    statement = {"sources": direct.sources, "statement_type": "cash paid for taxes", "statement_completeness": direct.sources[0]["table_completeness"]}
    result = render_evidence_answer({"requirements": [req]}, [RequirementAnswer(requirement_id=req["requirement_id"], statement_id="T1")],
        [], {}, {"S1": direct.sources[0]}, {}, {"T1": statement})
    assert result["complete"]


def test_temporal_plan_orders_reported_periods_independently_of_sources():
    plan = RequestPlanInput.model_validate({"requirements": [requirement(), requirement("2025")],
        "calculations": [{"operation": "percentage_change", "input_requirement_ids": ["revenue_2025", "revenue_2024"]}]})
    assert plan.calculations[0].input_requirement_ids == ["revenue_2024", "revenue_2025"]
    assert {item.source_year for item in plan.requirements} == {"2025"}


def test_finalization_requests_one_missing_fact_recovery_before_returning_partial():
    scope = Scope(("MSFT",), ("2025",), "10K")
    locked = {"tickers": ["MSFT"], "years": ["2025"], "doc_type": "10K", "requested_groups": [["MSFT", "2025"]]}
    runtime = SimpleNamespace(state={"messages": [{"type": "human", "content":
        "Application-resolved scope (locked): " + json.dumps(locked) + "\nUser question: Report revenue FY2024."}]})
    tools = {tool.name: tool for tool in create_financial_tools(SimpleNamespace(), FinancialCalculator(), permitted_scope=scope)}
    plan = tools["plan_financial_request"].func(requirements=[requirement()], runtime=runtime)
    assert json.loads(plan)["plan_validated"], plan
    runtime.state["messages"].append(AIMessage(content="", tool_calls=[
        {"name": "plan_financial_request", "args": {}, "id": "p", "type": "tool_call"}]))
    runtime.state["messages"].append(ToolMessage(name="plan_financial_request", tool_call_id="p", content=plan))
    recovery = tools["finalize_financial_answer"].func(runtime=runtime)
    assert json.loads(recovery)["recovery_required"]
    runtime.state["messages"].append(ToolMessage(name="finalize_financial_answer", tool_call_id="f", content=recovery))
    partial = json.loads(tools["finalize_financial_answer"].func(runtime=runtime))
    assert partial["answer_validated"] and not partial["complete"]
    assert partial["decision"] == "insufficient_evidence"


@pytest.mark.parametrize("placeholder", ["null", "None", "N/A", "unknown"])
def test_nullable_currency_does_not_conflict_with_an_evidence_currency(placeholder):
    candidate = CandidateFinancialFact(ticker="MSFT", metric="Revenue", period="2024", raw_value="$100 million",
        currency=placeholder, source_id="S1", evidence_excerpt="Revenue 2024 $100 million")
    result = ExtractedFactValidator().validate([candidate], [{**SOURCE, "evidence_text": candidate.evidence_excerpt}])
    assert result.valid_facts[0].currency == "USD"
