"""Evidence planning and bounded LangGraph orchestration for financial research."""

from __future__ import annotations

import json
import re
from typing import get_args
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError
from langgraph.prebuilt import create_react_agent

from src.agent.tools import RequestedObject, RequestPlanInput, StatementType, create_financial_tools
from src.financial.calculator import FinancialCalculator
from src.financial.coverage import normalize_reported_period
from src.financial.metric_terms import normalize_financial_term
from src.generation.llm_engine import get_llm_engine
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_understanding import DISPLAY_VERB_RE, understand_query
from src.retrieval.lexical_index import scoped_content_query
from src.retrieval.structured_lookup import StructuredDocumentLookup
from src.schemas import Scope
from src.observability import trace_component
from src.config import AGENT_MAX_ACTIONS


@trace_component("finrag.request_planning")
def build_request_plan(question: str, scope: Scope) -> RequestPlanInput:
    """Produce one structured plan directly from the request, before retrieval."""
    schema = RequestPlanInput.model_json_schema()
    requirement_schema = schema["$defs"]["PlannedRequirement"]
    schema["properties"].pop("calculation_intent")
    schema["properties"].pop("narrative_subject")
    schema["properties"]["synthesis"] = {"type": "boolean", "description": "Does an explanation output require QUALITATIVE interpretation across observations? Numeric values, source tables and calculations alone have synthesis=false. Inventory an explicitly requested qualitative explanation among the outputs; never add one merely because the question says compare. Do not draft an answer."}
    output_schema = schema["$defs"]["RequestedObject"]
    output_schema["properties"].pop("calculation")  # Formulas are declared once in calculations.
    output_schema["properties"]["kind"]["enum"] = ["numeric_value", "explanation", "source_table", "calculation"]
    output_schema["properties"]["kind"]["description"] = "Classify the requested result: a reported scalar, source explanation/list, actual source table, or calculation. A requested difference or dollar increase between two values is a calculation, even without the verb calculate. Units, scale and citations are properties, not separate explanations."
    output_schema["properties"]["subject"]["description"] = "The requested OUTPUT, such as operating income or its change, never just a company or business name. Keep the quantity, business/accounting basis and requested relationship distinct. Reuse an output across companies/sources; do not merge independent dimensions."
    requirement_schema["properties"].pop("kind")  # Inherit the requested output's type.
    requirement_schema["properties"]["object_id"] = {"type": "string", "pattern": "^o[1-9][0-9]*$",
        "description": "Reference the output supported by this SOURCE observation. Calculations reference operand requirement IDs; never create an evidence requirement for the computed result itself."}
    schema["properties"] = {"subjects": {"type": "array", "description": "FIRST identify ONLY actual assets, businesses, products or segments explicitly named in the original request. Exclude company, dates, actions and requested observations such as backlog or outlook. Inventory each subject once BEFORE splitting observations; omitted later subjects reuse its ID. Empty for company-wide metrics.",
        "items": {"type": "object", "properties": {"entity_type": {"type": "string", "enum": ["business_area", "asset_or_product", "company", "metric_or_action"],
            "description": "Classify meaning FIRST: business_area is the underlying economic activity (such as banking or cloud services); asset_or_product is something held or sold; company is the corporate actor. Any quantity, prediction, status or requested observation is metric_or_action, including backlog, deployment, margin, supply commentary and new-product outlook. An observation can NEVER be a business_area."},
            "subject_id": {"type": "string", "pattern": "^b[1-9][0-9]*$"},
            "name": {"type": "string", "minLength": 1, "maxLength": 160,
                "description": "Extract the underlying BUSINESS/ASSET part of compound phrases, not their observation head. 'Software revenue change' contains business software and observation revenue change; 'chip inventory outlook' contains asset chips and observation inventory outlook. Business areas need not include the word division. Do not call the entire commentary phrase a business."}},
            "required": ["entity_type", "subject_id", "name"], "additionalProperties": False}}, **schema["properties"]}
    requirement_schema["properties"].pop("subject")
    requirement_schema["properties"] = {"subject_id": {"anyOf": [{"type": "string"}, {"type": "null"}],
        "description": "Reference an inventoried business/asset subject ID. Reuse the antecedent's ID when a comparison clause omits its business. Null only for company-wide metrics. Do not create an observation as a new business."}, **requirement_schema["properties"]}
    # Units are properties of retrieved measurements, not pre-retrieval guesses.
    schema["$defs"]["PlannedRequirement"]["properties"].pop("value_type")
    understanding = understand_query(question)
    schema["$defs"]["PlannedRequirement"]["properties"]["ticker"]["enum"] = list(dict.fromkeys(ticker for ticker, year in scope.groups))
    schema["$defs"]["PlannedRequirement"]["properties"]["source_year"]["enum"] = list(dict.fromkeys(year for ticker, year in scope.groups))
    schema["properties"] = {key: schema["properties"][key] for key in
        ("subjects", "requested_objects", "requirements", "calculations", "synthesis", "clarification")}
    planner = get_llm_engine().with_structured_output(schema, method="json_schema", strict=True)
    messages = [
        SystemMessage(content="""Build a complete evidence plan for the ORIGINAL request and locked source groups. Never answer or invent values.

Inventory requested_objects using minimal literal request phrases and classify each requested OUTPUT ONCE. A scalar's currency, units, scale, period and citations are properties of that output, never additional explanations. Preserve separately requested reasons or explanations as distinct outputs. Source requirements reference these object IDs; calculation outputs are satisfied by formulas over source requirements, not by retrieving the computed result. Classify by the requested OUTPUT, not where evidence might be found:
- numeric_value: ONE scalar for ONE reported fiscal period. Expand a multi-year trend into separate values for each metric/year. A fact about business activity is not a numeric_value.
- explanation: source descriptions, lists, reasons, risks, policies, disclosed changes and outlooks. Preserve all requested details and categories. A numerical difference between two requested observations is arithmetic, not an explanation or a separately reported change.
- source_table: reproduce an actual source table/schedule, including a category breakdown. Preserve its specific subject, dimensions, qualifiers and all requested rows/columns. A comparison displayed in a table still uses numeric inputs/calculations, not source_table.
- calculation: a newly requested derived result; declare its formula once in calculations with this object_id.
Each independently named observation/dimension gets ONE output object, reused across the requested companies/sources. Do not create one umbrella output per company combining independent observations. Then use ONE flat requirements list for source observations and arithmetic operands only. A calculation output has a formula referencing those operands and NO separate retrieval requirement. Never add an umbrella requirement repeating a multi-part question alongside its subparts.
Each calculation declares its operation, ordered input_requirement_ids, rounding and CAGR periods when needed. Every operand is a numeric_value requirement. Never replace a computed result by searching for its name verbatim. Ratios/margin percentages require numerator and denominator; temporal growth percentages require earlier and later values. Source-reported changes and their explanations are not new arithmetic. Comparing reported values alone NEVER implies subtraction, ratios or growth calculations; only explicitly requested derived results belong in calculations.

Qualitative comparisons belong in synthesis; numerical differences and increases belong in calculations. Each evidence_question asks only for its own source observation, never the cross-source comparison or computed result. Separate source_year (the document) from reported_period (the observation). Each numeric_value represents exactly one source cell/observation, never several years. Historical comparative columns may come from the one requested filing. Cover every locked company/year/document and every requested output without inventing supporting requirements. A source_table is one coherent table per source, not separate rows and descriptions; retain its grouping dimensions in metric.

Inventory ONLY literal asset, product, business or segment names from the question in subjects. Companies and observations such as margins, revenue, concentration, backlog or outlook are not business owners. Never infer an industry restriction for an entire company. Reuse a named business subject across clauses when later clauses omit it; independent named businesses stay separate. Company-wide requirements have null subject_id. Preserve actual business/accounting qualifiers in input metrics and evidence questions, not output labels or formatting instructions.

retrieval_query translates abstract question labels into concrete disclosure terminology, including equivalent phrases for the underlying activities, inputs or assets. Do not merely repeat the label or company/year, which are already filtered. For business-model capital requirements, search for productive assets, infrastructure investment and capital expenditures, rather than only the abstract label or liquidity. Funding sources are a separate relationship. A description of resource use needs its operating drivers and investments; funding availability or a total alone does not explain what the business requires. evidence_question is standalone for this requirement and preserves its business, period, requested categories and accounting basis. Lists and broad explanations need the whole requested scope, not one example. Search terminology may expand; requested outputs must not.

Set synthesis=true only for a requested qualitative comparison or interpretation, including temporal trends and how companies/sources differ; otherwise false. A table of historical numbers or separate source summaries does not complete a requested trend explanation or qualitative comparison. Do not add unrequested arithmetic. Clarification is null unless genuinely missing USER parameters prevent research.
Before returning, check each original clause against the outputs, sources, periods, formulas and evidence requirements. Evidence questions and invented supporting outputs cannot broaden or narrow the original request."""),
        HumanMessage(content=json.dumps({"question": question, "locked_source_groups": list(scope.retrieval_groups),
            "requested_table_output": understanding.wants_table, "requested_full_table_output": understanding.wants_complete_table})),
    ]
    for attempt in range(2):
        draft = planner.invoke(messages)
        try:
            for item in [*draft.get("requested_objects", []), *draft.get("requirements", [])]:
                item["kind"] = {"numeric_value": "fact", "explanation": "narrative", "source_table": "statement"}.get(item.get("kind"), item.get("kind"))
            requested = draft.get("requested_objects")
            objects = {item.object_id: item for item in map(RequestedObject.model_validate, requested or [])}
            if "synthesis" in draft:
                draft["synthesis"] = question if draft["synthesis"] and (requested is None or any(obj.kind == "narrative" for obj in objects.values())) else None
            if requested is not None:
                if len(objects) != len(requested) or draft.get("requirements") and not objects:
                    raise ValueError("Inventory each explicitly requested result with a unique object ID.")
            inventory = draft.pop("subjects", [])
            subjects = {item["subject_id"]: item["name"] for item in inventory if item["entity_type"] in {"business_area", "asset_or_product"}
                        and re.search(r"\b" + r"[\W_]+".join(map(re.escape, re.findall(r"\w+", item["name"]))) + r"\b", question, re.I)}
            draft["calculations"] = [*draft.get("calculations", []), *(obj.calculation.model_copy(update={"object_id": obj.object_id}).model_dump()
                for obj in objects.values() if obj.kind == "calculation" and obj.calculation is not None)]
            for calculation in draft["calculations"]:
                if not calculation.get("object_id"):
                    owners = {item.get("object_id") for item in draft.get("requirements", [])
                              if item["requirement_id"] in calculation["input_requirement_ids"]}
                    if len(owners) == 1:
                        calculation["object_id"] = owners.pop()
            for item in draft.get("requirements", []):
                item["qualifiers"] = [value for value in item.get("qualifiers", [])
                    if normalize_financial_term(value).issubset(normalize_financial_term(question))]
                if requested is not None:
                    obj = objects.get(item.get("object_id"))
                    if obj is None:
                        raise ValueError("Every evidence requirement must reference an actual requested object.")
                    item["kind"] = "fact" if obj.kind == "calculation" else obj.kind
                    if obj.kind == "statement":
                        subject = obj.subject if scoped_content_query(obj.subject, scope) else scoped_content_query(
                            item.get("retrieval_query") or obj.quote, scope)
                        item.update(metric=subject, reported_period=obj.reported_period, evidence_question=question)
                    elif obj.kind == "fact":
                        item["reported_period"] = item.get("reported_period") or obj.reported_period
                elif understanding.wants_complete_table and (
                    understand_query(item.get("evidence_question") or "").wants_complete_table
                    or any(StructuredDocumentLookup._terms(name.replace("_", " ")).issubset(
                        StructuredDocumentLookup._terms(item.get("metric", ""))) for name in get_args(StatementType))
                ):
                    item["kind"] = "statement"
                if requested is None and item["kind"] == "statement":
                    content = scoped_content_query(item.get("retrieval_query") or item.get("evidence_question") or item["metric"], scope)
                    if StructuredDocumentLookup._terms(item["metric"]) < StructuredDocumentLookup._terms(content):
                        item["metric"] = content
                subject_id = item.pop("subject_id", None)
                if subject_id is not None and subject_id not in subjects:
                    kind = next((item["entity_type"] for item in inventory if item["subject_id"] == subject_id), None)
                    subject_id = next(iter(subjects)) if kind == "metric_or_action" and len(subjects) == 1 else None
                subject = subjects.get(subject_id)
                if subject and normalize_financial_term(subject) == normalize_financial_term(item["metric"]):
                    subject = None
                output = objects.get(item.get("object_id"))
                if subject and output and output.kind == "calculation" and normalize_financial_term(subject) == (
                        normalize_financial_term(output.subject) - {"percent", "percentage", "ratio", "rate"}):
                    subject = None
                if subject:
                    item["subject"] = subject
                    item["qualifiers"] = list(dict.fromkeys([*(item.get("qualifiers") or []), subject]))
                    for field in ("evidence_question", "retrieval_query"):
                        if item.get(field):
                            item[field] = f"{subject}: {item[field]}"
            draft["narrative_subject"] = next(iter(subjects.values())) if len(subjects) == 1 and len(scope.retrieval_groups) > 1 else None
            plan = RequestPlanInput.model_validate(draft)
            if (understanding.wants_complete_table and DISPLAY_VERB_RE.search(understanding.normalized_query)
                    and plan.requirements and all(item.kind == "narrative" for item in plan.requirements)):
                raise ValueError("The original request asks to display a source table. Preserve that output requirement with kind=statement and its full table subject; narrative summaries cannot replace the table.")
            if plan.clarification and (not plan.clarification.rstrip().endswith("?")
                    or set(re.findall(r"\w+", plan.clarification.casefold())).issubset(
                        set(re.findall(r"\w+", question.casefold())))):
                raise ValueError("Clarification must request genuinely missing user input, not repeat the research question or give a caveat. Plan the requested evidence instead.")
            plan = plan.model_copy(update={"requirements": [item.model_copy(update={
                "reported_period": normalize_reported_period(item.reported_period or "", question)
                if item.kind == "fact"
                else item.reported_period}) for item in plan.requirements]})
            plan = plan.validate_scope(scope, bind_single_source=True)
            return plan
        except ValueError as error:
            if attempt:
                raise
            messages.extend([
                AIMessage(content=json.dumps(draft)),
                HumanMessage(content="Re-read the original question and repair the invalid parts of the plan. Do not invent calculations to repair a source-reported observation; use narrative evidence for reported changes. Preserve genuinely requested arithmetic and its separate operands. Errors: " +
                             json.dumps(error.errors(include_context=False, include_url=False, include_input=False)
                                        if isinstance(error, ValidationError) else [{"msg": str(error)}])),
            ])


def compact_agent_context(state: dict) -> dict:
    """Keep evidence in graph state; give the routing model coverage and diagnostics."""
    messages = state.get("messages", [])
    plan = None
    covered = set()
    fact_ids_by_requirement = {}
    for message in messages:
        if isinstance(message, ToolMessage) and message.name in {
            "plan_financial_request", "search_financial_documents", "get_financial_statement",
        }:
            try:
                payload = json.loads(message.content)
            except (TypeError, json.JSONDecodeError):
                continue
            if message.name == "plan_financial_request" and payload.get("plan_validated"):
                plan = payload.get("plan")
            else:
                covered.update(payload.get("coverage", {}).get("covered_requirement_ids", []))
                for key, values in payload.get("coverage", {}).get("fact_ids_by_requirement", {}).items():
                    fact_ids_by_requirement.setdefault(key, set()).update(values)
    if not plan:
        return {"llm_input_messages": messages}
    compacted = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.name in {"search_financial_documents", "get_financial_statement"}:
            try:
                payload = json.loads(message.content)
            except (TypeError, json.JSONDecodeError):
                compacted.append(message)
                continue
            if "coverage" in payload:
                payload["coverage"] = {
                    "fact_ids_by_requirement": {key: sorted(values) for key, values in fact_ids_by_requirement.items()},
                    "covered_requirement_ids": sorted(covered),
                    "missing_requirement_ids": [item["requirement_id"] for item in plan["requirements"]
                                                if item.get("kind", "fact") == "fact" and item["requirement_id"] not in covered],
                }
            # Extraction, quote selection and table rendering read original tool
            # messages. The routing model only chooses the next action.
            payload["evidence_stored"] = bool(payload.pop("context", None) or payload.get("sources"))
            for source in payload.get("sources", []):
                source.pop("evidence_text", None)
                source.pop("full_evidence_text", None)
            for fact in payload.get("validated_facts", []):
                fact.pop("evidence_excerpt", None)
                fact.pop("evidence_context", None)
            for rejected in payload.get("rejected_facts", []):
                rejected.pop("evidence_excerpt", None)
            message = message.model_copy(update={"content": json.dumps(payload, separators=(",", ":"))})
        compacted.append(message)
    return {"llm_input_messages": compacted}


def create_finrag_agent(
    retriever: HybridRetriever,
    calculator: FinancialCalculator,
    *,
    permitted_scope: Scope | None = None,
    plan_builder=build_request_plan,
):
    """Build a stateless compiled agent graph for one query at a time.

    Conversation context is supplied as bounded user/assistant history by the
    application. Omitting a checkpointer prevents old retrieval payloads and
    validated facts from leaking into later requests or growing without bound.
    """
    tools = create_financial_tools(
        retriever,
        calculator,
        permitted_scope=permitted_scope,
        plan_builder=plan_builder,
    )
    def route_agent_model(state, runtime):
        messages = state.get("messages", ())
        calls = {call["id"]: call for message in messages if isinstance(message, AIMessage)
                 for call in message.tool_calls}
        outcomes = []
        for message in messages:
            if isinstance(message, ToolMessage):
                try:
                    payload = json.loads(message.content)
                except (TypeError, json.JSONDecodeError):
                    continue  # Schema errors are diagnostic text, not terminal results.
                if isinstance(payload, dict):
                    outcomes.append((message.name, payload, calls.get(message.tool_call_id, {})))
        plan = next((payload["plan"] for name, payload, _ in reversed(outcomes)
                     if name == "plan_financial_request" and payload.get("plan_validated")), None)
        finals = [payload for name, payload, _ in outcomes if name == "finalize_financial_answer"]
        terminal = next((payload for _, payload, _ in reversed(outcomes)
            if payload.get("plan_attempt_exhausted") or payload.get("answer_validated")
            or payload.get("final_answer_attempt_exhausted")
            or payload.get("error_code") == "request_aborted"
            or (str(payload.get("error_code", "")).startswith("provider_") and payload.get("retryable") is False)), None)
        actions = []
        content = ""
        if terminal or sum(not payload.get("recovery_required") for payload in finals) >= 2:
            content = (terminal or {}).get("answer", "Research has finished with the available evidence.")
        elif plan and plan.get("clarification"):
            content = plan["clarification"]
        elif plan is None:
            actions.append(("plan_financial_request", {}))
        else:
            requirements = plan["requirements"]
            attempted, refined, searched = set(), set(), set()
            for name, payload, call in outcomes:
                if name not in {"search_financial_documents", "get_financial_statement"}:
                    continue
                args = call.get("args", {})
                retries = sum(other.get("name") == name and other.get("args") == args for other in calls.values())
                if payload.get("retryable") and retries < 2:
                    continue
                group = (args.get("ticker"), args.get("filing_year"), args.get("doc_type", "10K"))
                searched.add(group)
                ids = args.get("unresolved_requirement_ids", [])
                if payload.get("error") and payload.get("retryable") is False:
                    refined.update(ids)
                    refined.update(item["requirement_id"] for item in requirements
                        if (item["ticker"], item["source_year"], item["doc_type"]) == group)
                if ids:
                    attempted.update(ids)
                    if name == "search_financial_documents":
                        refined.update(ids)
                else:
                    attempted.update(item["requirement_id"] for item in requirements
                        if (item["ticker"], item["source_year"], item["doc_type"]) == group)
            pending = [item for item in requirements if item["requirement_id"] not in attempted]
            if not pending and finals and finals[-1].get("recovery_required"):
                missing = set(finals[-1]["missing_requirement_ids"]) - refined
                pending = [item for item in requirements if item["requirement_id"] in missing]
            groups = dict.fromkeys((item["ticker"], item["source_year"], item["doc_type"]) for item in pending)
            gaps = {item["requirement_id"]: item.get("missing_details", []) for item in
                    ((finals[-1].get("request_coverage") or {}).get("assessments", []) if finals else [])}
            if finals:
                for key, reason in finals[-1].get("missing_evidence", {}).items():
                    if not gaps.get(key):
                        gaps[key] = [reason]
            for ticker, year, kind in groups:
                if sum(call["name"] != "finalize_financial_answer" for call in calls.values()) + len(actions) >= AGENT_MAX_ACTIONS:
                    break
                items = [item for item in pending if (item["ticker"], item["source_year"], item["doc_type"]) == (ticker, year, kind)]
                table = next((item for item in items if item["kind"] == "statement"), None)
                if table:
                    actions.append(("get_financial_statement", {"ticker": ticker, "filing_year": year,
                        "statement_type": table["metric"], "unresolved_requirement_ids": [table["requirement_id"]]}))
                else:
                    args = {"ticker": ticker, "filing_year": year, "doc_type": kind,
                        "query": " ".join(dict.fromkeys(" ".join(filter(None, [item.get("subject"), item.get('retrieval_query'), *gaps[item["requirement_id"]]]))
                            if gaps.get(item["requirement_id"]) else item.get("retrieval_query") or item["metric"] for item in items))[:6000]}
                    if (ticker, year, kind) in searched:
                        args["unresolved_requirement_ids"] = [item["requirement_id"] for item in items]
                    actions.append(("search_financial_documents", args))
            if not actions:
                actions.append(("finalize_financial_answer", {}))
        action = AIMessage(content=content, tool_calls=[{"name": name, "args": args,
            "id": f"research_{len(messages)}_{index}", "type": "tool_call"} for index, (name, args) in enumerate(actions)])
        return RunnableLambda(lambda _: action)

    return create_react_agent(
        route_agent_model,
        tools,
        version="v2",
        name="finrag_agent",
    )
