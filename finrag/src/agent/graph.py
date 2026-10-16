"""LangGraph ReAct agent factory for the FinRAG financial research tools."""

from __future__ import annotations

import json
import re
from typing import Literal, get_args
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from langgraph.prebuilt import create_react_agent

from src.agent.prompts import SYSTEM_PROMPT
from src.agent.tools import RequestPlanInput, StatementType, create_financial_tools
from src.financial.calculator import FinancialCalculator
from src.financial.coverage import normalize_reported_period
from src.generation.llm_engine import get_llm_engine
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_understanding import DISPLAY_VERB_RE, understand_query
from src.retrieval.structured_lookup import StructuredDocumentLookup
from src.schemas import Scope
from src.observability import trace_component


class RequestedObject(BaseModel):
    """An actual requested result, distinct from its evidence inputs or table rows."""
    model_config = ConfigDict(extra="forbid", strict=True)
    object_id: str = Field(pattern=r"^o[1-9][0-9]*$")
    quote: str = Field(min_length=3, max_length=1000, description="Copy the exact original request phrase asking for this result. Do not invent an explanation or detail request.")
    kind: Literal["fact", "narrative", "statement", "calculation"]
    subject: str = Field(min_length=2, max_length=120, description="Short specific metric, narrative subject, table title or calculation, not the whole question. For a source table, identify the whole table; requested row breakdowns and descriptions are parts of it, not different tables. Exclude presentation instructions, company and source year.")
    reported_period: str | None = Field(default=None, max_length=80, description="Single requested observation period when applicable. Null for a full table with all its columns or a definition table. A filing's year alone does not request a period column. Never invent earlier periods.")


@trace_component("finrag.request_planning")
def build_request_plan(question: str, scope: Scope) -> RequestPlanInput:
    """Produce one structured plan directly from the request, before retrieval."""
    schema = RequestPlanInput.model_json_schema()
    requirement_schema = schema["$defs"]["PlannedRequirement"]
    schema["properties"]["requested_objects"] = {"type": "array", "maxItems": 40,
        "description": "Inventory ONLY results explicitly requested by the user BEFORE building evidence requirements. One whole source table is one object even when multiple rows, descriptions or year columns are requested. A scalar request is a fact object, without an additional commentary object. Calculation objects may require multiple numeric evidence inputs.",
        "items": RequestedObject.model_json_schema()}
    requirement_schema["properties"]["object_id"] = {"type": "string", "pattern": r"^o[1-9][0-9]*$",
        "description": "Reference the requested object this evidence requirement satisfies. Never add unrequested explanation requirements. Table rows/columns use the same table object."}
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
    if not understanding.wants_complete_table:
        schema["$defs"]["PlannedRequirement"]["properties"]["kind"]["enum"] = ["fact", "narrative"]
        schema["properties"]["requested_objects"]["items"]["properties"]["kind"]["enum"] = ["fact", "narrative", "calculation"]
    schema["$defs"]["PlannedRequirement"]["properties"]["ticker"]["enum"] = list(dict.fromkeys(ticker for ticker, year in scope.groups))
    schema["$defs"]["PlannedRequirement"]["properties"]["source_year"]["enum"] = list(dict.fromkeys(year for ticker, year in scope.groups))
    planner = get_llm_engine().with_structured_output(schema, method="json_schema", strict=True)
    messages = [
        SystemMessage(content="""Build a minimal COMPLETE research plan for the original question and locked source groups.
Never answer, predict values or calculate. source_year is the DOCUMENT year; reported_period is the actual/forecast observation. Earlier values inside one filing do not request another document.
Inventory requested_objects from exact user phrases FIRST. Requirements reference those objects. A value request needs values, not extra discussion. A full table with requested rows, explanations or comparative columns is ONE table object per source, not separate row/description tables. Calculations need their actual fact operands; merely reporting a value adds no calculation.

Inventory actual assets/businesses in subjects BEFORE splitting requirements. Decompose compound phrases: the underlying economic activity or product is distinct from its deployment, supply commentary, revenue or outlook. Name ONLY that business/asset component, not the whole observation phrase. Classify companies and metric/actions separately; they cannot be business scopes. Each observation references its underlying business/asset subject_id. Backlog, new products, deployment, changes and outlook are observations, NEVER separate business areas. A later comparison clause with no new business reuses the earlier business ID even when its source year changes. Explicitly different named topics get separate subject IDs. Company-wide metrics have null subject_id.
For example, comparing Acme's Widget division calls, with earlier supply commentary and later product/margin outlook, needs the Widget division scope in BOTH evidence questions, not Acme-wide product commentary. An independently named Services business would have its own scope.

Each explicitly requested scalar amount, count, balance or rate is kind=fact, even when several scalars are listed together. Never use narrative merely to quote a numeric table instead of extracting its cells. Calculator operands are also facts.
Use ONE narrative requirement per coherent descriptive observation in each source. Keep its source-reported changes, rates, bases, explanations and lists together. Do not add another requirement for a baseline year, a growth rate already covered, or an unrequested numeric level. Narrative evidence may contain quoted numbers. kind=statement is for ANY explicitly requested source table or schedule, including primary statements and footnote tables. Preserve the requested title/subject, qualifiers, rows and period columns; a summary cannot satisfy an exact table request. The evidence_question must explicitly retain the request for the exact/full table. A value or explanation under a table heading does not request that table. A computed comparison displayed as a table still uses fact requirements and calculator results.
Metric retains the actual business/asset/segment scope, excluding company, document and dates. Keep measurement qualifiers and accounting bases. Do not substitute a component, adjusted measure or all-assets aggregate. Units come from evidence, not metric-name guesses.
Each retrieval_query must include ALL requested observations, not only a topic noun. Each evidence_question is standalone, includes its business scope, and preserves all requested details for that source. A source-reported change query needs both the level and reported change commentary/columns in its search and evidence question. Only genuinely independent scalar observations need separate fact requirements.
narrative_subject is the shared business/product noun phrase for an elided comparison, not its requested actions or observations; otherwise null.

calculation_intent distinguishes READING a source-reported change from NEW requested arithmetic. Source-reported changes and drivers use quotations. Derived calculations need separate numeric operands and an approved operation; never use a computed change as an input.
Plan every requested calculation: temporal changes use earlier then later values for ONE company; cross-company difference uses first minus second. X as a percentage of Y uses percentage_of, X then Y; ratio divides without percentage scaling. Dollar and percentage change need absolute_change and percentage_change. CAGR includes elapsed periods. Preserve source signs, periods and requested decimal_places. Merely stating values does not request absolute_value.
Use synthesis only for requested qualitative interpretation beyond observations/calculations. Clarification is null unless genuinely missing USER parameters prevent research; do not ask the user to research.
Before returning, check every original clause for source, business, period and requested detail. Remove duplicates and unrequested observations. Evidence questions must not broaden a named subject or discard a requested detail."""),
        HumanMessage(content=json.dumps({"question": question, "locked_source_groups": list(scope.retrieval_groups),
            "requested_table_output": understanding.wants_table, "requested_full_table_output": understanding.wants_complete_table})),
    ]
    for attempt in range(2):
        draft = planner.invoke(messages)
        try:
            requested = draft.pop("requested_objects", None)
            objects = {item.object_id: item for item in map(RequestedObject.model_validate, requested or [])}
            if requested is not None:
                if len(objects) != len(requested) or draft.get("requirements") and not objects:
                    raise ValueError("Inventory each explicitly requested result with a unique object ID.")
                if len(objects) != len(requested):
                    raise ValueError("Requested objects must use unique object IDs.")
            inventory = draft.pop("subjects", [])
            subjects = {item["subject_id"]: item["name"] for item in inventory if item["entity_type"] in {"business_area", "asset_or_product"}}
            object_metrics = {}
            for item in draft.get("requirements", []):
                if requested is not None:
                    obj = objects.get(item.get("object_id"))
                    if obj is None:
                        raise ValueError("Every evidence requirement must reference an actual requested object.")
                    metric = object_metrics.setdefault(obj.object_id, item.get("metric") or obj.subject)
                    item["kind"] = "fact" if obj.kind == "calculation" else obj.kind
                    if obj.kind == "statement":
                        item.update(metric=metric, reported_period=obj.reported_period, evidence_question=question)
                    elif obj.kind == "fact":
                        item.update(metric=metric)
                        item["reported_period"] = item.get("reported_period") or obj.reported_period
                elif understanding.wants_complete_table and (
                    understand_query(item.get("evidence_question") or "").wants_complete_table
                    or any(StructuredDocumentLookup._terms(name.replace("_", " ")).issubset(
                        StructuredDocumentLookup._terms(item.get("metric", ""))) for name in get_args(StatementType))
                ):
                    item["kind"] = "statement"
                subject_id = item.pop("subject_id", None)
                if subject_id is not None and subject_id not in subjects:
                    kind = next((item["entity_type"] for item in inventory if item["subject_id"] == subject_id), None)
                    if kind == "company" or kind == "metric_or_action" and not subjects:
                        subject_id = None
                    elif kind == "metric_or_action" and len(subjects) == 1:
                        subject_id = next(iter(subjects))
                    else:
                        raise ValueError("Assign each observation to its actual business/asset scope: " + json.dumps(subjects))
                subject = subjects.get(subject_id)
                if subject:
                    item["subject"] = subject
                    item["qualifiers"] = list(dict.fromkeys([*(item.get("qualifiers") or []), subject]))
                    for field in ("evidence_question", "retrieval_query"):
                        if item.get(field):
                            item[field] = f"{subject}: {item[field]}"
            calculation_inputs = {
                requirement_id
                for calculation in draft.get("calculations", [])
                for requirement_id in calculation.get("input_requirement_ids", [])
            }
            for item in draft.get("requirements", []):
                if item.get("requirement_id") in calculation_inputs:
                    item["kind"] = "fact"
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
                else None if item.kind != "statement" else item.reported_period}) for item in plan.requirements]})
            plan = plan.complete_explicit_calculations(question)
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
    model = get_llm_engine(parallel_tool_calls=True)
    def route_agent_model(state, runtime):
        outcomes = []
        for message in state.get("messages", ()):
            if isinstance(message, ToolMessage) and message.name in {
                "plan_financial_request", "finalize_financial_answer"}:
                try:
                    payload = json.loads(message.content)
                except (TypeError, json.JSONDecodeError):
                    continue  # Schema errors are diagnostic text, not terminal results.
                if isinstance(payload, dict):
                    outcomes.append(payload)
        plan = next((payload["plan"] for payload in reversed(outcomes)
                     if payload.get("plan_validated")), None)
        has_plan = plan is not None
        # A planned research request ends through verification, never through an
        # unverified prose response. Clarifications and terminal results may stop.
        if (plan and plan.get("clarification")) or any(
            payload.get("plan_attempt_exhausted") or payload.get("answer_validated")
            or payload.get("final_answer_attempt_exhausted") for payload in outcomes
        ):
            return model
        active_tools = [tool for tool in tools if (tool.name == "plan_financial_request") != has_plan]
        return model.bind_tools(active_tools, tool_choice="required" if has_plan else "plan_financial_request")

    return create_react_agent(
        route_agent_model,
        tools,
        prompt=SYSTEM_PROMPT,
        pre_model_hook=compact_agent_context,
        version="v2",
        name="finrag_agent",
    )
