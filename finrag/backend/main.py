from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import asdict
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from time import perf_counter, monotonic
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.models import (
    CatalogueResponse,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    PendingClarificationPayload,
    RequestMetricsView,
    ScopePayload,
    ScopeView,
    SourceView,
)
from backend.service_manager import ServiceFactory, ServiceManager
from src.app.chat_service import ChatService
from src.generation.citations import compact_citations
from src.retrieval.clarification_policy import merge_clarification_reply
from src.schemas import ChatResult, PendingClarification, Scope
from src.telemetry import RequestMetrics, collect_usage
from src.request_runtime import RequestAborted, RequestControl, request_control
from src.config import (
    LANGSMITH_API_KEY_CONFIGURED,
    LANGSMITH_PROJECT,
    LANGSMITH_TRACING_ENABLED,
    MAX_CONCURRENT_REQUESTS,
    REQUEST_DEADLINE_SECONDS,
)


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
SSE_HEARTBEAT_SECONDS = 10.0


class ApiProblem(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _configured_cors_origins(explicit: list[str] | tuple[str, ...] | None) -> list[str]:
    if explicit is None:
        raw = os.getenv("FINRAG_CORS_ORIGINS", "")
        values = [value.strip() for value in raw.split(",") if value.strip()]
    else:
        values = [str(value).strip() for value in explicit if str(value).strip()]

    origins: list[str] = []
    for value in values:
        if value == "*":
            raise ValueError("FINRAG_CORS_ORIGINS cannot contain a wildcard.")
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError(f"Invalid CORS origin: {value!r}.")
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in origins:
            origins.append(origin)
    return origins


def _request_ids(request: Request) -> tuple[str, str]:
    request_id = getattr(request.state, "request_id", uuid4().hex)
    trace_id = getattr(request.state, "trace_id", uuid4().hex[:16])
    return request_id, trace_id


def _error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    details: list[dict[str, str]] | None = None,
) -> JSONResponse:
    request_id, trace_id = _request_ids(request)
    payload = {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
            "trace_id": trace_id,
        }
    }
    if details:
        payload["error"]["details"] = details
    return JSONResponse(status_code=status_code, content=payload)


def _manager(request: Request) -> ServiceManager:
    manager = getattr(request.app.state, "service_manager", None)
    if manager is None or not manager.ready:
        raise ApiProblem(503, "service_unavailable", "FinRAG is not ready.")
    return manager


def _service_for_request(manager: ServiceManager, payload: ChatRequest) -> ChatService:
    try:
        return manager.get(payload.index_version)
    except KeyError as exc:
        raise ApiProblem(
            422,
            "invalid_index_version",
            "The selected index is not available from this server.",
        ) from exc


def _available_dimensions(service: ChatService) -> tuple[set[str], set[str], set[str]]:
    keys = {
        (str(ticker), str(year), str(doc_type))
        for ticker, year, doc_type in service.available_keys
    }
    return (
        {ticker for ticker, _, _ in keys},
        {year for _, year, _ in keys},
        {doc_type for _, _, doc_type in keys},
    )


def _validate_scope_values(
    value: ScopePayload,
    *,
    service: ChatService,
    field_name: str,
) -> None:
    tickers, years, document_types = _available_dimensions(service)
    invalid_tickers = sorted(set(value.tickers) - tickers)
    invalid_years = sorted(set(value.years) - years)
    requested_types = set(value.required_doc_types)
    if value.doc_type:
        requested_types.add(value.doc_type)
    invalid_types = sorted(requested_types - document_types)
    invalid_group_tickers = sorted(
        {ticker for ticker, _ in value.requested_groups} - tickers
    )
    invalid_group_years = sorted({year for _, year in value.requested_groups} - years)

    if invalid_tickers or invalid_group_tickers:
        raise ApiProblem(
            422,
            "invalid_scope",
            f"{field_name} contains a company that is not in the selected index.",
        )
    if invalid_years or invalid_group_years:
        raise ApiProblem(
            422,
            "invalid_scope",
            f"{field_name} contains a fiscal year that is not in the selected index.",
        )
    if invalid_types:
        raise ApiProblem(
            422,
            "invalid_scope",
            f"{field_name} contains a document type that is not in the selected index.",
        )


def _scope_from_payload(value: ScopePayload) -> Scope:
    return Scope(
        tickers=tuple(value.tickers),
        years=tuple(value.years),
        doc_type=value.doc_type,
        requested_groups=tuple(value.requested_groups),
        required_doc_types=tuple(value.required_doc_types),
    )


def _pending_from_payload(
    value: PendingClarificationPayload | None,
) -> PendingClarification | None:
    if value is None:
        return None
    return PendingClarification(
        original_question=value.original_question,
        scope=_scope_from_payload(value.scope),
        candidate_tickers=tuple(value.candidate_tickers),
        missing_fields=tuple(value.missing_fields),
        query_expansions=tuple(value.query_expansions),
    )


def _prepare_call(
    service: ChatService, payload: ChatRequest
) -> tuple[Scope, Scope, list[dict[str, str]], PendingClarification | None]:
    previous = payload.previous_scope
    _validate_scope_values(previous, service=service, field_name="previous_scope")
    pending_payload = payload.pending_clarification
    pending = _pending_from_payload(pending_payload)
    # A new substantive question supersedes an old clarification. Detect that
    # before validating the stored scope so stale state from another index
    # cannot block an unrelated request.
    if pending is not None and merge_clarification_reply(pending, payload.question, available_keys=service.available_keys) is None:
        pending = None
        pending_payload = None
    if pending_payload is not None:
        _validate_scope_values(
            pending_payload.scope,
            service=service,
            field_name="pending_clarification.scope",
        )
        available_tickers, _, _ = _available_dimensions(service)
        if set(pending_payload.candidate_tickers) - available_tickers:
            raise ApiProblem(
                422,
                "invalid_scope",
                "pending_clarification contains an unavailable candidate company.",
            )

    tickers, years, document_types = _available_dimensions(service)
    filters = payload.filters
    if filters.company and filters.company not in tickers:
        raise ApiProblem(422, "invalid_filter", "The company filter is unavailable.")
    if filters.fiscal_year and filters.fiscal_year not in years:
        raise ApiProblem(422, "invalid_filter", "The fiscal-year filter is unavailable.")
    if filters.document_type and filters.document_type not in document_types:
        raise ApiProblem(422, "invalid_filter", "The document-type filter is unavailable.")

    ui_scope = Scope(
        tickers=(filters.company,) if filters.company else (),
        years=(filters.fiscal_year,) if filters.fiscal_year else (),
        doc_type=filters.document_type,
    )
    history = [message.model_dump() for message in payload.history]
    return ui_scope, _scope_from_payload(previous), history, pending


def _clean_text(value: object, *, max_length: int = 1_000) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text[:max_length] if text else None


def _safe_filename(value: object) -> str | None:
    text = _clean_text(value, max_length=500)
    if text is None:
        return None
    return PurePosixPath(text.replace("\\", "/")).name


def _scope_view(scope: Scope) -> ScopeView:
    return ScopeView(
        tickers=list(scope.tickers),
        years=list(scope.years),
        doc_type=scope.doc_type,
        requested_groups=list(scope.requested_groups),
        required_doc_types=list(scope.required_doc_types),
        label=scope.label(),
        complete=scope.complete,
    )


def _pending_view(
    pending: PendingClarification | None,
) -> PendingClarificationPayload | None:
    if pending is None:
        return None
    return PendingClarificationPayload(
        original_question=pending.original_question,
        scope=ScopePayload(
            tickers=list(pending.scope.tickers),
            years=list(pending.scope.years),
            doc_type=pending.scope.doc_type,
            requested_groups=list(pending.scope.requested_groups),
            required_doc_types=list(pending.scope.required_doc_types),
        ),
        candidate_tickers=list(pending.candidate_tickers),
        missing_fields=list(pending.missing_fields),
        query_expansions=list(pending.query_expansions),
    )


def _source_view(source: dict[str, Any]) -> SourceView | None:
    source_id = _clean_text(source.get("id"), max_length=32)
    if source_id is None:
        return None
    return SourceView(
        id=source_id,
        ticker=_clean_text(source.get("ticker"), max_length=32),
        fiscal_year=_clean_text(source.get("fiscal_year"), max_length=16),
        fiscal_period=_clean_text(source.get("fiscal_period"), max_length=120),
        fiscal_year_end=_clean_text(source.get("fiscal_year_end"), max_length=120),
        fiscal_q4_months=_clean_text(source.get("fiscal_q4_months"), max_length=120),
        doc_type=_clean_text(source.get("doc_type"), max_length=32),
        source=_safe_filename(source.get("source")),
        section=_clean_text(source.get("section"), max_length=500),
        pdf_page=_clean_text(source.get("pdf_page"), max_length=64),
        period_end_date=_clean_text(source.get("period_end_date"), max_length=64),
        call_date=_clean_text(source.get("call_date"), max_length=64),
        speaker=_clean_text(source.get("speaker"), max_length=200),
        speaker_role=_clean_text(source.get("speaker_role"), max_length=200),
        evidence_text=str(source.get("evidence_text") or ""),
    )


def _trace_scope(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "tickers": [str(item) for item in value.get("tickers", [])[:12]],
        "years": [str(item) for item in value.get("years", [])[:12]],
        "doc_type": _clean_text(value.get("doc_type"), max_length=32),
        "required_doc_types": [
            str(item) for item in value.get("required_doc_types", [])[:2]
        ],
        "requested_groups": [
            [str(part) for part in group[:2]]
            for group in value.get("requested_groups", [])[:24]
            if isinstance(group, (list, tuple))
        ],
    }


def _sanitize_trace(trace: object) -> dict[str, Any]:
    """Whitelist bounded tool diagnostics; omit retrieved context and model reasoning."""
    if not isinstance(trace, dict):
        return {}
    sanitized: dict[str, Any] = {}
    cache = trace.get("answer_cache")
    if isinstance(cache, dict):
        sanitized["answer_cache"] = {"hit": bool(cache.get("hit")),
            "source_trace_id": _clean_text(cache.get("source_trace_id"), max_length=64)}
    for key in ("request_id", "trace_id", "index_version"):
        value = trace.get(key)
        valid_value = (
            isinstance(value, str)
            and (
                REQUEST_ID_RE.fullmatch(value)
                if key != "index_version"
                else re.fullmatch(r"[A-Za-z0-9._-]{1,64}", value)
            )
        )
        if valid_value:
            sanitized[key] = value
    pipeline = trace.get("retrieval_pipeline")
    if isinstance(pipeline, list):
        sanitized["retrieval_pipeline"] = [str(item)[:80] for item in pipeline[:12]]

    reranker = trace.get("reranker")
    if isinstance(reranker, dict):
        sanitized["reranker"] = {
            "configured": bool(reranker.get("configured")),
            "provider": _clean_text(reranker.get("provider"), max_length=40),
            "model": _clean_text(reranker.get("model"), max_length=120),
        }

    understanding = trace.get("understanding")
    if isinstance(understanding, dict):
        sanitized["understanding"] = {
            "tickers": [str(item) for item in understanding.get("tickers", [])[:12]],
            "source_years": [
                str(item) for item in understanding.get("source_years", [])[:12]
            ],
            "reference_years": [
                str(item) for item in understanding.get("reference_years", [])[:12]
            ],
            "requested_doc_types": [
                str(item) for item in understanding.get("requested_doc_types", [])[:2]
            ],
            "generic_followup": bool(understanding.get("generic_followup")),
            "semantic_fallback_used": bool(
                understanding.get("semantic_fallback_used")
            ),
            "ambiguous_fields": [
                str(item)[:80]
                for item in understanding.get("ambiguous_fields", [])[:8]
            ],
            "query_expansions": [
                str(item)[:100]
                for item in understanding.get("query_expansions", [])[:3]
            ],
        }

    semantic = trace.get("semantic_fallback")
    if isinstance(semantic, dict):
        sanitized["semantic_fallback"] = {
            "configured": bool(semantic.get("configured")),
            "used": bool(semantic.get("used")),
            "reason": _clean_text(semantic.get("reason"), max_length=80),
            "candidate_tickers": [
                str(item)[:16] for item in semantic.get("candidate_tickers", [])[:3]
            ],
            "unsupported_companies": [
                str(item)[:80]
                for item in semantic.get("unsupported_companies", [])[:4]
            ],
            "accepted_fields": [
                str(item)[:80] for item in semantic.get("accepted_fields", [])[:8]
            ],
            "ambiguous_fields": [
                str(item)[:80] for item in semantic.get("ambiguous_fields", [])[:8]
            ],
            "query_expansions": [
                str(item)[:100] for item in semantic.get("query_expansions", [])[:3]
            ],
            "latency_ms": max(0.0, float(semantic.get("latency_ms", 0.0))),
            "error_type": _clean_text(semantic.get("error_type"), max_length=80),
        }

    clarification = trace.get("clarification")
    if isinstance(clarification, dict):
        sanitized["clarification"] = {
            "pending_received": bool(clarification.get("pending_received")),
            "reply_applied": bool(clarification.get("reply_applied")),
            "pending_returned": bool(clarification.get("pending_returned")),
            "candidate_tickers": [
                str(item)[:16]
                for item in clarification.get("candidate_tickers", [])[:3]
            ],
            "missing_fields": [
                str(item)[:80]
                for item in clarification.get("missing_fields", [])[:4]
            ],
        }

    for key in ("ui_scope", "previous_scope", "resolved_scope"):
        scope = _trace_scope(trace.get(key))
        if scope is not None:
            sanitized[key] = scope

    inherited = trace.get("inherited_fields")
    if isinstance(inherited, list):
        sanitized["inherited_fields"] = [str(item)[:80] for item in inherited[:8]]

    plan = trace.get("request_plan")
    if isinstance(plan, dict):
        sanitized["request_plan"] = {
            "requirements": [{
                **{key: _clean_text(item.get(key), max_length=240) for key in
                   ("requirement_id", "ticker", "source_year", "doc_type", "reported_period", "metric", "kind")},
                "qualifiers": [str(value)[:80] for value in item.get("qualifiers", [])[:12]],
            } for item in plan.get("requirements", [])[:40] if isinstance(item, dict)],
            "calculations": [{
                "operation": _clean_text(item.get("operation"), max_length=80),
                "input_requirement_ids": [str(value)[:80] for value in item.get("input_requirement_ids", [])[:20]],
                "periods": item.get("periods"),
            } for item in plan.get("calculations", [])[:20] if isinstance(item, dict)],
        }

    answer_validation = trace.get("answer_validation")
    if isinstance(answer_validation, dict):
        sanitized["answer_validation"] = {
            "checked": bool(answer_validation.get("checked")),
            "valid": bool(answer_validation.get("valid")),
            "reason": _clean_text(answer_validation.get("reason"), max_length=100),
            "attempts": answer_validation.get("attempts"),
            "missing_requirement_ids": [
                str(item)[:80] for item in answer_validation.get("missing_requirement_ids", [])[:40]
            ],
            "cited_source_ids": [
                str(item)[:32]
                for item in answer_validation.get("cited_source_ids", [])[:24]
            ],
            "missing_groups": [
                [str(part)[:40] for part in group[:3]]
                for group in answer_validation.get("missing_groups", [])[:12]
                if isinstance(group, (list, tuple))
            ],
            "missing_calculation_source_ids": [
                str(item)[:32]
                for item in answer_validation.get(
                    "missing_calculation_source_ids", []
                )[:24]
            ],
        }

    citation_repair = trace.get("citation_repair")
    if isinstance(citation_repair, dict):
        sanitized["citation_repair"] = {
            "attempted": bool(citation_repair.get("attempted")),
            "succeeded": bool(citation_repair.get("succeeded")),
        }
    answer_recovery = trace.get("answer_recovery")
    if isinstance(answer_recovery, dict):
        sanitized["answer_recovery"] = {
            "reason": _clean_text(answer_recovery.get("reason"), max_length=80),
            "used_partial_evidence": bool(
                answer_recovery.get("used_partial_evidence")
            ),
        }

    agent = trace.get("agent")
    if isinstance(agent, dict):
        agent_view: dict[str, Any] = {
            "name": _clean_text(agent.get("name"), max_length=80),
            "backend": _clean_text(agent.get("backend"), max_length=40),
            "selected": bool(agent.get("selected")),
            "status": _clean_text(agent.get("status"), max_length=40),
            "route": _clean_text(agent.get("route"), max_length=80),
            "tools": [str(item)[:80] for item in agent.get("tools", [])[:6]],
            "tool_calls": [str(item)[:80] for item in agent.get("tool_calls", [])[:24]],
        }
        call_details = []
        for call in agent.get("tool_call_details", [])[:24]:
            if not isinstance(call, dict):
                continue
            arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
            safe_arguments = {}
            for key in (
                "ticker", "filing_year", "doc_type", "query", "statement_type",
                "fact_id_1", "fact_id_2", "fact_ids", "operation", "periods",
                "decimal_places",
            ):
                value = arguments.get(key)
                if isinstance(value, str):
                    safe_arguments[key] = _clean_text(value, max_length=240)
                elif isinstance(value, (int, float, bool)):
                    safe_arguments[key] = value
                elif isinstance(value, list):
                    safe_arguments[key] = [str(item)[:100] for item in value[:20]]
            call_details.append({
                "id": _clean_text(call.get("id"), max_length=80),
                "name": _clean_text(call.get("name"), max_length=80),
                "arguments": safe_arguments,
            })
        agent_view["tool_call_details"] = call_details

        result_views = []
        for result in agent.get("tool_results", [])[:24]:
            if not isinstance(result, dict):
                continue
            view = {
                key: result.get(key)
                for key in (
                    "source_count", "context_chars", "validated_fact_count",
                    "rejected_fact_count"
                )
                if isinstance(result.get(key), int)
            }
            view.update({
                "name": _clean_text(result.get("name"), max_length=80),
                "status": _clean_text(result.get("status"), max_length=40),
                "tool_call_id": _clean_text(result.get("tool_call_id"), max_length=80),
                "source_ids": [str(item)[:32] for item in result.get("source_ids", [])[:24]],
            })
            for key in (
                "already_retrieved", "context_budget_exhausted",
                "action_budget_exhausted", "recovery_attempted", "recovery_exhausted",
                "answer_validated", "resolved_in_final_answer",
            ):
                if key in result:
                    view[key] = bool(result.get(key))
            for key in ("ticker", "filing_year", "doc_type", "statement_type", "statement_id"):
                if result.get(key) is not None:
                    view[key] = _clean_text(result[key], max_length=40)
            coverage = result.get("coverage")
            if isinstance(coverage, dict):
                view["coverage"] = {
                    key: [str(item)[:80] for item in coverage.get(key, [])[:40]]
                    for key in ("covered_requirement_ids", "missing_requirement_ids")
                }
            completeness = result.get("statement_completeness")
            if isinstance(completeness, dict):
                view["statement_completeness"] = {
                    key: completeness[key] for key in
                    ("complete", "truncated", "source_rows", "returned_rows", "rows", "scope")
                    if isinstance(completeness.get(key), (str, int, bool))
                }
                view["statement_completeness"]["period_columns"] = [
                    str(item)[:16] for item in completeness.get("period_columns", [])[:12]
                ]
            if isinstance(result.get("scope_adjustment"), dict):
                view["scope_adjustment"] = {
                    key: _clean_text(value, max_length=40)
                    for key, value in result["scope_adjustment"].items()
                    if key in {"requested_source_year", "locked_source_year"}
                }
            facts = []
            for fact in result.get("validated_facts", [])[:12]:
                if not isinstance(fact, dict):
                    continue
                facts.append({
                    key: _clean_text(fact.get(key), max_length=180)
                    for key in (
                        "fact_id", "ticker", "metric", "period", "raw_value",
                        "normalized_unit", "currency", "scale", "source_id",
                    )
                    if fact.get(key) is not None
                })
            view["validated_facts"] = facts
            if result.get("error"):
                view["error"] = _clean_text(result.get("error"), max_length=240)
                view["error_code"] = _clean_text(result.get("error_code"), max_length=80)
                view["retryable"] = bool(result.get("retryable"))
                view["resolved_in_final_answer"] = bool(result.get("resolved_in_final_answer"))
            result_views.append(view)
        agent_view["tool_results"] = result_views

        calculator_views = []
        for result in agent.get("calculator_results", [])[:12]:
            if not isinstance(result, dict):
                continue
            calculator_views.append({
                key: _clean_text(result.get(key), max_length=240)
                for key in (
                    "tool_call_id", "operation", "formula", "substituted_formula",
                    "result", "result_unit", "currency", "scale", "error",
                    "error_code", "instruction",
                )
                if result.get(key) is not None
            } | {
                key: [str(item)[:32] for item in result.get(key, [])[:20]]
                for key in ("input_fact_ids", "source_ids", "requested_fact_ids", "missing_fact_ids")
                if isinstance(result.get(key), list)
            } | {
                "calculation_skipped": bool(result.get("calculation_skipped"))
            })
        agent_view["calculator_results"] = calculator_views
        sanitized["agent"] = agent_view

    retrieval = trace.get("retrieval")
    if isinstance(retrieval, dict):
        sanitized["retrieval"] = {
            "tool": _clean_text(retrieval.get("tool"), max_length=80),
            "source_ids": [str(item)[:32] for item in retrieval.get("source_ids", [])[:24]],
            "source_groups": [
                [str(part)[:64] for part in group[:3]]
                for group in retrieval.get("source_groups", [])[:24]
                if isinstance(group, (list, tuple))
            ],
        }
    return sanitized


def _chat_response(
    result: ChatResult,
    *,
    metrics: RequestMetrics,
    request: Request,
    include_trace: bool,
    allow_traces: bool,
) -> ChatResponse:
    request_id, fallback_trace_id = _request_ids(request)
    result_trace = result.trace if isinstance(result.trace, dict) else {}
    candidate_trace_id = str(result_trace.get("trace_id", ""))
    trace_id = (
        candidate_trace_id
        if REQUEST_ID_RE.fullmatch(candidate_trace_id)
        else fallback_trace_id
    )
    sources = [
        view
        for source in result.sources
        if isinstance(source, dict)
        for view in [_source_view(source)]
        if view is not None
    ]
    decision = getattr(result.decision, "value", str(result.decision))
    answer, source_data = compact_citations(str(result.answer), [source.model_dump() for source in sources])
    return ChatResponse(
        request_id=request_id,
        trace_id=trace_id,
        decision=str(decision),
        answer=answer,
        scope=_scope_view(result.scope),
        inherited_fields=[str(item) for item in result.inherited_fields],
        sources=[SourceView.model_validate(source) for source in source_data],
        metrics=RequestMetricsView(
            latency_ms=metrics.latency_ms,
            llm_calls=metrics.llm_calls,
            input_tokens=metrics.input_tokens,
            cached_input_tokens=metrics.cached_input_tokens,
            output_tokens=metrics.output_tokens,
            total_tokens=metrics.total_tokens,
            estimated_cost_usd=metrics.estimated_cost_usd,
            models=[str(model)[:120] for model in metrics.models[:8]],
            stages=metrics.stages,
            embedding_calls=metrics.embedding_calls,
            embedding_input_tokens=metrics.embedding_input_tokens,
            embedding_cost_usd=metrics.embedding_cost_usd,
            total_token_cost_usd=metrics.total_token_cost_usd,
            known_token_cost_usd=metrics.known_token_cost_usd,
            cost_complete=metrics.cost_complete,
            cost_issues=list(metrics.cost_issues),
            pricing_revision=metrics.pricing_revision,
        ),
        pending_clarification=_pending_view(result.pending_clarification),
        trace=_sanitize_trace(result_trace) if include_trace and allow_traces else None,
    )


async def _measured_ask(
    service: Any,
    question: str,
    *,
    ui_scope: Scope,
    previous_scope: Scope,
    history: list[dict],
    pending_clarification: PendingClarification | None,
    conversation_id: str | None = None,
    request_id: str | None = None,
    trace_id: str | None = None,
    limiter: asyncio.Semaphore | None = None,
    pending_workers: dict | None = None,
    request: Request | None = None,
) -> tuple[ChatResult, RequestMetrics]:
    with collect_usage() as usage:
        started = perf_counter()
        ask_kwargs: dict[str, Any] = {
            "ui_scope": ui_scope,
            "previous_scope": previous_scope,
            "history": history,
            "pending_clarification": pending_clarification,
            "request_id": request_id,
            "trace_id": trace_id,
        }
        if conversation_id is not None:
            ask_kwargs["thread_id"] = conversation_id
        control = RequestControl(monotonic() + REQUEST_DEADLINE_SECONDS)
        async def work():
            queued = perf_counter()
            async with limiter or asyncio.Semaphore(1):
                usage.record_stage("finrag.api_queue", (perf_counter() - queued) * 1000)
                executing = perf_counter()
                try:
                    with request_control(control):
                        return await asyncio.to_thread(service.ask, question, **ask_kwargs)
                finally:
                    usage.record_stage("finrag.api_execution", (perf_counter() - executing) * 1000)
        worker = asyncio.create_task(work())
        if pending_workers is not None:
            pending_workers[worker] = control
        def finished(task):
            if pending_workers is not None:
                pending_workers.pop(task, None)
            if not task.cancelled():
                task.exception()  # Observe failures from disconnected clients.
            if control.cancelled.is_set():
                LOGGER.info("Request worker completed after cancellation: %s", json.dumps({
                    "request_id": request_id,
                    "metrics": asdict(usage.snapshot(latency_ms=(perf_counter() - started) * 1000)),
                }))
        worker.add_done_callback(finished)
        try:
            while request is not None and not worker.done():
                if await request.is_disconnected():
                    raise asyncio.CancelledError
                if monotonic() >= control.deadline:
                    raise TimeoutError
                await asyncio.wait({worker}, timeout=0.25)
            result = await asyncio.wait_for(
                asyncio.shield(worker), max(0.001, control.deadline - monotonic())
            )
        except asyncio.CancelledError:
            control.cancelled.set()
            raise
        except (TimeoutError, RequestAborted) as error:
            control.cancelled.set()
            raise ApiProblem(504, "request_timeout", "The request exceeded its time limit.") from error
        latency_ms = (perf_counter() - started) * 1000
    return result, usage.snapshot(latency_ms=latency_ms)


def _sse(event: str, data: dict[str, Any], *, event_id: int | None = None) -> str:
    parts = [f"event: {event}"]
    if event_id is not None:
        parts.append(f"id: {event_id}")
    parts.append(
        "data: " + json.dumps(data, ensure_ascii=True, separators=(",", ":"))
    )
    return "\n".join(parts) + "\n\n"


def create_app(
    *,
    service_factory: ServiceFactory = ChatService,
    index_versions: tuple[str, ...] | list[str] | None = None,
    allow_traces: bool | None = None,
    cors_origins: list[str] | tuple[str, ...] | None = None,
) -> FastAPI:
    manager = ServiceManager(
        service_factory=service_factory,
        index_versions=index_versions,
        allow_traces=allow_traces,
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        await asyncio.to_thread(manager.start)
        application.state.service_manager = manager
        application.state.request_limiter = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        application.state.pending_workers = {}
        try:
            yield
        finally:
            pending_workers = dict(application.state.pending_workers)
            for control in pending_workers.values():
                control.cancelled.set()
            if pending_workers:
                await asyncio.gather(*pending_workers, return_exceptions=True)
            await asyncio.to_thread(manager.close)
            application.state.service_manager = None

    application = FastAPI(
        title="FinRAG API",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    origins = _configured_cors_origins(cors_origins)
    if origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", "X-Request-ID"],
            max_age=600,
        )

    @application.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = uuid4().hex
        request.state.trace_id = uuid4().hex[:16]
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Trace-ID"] = request.state.trace_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'; "
            "object-src 'none'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
            "connect-src 'self'"
        )
        return response

    @application.exception_handler(ApiProblem)
    async def api_problem_handler(request: Request, error: ApiProblem) -> JSONResponse:
        return _error_response(
            request,
            status_code=error.status_code,
            code=error.code,
            message=error.message,
        )

    @application.exception_handler(RequestValidationError)
    async def validation_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        details = [
            {
                "field": ".".join(str(part) for part in item.get("loc", ())),
                "message": str(item.get("msg", "Invalid value.")),
                "type": str(item.get("type", "validation_error")),
            }
            for item in error.errors()
        ]
        return _error_response(
            request,
            status_code=422,
            code="validation_error",
            message="The request payload is invalid.",
            details=details,
        )

    @application.exception_handler(HTTPException)
    async def http_error_handler(request: Request, error: HTTPException) -> JSONResponse:
        message = str(error.detail) if isinstance(error.detail, str) else "Request failed."
        return _error_response(
            request,
            status_code=error.status_code,
            code="http_error",
            message=message,
        )

    @application.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, error: Exception) -> JSONResponse:
        request_id, _ = _request_ids(request)
        LOGGER.exception("Unhandled API error (request_id=%s).", request_id)
        return _error_response(
            request,
            status_code=500,
            code="internal_error",
            message="The request could not be completed.",
        )

    @application.get(
        "/api/health",
        response_model=HealthResponse,
        response_model_exclude_none=True,
    )
    async def health(request: Request) -> HealthResponse:
        service_manager = _manager(request)
        return HealthResponse(
            status="ok",
            service_ready=service_manager.ready,
            default_index=service_manager.default_index,
            index_count=len(service_manager.index_versions),
            langsmith_tracing_enabled=LANGSMITH_TRACING_ENABLED,
            langsmith_api_key_configured=LANGSMITH_API_KEY_CONFIGURED,
            langsmith_project=LANGSMITH_PROJECT,
        )

    @application.get(
        "/api/catalogue",
        response_model=CatalogueResponse,
        response_model_exclude_none=True,
    )
    async def catalogue(request: Request) -> CatalogueResponse:
        service_manager = _manager(request)
        return CatalogueResponse(
            default_index=service_manager.default_index,
            trace_available=service_manager.allow_traces,
            indexes=service_manager.catalogue(),
        )

    @application.post(
        "/api/chat",
        response_model=ChatResponse,
        response_model_exclude_none=True,
        responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    )
    async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        service_manager = _manager(request)
        service = _service_for_request(service_manager, payload)
        ui_scope, previous_scope, history, pending = _prepare_call(service, payload)
        request_id, trace_id = _request_ids(request)
        result, metrics = await _measured_ask(
            service,
            payload.question,
            ui_scope=ui_scope,
            previous_scope=previous_scope,
            history=history,
            pending_clarification=pending,
            conversation_id=payload.conversation_id,
            request_id=request_id,
            trace_id=trace_id,
            limiter=request.app.state.request_limiter,
            pending_workers=request.app.state.pending_workers,
            request=request,
        )
        return _chat_response(
            result,
            metrics=metrics,
            request=request,
            include_trace=payload.include_trace,
            allow_traces=service_manager.allow_traces,
        )

    @application.post(
        "/api/chat/stream",
        response_class=StreamingResponse,
        responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    )
    async def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
        service_manager = _manager(request)
        service = _service_for_request(service_manager, payload)
        ui_scope, previous_scope, history, pending = _prepare_call(service, payload)
        request_id, trace_id = _request_ids(request)

        async def events():
            ask_task: asyncio.Task[tuple[ChatResult, RequestMetrics]] | None = None
            try:
                if await request.is_disconnected():
                    return
                yield _sse(
                    "stage",
                    {
                        "stage": "request_validated",
                        "message": "Request and filters validated",
                    },
                    event_id=1,
                )
                if await request.is_disconnected():
                    return
                yield _sse(
                    "stage",
                    {
                        "stage": "finrag_pipeline",
                        "message": "Resolving scope and retrieving grounded evidence",
                    },
                    event_id=2,
                )
                ask_task = asyncio.create_task(
                    _measured_ask(
                        service,
                        payload.question,
                        ui_scope=ui_scope,
                        previous_scope=previous_scope,
                        history=history,
                        pending_clarification=pending,
                        conversation_id=payload.conversation_id,
                        request_id=request_id,
                        trace_id=trace_id,
                        limiter=request.app.state.request_limiter,
                        pending_workers=request.app.state.pending_workers,
                    )
                )
                while not ask_task.done():
                    done, _ = await asyncio.wait(
                        {ask_task}, timeout=SSE_HEARTBEAT_SECONDS
                    )
                    if await request.is_disconnected():
                        ask_task.cancel()
                        return
                    if not done:
                        yield ": heartbeat\n\n"

                result, metrics = await ask_task
                if await request.is_disconnected():
                    return
                yield _sse(
                    "stage",
                    {
                        "stage": "response_ready",
                        "message": "Validating response metadata and citations",
                    },
                    event_id=3,
                )
                response = _chat_response(
                    result,
                    metrics=metrics,
                    request=request,
                    include_trace=payload.include_trace,
                    allow_traces=service_manager.allow_traces,
                )
                yield _sse(
                    "result",
                    response.model_dump(mode="json", exclude_none=True),
                    event_id=4,
                )
            except asyncio.CancelledError:
                if ask_task is not None and not ask_task.done():
                    ask_task.cancel()
                raise
            except Exception:
                error_request_id, error_trace_id = _request_ids(request)
                LOGGER.exception(
                    "Unhandled streaming API error (request_id=%s).",
                    error_request_id,
                )
                if not await request.is_disconnected():
                    yield _sse(
                        "error",
                        {
                            "error": {
                                "code": "stream_error",
                                "message": "The request could not be completed.",
                                "request_id": error_request_id,
                                "trace_id": error_trace_id,
                            }
                        },
                    )
            finally:
                if ask_task is not None and not ask_task.done():
                    ask_task.cancel()

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @application.get("/", include_in_schema=False)
    async def frontend() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html", media_type="text/html")

    application.mount(
        "/static",
        StaticFiles(directory=FRONTEND_DIR, html=False, check_dir=True),
        name="static",
    )
    return application


app = create_app()
