from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
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
    ScopePayload,
    ScopeView,
    SourceView,
)
from backend.service_manager import ServiceFactory, ServiceManager
from src.app.chat_service import ChatService
from src.schemas import ChatResult, PendingClarification, Scope


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
    pending = payload.pending_clarification
    if pending is not None:
        _validate_scope_values(
            pending.scope,
            service=service,
            field_name="pending_clarification.scope",
        )
        available_tickers, _, _ = _available_dimensions(service)
        if set(pending.candidate_tickers) - available_tickers:
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
    return ui_scope, _scope_from_payload(previous), history, _pending_from_payload(pending)


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


def _finite_score(value: object) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if score == score and abs(score) != float("inf") else None


def _sanitize_trace(trace: object) -> dict[str, Any]:
    """Whitelist retrieval diagnostics; omit prompts, context, queries, and previews."""
    if not isinstance(trace, dict):
        return {}
    sanitized: dict[str, Any] = {}
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

    retrieval = trace.get("retrieval")
    if isinstance(retrieval, dict):
        retrieval_view: dict[str, Any] = {
            "candidate_count": max(0, int(retrieval.get("candidate_count", 0))),
            "source_ids": [str(item)[:32] for item in retrieval.get("source_ids", [])[:24]],
            "source_groups": [
                [str(part)[:64] for part in group[:3]]
                for group in retrieval.get("source_groups", [])[:24]
                if isinstance(group, (list, tuple))
            ],
            "reranker_applied": bool(retrieval.get("reranker_applied")),
        }
        trace_sources: list[dict[str, Any]] = []
        allowed_text = (
            "id",
            "ticker",
            "fiscal_year",
            "doc_type",
            "section",
            "source",
            "rerank_provider",
            "rerank_model",
        )
        allowed_scores = (
            "dense_score",
            "lexical_score",
            "lexical_coverage",
            "rrf_score",
            "pre_rerank_score",
            "rerank_score",
            "final_score",
        )
        for item in retrieval.get("sources", [])[:24]:
            if not isinstance(item, dict):
                continue
            entry = {
                key: _safe_filename(item.get(key))
                if key == "source"
                else _clean_text(item.get(key), max_length=500)
                for key in allowed_text
            }
            entry.update({key: _finite_score(item.get(key)) for key in allowed_scores})
            compression = item.get("compression")
            if isinstance(compression, dict):
                entry["compression"] = {
                    "status": _clean_text(compression.get("status"), max_length=80),
                    "original_len": max(0, int(compression.get("original_len", 0))),
                    "compressed_len": max(0, int(compression.get("compressed_len", 0))),
                    "reduction_pct": _finite_score(compression.get("reduction_pct")),
                }
            trace_sources.append(entry)
        retrieval_view["sources"] = trace_sources
        sanitized["retrieval"] = retrieval_view

    generation = trace.get("generation")
    if isinstance(generation, dict):
        sanitized["generation"] = {
            "attempts": max(0, int(generation.get("attempts", 0))),
            "validation_reason": _clean_text(
                generation.get("validation_reason"), max_length=200
            ),
        }
    return sanitized


def _chat_response(
    result: ChatResult,
    *,
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
    return ChatResponse(
        request_id=request_id,
        trace_id=trace_id,
        decision=str(decision),
        answer=str(result.answer),
        scope=_scope_view(result.scope),
        inherited_fields=[str(item) for item in result.inherited_fields],
        sources=sources,
        pending_clarification=_pending_view(result.pending_clarification),
        trace=_sanitize_trace(result_trace) if include_trace and allow_traces else None,
    )


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
        try:
            yield
        finally:
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
        result = await asyncio.to_thread(
            service.ask,
            payload.question,
            ui_scope=ui_scope,
            previous_scope=previous_scope,
            history=history,
            pending_clarification=pending,
        )
        return _chat_response(
            result,
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

        async def events():
            ask_task: asyncio.Task[ChatResult] | None = None
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
                    asyncio.to_thread(
                        service.ask,
                        payload.question,
                        ui_scope=ui_scope,
                        previous_scope=previous_scope,
                        history=history,
                        pending_clarification=pending,
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

                result = await ask_task
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
                request_id, trace_id = _request_ids(request)
                LOGGER.exception("Unhandled streaming API error (request_id=%s).", request_id)
                if not await request.is_disconnected():
                    yield _sse(
                        "error",
                        {
                            "error": {
                                "code": "stream_error",
                                "message": "The request could not be completed.",
                                "request_id": request_id,
                                "trace_id": trace_id,
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
