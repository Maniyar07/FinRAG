from __future__ import annotations

import json
import logging
import re
from dataclasses import replace
from datetime import datetime, timezone
from time import perf_counter
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.graph import create_finrag_agent
from src.app.answer_cache import AnswerCache
from src.financial.calculator import FinancialCalculator
from src.generation.llm_engine import get_llm_engine
from src.generation.answer_guardrails import (
    INSUFFICIENT_EVIDENCE_RESPONSE,
    UNVERIFIABLE_RESPONSE,
    validate_answer_payload,
)
from src.generation.citations import strip_citations
from src.ingestion.manifest import available_keys, load_manifest
from src.config import (
    AGENT_MAX_ACTIONS,
    ANSWER_CACHE_MAX_ENTRIES,
    ANSWER_CACHE_TTL_SECONDS,
    LOGS_DIR,
    get_index_paths,
)
from src.observability import annotate_current_trace, trace_component
from src.retrieval.context_builder import ContextBuilder
from src.retrieval.clarification_policy import (
    merge_clarification_reply,
    pending_from_resolution,
)
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.lexical_index import LexicalIndex
from src.retrieval.query_understanding import understand_query
from src.retrieval.semantic_query_parser import (
    apply_semantic_result,
    build_semantic_query_parser_from_environment,
    semantic_clarification_message,
    should_use_semantic_fallback,
)
from src.retrieval.scope_policy import resolve_scope
from src.retrieval.vector_store import FinancialVectorStore
from src.runtime_errors import RuntimeErrorInfo, classify_runtime_error, runtime_error_message
from src.request_runtime import RequestAborted, check_request
from src.telemetry import usage_stage
from src.schemas import (
    ChatResult,
    Decision,
    PendingClarification,
    Scope,
    ScopeResolution,
)
from src.retrieval.reranker import build_cohere_reranker_from_environment


class ChatService:
    """Resolve request scope and invoke the LangGraph financial agent."""

    def __init__(
        self,
        *,
        index_version: str | None = None,
        agent_graph: Any | None = None,
    ):
        paths = get_index_paths(index_version)
        self.require_validated_answer = agent_graph is None
        self.index_version = paths.version
        self.manifest = load_manifest(paths)
        self.available_keys = available_keys(self.manifest)
        self.vector_manager = FinancialVectorStore(index_version=paths.version)
        try:
            expected_points = int(self.manifest.get("counts", {}).get("child_chunks", -1))
            actual_points = self.vector_manager.point_count()
            if expected_points < 0 or actual_points != expected_points:
                raise RuntimeError(
                    f"Index point count differs from its manifest: "
                    f"expected={expected_points}, actual={actual_points}."
                )
            lexical_index = LexicalIndex(paths.lexical_index_path)
            expected_parents = int(
                self.manifest.get("counts", {}).get("parent_chunks", -1)
            )
            actual_parents = sum(1 for _ in paths.parent_docstore_dir.glob("*.json"))
            expected_children = expected_points
            if expected_parents < 0 or actual_parents != expected_parents:
                raise RuntimeError(
                    "Parent-store count differs from its manifest: "
                    f"expected={expected_parents}, actual={actual_parents}."
                )
            if len(lexical_index.records) != expected_children:
                raise RuntimeError(
                    "Lexical-index count differs from its manifest: "
                    f"expected={expected_children}, actual={len(lexical_index.records)}."
                )
            # Reranking is optional and fail-open. When it is disabled, has no
            # API key, or its API request fails, HybridRetriever keeps the
            # original dense + BM25 + RRF ordering.
            reranker = build_cohere_reranker_from_environment()
            self.retriever = HybridRetriever(
                self.vector_manager.get_vectorstore(),
                lexical_index=lexical_index,
                context_builder=ContextBuilder(paths.parent_docstore_dir),
                reranker=reranker,
            )
            permitted_groups = tuple(
                sorted(
                    {
                        (str(ticker), str(year))
                        for ticker, year, _ in self.available_keys
                    }
                )
            )
            permitted_doc_types = tuple(
                sorted({str(doc_type).upper() for _, _, doc_type in self.available_keys})
            )
            permitted_scope = (
                Scope(
                    requested_groups=permitted_groups,
                    required_doc_types=permitted_doc_types,
                )
                if permitted_groups
                else None
            )
            self.agent_graph = (
                agent_graph
                if agent_graph is not None
                else create_finrag_agent(
                    self.retriever,
                    FinancialCalculator(),
                    permitted_scope=permitted_scope,
                )
            )
            self.semantic_parser = build_semantic_query_parser_from_environment()
            self.logger = self._build_logger()
            self.answer_cache = (AnswerCache(ANSWER_CACHE_MAX_ENTRIES, ANSWER_CACHE_TTL_SECONDS)
                if ANSWER_CACHE_MAX_ENTRIES and self.require_validated_answer else None)
        except Exception:
            self.vector_manager.close()
            raise

    @staticmethod
    def _build_logger() -> logging.Logger:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("finrag.query")
        logger.setLevel(logging.INFO)
        log_path = (LOGS_DIR / "query_traces.jsonl").resolve()
        if not any(
            isinstance(handler, logging.FileHandler)
            and getattr(handler, "baseFilename", None) == str(log_path)
            for handler in logger.handlers
        ):
            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
        logger.propagate = False
        return logger

    @staticmethod
    def _scope_data(scope: Scope) -> dict:
        return {
            "tickers": list(scope.tickers),
            "years": list(scope.years),
            "doc_type": scope.doc_type,
            "required_doc_types": list(scope.required_doc_types),
            "requested_groups": [list(group) for group in scope.groups],
        }

    def _record(self, trace: dict) -> None:
        self.logger.info(json.dumps({"timestamp_utc": datetime.now(timezone.utc).isoformat(), **trace},
                                    ensure_ascii=True, sort_keys=True))

    def _company_confirmation_message(self) -> str:
        tickers = tuple(sorted({str(key[0]) for key in self.available_keys}))
        labels = ", ".join(tickers) or "the companies in the selected index"
        return (
            "Please confirm every company in the comparison. The selected index "
            f"supports {labels}."
        )

    def _reranker_trace(self) -> dict:
        """Describe reranker state without logging the Cohere API key."""
        reranker = getattr(self.retriever, "reranker", None)
        if reranker is None:
            return {
                "configured": False,
                "provider": None,
                "model": None,
            }
        settings = getattr(reranker, "settings", None)
        return {
            "configured": True,
            "provider": "cohere",
            "model": getattr(settings, "model", None),
        }

    @staticmethod
    def _message_value(message: Any, key: str, default: Any = None) -> Any:
        if isinstance(message, dict):
            return message.get(key, default)
        return getattr(message, key, default)

    @classmethod
    def _message_kind(cls, message: Any) -> str | None:
        kind = cls._message_value(message, "type")
        if kind is None:
            kind = cls._message_value(message, "role")
        return str(kind).casefold() if kind is not None else None

    @classmethod
    def _message_text(cls, message: Any) -> str:
        content = cls._message_value(message, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    chunks.append(item["text"])
                else:
                    text = getattr(item, "text", None)
                    if isinstance(text, str):
                        chunks.append(text)
            return "\n".join(chunk for chunk in chunks if chunk).strip()
        return str(content or "").strip()

    @classmethod
    def _agent_sources(cls, messages: list[Any]) -> list[dict]:
        """Collect the evidence payloads returned by graph search tools."""
        sources: list[dict] = []
        seen: dict[tuple[str, str, str, str, str], dict] = {}
        for message in messages:
            if cls._message_kind(message) != "tool":
                continue
            try:
                payload = json.loads(cls._message_text(message))
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or not isinstance(payload.get("sources"), list):
                continue
            evidence_by_id: dict[str, str] = {}
            context = payload.get("context")
            if isinstance(context, str):
                blocks = re.split(r"(?=^\[SOURCE\s+S\d+\s*\|)", context, flags=re.MULTILINE)
                for block in blocks:
                    header = re.match(r"^\[SOURCE\s+(S\d+)\s*\|[^\]]*\]\s*\n?", block)
                    if header:
                        evidence_by_id[header.group(1)] = block[header.end():].strip()
            for source in payload["sources"]:
                if not isinstance(source, dict):
                    continue
                item = dict(source)
                item.setdefault("ticker", payload.get("ticker"))
                item.setdefault(
                    "fiscal_year", payload.get("filing_year", payload.get("year"))
                )
                item.setdefault("doc_type", payload.get("doc_type"))
                item.setdefault(
                    "evidence_text",
                    evidence_by_id.get(str(item.get("id") or ""), ""),
                )
                identity = (
                    str(item.get("ticker") or ""),
                    str(item.get("fiscal_year") or ""),
                    str(item.get("doc_type") or ""),
                    str(item.get("id") or ""),
                    str(item.get("parent_id") or item.get("id") or ""),
                )
                if identity in seen:
                    previous = seen[identity]
                    old, new = str(previous.get("evidence_text") or ""), str(item.get("evidence_text") or "")
                    if old in new:
                        previous["evidence_text"] = new
                    elif new and new not in old:
                        previous["evidence_text"] = old + "\n\n" + new
                    continue
                seen[identity] = item
                sources.append(item)
        return sources

    @classmethod
    def _agent_tool_diagnostics(cls, messages: list[Any]) -> dict[str, Any]:
        """Capture compact tool inputs and outputs without duplicating filing text."""
        calls: list[str] = []
        call_details: list[dict[str, Any]] = []
        results: list[dict] = []
        calculator_results: list[dict] = []
        for message in messages:
            kind = cls._message_kind(message)
            if kind in {"ai", "assistant"}:
                for call in cls._message_value(message, "tool_calls", ()) or ():
                    if isinstance(call, dict):
                        name = call.get("name")
                        function = call.get("function")
                        if name is None and isinstance(function, dict):
                            name = function.get("name")
                        arguments = call.get("args")
                        if arguments is None and isinstance(function, dict):
                            arguments = function.get("arguments")
                        call_id = call.get("id") or call.get("tool_call_id")
                    else:
                        name = getattr(call, "name", None)
                        arguments = getattr(call, "args", None)
                        call_id = getattr(call, "id", None)
                    if name:
                        tool_name = str(name)
                        calls.append(tool_name)
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError:
                                arguments = {"raw": arguments[:1000]}
                        call_details.append(
                            {
                                "id": str(call_id) if call_id is not None else None,
                                "name": tool_name,
                                "arguments": arguments if isinstance(arguments, (dict, list)) else {},
                            }
                        )
                continue
            if kind != "tool":
                continue

            content = cls._message_text(message)
            try:
                payload = json.loads(content)
            except (TypeError, json.JSONDecodeError):
                payload = None
            entry = {
                "name": cls._message_value(message, "name"),
                "status": cls._message_value(message, "status", "success"),
                "tool_call_id": cls._message_value(message, "tool_call_id"),
            }
            if isinstance(payload, dict):
                raw_sources = payload.get("sources", ())
                raw_facts = payload.get("validated_facts", ())
                raw_sources = raw_sources if isinstance(raw_sources, list) else []
                raw_facts = raw_facts if isinstance(raw_facts, list) else []
                entry["source_count"] = len(raw_sources)
                context = payload.get("context")
                if isinstance(context, str):
                    entry["context_chars"] = len(context)
                entry["source_ids"] = [
                    str(source.get("id"))
                    for source in raw_sources
                    if isinstance(source, dict) and source.get("id") is not None
                ]
                if not entry["source_ids"] and isinstance(payload.get("source_ids"), list):
                    entry["source_ids"] = [str(source_id) for source_id in payload["source_ids"]]
                entry["validated_fact_count"] = len(raw_facts)
                if payload.get("plan_validated") is not None:
                    entry["plan_validated"] = payload["plan_validated"]
                    if isinstance(payload.get("plan"), dict):
                        entry["request_plan"] = payload["plan"]
                if payload.get("context_tokens") is not None:
                    entry["context_tokens"] = payload["context_tokens"]
                for field in ("coverage", "recovery_attempted", "recovery_exhausted",
                              "statement_completeness", "statement_id", "answer_validated",
                              "complete", "missing_requirement_ids", "final_answer_attempts", "request_coverage"):
                    if field in payload:
                        entry[field] = payload[field]
                if payload.get("validation_error"):
                    entry["error"] = payload["validation_error"]
                    entry["status"] = "error"
                    entry["error_code"] = "answer_validation_rejected"
                if payload.get("candidate_count") is not None:
                    entry["candidate_count"] = payload["candidate_count"]
                if isinstance(payload.get("covered_groups"), list):
                    entry["covered_groups"] = payload["covered_groups"]
                if isinstance(payload.get("retrieval_diagnostics"), dict):
                    entry["retrieval_diagnostics"] = payload[
                        "retrieval_diagnostics"
                    ]
                fact_fields = (
                    "fact_id", "ticker", "metric", "period", "raw_value",
                    "numeric_value", "base_value", "value_type", "normalized_unit",
                    "currency", "scale", "accounting_basis", "source_id",
                    "row_label", "column_label", "validation_checks",
                    "requirement_id", "evidence_excerpt",
                    "value_qualifier",
                    "evidence_context",
                )
                entry["validated_facts"] = [
                    {key: fact.get(key) for key in fact_fields if key in fact}
                    for fact in raw_facts
                    if isinstance(fact, dict)
                ]
                entry["sources"] = [
                    {
                        key: source.get(key)
                        for key in (
                            "id", "ticker", "fiscal_year", "doc_type", "source",
                            "section", "pdf_page", "parent_id", "evidence_mode",
                        )
                        if key in source
                    }
                    for source in raw_sources
                    if isinstance(source, dict)
                ]
                for field in (
                    "ticker", "filing_year", "year", "doc_type", "statement_type"
                ):
                    if payload.get(field) is not None:
                        entry[field] = payload[field]
                if payload.get("rejected_fact_count") is not None:
                    entry["rejected_fact_count"] = payload["rejected_fact_count"]
                if isinstance(payload.get("scope_adjustment"), dict):
                    entry["scope_adjustment"] = payload["scope_adjustment"]
                raw_rejections = payload.get("rejected_facts", ())
                if isinstance(raw_rejections, list):
                    entry["rejected_facts"] = [
                        {
                            key: rejected.get(key)
                            for key in (
                                "source_id", "metric", "reason", "period",
                                "raw_value", "row_label", "column_label",
                            )
                            if key in rejected
                        }
                        for rejected in raw_rejections
                        if isinstance(rejected, dict)
                    ]

                calc_fields = (
                        "operation", "formula", "substituted_formula", "result",
                        "result_unit", "currency", "scale", "input_fact_ids",
                        "source_ids", "requested_fact_ids", "missing_fact_ids",
                        "available_validated_facts", "error", "instruction",
                        "calculation_skipped",
                        "error_code", "error_type", "retryable", "error_detail",
                        "calculation_id", "periods", "planned_input_requirement_ids", "input_order_adjusted",
                )
                if "calculator" in str(entry["name"] or "").casefold() or (
                    "operation" in payload and "result" in payload
                ):
                    calculator_result = {
                        key: payload.get(key) for key in calc_fields if key in payload
                    }
                    entry["calculator_result"] = calculator_result
                    calculator_results.append(
                        {"tool_call_id": entry["tool_call_id"], **calculator_result}
                    )
                if isinstance(payload.get("automatic_calculator_results"), list):
                    entry["automatic_calculator_results"] = payload["automatic_calculator_results"]
                    calculator_results.extend({"tool_call_id": entry["tool_call_id"], "execution_mode": "planned",
                        **{key: calculation[key] for key in calc_fields if key in calculation}}
                        for calculation in payload["automatic_calculator_results"] if isinstance(calculation, dict))
                if payload.get("error"):
                    entry["error"] = str(payload["error"])[:300]
                    entry["status"] = "error"
                    for field in (
                        "error_code", "error_type", "retryable", "error_detail"
                    ):
                        if field in payload:
                            entry[field] = payload[field]
                elif str(entry.get("status") or "").casefold() == "error" and not entry.get("error"):
                    entry["error"] = content[:300]
            elif entry["status"] == "error":
                # Framework validation messages put the actual field error after
                # echoed arguments; preserve that detail for evaluation/debugging.
                entry["error"] = content[-600:]
            results.append(entry)
        return {
            "tool_calls": calls,
            "tool_call_details": call_details,
            "tool_results": results,
            "calculator_results": calculator_results,
        }

    def _provider_failure_result(self, diagnostics, trace, scope, sources, inherited_fields):
        failure = next((item for item in reversed(diagnostics["tool_results"])
            if str(item.get("error_code", "")).startswith("provider_")
            and not item.get("resolved_in_final_answer")), None)
        if failure is None:
            return None
        info = RuntimeErrorInfo(failure["error_code"], bool(failure.get("retryable")))
        trace.update(error_code=info.code, retryable=info.retryable,
                     error_type=failure.get("error_type", "ProviderError"))
        trace["agent"] = {**trace.get("agent", {}), **diagnostics, "status": "failed"}
        trace["answer_validation"] = {"checked": True, "valid": False, "reason": "provider_failure"}
        answer = runtime_error_message(info)
        self._record({**trace, "decision": Decision.ERROR.value, "answer": answer})
        return ChatResult(Decision.ERROR, answer, scope, sources, inherited_fields, trace)

    def process_query(
        self,
        inputs: dict[str, Any],
        *,
        thread_id: str,
        scope: Scope,
        inherited_fields: tuple[str, ...] = (),
        trace: dict | None = None,
    ) -> ChatResult:
        """Invoke the compiled LangGraph agent and adapt its answer for the UI."""
        trace = dict(trace or {})
        index_version = str(getattr(self, "index_version", "unknown") or "unknown")
        messages: list[Any] = []
        finalized_payload: dict | None = None
        recovery_reason: str | None = None
        # A graph run is isolated even if an injected graph has a checkpointer.
        # Conversation continuity is passed explicitly as bounded chat history.
        graph_run_id = uuid4().hex
        config = {
            "configurable": {"thread_id": graph_run_id},
            # Each turn includes the context hook, model node and tool node.
            # Reserve finalization and its one repair in addition to tool actions.
            "recursion_limit": 3 * (AGENT_MAX_ACTIONS + 2) + 1,
            "run_name": "finrag_agent",
            "tags": [
                "finrag",
                "query",
                "langgraph",
                f"index:{index_version}",
                *(f"ticker:{ticker}" for ticker in scope.tickers),
                *(f"year:{year}" for year in scope.years),
                *(
                    f"doc_type:{doc_type}"
                    for doc_type in (
                        scope.required_doc_types
                        or ((scope.doc_type,) if scope.doc_type else ())
                    )
                ),
            ],
            "metadata": {
                "application": "finrag",
                "request_id": str(trace.get("request_id") or ""),
                "trace_id": str(trace.get("trace_id") or ""),
                "index_version": index_version,
                "scope_tickers": list(scope.tickers),
                "scope_years": list(scope.years),
                "scope_doc_types": list(
                    scope.required_doc_types
                    or ((scope.doc_type,) if scope.doc_type else ())
                ),
                "scope_tickers_csv": ",".join(scope.tickers),
                "scope_years_csv": ",".join(scope.years),
                "scope_doc_types_csv": ",".join(
                    scope.required_doc_types
                    or ((scope.doc_type,) if scope.doc_type else ())
                ),
                "scope_group_count": len(scope.retrieval_groups),
                "comparison": scope.is_comparison,
            },
        }
        try:
            stream = getattr(self.agent_graph, "stream", None)
            if callable(stream):
                state: Any = {}
                for state in stream(inputs, config=config, stream_mode="values"):
                    check_request()
                    if isinstance(state, dict) and isinstance(state.get("messages"), list):
                        messages = state["messages"]
                        for message in reversed(messages):
                            if self._message_kind(message) != "tool":
                                break
                            try:
                                payload = json.loads(self._message_text(message))
                            except (TypeError, json.JSONDecodeError):
                                continue
                            if isinstance(payload, dict) and payload.get(
                                "answer_validated"
                            ):
                                finalized_payload = payload
                                break
                            if isinstance(payload, dict) and payload.get(
                                "final_answer_attempt_exhausted"
                            ):
                                recovery_reason = "answer_validation_exhausted"
                                break
                            if isinstance(payload, dict) and payload.get("plan_attempt_exhausted"):
                                recovery_reason = "request_plan_validation_failed"
                                break
                            if (isinstance(payload, dict)
                                    and (payload.get("context_budget_exhausted") or payload.get("action_budget_exhausted"))
                                    and not getattr(self, "require_validated_answer", False)
                                    and not any(self._message_value(item, "name") == "plan_financial_request" for item in messages)):
                                recovery_reason = "tool_action_budget" if payload.get("action_budget_exhausted") else "retrieval_context_budget"
                                break
                    if recovery_reason or finalized_payload:
                        break
            else:
                state = self.agent_graph.invoke(inputs, config=config)
                messages = state.get("messages", []) if isinstance(state, dict) else []
            if recovery_reason:
                raise RuntimeError("The agent reached its request tool budget.")
            if not isinstance(messages, list):
                raise RuntimeError("The financial agent returned an invalid message state.")

            final_message = next(
                (
                    message
                    for message in reversed(messages)
                    if self._message_kind(message) in {"ai", "assistant"}
                    and not self._message_value(message, "tool_calls", ())
                    and self._message_text(message)
                ),
                None,
            )
            for message in messages:
                if self._message_kind(message) == "tool" and self._message_value(message, "name") == "finalize_financial_answer":
                    try:
                        payload = json.loads(self._message_text(message))
                    except (TypeError, json.JSONDecodeError):
                        continue
                    if isinstance(payload, dict) and payload.get("answer_validated"):
                        finalized_payload = payload
            if final_message is None and finalized_payload is None:
                raise RuntimeError("The financial agent returned no assistant answer.")
            answer = str(finalized_payload["answer"]) if finalized_payload else self._message_text(final_message)
            plan_clarification = None
            request_plan = None
            for message in messages:
                if self._message_kind(message) != "tool":
                    continue
                try:
                    payload = json.loads(self._message_text(message))
                except (TypeError, json.JSONDecodeError):
                    continue
                if not isinstance(payload, dict):
                    continue
                if payload.get("plan_validated") and isinstance(payload.get("plan"), dict):
                    request_plan = payload["plan"]
                planned_question = (
                    payload.get("plan", {}).get("clarification")
                    if isinstance(payload.get("plan"), dict)
                    else None
                )
                if payload.get("clarification_required") or planned_question:
                    plan_clarification = str(
                        planned_question or payload.get("message") or answer
                    )
                    break
            if plan_clarification:
                original_question = self._latest_question_from_messages(
                    inputs.get("messages", ())
                )
                pending = PendingClarification(
                    original_question=original_question,
                    scope=scope,
                    missing_fields=("requested metric or analysis",),
                )
                trace["clarification"] = {
                    **trace.get("clarification", {}),
                    "pending_returned": True,
                    "source": "validated_request_plan",
                }
                result = ChatResult(
                    Decision.CLARIFY,
                    plan_clarification,
                    scope,
                    inherited_fields=inherited_fields,
                    trace=trace,
                    pending_clarification=pending,
                )
                self._record({**trace, "decision": result.decision.value})
                return result
            sources = self._agent_sources(messages)
            tool_diagnostics = self._agent_tool_diagnostics(messages)
            if request_plan is not None:
                trace["request_plan"] = request_plan
            if not (finalized_payload and finalized_payload.get("complete")):
                failure = self._provider_failure_result(tool_diagnostics, trace, scope, sources, inherited_fields)
                if failure is not None:
                    return failure
            if finalized_payload is not None:
                citation_check = validate_answer_payload(answer, sources)
                complete = bool(finalized_payload.get("complete"))
                decision = (
                    Decision.ANSWERED if complete and citation_check.valid
                    else Decision.VALIDATION_FAILED if complete
                    else Decision.INSUFFICIENT_EVIDENCE
                )
                if complete and not citation_check.valid:
                    answer = UNVERIFIABLE_RESPONSE
                if complete and citation_check.valid:
                    for result in tool_diagnostics["tool_results"]:
                        if result.get("error"):
                            result["resolved_in_final_answer"] = True
                trace["answer_validation"] = {
                    "checked": True, "valid": complete and citation_check.valid,
                    "reason": (citation_check.reason if complete and not citation_check.valid
                               else "validated_evidence_claims" if complete else "incomplete_request_plan"),
                    "missing_requirement_ids": finalized_payload.get("missing_requirement_ids", []),
                    "cited_source_ids": finalized_payload.get("cited_source_ids", []),
                    "attempts": finalized_payload.get("final_answer_attempts", 1),
                }
                trace["agent"] = {
                    **trace.get("agent", {}), "backend": "langgraph", "selected": True,
                    "status": "completed", "thread_id": graph_run_id, **tool_diagnostics,
                }
                self._record({**trace, "decision": decision.value, "answer": answer})
                return ChatResult(decision, answer, scope, sources, inherited_fields, trace)
            answer_validation = {
                "checked": False,
                "valid": False,
                "reason": None,
                "cited_source_ids": sorted(
                    set(re.findall(r"\[(S\d+)\]", answer, flags=re.IGNORECASE))
                ),
            }
            nonretryable_tool_errors = [
                result
                for result in tool_diagnostics["tool_results"]
                if result.get("status") == "error"
                and result.get("retryable") is not True
            ]
            normalized_answer = answer.casefold()
            if nonretryable_tool_errors:
                decision = Decision.VALIDATION_FAILED
                answer_validation.update(
                    checked=True,
                    reason="nonretryable_tool_error",
                    tool_error_codes=[
                        result.get("error_code")
                        for result in nonretryable_tool_errors
                        if result.get("error_code")
                    ],
                )
                answer = UNVERIFIABLE_RESPONSE
            elif not sources:
                decision = Decision.INSUFFICIENT_EVIDENCE
                answer = INSUFFICIENT_EVIDENCE_RESPONSE
            elif UNVERIFIABLE_RESPONSE.casefold() in normalized_answer:
                decision = Decision.VALIDATION_FAILED
            elif INSUFFICIENT_EVIDENCE_RESPONSE.casefold() in normalized_answer:
                decision = Decision.INSUFFICIENT_EVIDENCE
            else:
                citation_check = validate_answer_payload(answer, sources)
                if not citation_check.valid and citation_check.reason == "missing_source_ids":
                    repaired = self._repair_uncited_answer(
                        question=self._latest_question_from_messages(inputs.get("messages", ())),
                        draft=answer,
                        sources=sources,
                        diagnostics=tool_diagnostics,
                    )
                    trace["citation_repair"] = {
                        "attempted": True,
                        "succeeded": bool(repaired),
                    }
                    if repaired:
                        answer = repaired
                        citation_check = validate_answer_payload(answer, sources)
                answer_validation = {
                    "checked": True,
                    "valid": citation_check.valid,
                    "reason": citation_check.reason,
                    "cited_source_ids": list(citation_check.source_ids),
                }
                if citation_check.valid:
                    answer = citation_check.answer
                    cited_ids = set(citation_check.source_ids)
                    cited_sources = [
                        source for source in sources
                        if str(source.get("id", "")).upper() in cited_ids
                    ]
                    required_types = scope.required_doc_types or (scope.doc_type,)
                    missing_groups = [
                        (ticker, year, doc_type)
                        for ticker, year in scope.groups
                        for doc_type in required_types
                        if not any(
                            str(source.get("ticker", "")).upper() == ticker
                            and str(source.get("fiscal_year", "")) == year
                            and (
                                doc_type is None
                                or str(source.get("doc_type", "")).upper().replace("-", "")
                                == str(doc_type).upper().replace("-", "")
                            )
                            for source in cited_sources
                        )
                    ]
                    calculator_source_ids = {
                        str(source_id).upper()
                        for result in tool_diagnostics["calculator_results"]
                        if result.get("result") is not None
                        for source_id in (result.get("source_ids") or ())
                    }
                    missing_calculation_citations = sorted(
                        calculator_source_ids - cited_ids
                    )
                    planned_source_groups = {
                        (
                            str(requirement.get("ticker", "")).upper(),
                            str(requirement.get("source_year", "")),
                            str(requirement.get("doc_type", "")).upper().replace("-", ""),
                        )
                        for requirement in (request_plan or {}).get("requirements", ())
                        if isinstance(requirement, dict)
                    }
                    missing_planned_groups = [
                        group for group in sorted(planned_source_groups)
                        if not any(
                            str(source.get("ticker", "")).upper() == group[0]
                            and str(source.get("fiscal_year", "")) == group[1]
                            and str(source.get("doc_type", "")).upper().replace("-", "") == group[2]
                            for source in cited_sources
                        )
                    ]
                    remaining_calculations = list(tool_diagnostics["calculator_results"])
                    missing_planned_calculations = []
                    for planned in (request_plan or {}).get("calculations", ()):
                        match_index = next((
                            index for index, result in enumerate(remaining_calculations)
                            if result.get("operation") == planned.get("operation")
                            and result.get("result") is not None
                            and len(result.get("input_fact_ids") or ())
                            == len(planned.get("input_requirement_ids") or ())
                        ), None)
                        if match_index is None:
                            missing_planned_calculations.append(planned.get("operation"))
                        else:
                            remaining_calculations.pop(match_index)
                    if (
                        missing_groups
                        or missing_calculation_citations
                        or missing_planned_groups
                        or missing_planned_calculations
                        or request_plan is not None
                        or getattr(self, "require_validated_answer", False)
                    ):
                        reason = (
                            "missing_planned_calculation"
                            if missing_planned_calculations
                            else "missing_planned_source_group_citation"
                            if missing_planned_groups
                            else "missing_scope_group_citation"
                            if missing_groups
                            else "missing_calculation_source_citation"
                            if missing_calculation_citations
                            else "missing_validated_answer"
                        )
                        answer_validation.update(
                            valid=False,
                            reason=reason,
                            missing_groups=[list(group) for group in missing_groups],
                            missing_calculation_source_ids=missing_calculation_citations,
                            missing_planned_source_groups=[
                                list(group) for group in missing_planned_groups
                            ],
                            missing_planned_calculations=missing_planned_calculations,
                        )
                        answer = UNVERIFIABLE_RESPONSE
                        decision = Decision.VALIDATION_FAILED
                    else:
                        decision = Decision.ANSWERED
                else:
                    answer = UNVERIFIABLE_RESPONSE
                    decision = Decision.VALIDATION_FAILED
            trace["agent"] = {
                **trace.get("agent", {}),
                "name": "langgraph_financial_agent",
                "backend": "langgraph",
                "selected": True,
                "status": "completed",
                "route": "langgraph_react",
                "thread_id": graph_run_id,
                **tool_diagnostics,
            }
            trace["answer_validation"] = answer_validation
            trace["retrieval"] = {
                "tool": "search_financial_documents",
                "source_ids": [source.get("id") for source in sources],
                "source_groups": [
                    [source.get("ticker"), source.get("fiscal_year"), source.get("doc_type")]
                    for source in sources
                ],
            }
            self._record({**trace, "decision": decision.value, "answer": answer})
            return ChatResult(
                decision,
                answer,
                scope,
                sources,
                inherited_fields,
                trace,
            )
        except Exception as error:
            if isinstance(error, RequestAborted):
                raise
            error_info = classify_runtime_error(error)
            if recovery_reason is None:
                trace["error_type"] = type(error).__name__
                trace["error_code"] = error_info.code
                trace["retryable"] = error_info.retryable
                if error_info.detail:
                    trace["error_detail"] = error_info.detail
            trace["agent"] = {
                **trace.get("agent", {}),
                "name": "langgraph_financial_agent",
                "backend": "langgraph",
                "selected": True,
                "status": "validation_failed" if recovery_reason in {"answer_validation_exhausted", "request_plan_validation_failed"} else "budget_limited" if recovery_reason else "failed",
                "route": "langgraph_react",
            }
            if messages:
                partial_diagnostics = self._agent_tool_diagnostics(messages)
                trace["agent"].update(partial_diagnostics)
                partial_sources = self._agent_sources(messages)
                failure = self._provider_failure_result(partial_diagnostics, trace, scope, partial_sources, inherited_fields)
                if failure is not None:
                    return failure
                has_plan = any(result.get("plan_validated") for result in partial_diagnostics["tool_results"])
                if has_plan or getattr(self, "require_validated_answer", False):
                    decision = (Decision.VALIDATION_FAILED if recovery_reason in {"answer_validation_exhausted", "request_plan_validation_failed"}
                                else Decision.INSUFFICIENT_EVIDENCE if recovery_reason or type(error).__name__ == "GraphRecursionError"
                                else Decision.ERROR)
                    trace["answer_validation"] = {
                        "checked": True, "valid": False,
                        "reason": recovery_reason or "graph_stopped_before_validated_answer",
                    }
                    self._record({**trace, "decision": decision.value})
                    return ChatResult(
                        decision, "The request plan could not be validated before retrieval. Please rephrase this request." if recovery_reason == "request_plan_validation_failed" else
                        UNVERIFIABLE_RESPONSE if decision == Decision.VALIDATION_FAILED else
                        INSUFFICIENT_EVIDENCE_RESPONSE if decision == Decision.INSUFFICIENT_EVIDENCE else
                        runtime_error_message(error_info),
                        scope, partial_sources, inherited_fields, trace,
                    )
                trace["retrieval"] = {
                    "tool": "search_financial_documents",
                    "source_ids": [source.get("id") for source in partial_sources],
                    "source_groups": [
                        [source.get("ticker"), source.get("fiscal_year"), source.get("doc_type")]
                        for source in partial_sources
                    ],
                }
                recoverable_run = (
                    type(error).__name__ == "GraphRecursionError"
                    or recovery_reason is not None
                )
                if recoverable_run and partial_sources:
                    partial_answer = self._repair_uncited_answer(
                        question=self._latest_question_from_messages(
                            inputs.get("messages", ())
                        ),
                        draft="",
                        sources=partial_sources,
                        diagnostics=partial_diagnostics,
                    )
                    answer_check = validate_answer_payload(
                        partial_answer, partial_sources
                    )
                    cited_ids = set(answer_check.source_ids)
                    cited_sources = [
                        source for source in partial_sources
                        if str(source.get("id", "")).upper() in cited_ids
                    ]
                    required_types = scope.required_doc_types or (scope.doc_type,)
                    missing_groups = [
                        (ticker, year, doc_type)
                        for ticker, year in scope.groups
                        for doc_type in required_types
                        if not any(
                            str(source.get("ticker", "")).upper() == ticker
                            and str(source.get("fiscal_year", "")) == year
                            and (
                                doc_type is None
                                or str(source.get("doc_type", "")).upper().replace("-", "")
                                == str(doc_type).upper().replace("-", "")
                            )
                            for source in cited_sources
                        )
                    ]
                    required_calculation_sources = {
                        str(source_id).upper()
                        for result in partial_diagnostics["calculator_results"]
                        if result.get("result") is not None
                        for source_id in (result.get("source_ids") or ())
                    }
                    if (
                        answer_check.valid
                        and not missing_groups
                        and required_calculation_sources.issubset(cited_ids)
                    ):
                        trace["agent"]["status"] = "recovered"
                        trace["answer_validation"] = {
                            "checked": True,
                            "valid": True,
                            "reason": "graph_recursion_recovered",
                            "cited_source_ids": sorted(cited_ids),
                        }
                        trace["answer_recovery"] = {
                            "reason": recovery_reason or "graph_recursion",
                            "used_partial_evidence": True,
                        }
                        self._record({**trace, "decision": Decision.ANSWERED.value})
                        return ChatResult(
                            Decision.ANSWERED,
                            answer_check.answer,
                            scope,
                            partial_sources,
                            inherited_fields,
                            trace,
                        )
            self._record({**trace, "decision": Decision.ERROR.value})
            return ChatResult(
                Decision.ERROR,
                runtime_error_message(error_info),
                scope,
                inherited_fields=inherited_fields,
                trace=trace,
            )

    @staticmethod
    def _latest_question_from_messages(messages: object) -> str:
        if not isinstance(messages, (list, tuple)):
            return ""
        for message in reversed(messages):
            role = ChatService._message_value(message, "role") or ChatService._message_value(message, "type")
            if str(role).casefold() not in {"user", "human"}:
                continue
            content = ChatService._message_text(message)
            marker = "User question:"
            return content.rsplit(marker, 1)[-1].strip() if marker in content else content
        return ""

    def _repair_uncited_answer(
        self,
        *,
        question: str,
        draft: str,
        sources: list[dict],
        diagnostics: dict[str, Any],
    ) -> str:
        """Repair citation omissions once, using only the current run's evidence."""
        evidence = {
            "validated_facts": [
                fact
                for result in diagnostics.get("tool_results", ())
                for fact in result.get("validated_facts", ())
            ],
            "calculator_results": [
                result for result in diagnostics.get("calculator_results", ())
                if result.get("result") is not None
            ],
            "sources": [
                {
                    key: source.get(key)
                    for key in (
                        "id", "ticker", "fiscal_year", "doc_type", "source",
                        "section", "evidence_text",
                    )
                    if source.get(key) is not None
                }
                for source in sources
            ],
        }
        if not evidence["sources"]:
            return ""
        model = getattr(self, "_answer_repair_model", None)
        if model is None:
            model = get_llm_engine()
            self._answer_repair_model = model
        try:
            response = model.invoke([
                SystemMessage(content=(
                    "Compose a concise answer or rewrite the draft using only "
                    "the supplied validated facts, calculator results, and source evidence. "
                    "Cite every factual claim inline with the exact source ID in brackets. "
                    "For calculations, use only the supplied calculator result and cite "
                    "its source IDs. Do not calculate, infer missing values, or add claims. "
                    "If a requested calculation has no calculator result, do not calculate "
                    "it yourself; state which result or input is unavailable. If the evidence "
                    "cannot answer part of the question, state that clearly."
                )),
                HumanMessage(content=json.dumps({
                    "question": question,
                    "draft": draft,
                    "evidence": evidence,
                }, ensure_ascii=False)),
            ])
            return self._message_text(response)
        except Exception:
            return ""

    @trace_component("finrag.chat_request", run_type="chain", root=True)
    def ask(
        self,
        question: str,
        *,
        ui_scope: Scope | None = None,
        previous_scope: Scope | None = None,
        history: list[dict] | None = None,
        pending_clarification: PendingClarification | None = None,
        thread_id: str | None = None,
        request_id: str | None = None,
        trace_id: str | None = None,
    ) -> ChatResult:
        trace_id = str(trace_id or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", trace_id):
            trace_id = uuid4().hex[:12]
        index_version = str(getattr(self, "index_version", "unknown") or "unknown")
        trace = {
            "trace_id": trace_id,
            "request_id": str(request_id or ""),
            "index_version": index_version,
            "question": question.strip()[:1000],
            "retrieval_pipeline": ["dense", "bm25", "rrf", "optional_cohere_rerank"],
            "reranker": self._reranker_trace(),
            "agent": {
                "name": "langgraph_financial_agent",
                "backend": "langgraph",
                "selected": False,
                "status": "not_started",
                "route": "scope_policy",
                "tools": [
                    "plan_financial_request",
                    "search_financial_documents",
                    "financial_calculator",
                    "get_financial_statement",
                    "finalize_financial_answer",
                ],
            },
        }
        annotate_current_trace(
            {
                "request_id": trace["request_id"],
                "trace_id": trace_id,
                "thread_id": str(thread_id or ""),
                "index_version": index_version,
            },
            tags=("query", f"index:{index_version}"),
        )
        question = question.strip()
        if not question:
            result = ChatResult(
                Decision.CLARIFY, "Please enter a question.", trace=trace
            )
            self._record({**trace, "decision": result.decision.value})
            return result

        clarification_turn = (
            merge_clarification_reply(pending_clarification, question, available_keys=self.available_keys)
            if pending_clarification is not None
            else None
        )
        if clarification_turn is not None:
            effective_question = pending_clarification.original_question
            understanding = clarification_turn.understanding
            candidate_tickers = clarification_turn.candidate_tickers
        else:
            effective_question = question
            understanding = understand_query(question, available_keys=self.available_keys)
            candidate_tickers = ()
        trace["ui_scope"] = self._scope_data(ui_scope or Scope())
        trace["previous_scope"] = self._scope_data(previous_scope or Scope())
        trace["clarification"] = {
            "pending_received": pending_clarification is not None,
            "reply_applied": clarification_turn is not None,
        }
        resolution = resolve_scope(
            understanding,
            ui_scope=ui_scope,
            previous_scope=previous_scope,
            available_keys=self.available_keys,
        )
        semantic_parser = getattr(self, "semantic_parser", None)
        semantic_trace = {
            "configured": semantic_parser is not None,
            "used": False,
        }
        trigger = (
            None
            if clarification_turn is not None
            else should_use_semantic_fallback(
                question,
                understanding,
                resolution,
                previous_scope=previous_scope,
            )
        )
        if semantic_parser is not None and trigger is not None:
            semantic_trace.update(
                {
                    "used": True,
                    "reason": trigger.reason,
                }
            )
            started = perf_counter()
            try:
                semantic = semantic_parser.parse(
                    question=question,
                    ui_scope=ui_scope,
                    previous_scope=previous_scope,
                    available_keys=self.available_keys,
                )
                application = apply_semantic_result(
                    question,
                    understanding,
                    semantic,
                    previous_scope=previous_scope,
                    available_keys=self.available_keys,
                )
                understanding = application.understanding
                candidate_tickers = application.candidate_tickers
                resolution = resolve_scope(
                    understanding,
                    ui_scope=ui_scope,
                    previous_scope=previous_scope,
                    available_keys=self.available_keys,
                )
                semantic_trace.update(
                    {
                        "candidate_tickers": list(application.candidate_tickers),
                        "unsupported_companies": list(
                            application.unsupported_companies
                        ),
                        "accepted_fields": list(application.accepted_fields),
                        "ambiguous_fields": list(application.ambiguous_fields),
                        "query_expansions": list(understanding.query_expansions),
                    }
                )
                terminal_rejection = resolution.decision in {
                    Decision.OUT_OF_SCOPE,
                    Decision.SCOPE_CONFLICT,
                    Decision.DATA_UNAVAILABLE,
                }
                if application.needs_clarification and not terminal_rejection:
                    resolution = ScopeResolution(
                        Decision.CLARIFY,
                        resolution.scope,
                        semantic_clarification_message(
                            application, resolution.message
                        ),
                        resolution.inherited_fields,
                    )
                elif (
                    trigger.fail_closed
                    and not application.unsupported_companies
                    and resolution.decision == Decision.SEARCH
                ):
                    resolution = ScopeResolution(
                        Decision.CLARIFY,
                        resolution.scope,
                        self._company_confirmation_message(),
                        resolution.inherited_fields,
                    )
            except Exception as error:
                semantic_trace["error_type"] = type(error).__name__
                if trigger.fail_closed:
                    resolution = ScopeResolution(
                        Decision.CLARIFY,
                        resolution.scope,
                        self._company_confirmation_message(),
                        resolution.inherited_fields,
                    )
            finally:
                semantic_trace["latency_ms"] = round(
                    (perf_counter() - started) * 1000, 2
                )
        trace["semantic_fallback"] = semantic_trace
        trace["understanding"] = {
            "tickers": list(understanding.tickers),
            "source_years": list(understanding.years),
            "reference_years": list(understanding.reference_years),
            "requested_doc_types": list(understanding.requested_doc_types),
            "generic_followup": understanding.generic_followup,
            "semantic_fallback_used": understanding.semantic_fallback_used,
            "ambiguous_fields": list(understanding.ambiguous_fields),
            "query_expansions": list(understanding.query_expansions),
        }
        trace["resolved_scope"] = self._scope_data(resolution.scope)
        trace["inherited_fields"] = list(resolution.inherited_fields)
        scope_doc_types = resolution.scope.required_doc_types or (
            (resolution.scope.doc_type,) if resolution.scope.doc_type else ()
        )
        annotate_current_trace(
            {
                "scope_decision": resolution.decision.value,
                "scope_tickers": resolution.scope.tickers,
                "scope_years": resolution.scope.years,
                "scope_doc_types": scope_doc_types,
                "scope_tickers_csv": ",".join(resolution.scope.tickers),
                "scope_years_csv": ",".join(resolution.scope.years),
                "scope_doc_types_csv": ",".join(scope_doc_types),
                "scope_group_count": len(resolution.scope.retrieval_groups),
                "comparison": resolution.scope.is_comparison,
            },
            tags=tuple(
                [*(f"ticker:{ticker}" for ticker in resolution.scope.tickers)]
                + [*(f"year:{year}" for year in resolution.scope.years)]
                + [*(f"doc_type:{doc_type}" for doc_type in scope_doc_types)]
            ),
        )
        if resolution.decision != Decision.SEARCH:
            pending_result = None
            result_scope = resolution.scope
            inherited_fields = resolution.inherited_fields
            if resolution.decision == Decision.CLARIFY:
                pending_result = pending_from_resolution(
                    original_question=effective_question,
                    understanding=understanding,
                    resolution=resolution,
                    candidate_tickers=candidate_tickers,
                )
                result_scope = pending_result.scope
                trace["resolved_scope"] = self._scope_data(result_scope)
                trace["clarification"].update(
                    {
                        "pending_returned": True,
                        "candidate_tickers": list(
                            pending_result.candidate_tickers
                        ),
                        "missing_fields": list(pending_result.missing_fields),
                    }
                )
                if not result_scope.tickers:
                    inherited_fields = tuple(
                        field
                        for field in inherited_fields
                        if field != "company"
                    )
            else:
                trace["clarification"]["pending_returned"] = False
            result = ChatResult(
                resolution.decision,
                resolution.message,
                result_scope,
                inherited_fields=inherited_fields,
                trace=trace,
                pending_clarification=pending_result,
            )
            self._record({**trace, "decision": result.decision.value})
            return result

        agent_question = effective_question
        if clarification_turn is not None and question != effective_question:
            agent_question = (
                f"{effective_question}\n\nUser clarification: {question}"
            )
        locked_scope = {
            **self._scope_data(resolution.scope),
            "instruction": (
                "Use these company, source filing/call year, and document type values for every search. "
                "A reported period inside a source may differ from its source year."
            ),
        }
        user_message = (
            "Application-resolved scope (locked): "
            f"{json.dumps(locked_scope, ensure_ascii=True, sort_keys=True)}\n"
            f"User question: {agent_question}"
        )
        messages: list[dict[str, str]] = []
        # History resolves follow-up references only; retrieved tool evidence is
        # never carried across turns. Bound both message count and text size.
        for item in (history or [])[-6:]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").casefold()
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                content = " ".join(strip_citations(content).split())[:1_200]
                if content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_message})
        def research():
            return self.process_query({"messages": messages}, thread_id=thread_id or uuid4().hex,
                scope=resolution.scope, inherited_fields=resolution.inherited_fields, trace=trace)
        cache = getattr(self, "answer_cache", None)
        if cache is None:
            result = research()
        else:
            key = json.dumps([getattr(self, "index_version", "unknown"), messages,
                resolution.inherited_fields, thread_id], sort_keys=True)
            with usage_stage("finrag.answer_cache"):
                result, reused = cache.get_or_compute(key, research)
            if reused:
                result = replace(result, trace={**result.trace, **trace,
                    "agent": {**trace["agent"], "status": "cached", "route": "answer_cache"},
                    "answer_cache": {"hit": True, "source_trace_id": result.trace.get("trace_id")}})
                self._record({**result.trace, "decision": result.decision.value})
        annotate_current_trace(
            {
                "final_decision": result.decision.value,
                "answer_source_count": len(result.sources),
            }
        )
        return result

    def close(self) -> None:
        self.vector_manager.close()
