from __future__ import annotations

import json
import logging
import re
from dataclasses import replace
from time import perf_counter
from uuid import uuid4

from src.generation.answer_generator import AnswerGenerator
from src.generation.citations import expand_citations
from src.generation.narrative_quotes import NarrativeQuoteSelector
from src.generation.answer_guardrails import (
    INSUFFICIENT_EVIDENCE_RESPONSE,
    UNVERIFIABLE_RESPONSE,
    missing_comparison_groups,
    scope_coverage_message,
)
from src.ingestion.manifest import available_keys, load_manifest
from src.config import LOGS_DIR, MULTIHOP_ENABLED, get_index_paths
from src.app.multihop_coordinator import MultiHopCoordinator
from src.orchestration.executor import MultiHopExecutor
from src.orchestration.planner import MultiHopPlanner, should_use_multihop
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
from src.retrieval.structured_lookup import StructuredDocumentLookup, VerifiedTableRow
from src.retrieval.vector_store import FinancialVectorStore
from src.schemas import (
    ChatResult,
    Decision,
    PendingClarification,
    Scope,
    ScopeResolution,
    RetrievalBundle,
)
from src.retrieval.reranker import build_cohere_reranker_from_environment
from src.tools.document_search import DocumentSearchRequest, DocumentSearchTool


class ChatService:
    """Coordinate policy, retrieval, evidence checks, and answer generation."""

    def __init__(self, *, index_version: str | None = None):
        paths = get_index_paths(index_version)
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
            self.structured_lookup = StructuredDocumentLookup(paths.parent_docstore_dir)
            self.document_search_tool = DocumentSearchTool(self.retriever)
            self.generator = AnswerGenerator()
            self.semantic_parser = build_semantic_query_parser_from_environment()
            self.multihop_enabled = MULTIHOP_ENABLED
            self.multihop_planner = (
                MultiHopPlanner() if self.multihop_enabled else None
            )
            self.multihop_executor = (
                MultiHopExecutor(
                    document_search=self.document_search_tool,
                    statement_lookup=self.structured_lookup,
                )
                if self.multihop_enabled
                else None
            )
            self.narrative_quote_selector = (
                NarrativeQuoteSelector() if self.multihop_enabled else None
            )
            self.logger = self._build_logger()
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
            "requested_groups": [list(group) for group in scope.requested_groups],
        }

    def _record(self, trace: dict) -> None:
        self.logger.info(json.dumps(trace, ensure_ascii=True, sort_keys=True))

    def _company_confirmation_message(self) -> str:
        tickers = tuple(
            ticker
            for ticker in ("JPM", "MSFT", "TSLA")
            if any(key[0] == ticker for key in self.available_keys)
        )
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
    def _retrieval_source_trace(source: dict) -> dict:
        """Return scores and source identity without duplicating evidence text."""
        return {
            "id": source.get("id"),
            "ticker": source.get("ticker"),
            "fiscal_year": source.get("fiscal_year"),
            "doc_type": source.get("doc_type"),
            "compression": source.get("compression", {}),
            "section": source.get("section"),
            "source": source.get("source"),
            "parent_id": source.get("parent_id"),
            "dense_score": source.get("dense_score"),
            "lexical_score": source.get("lexical_score"),
            "lexical_coverage": source.get("lexical_coverage"),
            "rrf_score": source.get("rrf_score"),
            "pre_rerank_score": source.get("pre_rerank_score"),
            "rerank_score": source.get("rerank_score"),
            "rerank_provider": source.get("rerank_provider"),
            "rerank_model": source.get("rerank_model"),
            "final_score": source.get("score"),
        }

    @staticmethod
    def _with_statement_rows(
        bundle: RetrievalBundle, rows: tuple[VerifiedTableRow, ...]
    ) -> tuple[RetrievalBundle, tuple[VerifiedTableRow, ...]]:
        """Make verified indexed rows visible to generation with valid citations."""
        if not rows:
            return bundle, rows
        sources = list(bundle.sources)
        context = bundle.context
        next_id = max(
            (int(str(source.get("id", "S0"))[1:]) for source in sources
             if re.fullmatch(r"S\d+", str(source.get("id", "")))),
            default=0,
        ) + 1
        source_ids: dict[str, str] = {}
        verified = []
        for row in rows:
            parent_id = str(row.source.get("parent_id"))
            source_id = source_ids.get(parent_id)
            if source_id is None:
                source_id = f"S{next_id}"
                next_id += 1
                source_ids[parent_id] = source_id
                source = {**row.source, "id": source_id}
                sources.append(source)
                context += (
                    f"\n\n---\n\n[SOURCE {source_id} | TICKER: {source.get('ticker')} "
                    f"| FISCAL YEAR: {source.get('fiscal_year')} | TYPE: 10K "
                    f"| SECTION: {source.get('section')}]\n"
                    f"{source['evidence_text']}"
                )
            verified.append(replace(row, source={**row.source, "id": source_id}))
        return (
            RetrievalBundle(context, sources, bundle.scope, bundle.candidate_count,
                            bundle.covered_groups),
            tuple(verified),
        )

    @staticmethod
    def _retrieval_query(
        question: str, history: list[dict] | None, *, is_followup: bool
    ) -> str:
        if not is_followup or not history:
            return question
        for message in reversed(history):
            if message.get("role") == "user":
                previous = " ".join(str(message.get("content", "")).split())[:800]
                if previous:
                    return f"{question}\nPrevious user question: {previous}"
        return question

    def _run_multihop(
        self,
        *,
        question: str,
        scope: Scope,
        inherited_fields: tuple[str, ...],
        history: list[dict] | None,
        query_expansions: tuple[str, ...],
        wants_table: bool,
        trace: dict,
    ) -> ChatResult:
        planner = getattr(self, "multihop_planner", None)
        executor = getattr(self, "multihop_executor", None)
        if planner is None or executor is None:
            raise RuntimeError("Multi-hop orchestration is enabled but not initialized.")
        coordinator = MultiHopCoordinator(
            planner=planner,
            executor=executor,
            document_search_tool=self.document_search_tool,
            generator=self.generator,
            narrative_quote_selector=getattr(self, "narrative_quote_selector", None),
            record=self._record,
            retrieval_source_trace=self._retrieval_source_trace,
        )
        return coordinator.run(
            question=question,
            scope=scope,
            inherited_fields=inherited_fields,
            history=history,
            query_expansions=query_expansions,
            wants_table=wants_table,
            trace=trace,
        )
    def ask(
        self,
        question: str,
        *,
        ui_scope: Scope | None = None,
        previous_scope: Scope | None = None,
        history: list[dict] | None = None,
        pending_clarification: PendingClarification | None = None,
    ) -> ChatResult:
        trace_id = uuid4().hex[:12]
        trace = {
            "trace_id": trace_id,
            "question": question.strip()[:1000],
            "retrieval_pipeline": ["dense", "bm25", "rrf", "optional_cohere_rerank"],
            "reranker": self._reranker_trace(),
        }
        question = question.strip()
        if not question:
            result = ChatResult(
                Decision.CLARIFY, "Please enter a question.", trace=trace
            )
            self._record({**trace, "decision": result.decision.value})
            return result

        clarification_turn = (
            merge_clarification_reply(pending_clarification, question)
            if pending_clarification is not None
            else None
        )
        if clarification_turn is not None:
            effective_question = pending_clarification.original_question
            understanding = clarification_turn.understanding
            candidate_tickers = clarification_turn.candidate_tickers
        else:
            effective_question = question
            understanding = understand_query(question)
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

        trace["multihop"] = {
            "enabled": bool(getattr(self, "multihop_enabled", False)),
            "selected": False,
        }
        lookup = getattr(self, "structured_lookup", None)
        if lookup is not None:
            direct = lookup.answer(
                effective_question,
                resolution.scope,
                wants_complete_table=understanding.wants_complete_table,
            )
            if direct is not None:
                trace["retrieval"] = {
                    "mode": direct.mode,
                    "source_ids": [source["id"] for source in direct.sources],
                }
                result = ChatResult(
                    Decision.ANSWERED,
                    expand_citations(direct.answer, direct.sources),
                    resolution.scope,
                    direct.sources,
                    resolution.inherited_fields,
                    trace,
                )
                self._record({**trace, "decision": result.decision.value})
                return result

        retrieval_query = self._retrieval_query(
            understanding.normalized_query or effective_question,
            history,
            is_followup=understanding.generic_followup,
        )
        trace["retrieval_query"] = retrieval_query[:1800]
        use_multihop = (
            bool(getattr(self, "multihop_enabled", False))
            and not understanding.wants_complete_table
            and should_use_multihop(effective_question, resolution.scope)
        )
        trace["multihop"]["selected"] = use_multihop
        if use_multihop:
            return self._run_multihop(
                question=effective_question,
                scope=resolution.scope,
                inherited_fields=resolution.inherited_fields,
                history=history,
                query_expansions=understanding.query_expansions,
                wants_table=understanding.wants_table,
                trace=trace,
            )
        try:
            search_result = self.document_search_tool.execute(
                DocumentSearchRequest(
                    query=retrieval_query,
                    scope=resolution.scope,
                    query_expansions=understanding.query_expansions,
                    preserve_all=understanding.wants_complete_table,
                ),
                permitted_scope=resolution.scope,
            )
            bundle = search_result.bundle
            verified_rows: tuple[VerifiedTableRow, ...] = ()
            if (
                lookup is not None
                and hasattr(lookup, "statement_rows")
                and not understanding.wants_table
            ):
                verified_rows = lookup.statement_rows(effective_question, resolution.scope)
                bundle, verified_rows = self._with_statement_rows(bundle, verified_rows)
        except Exception as error:
            trace["error_type"] = type(error).__name__
            self._record({**trace, "decision": Decision.ERROR.value})
            return ChatResult(
                Decision.ERROR,
                "The document search could not be completed. Verify the selected index "
                "and try again.",
                resolution.scope,
                inherited_fields=resolution.inherited_fields,
                trace=trace,
            )
        source_traces = [
            self._retrieval_source_trace(source) for source in bundle.sources
        ]
        trace["retrieval"] = {
            "tool": self.document_search_tool.name,
            "purpose": search_result.purpose.value,
            "candidate_count": bundle.candidate_count,
            "source_ids": [source["id"] for source in bundle.sources],
            "source_groups": [
                [source["ticker"], source["fiscal_year"], source["doc_type"]]
                for source in bundle.sources
            ],
            # Configured means the Cohere stage exists. Applied means this
            # particular request returned Cohere scores instead of using the
            # fail-open RRF result.
            "reranker_applied": any(
                source.get("rerank_score") is not None for source in bundle.sources
            ),
            "sources": source_traces,
            "verified_statement_rows": len(verified_rows),
        }
        if not bundle.context or not bundle.sources:
            result = ChatResult(
                Decision.INSUFFICIENT_EVIDENCE,
                INSUFFICIENT_EVIDENCE_RESPONSE,
                resolution.scope,
                inherited_fields=resolution.inherited_fields,
                trace=trace,
            )
            self._record({**trace, "decision": result.decision.value})
            return result

        missing = missing_comparison_groups(bundle)
        if missing:
            result = ChatResult(
                Decision.INSUFFICIENT_EVIDENCE,
                scope_coverage_message(missing, resolution.scope),
                resolution.scope,
                bundle.sources,
                resolution.inherited_fields,
                trace,
            )
            self._record({**trace, "decision": result.decision.value})
            return result

        try:
            generation = self.generator.generate_with_trace(
                effective_question,
                bundle,
                history=history,
                wants_table=understanding.wants_table,
                wants_complete_table=understanding.wants_complete_table,
                verified_rows=verified_rows,
            )
            answer = generation.answer
            trace["generation"] = {
                "attempts": generation.attempts,
                "validation_reason": generation.validation_reason,
                "raw_output_previews": list(generation.raw_output_previews),
            }
        except Exception as error:
            trace["error_type"] = type(error).__name__
            self._record({**trace, "decision": Decision.ERROR.value})
            return ChatResult(
                Decision.ERROR,
                "The answer model could not complete this request. The retrieved evidence "
                "was not converted into an answer; please try again.",
                resolution.scope,
                bundle.sources,
                resolution.inherited_fields,
                trace,
            )
        decision = (
            Decision.VALIDATION_FAILED if answer == UNVERIFIABLE_RESPONSE
            else Decision.INSUFFICIENT_EVIDENCE if answer == INSUFFICIENT_EVIDENCE_RESPONSE
            else Decision.ANSWERED
        )
        result = ChatResult(
            decision,
            answer,
            resolution.scope,
            bundle.sources,
            resolution.inherited_fields,
            trace,
        )
        self._record({**trace, "decision": result.decision.value})
        return result

    def close(self) -> None:
        self.vector_manager.close()
