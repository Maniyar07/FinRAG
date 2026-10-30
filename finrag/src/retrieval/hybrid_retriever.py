"""Balanced dense + BM25 retrieval with reciprocal-rank fusion."""

from __future__ import annotations

from collections import defaultdict
from math import ceil
from typing import TypeAlias

from langchain_core.documents import Document
from qdrant_client.http import models

from src.config import (
    DENSE_CANDIDATE_K,
    LEXICAL_CANDIDATE_K,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_PARENTS,
    MAX_CONTEXT_TOKENS,
    MIN_DENSE_SCORE,
    MIN_LEXICAL_COVERAGE,
)
from src.retrieval.context_builder import ContextBuilder
from src.retrieval.lexical_index import LexicalHit, LexicalIndex, scoped_content_query
from src.observability import trace_component
from src.schemas import RankedChild, RetrievalBundle, Scope
from dataclasses import replace


GroupKey: TypeAlias = tuple[str, str, str | None]
DenseResult: TypeAlias = tuple[GroupKey, int, Document, float]
LexicalResult: TypeAlias = tuple[GroupKey, int, LexicalHit]

# RRF uses ranks rather than trying to compare cosine and BM25 score scales.
# 60 is the conventional RRF smoothing constant, not a company-specific rule.
RRF_CONSTANT = 60.0
# Expansion-only lexical hits are deliberately weaker than every candidate
# found by the original dense/BM25 query. They can recover missing terminology
# without displacing the baseline retrieval ordering.
EXPANSION_RRF_WEIGHT = 0.35



class HybridRetriever:
    """Search each requested group, fuse dense/BM25 ranks, and load parents."""

    def __init__(
        self,
        vectorstore,
        *,
        lexical_index: LexicalIndex,
        context_builder: ContextBuilder,
        reranker=None,
    ):
        self.vectorstore = vectorstore
        self.lexical_index = lexical_index
        self.context_builder = context_builder
        self.reranker = reranker

    @staticmethod
    def _filter_for_group(group: GroupKey) -> models.Filter:
        """Build one exact Qdrant filter for a comparison group."""
        ticker, year, doc_type = group
        conditions = [
            models.FieldCondition(
                key="metadata.ticker", match=models.MatchValue(value=ticker)
            ),
            models.FieldCondition(
                key="metadata.fiscal_year", match=models.MatchValue(value=year)
            ),
        ]
        if doc_type:
            conditions.append(
                models.FieldCondition(
                    key="metadata.doc_type",
                    match=models.MatchValue(value=doc_type),
                )
            )
        return models.Filter(must=conditions)

    @staticmethod
    def _scope_for_group(group: GroupKey) -> Scope:
        ticker, year, doc_type = group
        return Scope(tickers=(ticker,), years=(year,), doc_type=doc_type)

    @staticmethod
    def _candidate_limit(total_limit: int, group_count: int) -> int:
        """Divide the candidate budget fairly across requested groups."""
        return max(1, ceil(total_limit / max(group_count, 1)))

    @trace_component("finrag.retrieval.dense_search", run_type="retriever")
    def _dense_search(self, query: str, scope: Scope, *, embedding: list[float] | None = None,
                      total_limit: int = DENSE_CANDIDATE_K) -> list[DenseResult]:
        """Run a separate filtered dense search for every requested group."""
        groups = scope.retrieval_groups
        if not groups:
            return []

        per_group = self._candidate_limit(total_limit, len(groups))
        results: list[DenseResult] = []
        for group in groups:
            options = {"k": per_group, "filter": self._filter_for_group(group)}
            group_hits = (self.vectorstore.similarity_search_with_score_by_vector(embedding=embedding, **options)
                          if embedding is not None else self.vectorstore.similarity_search_with_score(query=query, **options))
            for rank, (document, score) in enumerate(group_hits, start=1):
                results.append((group, rank, document, float(score)))
        return results

    @trace_component("finrag.retrieval.lexical_search", run_type="retriever")
    def _lexical_search(
        self, query: str, scope: Scope, *, total_limit: int | None = None
    ) -> list[LexicalResult]:
        """Run a separate filtered BM25 search for every requested group."""
        groups = scope.retrieval_groups
        if not groups:
            return []

        per_group = self._candidate_limit(
            total_limit or LEXICAL_CANDIDATE_K, len(groups)
        )
        results: list[LexicalResult] = []
        for group in groups:
            group_hits = self.lexical_index.search(
                query,
                self._scope_for_group(group),
                k=per_group,
            )
            for rank, hit in enumerate(group_hits, start=1):
                results.append((group, rank, hit))
        return results

    @trace_component("finrag.retrieval.rank_fusion", run_type="retriever")
    def _fuse(
        self,
        query: str,
        scope: Scope,
        *,
        query_expansions: tuple[str, ...] = (),
        dense_results: list[DenseResult] | None = None,
    ) -> list[RankedChild]:
        """Fuse per-group ranks without comparing raw dense and BM25 scores."""
        candidates: dict[str, dict] = defaultdict(dict)
        rrf_scores: dict[str, float] = defaultdict(float)

        for group, rank, document, dense_score in (self._dense_search(query, scope) if dense_results is None else dense_results):
            child_id = str(document.metadata.get("child_id", ""))
            if not child_id:
                continue
            values = candidates[child_id]
            values.setdefault("text", document.page_content)
            values.setdefault("metadata", document.metadata)
            values.setdefault("group", group)
            values["dense"] = max(dense_score, float(values.get("dense", -1.0)))
            rrf_scores[child_id] += 1.0 / (RRF_CONSTANT + rank)

        for group, rank, hit in self._lexical_search(query, scope):
            child_id = str(hit.record.get("child_id", ""))
            if not child_id:
                continue
            values = candidates[child_id]
            values.setdefault("text", str(hit.record["text"]))
            values.setdefault("metadata", hit.record["metadata"])
            values.setdefault("group", group)
            values["lexical"] = max(
                float(hit.score), float(values.get("lexical", 0.0))
            )
            values["coverage"] = max(
                float(hit.coverage), float(values.get("coverage", 0.0))
            )
            rrf_scores[child_id] += 1.0 / (RRF_CONSTANT + rank)

        # Search suggested terminology as independent lexical queries. Never
        # boost or replace a candidate found by the original query; expansions
        # may only add lower-priority candidates for recall.
        baseline_ids = set(candidates)
        expansions = query_expansions[:3]
        expansion_limit = self._candidate_limit(
            LEXICAL_CANDIDATE_K, len(expansions)
        )
        for expansion in expansions:
            for group, rank, hit in self._lexical_search(
                expansion, scope, total_limit=expansion_limit
            ):
                child_id = str(hit.record.get("child_id", ""))
                if not child_id or child_id in baseline_ids:
                    continue
                values = candidates[child_id]
                values.setdefault("text", str(hit.record["text"]))
                values.setdefault("metadata", hit.record["metadata"])
                values.setdefault("group", group)
                values["lexical"] = max(
                    float(hit.score), float(values.get("lexical", 0.0))
                )
                values["coverage"] = max(
                    float(hit.coverage), float(values.get("coverage", 0.0))
                )
                rrf_scores[child_id] = max(
                    rrf_scores[child_id],
                    EXPANSION_RRF_WEIGHT / (RRF_CONSTANT + rank),
                )

        max_rrf = max(rrf_scores.values(), default=1.0)
        accepted: list[RankedChild] = []
        below_threshold: list[RankedChild] = []

        for child_id, values in candidates.items():
            metadata = values.get("metadata", {})
            parent_id = str(metadata.get("parent_id", ""))
            if not parent_id:
                continue

            dense_score = values.get("dense")
            coverage = float(values.get("coverage", 0.0))
            normalized_rrf = rrf_scores[child_id] / max_rrf
            ranked_child = RankedChild(
                child_id=child_id,
                parent_id=parent_id,
                text=str(values.get("text", "")),
                metadata={
                    **metadata,
                    "retrieval_group": values.get("group"),
                    # Preserve RRF independently of optional reranking so
                    # fail-open traces still expose the score actually used.
                    "rrf_score": normalized_rrf,
                },
                dense_score=dense_score,
                lexical_score=values.get("lexical"),
                lexical_coverage=coverage,
                fused_score=normalized_rrf,
            )

            passes_threshold = (
                dense_score is not None and dense_score >= MIN_DENSE_SCORE
            ) or coverage >= MIN_LEXICAL_COVERAGE
            (accepted if passes_threshold else below_threshold).append(ranked_child)

        def sort_key(child: RankedChild):
            return (-child.fused_score, child.child_id)

        accepted.sort(key=sort_key)
        below_threshold.sort(key=sort_key)

        # Do not silently lose a requested comparison group only because its
        # best candidate narrowly missed a global score threshold. Later
        # evidence verification still decides whether the source supports an answer.
        if scope.is_comparison:
            covered_groups = {
                tuple(child.metadata.get("retrieval_group", ())) for child in accepted
            }
            for group in scope.retrieval_groups:
                if group in covered_groups:
                    continue
                fallback = next(
                    (
                        child
                        for child in below_threshold
                        if tuple(child.metadata.get("retrieval_group", ())) == group
                    ),
                    None,
                )
                if fallback is not None:
                    accepted.append(fallback)
                    covered_groups.add(group)

        accepted.sort(key=sort_key)
        return accepted

    @trace_component("finrag.hybrid_retrieval", run_type="retriever")
    def retrieve(
        self,
        query: str,
        scope: Scope,
        *,
        query_expansions: tuple[str, ...] = (),
        preserve_all: bool = False,
        max_parents: int | None = None,
        max_chars: int | None = None,
        max_tokens: int | None = None,
        exclude_parent_ids: frozenset[str] = frozenset(),
        preferred_parent_ids: tuple[str, ...] = (),
        requirement_queries: tuple[str, ...] = (),
    ) -> RetrievalBundle:
        subjects = tuple(dict.fromkeys(value for query_text in requirement_queries
            if (value := scoped_content_query(query_text, scope))))
        # Embed the baseline and independent observations together. Local vector
        # searches then reuse these vectors across every locked source group.
        dense_by_query = None
        embedder = getattr(self.vectorstore, "embeddings", None)
        if (subjects and callable(getattr(embedder, "embed_documents", None))
                and callable(getattr(self.vectorstore, "similarity_search_with_score_by_vector", None))):
            queries = tuple(dict.fromkeys((query, *subjects)))
            vectors = embedder.embed_documents(list(queries))
            if len(vectors) != len(queries):
                raise ValueError("The embedding provider did not return every retrieval query.")
            facet_limit = self._candidate_limit(DENSE_CANDIDATE_K, len(subjects))
            dense_by_query = {value: self._dense_search(value, scope, embedding=vector,
                total_limit=DENSE_CANDIDATE_K if value == query else facet_limit)
                for value, vector in zip(queries, vectors)}
        ranked = self._fuse(
            query,
            scope,
            query_expansions=tuple(
                value.strip() for value in query_expansions if value.strip()
            )[:3],
            **({"dense_results": dense_by_query[query]} if dense_by_query is not None else {}),
        )
        subject_parents = []
        parent_queries = defaultdict(list)
        for subject in subjects:
            if dense_by_query is not None:
                hits = self._fuse(subject, scope, dense_results=dense_by_query[subject])
                existing = {child.child_id for child in ranked}
                ranked.extend(child for child in hits if child.child_id not in existing)
                matches = [(tuple(child.metadata.get("retrieval_group", ())), child.parent_id) for child in hits
                    if ((child.dense_score is not None and child.dense_score >= MIN_DENSE_SCORE)
                        or child.lexical_coverage >= MIN_LEXICAL_COVERAGE)]
            else:
                matches = [(group, str(hit.record.get("metadata", {}).get("parent_id", "")))
                    for group, _, hit in self._lexical_search(subject, scope) if hit.coverage >= MIN_LEXICAL_COVERAGE]
            for group in scope.retrieval_groups:
                parent_id = next((key for hit_group, key in matches if hit_group == group and key not in exclude_parent_ids), "")
                if parent_id:
                    subject_parents.append(parent_id)
                    parent_queries[parent_id].append(subject)
        ranked = [replace(child, metadata={**child.metadata, "evidence_queries": tuple(parent_queries[child.parent_id])})
                  if child.parent_id in parent_queries else child for child in ranked]
        if self.reranker is not None:
            ranked = self.reranker.rerank(query, ranked, scope)
        preferred_parent_ids = tuple(dict.fromkeys((*preferred_parent_ids, *(value for value in subject_parents if value))))
        # Planned evidence matches prioritize their intact parent, never a sliced row.
        priority = max((child.fused_score for child in ranked), default=0) + 1
        preferred = set(preferred_parent_ids) - exclude_parent_ids
        priorities = {parent_id: priority + len(preferred_parent_ids) - index
                      for index, parent_id in enumerate(preferred_parent_ids)}
        present = {child.parent_id for child in ranked}
        ranked = [replace(child, fused_score=priorities[child.parent_id],
                          metadata={**child.metadata, "retrieval_priority": "planned_evidence_match"})
                  if child.parent_id in preferred else child for child in ranked]
        missing = sorted(preferred - present)
        if missing:
            for parent_id, parent in zip(missing, self.context_builder.parent_store.get_many(missing)):
                if parent is not None and any(
                        (parent.metadata.get("ticker"), str(parent.metadata.get("fiscal_year")),
                         str(parent.metadata.get("doc_type", "")).upper().replace("-", ""))
                        == group for group in scope.retrieval_groups):
                    ranked.append(RankedChild("indexed:" + parent_id, parent_id, parent.page_content,
                        {**parent.metadata, "retrieval_priority": "planned_evidence_match",
                         "evidence_queries": tuple(parent_queries[parent_id])}, fused_score=priorities[parent_id]))
        parent_limit = max_parents if max_parents is not None else (
            MAX_CONTEXT_PARENTS
            if not scope.is_comparison
            else max(MAX_CONTEXT_PARENTS, len(scope.retrieval_groups) * 2)
        )
        char_limit = max_chars if max_chars is not None else MAX_CONTEXT_CHARS
        token_limit = max_tokens if max_tokens is not None else MAX_CONTEXT_TOKENS
        if parent_limit <= 0 or char_limit <= 0 or token_limit <= 0:
            raise ValueError("Retrieval context limits must be positive.")
        return self.context_builder.build(
            ranked,
            scope,
            max_parents=parent_limit,
            max_chars=char_limit,
            max_tokens=token_limit,
            query=scoped_content_query(query, scope) or query,
            preserve_all=preserve_all,
            exclude_parent_ids=exclude_parent_ids,
            requirement_queries=requirement_queries,
        )
