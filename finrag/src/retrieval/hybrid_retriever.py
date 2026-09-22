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
    MIN_DENSE_SCORE,
    MIN_LEXICAL_COVERAGE,
)
from src.retrieval.context_builder import ContextBuilder
from src.retrieval.lexical_index import LexicalHit, LexicalIndex
from src.schemas import RankedChild, RetrievalBundle, Scope


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

    def _dense_search(self, query: str, scope: Scope) -> list[DenseResult]:
        """Run a separate filtered dense search for every requested group."""
        groups = scope.retrieval_groups
        if not groups:
            return []

        per_group = self._candidate_limit(DENSE_CANDIDATE_K, len(groups))
        results: list[DenseResult] = []
        for group in groups:
            group_hits = self.vectorstore.similarity_search_with_score(
                query=query,
                k=per_group,
                filter=self._filter_for_group(group),
            )
            for rank, (document, score) in enumerate(group_hits, start=1):
                results.append((group, rank, document, float(score)))
        return results

    def _lexical_search(self, query: str, scope: Scope) -> list[LexicalResult]:
        """Run a separate filtered BM25 search for every requested group."""
        groups = scope.retrieval_groups
        if not groups:
            return []

        per_group = self._candidate_limit(LEXICAL_CANDIDATE_K, len(groups))
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

    def _fuse(
        self,
        query: str,
        scope: Scope,
        *,
        query_expansions: tuple[str, ...] = (),
    ) -> list[RankedChild]:
        """Fuse per-group ranks without comparing raw dense and BM25 scores."""
        candidates: dict[str, dict] = defaultdict(dict)
        rrf_scores: dict[str, float] = defaultdict(float)

        for group, rank, document, dense_score in self._dense_search(query, scope):
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
        for expansion in query_expansions[:3]:
            for group, rank, hit in self._lexical_search(expansion, scope):
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
            ranked_child = RankedChild(
                child_id=child_id,
                parent_id=parent_id,
                text=str(values.get("text", "")),
                metadata={**metadata, "retrieval_group": values.get("group")},
                dense_score=dense_score,
                lexical_score=values.get("lexical"),
                lexical_coverage=coverage,
                fused_score=rrf_scores[child_id] / max_rrf,
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

    def retrieve(
        self,
        query: str,
        scope: Scope,
        *,
        query_expansions: tuple[str, ...] = (),
        preserve_all: bool = False,
    ) -> RetrievalBundle:
        ranked = self._fuse(
            query,
            scope,
            query_expansions=tuple(
                value.strip() for value in query_expansions if value.strip()
            )[:3],
        )
        if self.reranker is not None:
            ranked = self.reranker.rerank(query, ranked, scope)
        parent_limit = (
            MAX_CONTEXT_PARENTS
            if not scope.is_comparison
            else max(MAX_CONTEXT_PARENTS, len(scope.retrieval_groups) * 2)
        )
        return self.context_builder.build(
            ranked,
            scope,
            max_parents=parent_limit,
            max_chars=MAX_CONTEXT_CHARS,
            query=query,
            preserve_all=preserve_all,
        )
