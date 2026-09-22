"""Optional Cohere API reranking with comparison-group protection.

The module uses Python's standard HTTP client, so it does not install PyTorch,
sentence-transformers, or native DLLs. If Cohere is disabled or unavailable,
the caller keeps the original RRF ranking.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, replace
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.schemas import RankedChild, Scope


LOGGER = logging.getLogger(__name__)
COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"
RETRYABLE_HTTP_CODES = frozenset({408, 429, 500, 502, 503, 504})


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _clean_text(value: object) -> str:
    """Convert optional metadata to text without producing the string 'None'."""
    if value is None:
        return ""
    return str(value).strip()


def _normalize_doc_type(value: object) -> str | None:
    """Normalize known document-type formatting while preserving missing values."""
    text = _clean_text(value).upper().replace("-", "")
    return text or None


@dataclass(frozen=True)
class CohereRerankSettings:
    """Environment-backed settings for the optional API stage."""

    api_key: str
    model: str
    max_candidates: int
    top_n: int
    minimum_per_group: int
    timeout_seconds: float
    retries: int
    max_document_chars: int

    @classmethod
    def from_environment(cls) -> CohereRerankSettings:
        max_candidates = _positive_int("FINRAG_RERANK_CANDIDATE_K", 20)
        if max_candidates > 100:
            raise ValueError("FINRAG_RERANK_CANDIDATE_K cannot exceed 100.")

        top_n = _positive_int("FINRAG_RERANK_TOP_N", 5)
        if top_n > max_candidates:
            raise ValueError(
                "FINRAG_RERANK_TOP_N cannot exceed FINRAG_RERANK_CANDIDATE_K."
            )

        retries = int(os.getenv("FINRAG_RERANK_RETRIES", "1"))
        if retries < 0 or retries > 2:
            raise ValueError("FINRAG_RERANK_RETRIES must be between 0 and 2.")

        minimum_per_group = _positive_int("FINRAG_RERANK_MIN_PER_GROUP", 2)
        if minimum_per_group > max_candidates:
            raise ValueError(
                "FINRAG_RERANK_MIN_PER_GROUP cannot exceed "
                "FINRAG_RERANK_CANDIDATE_K."
            )

        return cls(
            api_key=os.getenv("COHERE_API_KEY", "").strip(),
            model=os.getenv(
                "FINRAG_RERANK_MODEL", "rerank-v4.0-fast"
            ).strip(),
            max_candidates=max_candidates,
            top_n=top_n,
            minimum_per_group=minimum_per_group,
            timeout_seconds=_positive_float("FINRAG_RERANK_TIMEOUT_SECONDS", 12.0),
            retries=retries,
            max_document_chars=_positive_int("FINRAG_RERANK_MAX_DOCUMENT_CHARS", 6000),
        )


GroupKey = tuple[str, str, str | None]


def _scope_groups(scope: Scope) -> tuple[GroupKey, ...]:
    """Read modern Scope groups while remaining compatible with Phase 1."""
    retrieval_groups = getattr(scope, "retrieval_groups", ())
    if retrieval_groups:
        return tuple(
            (
                _clean_text(ticker).upper(),
                _clean_text(year),
                _normalize_doc_type(doc_type),
            )
            for ticker, year, doc_type in retrieval_groups
        )

    tickers = tuple(getattr(scope, "tickers", ()))
    years = tuple(getattr(scope, "years", ()))
    doc_type = getattr(scope, "doc_type", None)
    return tuple(
        (
            _clean_text(ticker).upper(),
            _clean_text(year),
            _normalize_doc_type(doc_type),
        )
        for ticker in tickers
        for year in years
    )


def _child_group(child: RankedChild, groups: tuple[GroupKey, ...]) -> GroupKey:
    ticker = _clean_text(child.metadata.get("ticker")).upper()
    year = _clean_text(child.metadata.get("fiscal_year"))
    # Missing metadata remains None; it never becomes the misleading string
    # "None" and therefore cannot falsely match a required document type.
    doc_type = _normalize_doc_type(child.metadata.get("doc_type"))

    for group in groups:
        group_ticker, group_year, group_doc_type = group
        if (
            ticker == group_ticker
            and year == group_year
            and (group_doc_type is None or doc_type == group_doc_type)
        ):
            return group
    return ticker, year, doc_type


class CohereReranker:
    """Rerank fused children and preserve requested comparison groups."""

    def __init__(self, settings: CohereRerankSettings):
        if not settings.api_key:
            raise ValueError("COHERE_API_KEY is required when reranking is enabled.")
        if not settings.model:
            raise ValueError("FINRAG_RERANK_MODEL cannot be empty.")
        self.settings = settings

    @staticmethod
    def _deduplicate_parents(children: list[RankedChild]) -> list[RankedChild]:
        """Keep the best fused child for each parent before paid reranking."""
        selected: list[RankedChild] = []
        seen_parent_ids: set[str] = set()
        for child in children:
            if child.parent_id in seen_parent_ids:
                continue
            seen_parent_ids.add(child.parent_id)
            selected.append(child)
        return selected

    def _candidate_pool(
        self, children: list[RankedChild], scope: Scope
    ) -> tuple[list[RankedChild], tuple[GroupKey, ...]]:
        """Build a bounded pool without allowing one group to consume it."""
        unique_children = self._deduplicate_parents(children)
        groups = _scope_groups(scope)
        if len(groups) <= 1:
            return unique_children[: self.settings.max_candidates], groups

        buckets: dict[GroupKey, list[RankedChild]] = {group: [] for group in groups}
        unmatched: list[RankedChild] = []
        for child in unique_children:
            group = _child_group(child, groups)
            if group in buckets:
                buckets[group].append(child)
            else:
                unmatched.append(child)

        selected: list[RankedChild] = []
        selected_ids: set[str] = set()
        while len(selected) < self.settings.max_candidates and any(buckets.values()):
            made_progress = False
            for group in groups:
                if buckets[group] and len(selected) < self.settings.max_candidates:
                    child = buckets[group].pop(0)
                    selected.append(child)
                    selected_ids.add(child.child_id)
                    made_progress = True
            if not made_progress:
                break

        # Normally every child maps to a requested group. This fallback keeps
        # compatible records whose older metadata is incomplete.
        for child in unmatched:
            if len(selected) >= self.settings.max_candidates:
                break
            if child.child_id not in selected_ids:
                selected.append(child)
        return selected, groups

    def _api_document(self, child: RankedChild) -> str:
        """Add compact source metadata without changing stored chunk text."""
        metadata = child.metadata
        header = (
            "[SOURCE METADATA | "
            f"ticker={metadata.get('ticker', 'unknown')} | "
            f"filing_year={metadata.get('fiscal_year', 'unknown')} | "
            f"document_type={metadata.get('doc_type', 'unknown')} | "
            f"section={metadata.get('section', 'unknown')} | "
            f"file={metadata.get('source', 'unknown')}]"
        )
        available = max(self.settings.max_document_chars - len(header) - 1, 1)
        return f"{header}\n{child.text[:available]}"

    def _request_scores(self, query: str, pool: list[RankedChild]) -> list[float]:
        payload = json.dumps(
            {
                "model": self.settings.model,
                "query": query,
                "documents": [self._api_document(child) for child in pool],
                # Request every candidate score; group-aware top-N is applied locally.
                "top_n": len(pool),
            }
        ).encode("utf-8")
        request = Request(
            COHERE_RERANK_URL,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "FinRAG/1.0",
            },
        )

        last_error: Exception | None = None
        for attempt in range(self.settings.retries + 1):
            try:
                with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                    result = json.loads(response.read().decode("utf-8"))
                return self._validate_scores(result, len(pool))
            except HTTPError as exc:
                last_error = exc
                if exc.code not in RETRYABLE_HTTP_CODES:
                    break
            except (URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc

            if attempt < self.settings.retries:
                time.sleep(0.5 * (2**attempt))

        raise RuntimeError(
            f"Cohere reranking failed after {self.settings.retries + 1} attempt(s): "
            f"{type(last_error).__name__ if last_error else 'unknown error'}"
        ) from last_error

    @staticmethod
    def _validate_scores(response: dict, candidate_count: int) -> list[float]:
        results = response.get("results")
        if not isinstance(results, list) or len(results) != candidate_count:
            raise ValueError("Cohere returned an incomplete reranking result.")

        scores: list[float | None] = [None] * candidate_count
        for result in results:
            index = result.get("index")
            score = result.get("relevance_score")
            if not isinstance(index, int) or not 0 <= index < candidate_count:
                raise ValueError("Cohere returned an invalid candidate index.")
            try:
                numeric_score = float(score)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Cohere returned a non-numeric relevance score."
                ) from exc
            if not math.isfinite(numeric_score) or scores[index] is not None:
                raise ValueError("Cohere returned an invalid relevance score.")
            scores[index] = numeric_score

        if any(score is None for score in scores):
            raise ValueError("Cohere omitted one or more candidate scores.")
        return [float(score) for score in scores]

    def _select_balanced(
        self,
        reranked: list[RankedChild],
        groups: tuple[GroupKey, ...],
    ) -> list[RankedChild]:
        """Keep normal top-N for simple queries and quotas for comparisons."""
        if len(groups) <= 1:
            return reranked[: self.settings.top_n]

        target = min(
            len(reranked),
            max(self.settings.top_n, len(groups) * self.settings.minimum_per_group),
        )
        selected: list[RankedChild] = []
        selected_ids: set[str] = set()

        for group in groups:
            group_hits = [
                child for child in reranked if _child_group(child, groups) == group
            ]
            for child in group_hits[: self.settings.minimum_per_group]:
                if child.child_id not in selected_ids:
                    selected.append(child)
                    selected_ids.add(child.child_id)

        for child in reranked:
            if len(selected) >= target:
                break
            if child.child_id not in selected_ids:
                selected.append(child)
                selected_ids.add(child.child_id)

        return sorted(selected, key=lambda child: (-child.fused_score, child.child_id))

    def rerank(
        self,
        query: str,
        children: list[RankedChild],
        scope: Scope,
    ) -> list[RankedChild]:
        """Return Cohere-ranked children, or unchanged RRF children on failure."""
        if not children:
            return []

        pool, groups = self._candidate_pool(children, scope)
        if not pool:
            return children

        try:
            scores = self._request_scores(query, pool)
        except RuntimeError as exc:
            LOGGER.warning("%s Falling back to dense + BM25 + RRF.", exc)
            return children

        reranked = [
            replace(
                child,
                metadata={
                    **child.metadata,
                    # Keep the normalized RRF score before fused_score is
                    # replaced by Cohere's relevance score.
                    "rrf_score": child.fused_score,
                    "pre_rerank_score": child.fused_score,
                    "rerank_provider": "cohere",
                    "rerank_model": self.settings.model,
                    "rerank_score": score,
                    "rerank_applied": True,
                },
                fused_score=score,
            )
            for child, score in zip(pool, scores)
        ]
        reranked.sort(key=lambda child: (-child.fused_score, child.child_id))
        return self._select_balanced(reranked, groups)


def build_cohere_reranker_from_environment() -> CohereReranker | None:
    """Create the optional reranker without changing application wiring."""
    if not _env_bool("FINRAG_RERANK_ENABLED", default=False):
        return None

    settings = CohereRerankSettings.from_environment()
    if not settings.api_key:
        LOGGER.warning(
            "FINRAG_RERANK_ENABLED=true but COHERE_API_KEY is missing; "
            "continuing with dense + BM25 + RRF."
        )
        return None
    return CohereReranker(settings)
