from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from src.config import COMPRESSION_ENABLED, COMPRESSION_KEEP_BLOCKS, MAX_CONTEXT_TOKENS
from src.ingestion.parent_store import ParentStore
from src.retrieval.compression import compress, compress_to_budget
from src.retrieval.token_budget import count_tokens, token_counter_name
from src.observability import trace_component
from src.schemas import RankedChild, RetrievalBundle, Scope


GroupKey = tuple[str, str, str | None]


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_doc_type(value: object) -> str | None:
    text = _clean_text(value).upper().replace("-", "")
    return text or None


def _normalize_group(group: GroupKey) -> GroupKey:
    ticker, year, doc_type = group
    return (
        _clean_text(ticker).upper(),
        _clean_text(year),
        _normalize_doc_type(doc_type),
    )


def _child_group(
    child: RankedChild, requested_groups: tuple[GroupKey, ...]
) -> GroupKey | None:
    """Return the requested group matched by a child's safe metadata."""
    ticker = _clean_text(child.metadata.get("ticker")).upper()
    year = _clean_text(child.metadata.get("fiscal_year"))
    doc_type = _normalize_doc_type(child.metadata.get("doc_type"))

    for group in requested_groups:
        group_ticker, group_year, group_doc_type = group
        if (
            ticker == group_ticker
            and year == group_year
            and (group_doc_type is None or doc_type == group_doc_type)
        ):
            return group
    return None


def _rounded(value: object) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _page_label(metadata: dict) -> str:
    start = metadata.get("pdf_page_start")
    end = metadata.get("pdf_page_end")
    if start is None:
        return str(metadata.get("pdf_page", "not available"))
    return str(start) if end in (None, start) else f"{start}-{end}"


def _source_header(source_id: str, metadata: dict) -> str:
    fields = [
        f"SOURCE {source_id}",
        f"TICKER: {metadata.get('ticker')}",
        f"FISCAL YEAR: {metadata.get('fiscal_year')}",
        f"DOCUMENT PERIOD ONLY: {metadata.get('fiscal_period')} (not the period of every fact)",
        f"TYPE: {metadata.get('doc_type')}",
        f"FILE: {metadata.get('source')}",
        f"SECTION: {metadata.get('section')}",
        f"PDF PAGE: {_page_label(metadata)}",
    ]
    if metadata.get("period_end_date"):
        fields.append(f"PERIOD END: {metadata['period_end_date']}")
    if metadata.get("fiscal_q4_months"):
        fields.append(f"FISCAL Q4 MONTHS: {metadata['fiscal_q4_months']}")
    if metadata.get("call_date"):
        fields.append(f"CALL DATE: {metadata['call_date']}")
    if metadata.get("speaker"):
        fields.append(f"SPEAKER: {metadata['speaker']}")
    if metadata.get("speaker_role"):
        fields.append(f"SPEAKER ROLE: {metadata['speaker_role']}")
    return "[" + " | ".join(fields) + "]"


class ContextBuilder:
    def __init__(self, parent_store_path: Path):
        if not parent_store_path.is_dir():
            raise FileNotFoundError(f"Parent store not found: '{parent_store_path}'.")
        self.parent_store = ParentStore(parent_store_path)

    @staticmethod
    def _select_parent_ids(
        ranked_children: list[RankedChild], scope: Scope, limit: int
    ) -> tuple[list[str], dict[str, RankedChild]]:
        best: dict[str, RankedChild] = {}
        for child in ranked_children:
            current = best.get(child.parent_id)
            if current is None or child.fused_score > current.fused_score:
                best[child.parent_id] = child
        ranked = sorted(
            best.values(), key=lambda child: child.fused_score, reverse=True
        )
        groups_to_cover = tuple(
            _normalize_group(group) for group in scope.retrieval_groups
        )
        if not groups_to_cover:
            return [child.parent_id for child in ranked[:limit]], best
        if len(groups_to_cover) == 1:
            matched = [
                child
                for child in ranked
                if _child_group(child, groups_to_cover) is not None
            ]
            return [child.parent_id for child in matched[:limit]], best

        groups: dict[GroupKey, list[RankedChild]] = defaultdict(list)
        for child in ranked:
            group = _child_group(child, groups_to_cover)
            if group is not None:
                groups[group].append(child)

        selected: list[str] = []
        selected_ids: set[str] = set()
        # Round-robin selection prevents one company/year/source group from
        # consuming the entire parent budget. The progress flag prevents an
        # endless loop if malformed metadata matches no requested group.
        while len(selected) < limit:
            made_progress = False
            for key in groups_to_cover:
                if not groups[key] or len(selected) >= limit:
                    continue
                parent_id = groups[key].pop(0).parent_id
                if parent_id not in selected_ids:
                    selected.append(parent_id)
                    selected_ids.add(parent_id)
                    made_progress = True
            if not made_progress:
                break
        return selected, best

    @trace_component("finrag.retrieval.context_build")
    def build(
        self,
        ranked_children: list[RankedChild],
        scope: Scope,
        *,
        max_parents: int,
        max_chars: int,
        max_tokens: int = MAX_CONTEXT_TOKENS,
        query: str = "",
        preserve_all: bool = False,
        exclude_parent_ids: frozenset[str] = frozenset(),
    ) -> RetrievalBundle:
        parent_ids, best = self._select_parent_ids(
            ranked_children, scope, len(ranked_children)
        )
        selected_before_exclusions = len(parent_ids)
        excluded_count = sum(
            parent_id in exclude_parent_ids for parent_id in parent_ids
        )
        parent_ids = [
            parent_id
            for parent_id in parent_ids
            if parent_id not in exclude_parent_ids
        ]
        parents = self.parent_store.get_many(parent_ids)
        children_by_parent: dict[str, set[str]] = defaultdict(set)
        for child in ranked_children:
            children_by_parent[child.parent_id].add(child.child_id)
        context_parts: list[str] = []
        sources: list[dict] = []
        attempts: list[dict[str, object]] = []

        for parent_id, parent in zip(parent_ids, parents):
            if len(sources) >= max_parents:
                break
            if parent is None:
                attempts.append(
                    {"parent_id": parent_id, "outcome": "skipped_missing_parent"}
                )
                continue
            source_id = f"S{len(sources) + 1}"
            header = _source_header(source_id, parent.metadata)
            full_body = parent.page_content.strip()
            keep_parent_whole = preserve_all or len(children_by_parent[parent_id]) > 1
            body = full_body
            stats = {}
            if COMPRESSION_ENABLED and query:
                body, stats = compress(
                    body,
                    query,
                    keep_blocks=COMPRESSION_KEEP_BLOCKS,
                    preserve_all=keep_parent_whole,
                )
            block = f"{header}\n{body}"
            candidate_context = "\n\n---\n\n".join((*context_parts, block))
            initial_chars = len(candidate_context)
            initial_tokens = count_tokens(candidate_context)
            evidence_mode = (
                "full_parent" if body == full_body else "compressed_parent"
            )
            if initial_chars > max_chars or initial_tokens > max_tokens:
                if keep_parent_whole:
                    attempts.append({"parent_id": parent_id, "outcome": "skipped_full_parent_budget",
                                     "required_chars": initial_chars, "required_tokens": initial_tokens})
                    continue
                existing = "\n\n---\n\n".join(context_parts)
                separator = "\n\n---\n\n" if existing else ""
                prefix = f"{existing}{separator}{header}\n"
                focused, focused_stats = compress_to_budget(
                    full_body,
                    query,
                    max_chars=max_chars - len(prefix),
                    max_tokens=max_tokens - count_tokens(prefix),
                    keep_blocks=COMPRESSION_KEEP_BLOCKS,
                )
                if focused:
                    focused_block = f"{header}\n{focused}"
                    focused_context = "\n\n---\n\n".join(
                        (*context_parts, focused_block)
                    )
                    if (
                        len(focused_context) <= max_chars
                        and count_tokens(focused_context) <= max_tokens
                    ):
                        body = focused
                        stats = focused_stats
                        block = focused_block
                        candidate_context = focused_context
                        evidence_mode = "budget_excerpt"
                if (
                    len(candidate_context) > max_chars
                    or count_tokens(candidate_context) > max_tokens
                ):
                    attempts.append(
                        {
                            "parent_id": parent_id,
                            "ticker": parent.metadata.get("ticker"),
                            "fiscal_year": parent.metadata.get("fiscal_year"),
                            "outcome": "skipped_no_evidence_fit",
                            "initial_chars": initial_chars,
                            "initial_tokens": initial_tokens,
                            "compression_status": stats.get("status"),
                            "budget_excerpt_status": focused_stats.get("status"),
                        }
                    )
                    continue
            context_parts.append(block)
            child = best[parent_id]
            attempts.append(
                {
                    "parent_id": parent_id,
                    "ticker": parent.metadata.get("ticker"),
                    "fiscal_year": parent.metadata.get("fiscal_year"),
                    "outcome": "selected",
                    "evidence_mode": evidence_mode,
                    "context_chars_after": len(candidate_context),
                    "context_tokens_after": count_tokens(candidate_context),
                    "compression_status": stats.get("status"),
                }
            )
            sources.append(
                {
                    "id": source_id,
                    "evidence_text": body,
                    "full_evidence_text": full_body,
                    "compression": stats,
                    "rerank_applied": bool(
                        child.metadata.get("rerank_applied")
                        or child.metadata.get("rerank_score") is not None
                    ),
                    "rerank_provider": child.metadata.get("rerank_provider"),
                    "rerank_model": child.metadata.get("rerank_model"),
                    "rerank_score": _rounded(
                        child.metadata.get("rerank_score")
                    ),
                    "rrf_score": _rounded(
                        child.metadata.get(
                            "rrf_score",
                            child.metadata.get("pre_rerank_score"),
                        )
                    ),
                    "pre_rerank_score": _rounded(
                        child.metadata.get("pre_rerank_score")
                    ),
                    "parent_id": parent_id,
                    "parent_complete": body == full_body,
                    "evidence_mode": evidence_mode,
                    "score": round(child.fused_score, 4),
                    "dense_score": (
                        round(child.dense_score, 4)
                        if child.dense_score is not None
                        else None
                    ),
                    "lexical_score": _rounded(child.lexical_score),
                    "lexical_coverage": round(child.lexical_coverage, 4),
                    "retrieval_group": child.metadata.get("retrieval_group"),
                    "ticker": parent.metadata.get("ticker")
                    or child.metadata.get("ticker"),
                    "fiscal_year": parent.metadata.get("fiscal_year")
                    or child.metadata.get("fiscal_year"),
                    "fiscal_period": parent.metadata.get("fiscal_period"),
                    "fiscal_year_end": parent.metadata.get("fiscal_year_end"),
                    "fiscal_q4_months": parent.metadata.get("fiscal_q4_months"),
                    "doc_type": parent.metadata.get("doc_type")
                    or child.metadata.get("doc_type"),
                    "source": parent.metadata.get("source"),
                    "section": parent.metadata.get("section"),
                    "pdf_page": _page_label(parent.metadata),
                    "period_end_date": parent.metadata.get("period_end_date"),
                    "call_date": parent.metadata.get("call_date"),
                    "speaker": parent.metadata.get("speaker"),
                    "speaker_role": parent.metadata.get("speaker_role"),
                    "source_hash": parent.metadata.get("source_hash"),
                }
            )

        covered = tuple(
            sorted(
                {(str(item["ticker"]), str(item["fiscal_year"])) for item in sources}
            )
        )
        context = "\n\n---\n\n".join(context_parts)
        diagnostics = {
            "token_counter": token_counter_name(),
            "budget": {
                "max_parents": max_parents,
                "max_chars": max_chars,
                "max_tokens": max_tokens,
            },
            "ranked_child_count": len(ranked_children),
            "unique_parent_count": selected_before_exclusions,
            "excluded_parent_count": excluded_count,
            "attempted_parent_count": len(attempts),
            "selected_source_count": len(sources),
            "final_context_chars": len(context),
            "final_context_tokens": count_tokens(context),
            "covered_groups": [list(group) for group in covered],
            "parent_attempts": attempts,
        }
        return RetrievalBundle(
            context=context,
            sources=sources,
            scope=scope,
            candidate_count=len(ranked_children),
            covered_groups=covered,
            context_tokens=count_tokens(context),
            diagnostics=diagnostics,
        )
