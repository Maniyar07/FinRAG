"""Deterministic multi-turn clarification state.

Clarification replies may fill scope fields, but they never bypass the normal
scope policy. The original financial question remains the retrieval and answer
generation query after all required fields have been collected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from src.retrieval.query_understanding import understand_query
from src.schemas import PendingClarification, QueryUnderstanding, Scope, ScopeResolution


AFFIRM_RE = re.compile(r"^(?:yes|yeah|yep|correct|right|confirm(?:ed)?|sure)\b", re.I)
DENY_RE = re.compile(r"^(?:no|nope|incorrect|wrong)\b", re.I)
DOCUMENT_FIELD_RE = re.compile(
    r"\b(?:document type|evidence basis)\b|\bchoose\b.{0,100}\b(?:10-k|transcript)\b",
    re.I,
)


@dataclass(frozen=True)
class ClarificationTurn:
    understanding: QueryUnderstanding
    candidate_tickers: tuple[str, ...] = ()


def _document_scope(understanding: QueryUnderstanding, current: Scope) -> tuple[
    str | None, tuple[str, ...]
]:
    requested = understanding.requested_doc_types
    if not requested:
        return current.doc_type, current.required_doc_types
    if len(requested) == 1:
        return requested[0], ()
    return None, requested


def merge_clarification_reply(
    pending: PendingClarification,
    reply: str,
) -> ClarificationTurn | None:
    """Merge a short scope reply, or return ``None`` for a new substantive query."""

    reply_understanding = understand_query(reply)
    normalized = reply.strip()
    affirmative = bool(AFFIRM_RE.search(normalized))
    negative = bool(DENY_RE.search(normalized))
    has_scope_value = bool(
        reply_understanding.tickers
        or reply_understanding.years
        or reply_understanding.requested_doc_types
    )
    if reply_understanding.topic is not None or not (
        affirmative or negative or has_scope_value
    ):
        return None

    candidates = pending.candidate_tickers
    if reply_understanding.tickers:
        tickers = reply_understanding.tickers
        candidates = ()
    elif affirmative and len(candidates) == 1:
        tickers = candidates
        candidates = ()
    else:
        tickers = pending.scope.tickers
        if negative and candidates:
            candidates = ()
            tickers = ()

    years = reply_understanding.years or pending.scope.years
    doc_type, required_doc_types = _document_scope(
        reply_understanding, pending.scope
    )
    original = understand_query(pending.original_question)
    scope_values_changed = bool(
        reply_understanding.tickers or reply_understanding.years
    )
    merged = replace(
        original,
        tickers=tickers,
        years=years,
        doc_type=doc_type,
        requested_doc_types=(
            required_doc_types
            if required_doc_types
            else (doc_type,) if doc_type else ()
        ),
        requested_groups=(
            () if scope_values_changed else pending.scope.requested_groups
        ),
        generic_followup=False,
        domain_relevant=True,
        query_expansions=pending.query_expansions,
        semantic_fallback_used=bool(pending.query_expansions),
        ambiguous_fields=(),
    )
    return ClarificationTurn(merged, candidates)


def pending_from_resolution(
    *,
    original_question: str,
    understanding: QueryUnderstanding,
    resolution: ScopeResolution,
    candidate_tickers: tuple[str, ...] = (),
) -> PendingClarification:
    """Create safe pending state from a deterministic clarification result."""

    fields = list(understanding.ambiguous_fields)
    scope = resolution.scope
    if (
        candidate_tickers
        and not understanding.tickers
        and set(scope.tickers) != set(candidate_tickers)
    ):
        # A proposed company must not coexist with an inherited, confirmed-looking
        # ticker. The user still needs to confirm the candidate.
        scope = replace(scope, tickers=(), requested_groups=())
    if not scope.tickers:
        fields.append("company")
    if not scope.years:
        fields.append("year")
    if DOCUMENT_FIELD_RE.search(resolution.message):
        fields.append("document type")

    return PendingClarification(
        original_question=original_question[:4000],
        scope=scope,
        candidate_tickers=tuple(dict.fromkeys(candidate_tickers)),
        missing_fields=tuple(dict.fromkeys(fields)),
        query_expansions=understanding.query_expansions,
    )
