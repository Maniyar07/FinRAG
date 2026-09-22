from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Decision(str, Enum):
    SEARCH = "search"
    CLARIFY = "clarify"
    SCOPE_CONFLICT = "scope_conflict"
    OUT_OF_SCOPE = "out_of_scope"
    DATA_UNAVAILABLE = "data_unavailable"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    ANSWERED = "answered"
    ERROR = "error"


@dataclass(frozen=True)
class Scope:
    tickers: tuple[str, ...] = ()
    years: tuple[str, ...] = ()
    doc_type: str | None = None
    requested_groups: tuple[tuple[str, str], ...] = ()
    required_doc_types: tuple[str, ...] = ()

    @property
    def groups(self) -> tuple[tuple[str, str], ...]:
        if self.requested_groups:
            return self.requested_groups
        return tuple((ticker, year) for ticker in self.tickers for year in self.years)

    @property
    def complete(self) -> bool:
        return bool(self.groups)

    @property
    def is_comparison(self) -> bool:
        return len(self.groups) > 1 or len(self.required_doc_types) > 1

    @property
    def retrieval_groups(self) -> tuple[tuple[str, str, str | None], ...]:
        """Return independently searched groups needed for balanced evidence."""
        document_types: tuple[str | None, ...]
        if self.required_doc_types:
            document_types = self.required_doc_types
        elif self.doc_type:
            document_types = (self.doc_type,)
        else:
            document_types = (None,)
        return tuple(
            (ticker, year, document_type)
            for ticker, year in self.groups
            for document_type in document_types
        )

    def label(self) -> str:
        if self.required_doc_types:
            document_label = " and ".join(
                "10-K" if value == "10K" else "transcript"
                for value in self.required_doc_types
            )
        else:
            document_label = (
                "10-K" if self.doc_type == "10K"
                else "transcript" if self.doc_type == "TRANSCRIPT"
                else "10-K and transcript"
            )
        if self.requested_groups:
            pairs = " vs ".join(f"{ticker} {year}" for ticker, year in self.requested_groups)
            return f"{pairs} | {document_label}"
        ticker = ", ".join(self.tickers) or "company not selected"
        year = ", ".join(self.years) or "year not selected"
        return f"{ticker} | {year} | {document_label}"


@dataclass(frozen=True)
class QueryUnderstanding:
    tickers: tuple[str, ...]
    years: tuple[str, ...]
    doc_type: str | None
    all_tickers: bool = False
    all_years: bool = False
    comparison: bool = False
    explicit_out_of_scope: bool = False
    topic: str | None = None
    unsupported_years: tuple[str, ...] = ()
    unsupported_companies: tuple[str, ...] = ()
    domain_relevant: bool = False
    generic_followup: bool = False
    requested_groups: tuple[tuple[str, str], ...] = ()
    requested_doc_types: tuple[str, ...] = ()
    reference_years: tuple[str, ...] = ()
    latest_available_year: bool = False
    wants_table: bool = False
    wants_complete_table: bool = False
    normalized_query: str = ""
    semantic_fallback_used: bool = False
    query_expansions: tuple[str, ...] = ()
    ambiguous_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScopeResolution:
    decision: Decision
    scope: Scope
    message: str = ""
    inherited_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class PendingClarification:
    original_question: str
    scope: Scope = field(default_factory=Scope)
    candidate_tickers: tuple[str, ...] = ()
    missing_fields: tuple[str, ...] = ()
    query_expansions: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceFile:
    path: Path
    ticker: str
    fiscal_year: str
    doc_type: str

    @property
    def key(self) -> tuple[str, str, str]:
        return self.ticker, self.fiscal_year, self.doc_type

    
@dataclass(frozen=True)
class RankedChild:
    child_id: str
    parent_id: str
    text: str
    metadata: dict
    dense_score: float | None = None
    lexical_score: float | None = None
    lexical_coverage: float = 0.0
    fused_score: float = 0.0


@dataclass(frozen=True)
class RetrievalBundle:
    context: str
    sources: list[dict]
    scope: Scope
    candidate_count: int
    covered_groups: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ChatResult:
    decision: Decision
    answer: str
    scope: Scope = field(default_factory=Scope)
    sources: list[dict] = field(default_factory=list)
    inherited_fields: tuple[str, ...] = ()
    trace: dict = field(default_factory=dict)
    pending_clarification: PendingClarification | None = None
