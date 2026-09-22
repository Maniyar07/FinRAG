"""A bounded tool interface over FinRAG's existing hybrid retriever."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from src.schemas import RetrievalBundle, Scope


MAX_QUERY_CHARS = 6_000
MAX_EXPANSIONS = 3
MAX_EXPANSION_CHARS = 100


class DocumentSearchPurpose(str, Enum):
    """Stable intent labels for traces and future multi-hop planning."""

    GENERAL = "general"
    MANAGEMENT_EXPLANATION = "management_explanation"
    RISK = "risk"
    GUIDANCE = "guidance"
    ACCOUNTING_POLICY = "accounting_policy"
    TREND = "trend"
    COMPARISON = "comparison"


@dataclass(frozen=True)
class DocumentSearchRequest:
    """Validated input for one focused search of the indexed corpus."""

    query: str
    scope: Scope
    purpose: DocumentSearchPurpose = DocumentSearchPurpose.GENERAL
    query_expansions: tuple[str, ...] = ()
    preserve_all: bool = False

    def __post_init__(self) -> None:
        query = " ".join(str(self.query).split())
        if not query:
            raise ValueError("Document search query cannot be empty.")
        if len(query) > MAX_QUERY_CHARS:
            raise ValueError(
                f"Document search query cannot exceed {MAX_QUERY_CHARS} characters."
            )
        if not isinstance(self.scope, Scope):
            raise TypeError("Document search scope must be a Scope instance.")
        if not self.scope.complete:
            raise ValueError("Document search requires a complete company/year scope.")
        if self.scope.doc_type and self.scope.required_doc_types:
            raise ValueError(
                "Document search scope cannot set both doc_type and required_doc_types."
            )

        try:
            purpose = DocumentSearchPurpose(self.purpose)
        except ValueError as error:
            raise ValueError(f"Unsupported document search purpose: {self.purpose}.") from error

        expansions = tuple(
            " ".join(str(value).split())
            for value in self.query_expansions
            if str(value).strip()
        )
        if len(expansions) > MAX_EXPANSIONS:
            raise ValueError(
                f"Document search accepts at most {MAX_EXPANSIONS} query expansions."
            )
        if any(len(value) > MAX_EXPANSION_CHARS for value in expansions):
            raise ValueError(
                "Each document search query expansion cannot exceed "
                f"{MAX_EXPANSION_CHARS} characters."
            )
        if len({value.casefold() for value in expansions}) != len(expansions):
            raise ValueError("Document search query expansions must be unique.")

        object.__setattr__(self, "query", query)
        object.__setattr__(self, "purpose", purpose)
        object.__setattr__(self, "query_expansions", expansions)


@dataclass(frozen=True)
class DocumentSearchResult:
    """Evidence returned by one document-search hop."""

    query: str
    purpose: DocumentSearchPurpose
    bundle: RetrievalBundle

    @property
    def context(self) -> str:
        return self.bundle.context

    @property
    def sources(self) -> list[dict]:
        return self.bundle.sources

    @property
    def candidate_count(self) -> int:
        return self.bundle.candidate_count


class DocumentRetriever(Protocol):
    """The small part of HybridRetriever required by this tool."""

    def retrieve(
        self,
        query: str,
        scope: Scope,
        *,
        query_expansions: tuple[str, ...] = (),
        preserve_all: bool = False,
    ) -> RetrievalBundle: ...


def _document_types(scope: Scope) -> frozenset[str] | None:
    """Return exact permitted types, or None when the scope allows either type."""
    if scope.required_doc_types:
        return frozenset(scope.required_doc_types)
    if scope.doc_type:
        return frozenset((scope.doc_type,))
    return None


def _validate_scope_containment(requested: Scope, permitted: Scope) -> None:
    """Prevent a tool call from widening the scope resolved for the user."""
    if not permitted.complete:
        raise ValueError("Permitted document search scope must be complete.")

    requested_groups = set(requested.groups)
    permitted_groups = set(permitted.groups)
    if not requested_groups.issubset(permitted_groups):
        raise ValueError(
            "Document search cannot expand beyond the permitted company/year scope."
        )

    requested_types = _document_types(requested)
    permitted_types = _document_types(permitted)
    if permitted_types is None:
        return
    if requested_types is None or not requested_types.issubset(permitted_types):
        raise ValueError(
            "Document search cannot expand beyond the permitted document-type scope."
        )


class DocumentSearchTool:
    """Run focused, scope-safe searches through the existing retrieval pipeline."""

    name = "document_search"
    description = (
        "Search indexed 10-K filings and earnings transcripts for narrative evidence "
        "such as explanations, risks, guidance, accounting policies, and trends."
    )

    def __init__(self, retriever: DocumentRetriever) -> None:
        self.retriever = retriever

    def execute(
        self,
        request: DocumentSearchRequest,
        *,
        permitted_scope: Scope,
    ) -> DocumentSearchResult:
        if not isinstance(request, DocumentSearchRequest):
            raise TypeError("request must be a DocumentSearchRequest instance.")
        if not isinstance(permitted_scope, Scope):
            raise TypeError("permitted_scope must be a Scope instance.")

        _validate_scope_containment(request.scope, permitted_scope)
        bundle = self.retriever.retrieve(
            request.query,
            request.scope,
            query_expansions=request.query_expansions,
            preserve_all=request.preserve_all,
        )
        return DocumentSearchResult(
            query=request.query,
            purpose=request.purpose,
            bundle=bundle,
        )
