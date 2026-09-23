from __future__ import annotations

from collections.abc import Collection

from src.constants import TICKERS, YEARS
from src.schemas import Decision, QueryUnderstanding, Scope, ScopeResolution


OUT_OF_SCOPE_MESSAGE = (
    "This assistant is limited to the uploaded 2024-2025 SEC 10-K reports and "
    "Q4 earnings transcripts for JPM, MSFT, and TSLA. It cannot answer that request "
    "from this document collection."
)


def _format_values(values: tuple[str, ...]) -> str:
    return ", ".join(values)


def _conflict_message(field: str, query_values: tuple[str, ...], ui_values: tuple[str, ...]) -> str:
    return (
        f"The question requests {field} {_format_values(query_values)}, but the active "
        f"filter selects {_format_values(ui_values)}. Change the filter or revise the question."
    )


def _missing_sources(
    scope: Scope, available_keys: Collection[tuple[str, str, str]]
) -> list[tuple[str, str, str]]:
    missing: list[tuple[str, str, str]] = []
    for ticker, year in scope.groups:
        required_types = scope.required_doc_types or (
            (scope.doc_type,) if scope.doc_type else ()
        )
        if required_types:
            for document_type in required_types:
                key = (ticker, year, document_type)
                if key not in available_keys:
                    missing.append(key)
        elif not any((ticker, year, doc_type) in available_keys for doc_type in ("10K", "TRANSCRIPT")):
            missing.append((ticker, year, "10K or TRANSCRIPT"))
    return missing


def _latest_groups(
    tickers: tuple[str, ...],
    document_types: tuple[str, ...],
    available_keys: Collection[tuple[str, str, str]],
) -> tuple[tuple[str, str], ...]:
    groups: list[tuple[str, str]] = []
    for ticker in tickers:
        candidate_years = {
            year
            for source_ticker, year, source_type in available_keys
            if source_ticker == ticker
            and (not document_types or source_type in document_types)
        }
        if candidate_years:
            groups.append((ticker, max(candidate_years)))
    return tuple(groups)


def _has_document_type_ambiguity(
    scope: Scope,
    available_keys: Collection[tuple[str, str, str]] | None,
) -> bool:
    if available_keys is None:
        return True
    return any(
        (ticker, year, "10K") in available_keys
        and (ticker, year, "TRANSCRIPT") in available_keys
        for ticker, year in scope.groups
    )


def resolve_scope(
    understanding: QueryUnderstanding,
    *,
    ui_scope: Scope | None = None,
    previous_scope: Scope | None = None,
    available_keys: Collection[tuple[str, str, str]] | None = None,
) -> ScopeResolution:
    ui = ui_scope or Scope()
    previous = previous_scope or Scope()

    if understanding.explicit_out_of_scope:
        return ScopeResolution(Decision.OUT_OF_SCOPE, Scope(), OUT_OF_SCOPE_MESSAGE)
    if understanding.unsupported_years or understanding.unsupported_companies:
        details = []
        if understanding.unsupported_companies:
            details.append("company " + _format_values(understanding.unsupported_companies))
        if understanding.unsupported_years:
            details.append("year " + _format_values(understanding.unsupported_years))
        return ScopeResolution(
            Decision.OUT_OF_SCOPE,
            Scope(),
            f"The requested {' and '.join(details)} is outside the available corpus. "
            + OUT_OF_SCOPE_MESSAGE,
        )
    if (
        not understanding.domain_relevant
        and not understanding.generic_followup
        and not understanding.tickers
        and not understanding.years
    ):
        return ScopeResolution(Decision.OUT_OF_SCOPE, Scope(), OUT_OF_SCOPE_MESSAGE)

    query_tickers = tuple(TICKERS) if understanding.all_tickers else understanding.tickers
    query_years = tuple(YEARS) if understanding.all_years else understanding.years

    if ui.tickers and query_tickers and set(ui.tickers) != set(query_tickers):
        return ScopeResolution(
            Decision.SCOPE_CONFLICT,
            ui,
            _conflict_message("company", query_tickers, ui.tickers),
        )
    if ui.years and query_years and set(ui.years) != set(query_years):
        return ScopeResolution(
            Decision.SCOPE_CONFLICT,
            ui,
            _conflict_message("year", query_years, ui.years),
        )
    if (
        ui.doc_type
        and understanding.requested_doc_types
        and understanding.requested_doc_types != (ui.doc_type,)
    ):
        return ScopeResolution(
            Decision.SCOPE_CONFLICT,
            ui,
            _conflict_message(
                "document type", understanding.requested_doc_types, (ui.doc_type,)
            ),
        )

    inherited: list[str] = []
    if query_tickers:
        tickers = query_tickers
    elif ui.tickers:
        tickers = ui.tickers
    elif query_years and len(previous.tickers) > 1:
        tickers = ()
    elif understanding.generic_followup:
        tickers = previous.tickers
        if tickers:
            inherited.append("company")
    else:
        tickers = ()

    if query_years:
        years = query_years
    elif ui.years:
        years = ui.years
    elif query_tickers and len(previous.years) > 1:
        years = ()
    elif understanding.generic_followup:
        years = previous.years
        if years:
            inherited.append("year")
    else:
        years = ()

    required_doc_types: tuple[str, ...] = ()
    if len(understanding.requested_doc_types) > 1:
        doc_type = None
        required_doc_types = understanding.requested_doc_types
    elif understanding.doc_type:
        doc_type = understanding.doc_type
    elif ui.doc_type:
        doc_type = ui.doc_type
    elif understanding.generic_followup:
        doc_type = previous.doc_type
        required_doc_types = previous.required_doc_types
        if doc_type:
            inherited.append("document type")
        elif required_doc_types:
            inherited.append("document types")
    else:
        doc_type = None

    latest_groups: tuple[tuple[str, str], ...] = ()
    if understanding.latest_available_year and not query_years and available_keys is not None:
        latest_groups = _latest_groups(
            tickers,
            required_doc_types or ((doc_type,) if doc_type else ()),
            available_keys,
        )
        if latest_groups:
            years = tuple(dict.fromkeys(year for _, year in latest_groups))

    preserve_previous_groups = (
        understanding.generic_followup
        and not understanding.requested_groups
        and not query_tickers
        and not query_years
        and not ui.tickers
        and not ui.years
    )
    requested_groups = (
        understanding.requested_groups
        or latest_groups
        or (previous.requested_groups if preserve_previous_groups else ())
    )
    scope = Scope(
        tickers=tickers,
        years=years,
        doc_type=doc_type,
        requested_groups=requested_groups,
        required_doc_types=required_doc_types,
    )
    missing_fields = []
    if not scope.tickers:
        missing_fields.append("company (JPM, MSFT, or TSLA)")
    if not scope.years:
        missing_fields.append("year (2024 or 2025)")
    if missing_fields:
        return ScopeResolution(
            Decision.CLARIFY,
            scope,
            "Please specify the " + " and ".join(missing_fields) + ".",
            tuple(inherited),
        )

    if available_keys is not None:
        unavailable = _missing_sources(scope, available_keys)
        if unavailable:
            formatted = ", ".join(
                f"{ticker} {year} {document_type}"
                for ticker, year, document_type in unavailable
            )
            return ScopeResolution(
                Decision.DATA_UNAVAILABLE,
                scope,
                f"The requested source is not loaded: {formatted}. Choose an available "
                "company, year, or document type.",
                tuple(inherited),
            )

    if (
        understanding.topic == "performance"
        and not understanding.requested_doc_types
        and not ui.doc_type
        and not doc_type
        and not required_doc_types
        and _has_document_type_ambiguity(scope, available_keys)
    ):
        return ScopeResolution(
            Decision.CLARIFY,
            scope,
            "Please choose the evidence basis: the annual 10-K, the Q4 earnings "
            "transcript, or both with periods reported separately.",
            tuple(inherited),
        )

    return ScopeResolution(Decision.SEARCH, scope, inherited_fields=tuple(inherited))
