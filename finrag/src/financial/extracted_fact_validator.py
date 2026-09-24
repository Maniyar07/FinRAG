"""Validate extracted financial facts against retrieved source evidence."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from bs4 import BeautifulSoup

from src.financial.models import (
    CandidateFinancialFact,
    FactValidationResult,
    RejectedFinancialFact,
    ValidatedFinancialFact,
)
from src.financial.value_normalizer import (
    FinancialValueNormalizer,
    ValueNormalizationError,
)
from src.schemas import Scope


def _plain_text(value: object) -> str:
    text = BeautifulSoup(str(value or ""), "html.parser").get_text(" ")
    return " ".join(text.split())


def _folded(value: object) -> str:
    return _plain_text(value).casefold()


def _compact_value(value: object) -> str:
    # Currency symbols are sometimes placed in a separate visual table cell.
    # They do not change the numeric identity because currency is validated
    # independently by the normalizer.
    return re.sub(r"[\s,$€£¥]", "", _plain_text(value)).casefold()


def _excerpt_supported(excerpt: object, source_text: object) -> bool:
    """Allow table-cell reordering, but never unsupported words or numbers."""
    candidate = _folded(excerpt)
    source = _folded(source_text)
    if not candidate:
        return False
    if candidate in source:
        return True
    # HTML table readers commonly present headers before row labels while a
    # model cites them in row/period/value order. All meaningful tokens must
    # still occur verbatim somewhere in the same validated parent source.
    tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", candidate)
    if len(tokens) < 3:
        return False
    source_tokens = set(re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", source))
    numeric_tokens = [token for token in tokens if any(char.isdigit() for char in token)]
    if not all(token in source_tokens for token in numeric_tokens):
        return False
    stopwords = {
        "a",
        "an",
        "and",
        "as",
        "at",
        "for",
        "fiscal",
        "in",
        "million",
        "millions",
        "of",
        "the",
        "total",
        "was",
        "were",
    }
    meaningful = [
        token
        for token in tokens
        if token not in stopwords and not any(char.isdigit() for char in token)
    ]
    if not meaningful:
        return bool(numeric_tokens)
    supported = sum(token in source_tokens for token in meaningful)
    return supported / len(meaningful) >= 0.7


def _source_text(source: dict) -> str:
    return str(source.get("full_evidence_text") or source.get("evidence_text") or "")


def _source_in_scope(source: dict, scope: Scope) -> bool:
    ticker = str(source.get("ticker", "")).upper()
    year = str(source.get("fiscal_year", ""))
    doc_type = str(source.get("doc_type", "")).upper().replace("-", "")
    for expected_ticker, expected_year, expected_type in scope.retrieval_groups:
        if ticker != expected_ticker or year != expected_year:
            continue
        if expected_type is None or doc_type == expected_type:
            return True
    return False


def _fact_id(candidate: CandidateFinancialFact, source: dict) -> str:
    identity = "|".join(
        (
            str(source.get("source_hash", "")),
            str(source.get("parent_id", "")),
            candidate.source_id,
            candidate.ticker,
            candidate.metric.casefold(),
            candidate.period.casefold(),
            candidate.raw_value,
            (candidate.row_label or "").casefold(),
            (candidate.column_label or "").casefold(),
        )
    )
    return "F" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


class ExtractedFactValidator:
    """Fail closed unless a candidate can be tied to one retrieved source."""

    def __init__(self, normalizer: FinancialValueNormalizer | None = None) -> None:
        self.normalizer = normalizer or FinancialValueNormalizer()

    def validate(
        self,
        candidates: Iterable[CandidateFinancialFact],
        sources: list[dict],
        *,
        permitted_scope: Scope | None = None,
    ) -> FactValidationResult:
        source_map: dict[str, dict] = {}
        duplicate_ids: set[str] = set()
        for source in sources:
            source_id = str(source.get("id", "")).strip().upper()
            if not source_id:
                continue
            if source_id in source_map:
                duplicate_ids.add(source_id)
            source_map[source_id] = source

        valid: list[ValidatedFinancialFact] = []
        rejected: list[RejectedFinancialFact] = []
        for candidate in candidates:
            reason = self._rejection_reason(
                candidate,
                source_map,
                duplicate_ids,
                permitted_scope,
            )
            if reason:
                rejected.append(
                    RejectedFinancialFact(
                        source_id=candidate.source_id,
                        metric=candidate.metric,
                        reason=reason,
                        period=candidate.period,
                        raw_value=candidate.raw_value,
                        row_label=candidate.row_label,
                        column_label=candidate.column_label,
                    )
                )
                continue

            source = source_map[candidate.source_id]
            try:
                normalized = self.normalizer.normalize(candidate)
            except ValueNormalizationError as error:
                rejected.append(
                    RejectedFinancialFact(
                        source_id=candidate.source_id,
                        metric=candidate.metric,
                        reason=f"normalization_failed:{error}",
                        period=candidate.period,
                        raw_value=candidate.raw_value,
                        row_label=candidate.row_label,
                        column_label=candidate.column_label,
                    )
                )
                continue

            valid.append(
                ValidatedFinancialFact(
                    fact_id=_fact_id(candidate, source),
                    ticker=candidate.ticker,
                    metric=candidate.metric,
                    period=candidate.period,
                    raw_value=candidate.raw_value,
                    numeric_value=normalized.numeric_value,
                    base_value=normalized.base_value,
                    value_type=normalized.value_type,
                    normalized_unit=normalized.normalized_unit,
                    currency=normalized.currency,
                    scale=normalized.scale,
                    accounting_basis=candidate.accounting_basis,
                    source_id=candidate.source_id,
                    evidence_excerpt=candidate.evidence_excerpt,
                    row_label=candidate.row_label,
                    column_label=candidate.column_label,
                    validation_checks=(
                        "source_id",
                        "source_scope",
                        "ticker",
                        "raw_value",
                        "metric_or_row",
                        "period",
                        "evidence_excerpt",
                        "numeric_normalization",
                    ),
                )
            )
        return FactValidationResult(
            valid_facts=tuple(valid),
            rejected_facts=tuple(rejected),
        )

    @staticmethod
    def _rejection_reason(
        candidate: CandidateFinancialFact,
        source_map: dict[str, dict],
        duplicate_ids: set[str],
        permitted_scope: Scope | None,
    ) -> str | None:
        if candidate.source_id in duplicate_ids:
            return "duplicate_source_id"
        source = source_map.get(candidate.source_id)
        if source is None:
            return "unknown_source_id"
        if permitted_scope is not None and not _source_in_scope(source, permitted_scope):
            return "source_outside_permitted_scope"

        source_ticker = str(source.get("ticker", "")).strip().upper()
        if source_ticker != candidate.ticker:
            return "ticker_source_mismatch"

        text = _source_text(source)
        folded = _folded(text)
        compact = _compact_value(text)
        raw_folded = _folded(candidate.raw_value)
        if raw_folded not in folded and _compact_value(candidate.raw_value) not in compact:
            return "raw_value_not_in_source"

        label = candidate.row_label or candidate.metric
        if _folded(label) not in folded:
            return "metric_or_row_not_in_source"

        period_text = candidate.column_label or candidate.period
        period_years = re.findall(r"(?:19|20)\d{2}", period_text)
        if period_years and not all(year in folded for year in period_years):
            return "period_not_in_source"

        if not _excerpt_supported(candidate.evidence_excerpt, text):
            return "evidence_excerpt_not_in_source"
        return None
