"""Validate extracted financial facts against retrieved source evidence."""

from __future__ import annotations

import hashlib
import html
import re
from collections.abc import Iterable

from bs4 import BeautifulSoup

from src.financial.models import (
    CandidateFinancialFact,
    FactValidationResult,
    NormalizedFinancialValue,
    RejectedFinancialFact,
    ValidatedFinancialFact,
)
from src.financial.value_normalizer import (
    FinancialValueNormalizer,
    ValueNormalizationError,
)
from src.observability import trace_component
from src.financial.metric_terms import (
    has_financial_metric_conflict,
    is_metric_match,
    normalize_financial_term,
)
from src.financial.coverage import source_matches_requirement, normalize_reported_period
from src.retrieval.structured_lookup import StructuredDocumentLookup, MONETARY_STATEMENT_RE
from src.schemas import Scope


def _plain_text(value: object) -> str:
    text = str(value or "")
    text = BeautifulSoup(text, "html.parser").get_text(" ") if re.search(r"</?[a-z][^>]*>", text, re.I) else html.unescape(text)
    return " ".join(text.split())


def _folded(value: object) -> str:
    return _plain_text(value).casefold()


def _compact_value(value: object) -> str:
    # Currency symbols are sometimes placed in a separate visual table cell.
    # They do not change the numeric identity because currency is validated
    # independently by the normalizer.
    return re.sub(r"[\s,$€£¥]", "", _plain_text(value)).casefold()


def _excerpt_supported(excerpt: object, source_text: object, *, table: bool = False) -> bool:
    """Bind prose within one passage; table excerpts use independently bound cells."""
    candidate = re.sub(r"\bfy\s*(?=\d{4}\b)", "", _folded(excerpt))
    source = _folded(source_text)
    if not candidate:
        return False
    if candidate in source:
        return True
    tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", candidate)
    if len(tokens) < 3:
        return False
    if table:
        return set(tokens).issubset(set(re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", source)))
    pattern = r"\b" + r"\b.*?\b".join(map(re.escape, tokens)) + r"\b"
    return any(re.search(pattern, passage) for passage in re.split(r"(?<=[.!?])\s+", source))


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


def _fact_id(
    candidate: CandidateFinancialFact,
    source: dict,
    normalized: NormalizedFinancialValue,
) -> str:
    stable_source = (
        source.get("evidence_id") or source.get("parent_id") or source.get("source_hash") or source.get("id") or ""
    )
    identity = "|".join(
        (
            str(stable_source),
            str(source.get("ticker") or candidate.ticker).upper(),
            str(source.get("fiscal_year") or ""),
            str(source.get("doc_type") or "").upper().replace("-", ""),
            str(source.get("source") or ""),
            candidate.ticker,
            candidate.metric.casefold(),
            candidate.period.casefold(),
            candidate.raw_value,
            (candidate.row_label or "").casefold(),
            (candidate.column_label or "").casefold(),
            normalized.value_type.value,
            normalized.normalized_unit.casefold(),
            (normalized.currency or "").casefold(),
            (normalized.scale or "").casefold(),
            (candidate.accounting_basis or "").casefold(),
        )
    )
    return "F" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


class ExtractedFactValidator:
    """Fail closed unless a candidate can be tied to one retrieved source."""

    def __init__(self, normalizer: FinancialValueNormalizer | None = None) -> None:
        self.normalizer = normalizer or FinancialValueNormalizer()

    @trace_component("finrag.financial.fact_validate")
    def validate(
        self,
        candidates: Iterable[CandidateFinancialFact],
        sources: list[dict],
        *,
        permitted_scope: Scope | None = None,
        requirements: tuple[dict, ...] = (),
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
        table_grids: dict[str, list] = {}
        for candidate in candidates:
            evidence_context = ""
            requirement = next((item for item in requirements
                if item["requirement_id"] == candidate.requirement_id and item.get("kind", "fact") == "fact"), None)
            if requirement:
                matching = [item for item in requirements if item.get("kind", "fact") == "fact"
                    and source_matches_requirement(source_map.get(candidate.source_id, {}), item)
                    and normalize_financial_term(candidate.metric) == normalize_financial_term(item["metric"])
                    and normalize_reported_period(candidate.period) == normalize_reported_period(item.get("reported_period") or "")]
                if len(matching) == 1:
                    requirement = matching[0]
                    candidate = candidate.model_copy(update={"requirement_id": requirement["requirement_id"]})
            reason = self._rejection_reason(
                candidate,
                source_map,
                duplicate_ids,
                permitted_scope,
                requirement,
            )
            if candidate.requirement_id and requirement is None:
                reason = "unknown_requirement_id"
            try:
                if not reason:
                    source = source_map[candidate.source_id]
                    normalized = self.normalizer.normalize(candidate, prefer_presented_scale=True)
                    if requirement and not source_matches_requirement(source, requirement):
                        reason = "requirement_source_or_type_mismatch"
                    # When the table exposes one reported-year column, bind the
                    # claim to that row and cell, rather than matching a number
                    # elsewhere in the parent. Ambiguous headers stay on the
                    # existing extraction/evidence path.
                    years = set(re.findall(r"(?:19|20)\d{2}", f"{candidate.period} {candidate.column_label or ''}"))
                    bindings = []
                    matched_values = []
                    excerpt_supported = False
                    bound_metrics = set()
                    if candidate.source_id not in table_grids:
                        body = _source_text(source)
                        if "<table" not in body.lower():
                            body = str(source.get("evidence_text") or body)
                        table_grids[candidate.source_id] = [
                            (table, table.find_all("tr"), StructuredDocumentLookup.table_grid(table),
                             *StructuredDocumentLookup.reported_year_columns(table),
                             StructuredDocumentLookup.table_context(body, match.start()))
                            for match in re.finditer(r"<table\b.*?</table>", body, re.I | re.S)
                            for table in [BeautifulSoup(match.group(), "html.parser").find("table")]
                            if table is not None
                        ]
                    for table, rows, grid, width, year_columns, caption in table_grids[candidate.source_id] if len(years) == 1 else ():
                        columns = StructuredDocumentLookup.reported_value_columns(table, next(iter(years)), candidate.column_label or "")
                        if requirement and not is_metric_match(candidate.row_label or candidate.metric, requirement["metric"]):
                            requested_columns = StructuredDocumentLookup.reported_value_columns(table, next(iter(years)), requirement["metric"])
                            columns = tuple(column for column in columns if column in requested_columns)
                        if not columns:
                            continue
                        section_label = ""
                        for row, values in zip(rows, grid):
                            if values and len(set(values)) == 1 and values[0].strip() and not re.search(r"\d", values[0]):
                                section_label = values[0]
                            if (not row.find("td", recursive=False) or not values
                                    or _folded(values[0]) != _folded(candidate.row_label or candidate.metric)
                                    or len(values) - width not in {0, 1}):
                                continue
                            metric_label = values[0]
                            section_supported = True
                            if requirement and candidate.row_label:
                                requested_terms = normalize_financial_term(" ".join((requirement["metric"], *requirement.get("qualifiers", ()))))
                                row_terms = normalize_financial_term(metric_label)
                                section_terms = normalize_financial_term(section_label)
                                local_label = f"{section_label} {metric_label}".strip()
                                metric_conflict = (
                                    has_financial_metric_conflict(
                                        local_label, requirement["metric"],
                                    )
                                    and not is_metric_match(
                                        metric_label, requirement["metric"],
                                    )
                                )
                                section_supported = (
                                    not metric_conflict
                                    and (
                                        all(any(is_metric_match(term, target)
                                                for target in requested_terms)
                                            for term in section_terms)
                                        or all(any(is_metric_match(term, actual)
                                                   for actual in row_terms | section_terms)
                                               for term in requested_terms)
                                    )
                                )
                                local_terms = normalize_financial_term(" ".join((metric_label, section_label if section_supported else "",
                                    *(values[column] for row, values in zip(rows, grid)
                                      if row.find("th", recursive=False) for column in columns if column < len(values)))))
                                # A caption describes the whole table, not every
                                # component row. Terms also naming other rows
                                # cannot establish a larger aggregate here.
                                other_row_terms = set().union(*(normalize_financial_term(cells[0])
                                    for tr, cells in zip(rows, grid) if tr.find("td", recursive=False)
                                    and cells and _folded(cells[0]) != _folded(metric_label)))
                                qualifier_terms = normalize_financial_term(" ".join(requirement.get("qualifiers", ())))
                                scoped_row = bool(qualifier_terms) and qualifier_terms.issubset(row_terms | section_terms)
                                metric_terms = normalize_financial_term(requirement["metric"])
                                inherited_measure = metric_terms if scoped_row and any(
                                    tr.find("td", recursive=False) and cells and normalize_financial_term(cells[0]) == metric_terms
                                    for tr, cells in zip(rows, grid)) else set()
                                local_terms |= normalize_financial_term(caption) - (other_row_terms - inherited_measure)
                                metric_supported = not metric_conflict and all(any(is_metric_match(term, target)
                                    for target in requested_terms) for term in row_terms) and (is_metric_match(metric_label, requirement["metric"]) or all(
                                    any(is_metric_match(term, actual) for actual in local_terms)
                                    for term in requested_terms)) and not (
                                    "cost" in normalize_financial_term(section_label)
                                    and "revenue" in normalize_financial_term(requirement["metric"])
                                    and "cost" not in normalize_financial_term(requirement["metric"]))
                                if (re.search(r"\btotal\b", requirement["metric"], re.I)
                                        and not re.search(r"\btotal\b", metric_label, re.I)
                                        and not normalize_financial_term(section_label).issubset(requested_terms)
                                        and not ("financial statements" in str(source.get("section", "")).casefold()
                                                 and MONETARY_STATEMENT_RE.search(caption)
                                                 and is_metric_match(metric_label, requirement["metric"]))):
                                    metric_supported = False
                            elif not normalize_financial_term(candidate.metric).issubset(normalize_financial_term(metric_label)):
                                section_supported = not section_label or normalize_financial_term(section_label).issubset(normalize_financial_term(candidate.metric))
                                metric_label = f"{section_label} {metric_label}".strip()
                                metric_supported = section_supported and is_metric_match(metric_label, candidate.metric)
                            else:
                                metric_supported = is_metric_match(metric_label, candidate.metric)
                            bound_candidate = candidate
                            scale = re.search(r"\b(?:millions?|billions?|thousands?)\b", caption + " " + " ".join(
                                " ".join(values) for row, values in zip(rows, grid) if row.find("th", recursive=False)), re.I)
                            if normalized.value_type == "currency" and scale:
                                bound_candidate = candidate.model_copy(update={"scale": scale.group().lower()})
                            header_terms = normalize_financial_term(" ".join(header[column]
                                for tr, header in zip(rows, grid) if tr.find("th", recursive=False)
                                for column in columns if column < len(header)
                                and not re.search(r"(?:19|20)\d{2}", header[column])))
                            if re.search(r"\bper[\s-]+shares?\b", " ".join((section_label, metric_label, " ".join(header_terms))), re.I):
                                bound_candidate = bound_candidate.model_copy(update={"unit": "per share", "scale": None})
                            count_column = len(columns) == 1 and bool(header_terms & {"unit", "share"}) and not header_terms & {"per", "percent", "percentage", "rate", "ratio"}
                            if count_column:
                                count_scale = scale.group().lower() if scale else None
                                if re.search(r"\bexcept\b.*\b(?:units?|shares?)\b",
                                             re.sub(r"\bper[\s-]+(?:shares?|units?)\b", "", caption, flags=re.I), re.I):
                                    count_scale = None
                                bound_candidate = bound_candidate.model_copy(update={
                                    "unit": "count", "currency": None, "scale": count_scale})
                            bound_value = self.normalizer.normalize(bound_candidate, prefer_presented_scale=True)
                            cells = []
                            for column in columns:
                                position = column + len(values) - width
                                if position <= 0 or position >= len(values) or not re.search(r"\d", values[position]):
                                    continue
                                try:
                                    cell_value = self.normalizer.normalize(bound_candidate.model_copy(update={"raw_value": values[position]}))
                                    cells.append((column, cell_value))
                                except ValueNormalizationError:
                                    continue
                            matching_cells = [(column, value) for column, value in cells
                                              if value.base_value == bound_value.base_value and value.value_type == bound_value.value_type]
                            value_supported = len(matching_cells) == 1
                            if value_supported and requirement:
                                selected_column = matching_cells[0][0]
                                leaf_terms = normalize_financial_term(" ".join(header[selected_column]
                                    for tr, header in zip(rows, grid) if tr.find("th", recursive=False)
                                    and selected_column < len(header)
                                    and not re.search(r"(?:19|20)\d{2}", header[selected_column])))
                                target_terms = normalize_financial_term(requirement["metric"])
                                if count_column and bound_value.value_type != "count":
                                    reason = "table_unit_type_mismatch"
                                measures = {frozenset(normalize_financial_term(" ".join(header[column]
                                    for tr, header in zip(rows, grid) if tr.find("th", recursive=False)
                                    and column < len(header) and not re.search(r"(?:19|20)\d{2}", header[column]))))
                                    for column in year_columns[next(iter(years))]}
                                if len(measures - {frozenset()}) > 1 and not is_metric_match(metric_label, requirement["metric"]):
                                    selected_terms = normalize_financial_term(f"{metric_label} {section_label}") | leaf_terms
                                    metric_supported = metric_supported and all(any(is_metric_match(term, actual)
                                        for actual in selected_terms) for term in target_terms)
                                if all(any(is_metric_match(term, actual) for actual in normalize_financial_term(metric_label)) for term in target_terms) and leaf_terms and leaf_terms < target_terms:
                                    metric_supported = False
                            bindings.append((metric_supported, value_supported))
                            if metric_supported and value_supported:
                                # A unique copied value can resolve a missing leaf
                                # label. Never select among repeated values or years.
                                selected_column = matching_cells[0][0]
                                position = selected_column + len(values) - width
                                if (scale and MONETARY_STATEMENT_RE.search(caption)
                                        and bound_value.value_type != "currency"
                                        and not re.search(r"%|\b(?:percent(?:age)?|rates?|ratios?|bps|shares?|units?)\b",
                                                          metric_label + " " + section_label + " " + " ".join(header_terms) + " " + values[position], re.I)):
                                    reason = "table_unit_type_mismatch"
                                headers = [values[selected_column] for row, values in zip(rows, grid)
                                           if row.find("th", recursive=False) and selected_column < len(values)
                                           and values[selected_column].strip()]
                                if headers:
                                    bound_candidate = bound_candidate.model_copy(update={
                                        "column_label": " | ".join(dict.fromkeys(headers))})
                                evidence_context = " ".join((caption, section_label,
                                    *(" ".join(values) for row, values in zip(rows, grid) if row.find("th", recursive=False))))
                                excerpt_supported |= _excerpt_supported(candidate.evidence_excerpt,
                                    evidence_context + " " + " ".join(values), table=True)
                                matched_values.append((bound_candidate, bound_value))
                                row_terms = normalize_financial_term(values[0])
                                section_terms = normalize_financial_term(section_label)
                                for planned_requirement in requirements:
                                    target = planned_requirement["metric"]
                                    target_terms = normalize_financial_term(target)
                                    if (planned_requirement.get("kind", "fact") == "fact"
                                            and target_terms.issubset(row_terms | section_terms)
                                            and (target_terms.issubset(row_terms) or section_terms.issubset(target_terms))
                                            and is_metric_match(f"{section_label} {values[0]}", target)):
                                        bound_metrics.add(target)
                    table_row_present = candidate.row_label and any(
                        tr.find("td", recursive=False) and cells and _folded(cells[0]) == _folded(candidate.row_label)
                        for _, rows, grid, *_ in table_grids[candidate.source_id] for tr, cells in zip(rows, grid))
                    if table_row_present and not bindings:
                        reason = "table_row_not_bound"
                    elif bindings and not any(metric and value for metric, value in bindings):
                        reason = ("table_metric_label_mismatch" if any(value and not metric for metric, value in bindings)
                                  else "table_row_period_value_mismatch")
                    elif matched_values and not excerpt_supported:
                        reason = "evidence_excerpt_not_in_source"
                    elif len(bound_metrics) == 1:
                        candidate = candidate.model_copy(update={"metric": next(iter(bound_metrics))})
                    if matched_values and not reason:
                        if len({(value.base_value, value.value_type, value.currency) for _, value in matched_values}) != 1:
                            reason = "conflicting_table_units"
                        else:
                            scale_candidate, normalized = matched_values[0]
                            candidate = candidate.model_copy(update={"scale": scale_candidate.scale,
                                                                     "column_label": scale_candidate.column_label})
                    if requirement and requirement.get("value_type") and normalized.value_type != requirement["value_type"]:
                        reason = "requirement_source_or_type_mismatch"
            except ValueNormalizationError as error:
                reason = f"normalization_failed:{error}"
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

            valid.append(
                ValidatedFinancialFact(
                    fact_id=_fact_id(candidate, source, normalized),
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
                        *(("table_row_period_value",) if bindings else ()),
                        *(("table_metric_binding",) if len(bound_metrics) == 1 else ()),
                        *(("request_evidence_binding",) if requirement else ()),
                    ),
                    requirement_id=candidate.requirement_id,
                    value_qualifier=(qualifier.group(1) if (qualifier := re.search(
                        r"\b(over|exceeds?|exceeded|below|more than|less than|at least|at most|up to|approximately|roughly|about)\s+"
                        + re.escape(re.sub(r"^(?:over|exceeds?|exceeded|below|more than|less than|at least|at most|up to|approximately|roughly|about)\s+", "", candidate.raw_value, flags=re.I)),
                        candidate.evidence_excerpt, re.I)) else None),
                    evidence_context=evidence_context or next((_plain_text(paragraph) for paragraph in re.split(r"\n\s*\n", _source_text(source))
                        if _excerpt_supported(candidate.evidence_excerpt, paragraph)), None),
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
        requirement: dict | None = None,
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
        if _folded(label) not in folded and (not requirement or candidate.row_label):
            return "metric_or_row_not_in_source"

        period_text = candidate.column_label or candidate.period
        period_years = re.findall(r"(?:19|20)\d{2}", period_text)
        source_year = str(source.get("fiscal_year") or "")
        fiscal_years = {str(min((int(source_year) // 100 * 100 + int(short) + offset for offset in (-100, 0, 100)),
                               key=lambda year: abs(year - int(source_year))))
                        for short in re.findall(r"\bfy\s*(\d{2})\b", folded)} if re.fullmatch(r"\d{4}", source_year) else set()
        missing_years = [year for year in period_years if year not in folded and year not in fiscal_years]
        implicit_forecast = (
            requirement is not None and str(source.get("doc_type", "")).upper() == "TRANSCRIPT"
            and normalize_reported_period(candidate.period) == normalize_reported_period(requirement.get("reported_period") or "")
            and bool(re.search(r"\b(?:expect|guidance|outlook|should|will)\b", candidate.evidence_excerpt, re.I))
            and all(int(year) == int(source.get("fiscal_year", "0")) + 1 for year in missing_years)
            and any(quarter.upper() in candidate.evidence_excerpt.upper()
                    for quarter in re.findall(r"\bQ[1-4]\b", candidate.period, re.I))
        )
        if missing_years and not implicit_forecast and not all(
            year == str(source.get("fiscal_year", "")) for year in missing_years
        ):
            return "period_not_in_source"

        table_bound = candidate.row_label and any(
            _folded(cell.get_text(" ")) == _folded(candidate.row_label)
            for table in BeautifulSoup(text + str(source.get("evidence_text", "")), "html.parser").find_all("table")
            for row in table.find_all("tr") for cell in row.find_all("td", recursive=False)[:1])
        if not table_bound and not _excerpt_supported(candidate.evidence_excerpt, text):
            return "evidence_excerpt_not_in_source"
        if not table_bound:
            excerpt = re.sub(r"[,$€£¥]", "", _folded(candidate.evidence_excerpt))
            raw = re.sub(r"[,$€£¥]", "", _folded(candidate.raw_value))
            if not re.search(r"(?<![\d.])" + re.escape(raw) + r"(?!\d|\.\d)", excerpt):
                return "raw_value_not_in_excerpt"
            if requirement:
                terms = normalize_financial_term(candidate.evidence_excerpt)
                if not all(any(is_metric_match(term, actual) for actual in terms)
                           for term in normalize_financial_term(candidate.row_label or candidate.metric)):
                    return "excerpt_metric_mismatch"
        return None
