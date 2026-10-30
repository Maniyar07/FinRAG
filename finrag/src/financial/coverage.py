"""Match planned requirements to evidence without depending on an agent backend."""

from __future__ import annotations

import re
import calendar
from datetime import datetime
from difflib import SequenceMatcher
from collections.abc import Mapping, Sequence

from src.financial.metric_terms import has_financial_metric_conflict, is_metric_match, normalize_financial_term
from src.financial.models import ValidatedFinancialFact


def normalize_reported_period(value: str, question: str = "") -> str:
    """Preserve explicit dates; never infer a company's fiscal quarter from a month."""
    if question and re.search(r"\bFY\s*\d{4}|\bfiscal year\b|\byear ended\b", value, re.I):
        return value
    years = set(re.findall(r"(?:19|20)\d{2}", value))
    months = "|".join((*calendar.month_name[1:], *calendar.month_abbr[1:]))
    dates = set()
    for match in re.finditer(rf"\b(?:{months})\.?\s+\d{{1,2}},?\s+(?:19|20)\d{{2}}\b", question or value, re.I):
        text = match.group().replace(",", "").replace(".", "")
        for fmt in ("%B %d %Y", "%b %d %Y"):
            try:
                date = datetime.strptime(text, fmt).date()
                if not years or str(date.year) in years:
                    dates.add(date.isoformat())
                break
            except ValueError:
                continue
    return next(iter(dates)) if len(dates) == 1 else value


def source_matches_requirement(source: Mapping, requirement: Mapping) -> bool:
    return (
        str(source.get("ticker", "")).upper() == requirement["ticker"]
        and str(source.get("fiscal_year", "")) == requirement["source_year"]
        and str(source.get("doc_type", "")).upper().replace("-", "")
        == requirement["doc_type"]
    )


def fact_matches_requirement(
    fact: ValidatedFinancialFact, requirement: Mapping, source: Mapping,
) -> bool:
    """Require metric, source identity, reported period, and qualifiers together."""
    if fact.ticker != requirement["ticker"] or not source_matches_requirement(source, requirement):
        return False
    if requirement.get("value_type") and fact.value_type != requirement["value_type"]:
        return False
    bound = fact.requirement_id == requirement["requirement_id"]
    if fact.requirement_id and not bound:
        return False
    if has_financial_metric_conflict(" ".join(filter(None, [fact.row_label or fact.metric, fact.column_label])), requirement["metric"]):
        return False
    metric_bound = bound and "request_evidence_binding" in fact.validation_checks
    if not metric_bound and not is_metric_match(fact.metric, requirement["metric"]):
        return False
    # Character similarity alone cannot distinguish a long metric's amount from
    # its growth rate. Every requested semantic term must have extracted support.
    requested_terms = normalize_financial_term(requirement["metric"])
    fact_terms = normalize_financial_term(f"{fact.metric} {fact.row_label or ''}")
    if not metric_bound and not requested_terms.issubset(fact_terms):
        return False
    requested = normalize_reported_period(str(requirement.get("reported_period") or "")).upper()
    actual = normalize_reported_period(fact.period).upper() + " " + (fact.column_label or "").upper()
    requested_dates = set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", requested))
    if requested_dates and not requested_dates.issubset(set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", actual))):
        return False
    years = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", requested))
    if not years.issubset(set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", actual))):
        return False
    quarters = set(re.findall(r"\bQ[1-4]\b", requested))
    actual_quarters = set(re.findall(r"\bQ[1-4]\b", actual))
    if quarters and not quarters.issubset(actual_quarters):
        return False
    if re.search(r"\bFY|\bannual\b|\bfiscal year\b", requested, re.I) and actual_quarters:
        return False
    labels = " ".join((fact.row_label or fact.metric, fact.column_label or "",
                       fact.accounting_basis or "", fact.evidence_context or ""
                       if "table_row_period_value" in fact.validation_checks else fact.evidence_context or fact.evidence_excerpt))
    terms = normalize_financial_term(labels)
    # Single-letter labels are meaningful qualifiers even when they are common
    # stopwords in prose (for example a region or share class).
    terms.update(re.findall(r"\b[a-z]\b", labels.casefold()))
    return all(all(any(SequenceMatcher(None, requested, actual).ratio() >= 0.8 for actual in terms)
                   for requested in normalize_financial_term(value) | set(re.findall(r"\b[a-z]\b", value.casefold())))
               for value in requirement.get("qualifiers", ()))


def fact_coverage(
    requirements: Sequence[Mapping], facts: Sequence[ValidatedFinancialFact],
    sources: Sequence[Mapping],
) -> dict:
    """Narrative and statement requirements need their own evidence contracts."""
    by_source = {str(source.get("id")): source for source in sources}
    matches = {
        item["requirement_id"]: [
            fact.fact_id for fact in facts
            if fact_matches_requirement(fact, item, by_source.get(fact.source_id, {}))
        ]
        for item in requirements if item.get("kind", "fact") == "fact"
    }
    return {
        "fact_ids_by_requirement": matches,
        "covered_requirement_ids": [key for key, values in matches.items() if values],
        "missing_requirement_ids": [key for key, values in matches.items() if not values],
    }
