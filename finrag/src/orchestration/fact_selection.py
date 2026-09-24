"""Select validated financial facts for planned calculations."""

from __future__ import annotations

import re

from src.financial.calculator import CalculationOperation
from src.financial.models import ValidatedFinancialFact, ValueType
from src.orchestration.models import FactReference


WORD_RE = re.compile(r"[a-z0-9]+")
YEAR_RE = re.compile(r"(?:19|20)\d{2}")


def _terms(value: str) -> set[str]:
    terms = set(WORD_RE.findall(value.casefold()))
    normalized = {
        term[:-1] if term.endswith("s") and len(term) > 4 else term
        for term in terms
    }
    aliases = {
        "operating": "operation",
    }
    normalized = {aliases.get(term, term) for term in normalized}
    return normalized - {
        "and",
        "amount",
        "expense",
        "financial",
        "from",
        "reported",
        "total",
        "value",
    }


MATERIAL_METRIC_QUALIFIERS = frozenset(
    {
        "adjusted",
        "automotive",
        "combined",
        "restricted",
        "segment",
        "short",
    }
)


def _fact_matches_metric(fact: ValidatedFinancialFact, metric_hint: str) -> bool:
    """Match common accounting aliases without accepting a materially broader row."""
    expected = _terms(metric_hint)
    # The extracted row label is source evidence. The model-authored metric name
    # is only a fallback and must not override a contradictory table row.
    actual = _terms(fact.row_label or fact.metric)
    if "rate" in expected and fact.value_type != ValueType.PERCENT:
        return False
    tax_rate_alias = (
        expected == {"effective", "tax", "rate"}
        and actual == {"effective", "rate"}
    )
    if not expected or (not tax_rate_alias and not expected.issubset(actual)):
        return False
    unexpected_qualifiers = (actual & MATERIAL_METRIC_QUALIFIERS) - expected
    return not unexpected_qualifiers


def _fact_matches_period(
    fact: ValidatedFinancialFact,
    requested_period: str,
    *,
    allow_unknown: bool,
) -> bool:
    actual = f"{fact.period} {fact.column_label or ''}"
    if requested_period.casefold() in actual.casefold():
        return True
    requested_years = YEAR_RE.findall(requested_period)
    actual_years = YEAR_RE.findall(actual)
    if requested_years:
        if all(year in actual_years for year in requested_years):
            return True
        return allow_unknown and not actual_years
    return False


def _numeric_search_instruction(question: str) -> str:
    lowered = question.casefold()
    base = (
        "Find the exact reported table row, year column, units, and scale."
    )
    if "operating income" in lowered:
        return (
            f'{base} Treat "income from operations" as the same requested row. '
            "Prefer the consolidated income statement or operations table."
        )
    if "cash and cash equivalents" in lowered:
        return (
            f"{base} Use the exact Cash and cash equivalents row from the "
            "Consolidated Balance Sheets under Assets / Current assets. Exclude "
            "fair-value footnotes, restricted cash, combined cash and restricted-cash "
            "reconciliations, and short-term investments."
        )
    if "effective tax rate" in lowered:
        return (
            f"{base} Prefer the exact effective-rate row in the income-tax rate "
            "reconciliation table containing the Federal statutory rate and Effect of "
            "rows, not a rounded narrative summary."
        )
    if any(
        term in lowered
        for term in (
            "revenue",
            "income",
            "loss",
            "expense",
            "profit",
        )
    ):
        return f"{base} Prefer the consolidated income statement or operations table."
    return base


def _with_full_evidence(sources: list[dict]) -> list[dict]:
    """Use the stored parent only for a focused numeric recovery attempt."""
    expanded: list[dict] = []
    for source in sources:
        item = dict(source)
        full_text = str(source.get("full_evidence_text") or "").strip()
        if full_text:
            item["evidence_text"] = full_text
        expanded.append(item)
    return expanded


def _focused_numeric_sources(
    sources: list[dict],
    question: str,
    metric_hints: tuple[str, ...] = (),
) -> list[dict]:
    """Prefer parents that contain the requested primary statement/table."""
    lowered = question.casefold()
    if any(
        hint.casefold() in {"revenue", "total revenue", "total revenues", "operating income", "research and development expense"}
        for hint in metric_hints
    ):
        statements = [
            source for source in sources
            if "financial statements and supplementary data"
            in str(source.get("section") or "").casefold()
            and re.search(
                r"(?:income statements|statements of operations)",
                str(source.get("full_evidence_text") or source.get("evidence_text") or ""),
                re.IGNORECASE,
            )
        ]
        if statements:
            return _with_full_evidence(statements)
    if "effective tax rate" in lowered:
        exact_reconciliations = [
            source
            for source in sources
            if "<td>effective rate</td>"
            in str(
                source.get("full_evidence_text")
                or source.get("evidence_text")
                or ""
            ).casefold()
            and "federal statutory rate"
            in str(
                source.get("full_evidence_text")
                or source.get("evidence_text")
                or ""
            ).casefold()
        ]
        if exact_reconciliations:
            return _with_full_evidence(exact_reconciliations)
    preferred: list[dict] = []
    exact_rows = []
    for source in sources:
        full_text = str(
            source.get("full_evidence_text") or source.get("evidence_text") or ""
        ).casefold()
        for hint in metric_hints:
            row_label = re.sub(r"\s+expenses?$", "", hint, flags=re.IGNORECASE)
            if re.search(
                rf"<td[^>]*>\s*{re.escape(row_label.casefold())}\s*</td>",
                full_text,
            ):
                exact_rows.append(source)
                break
        if "effective tax rate" in lowered:
            if "effective rate" in full_text and (
                "federal statutory rate" in full_text
                or "tax at statutory federal rate" in full_text
            ):
                preferred.append(source)
        elif "cash and cash equivalents" in lowered:
            if (
                "consolidated balance sheets" in full_text
                and "cash and cash equivalents" in full_text
            ):
                preferred.append(source)
    return _with_full_evidence(preferred or exact_rows or sources)


def _select_fact(
    reference: FactReference,
    facts_by_requirement: dict[str, tuple[ValidatedFinancialFact, ...]],
    source_map: dict[str, dict],
) -> ValidatedFinancialFact:
    candidates = [
        fact
        for fact in facts_by_requirement.get(reference.requirement_id, ())
        if fact.ticker == reference.ticker
        and str(source_map.get(fact.source_id, {}).get("fiscal_year"))
        == reference.fiscal_year
    ]
    if reference.metric_hint:
        candidates = [
            fact
            for fact in candidates
            if _fact_matches_metric(fact, reference.metric_hint)
        ]

    requested_period = reference.period or reference.fiscal_year
    matching_period = [
        fact
        for fact in candidates
        if _fact_matches_period(fact, requested_period, allow_unknown=False)
    ]
    if matching_period:
        candidates = matching_period
    else:
        candidates = [
            fact
            for fact in candidates
            if _fact_matches_period(fact, requested_period, allow_unknown=True)
        ]

    exact_rows = [
        fact for fact in candidates if "exact_table_row" in fact.validation_checks
    ]
    if exact_rows:
        candidates = exact_rows

    # A consolidated statement outranks segment disclosures for company-wide
    # metrics. Keep conflicting figures ambiguous when no primary row exists.
    primary: list[ValidatedFinancialFact] = []
    if reference.metric_hint and reference.metric_hint.casefold() in {
        "revenue", "total revenue", "total revenues", "operating income",
        "research and development expense",
    }:
        primary = [
            fact for fact in candidates
            if "financial statements and supplementary data"
            in str(source_map.get(fact.source_id, {}).get("section") or "").casefold()
            and re.search(
                r"(?:income statements|statements of operations)",
                str(source_map.get(fact.source_id, {}).get("full_evidence_text")
                    or source_map.get(fact.source_id, {}).get("evidence_text") or ""),
                re.IGNORECASE,
            )
        ]
        if primary:
            candidates = primary
    monetary = [fact for fact in candidates if fact.value_type == ValueType.CURRENCY]
    if primary and monetary and len({fact.numeric_value for fact in monetary}) == 1:
        candidates = monetary

    unique = {fact.fact_id: fact for fact in candidates}
    if len(unique) > 1:
        values = {fact.numeric_value for fact in unique.values()}
        if len(values) == 1:
            return next(iter(unique.values()))
        highest_precision = max(
            max(0, -fact.numeric_value.as_tuple().exponent)
            for fact in unique.values()
        )
        precise = [
            fact
            for fact in unique.values()
            if max(0, -fact.numeric_value.as_tuple().exponent) == highest_precision
        ]
        if len(precise) == 1:
            return precise[0]
    if len(unique) != 1:
        raise ValueError(
            f"Fact reference {reference.requirement_id}/{reference.ticker}/"
            f"{reference.fiscal_year}/{requested_period} matched {len(unique)} "
            "validated facts."
        )
    return next(iter(unique.values()))


TEMPORAL_CHANGE_OPERATIONS = frozenset(
    {
        CalculationOperation.ABSOLUTE_CHANGE,
        CalculationOperation.PERCENTAGE_CHANGE,
        CalculationOperation.PERCENTAGE_POINT_CHANGE,
        CalculationOperation.BASIS_POINT_CHANGE,
        CalculationOperation.CAGR,
    }
)


def _ordered_calculation_facts(
    calculation: object,
    selected: tuple[ValidatedFinancialFact, ...],
) -> tuple[ValidatedFinancialFact, ...]:
    """Enforce old-to-new order for temporal calculations.

    Planner output remains useful for binding facts, but chronological arithmetic
    must not depend on the order produced by a language model.
    """
    operation = getattr(calculation, "operation", None)
    inputs = tuple(getattr(calculation, "inputs", ()))
    if operation not in TEMPORAL_CHANGE_OPERATIONS or len(selected) != 2:
        return selected
    if len(inputs) != 2 or inputs[0].ticker != inputs[1].ticker:
        return selected

    def time_key(item: tuple[object, ValidatedFinancialFact]) -> tuple[int, str]:
        reference, fact = item
        text = " ".join(
            str(value or "")
            for value in (
                getattr(reference, "period", None),
                getattr(reference, "fiscal_year", None),
                fact.column_label,
                fact.period,
            )
        )
        years = YEAR_RE.findall(text)
        year = int(years[0]) if years else 0
        return year, text.casefold()

    ordered = sorted(zip(inputs, selected), key=time_key)
    return tuple(fact for _, fact in ordered)
