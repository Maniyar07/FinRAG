"""Deterministic plan builders for common financial research questions."""

from __future__ import annotations

import re

from src.financial.calculator import CalculationOperation
from src.orchestration.models import (
    EvidenceGroup,
    EvidenceRequirement,
    EvidenceType,
    FactReference,
    MultiHopPlan,
    PlannedCalculation,
)
from src.schemas import Scope


YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
SUPPORTED_CHANGE_METRICS = (
    ("cash and cash equivalents", "cash_and_cash_equivalents"),
    ("net income", "net_income"),
    ("total assets", "total_assets"),
    ("operating income", "operating_income"),
    ("revenue", "revenue"),
)


def _change_operations(question: str) -> tuple[CalculationOperation, ...]:
    """Return every explicitly requested temporal calculation in stable order."""
    lowered = question.casefold()
    if "percentage-point" in lowered or "percentage point" in lowered:
        return (CalculationOperation.PERCENTAGE_POINT_CHANGE,)
    if "basis point" in lowered or "bps" in lowered:
        return (CalculationOperation.BASIS_POINT_CHANGE,)
    if "cagr" in lowered:
        return (CalculationOperation.CAGR,)
    if "ratio" in lowered:
        return (CalculationOperation.RATIO,)
    operations = tuple(
        operation
        for requested, operation in (
            ("absolute change", CalculationOperation.ABSOLUTE_CHANGE),
            ("percentage change", CalculationOperation.PERCENTAGE_CHANGE),
        )
        if requested in lowered
    )
    if operations:
        return operations
    if re.search(r"\bchanges?\b", lowered) and len(YEAR_RE.findall(lowered)) >= 2:
        return (CalculationOperation.ABSOLUTE_CHANGE,)
    raise ValueError("Deterministic comparison fallback found no calculation.")


def _focused_narrative_task(question: str, metrics: list[str]) -> str:
    """Keep a qualitative search focused on its topic, not the full calculation."""
    parts = re.split(
        r"\bthen\s+calculate\b", question, maxsplit=1, flags=re.IGNORECASE
    )
    before_calculation = parts[0].strip(" ,.;")
    if len(parts) == 2 and re.search(
        r"\b(?:explain|summarize|discuss|commentary|reasons?|drivers?|risks?)\b",
        before_calculation,
        re.IGNORECASE,
    ):
        narrative = before_calculation
    else:
        match = re.search(
            r"\b(?:explain|summarize|discuss)\b.+$",
            question,
            re.IGNORECASE,
        )
        narrative = match.group(0).strip(" ,.;") if match else ""
    topic_metrics = [
        metric for metric in metrics if metric.casefold() in narrative.casefold()
    ]
    if not topic_metrics and not re.search(r"\brisks?\b", narrative, re.IGNORECASE):
        topic_metrics = metrics
    topic = " and ".join(dict.fromkeys(topic_metrics))
    if not narrative:
        narrative = "summarize relevant management commentary"
    return f"{topic}: {narrative}" if topic else narrative


def allowed_document_types(scope: Scope) -> frozenset[str]:
    if scope.required_doc_types:
        return frozenset(scope.required_doc_types)
    if scope.doc_type:
        return frozenset((scope.doc_type,))
    return frozenset(("10K", "TRANSCRIPT"))


def _mixed_source_change_fallback(
    *,
    question: str,
    permitted_scope: Scope,
) -> MultiHopPlan:
    """Plan transcript commentary plus annual filing changes deterministically."""
    lowered = question.casefold().replace("-", " ")
    metrics = [
        (label, identifier)
        for label, identifier in SUPPORTED_CHANGE_METRICS
        if re.search(rf"\b{re.escape(label)}\b", lowered)
    ]
    if not metrics or len(permitted_scope.years) < 2:
        raise ValueError("Mixed-source fallback requires metrics and two years.")
    operations = _change_operations(question)
    all_groups = tuple(
        EvidenceGroup(ticker=ticker, fiscal_year=year)
        for ticker, year in permitted_scope.groups
    )
    latest_year = max(permitted_scope.years)
    transcript_groups = tuple(
        EvidenceGroup(ticker=ticker, fiscal_year=year)
        for ticker, year in permitted_scope.groups
        if year == latest_year
    )
    narrative_task = _focused_narrative_task(
        question, [label for label, _ in metrics]
    )
    requirements: list[EvidenceRequirement] = [
        EvidenceRequirement(
            requirement_id="management_commentary",
            question=narrative_task,
            evidence_type=EvidenceType.NARRATIVE,
            document_type="TRANSCRIPT",
            groups=transcript_groups,
        )
    ]
    for label, identifier in metrics:
        requirements.append(
            EvidenceRequirement(
                requirement_id=f"reported_{identifier}",
                question=(
                    f"Find the exact reported {label} table row for every requested "
                    "company and fiscal year, preserving period, units, and scale."
                ),
                evidence_type=EvidenceType.NUMERIC,
                document_type="10K",
                groups=all_groups,
            )
        )
    calculations: list[PlannedCalculation] = []
    for ticker in dict.fromkeys(group.ticker for group in all_groups):
        ticker_groups = sorted(
            (group for group in all_groups if group.ticker == ticker),
            key=lambda group: group.fiscal_year,
        )
        if len(ticker_groups) < 2:
            raise ValueError("Mixed-source changes require two periods per company.")
        old_group, new_group = ticker_groups[0], ticker_groups[-1]
        for label, identifier in metrics:
            for operation in operations:
                calculations.append(
                    PlannedCalculation(
                        calculation_id=(
                            f"{ticker.casefold()}_{identifier}_{operation.value}"
                        ),
                        label=(
                            f"{ticker} {label} {operation.value.replace('_', ' ')} "
                            f"from {old_group.fiscal_year} to {new_group.fiscal_year}"
                        ),
                        operation=operation,
                        inputs=(
                            FactReference(
                                requirement_id=f"reported_{identifier}",
                                ticker=ticker,
                                fiscal_year=old_group.fiscal_year,
                                metric_hint=label,
                            ),
                            FactReference(
                                requirement_id=f"reported_{identifier}",
                                ticker=ticker,
                                fiscal_year=new_group.fiscal_year,
                                metric_hint=label,
                            ),
                        ),
                        periods=(
                            int(new_group.fiscal_year) - int(old_group.fiscal_year)
                            if operation == CalculationOperation.CAGR else None
                        ),
                    )
                )
    return MultiHopPlan(
        original_question=question,
        requirements=tuple(requirements),
        calculations=tuple(calculations),
    )

def _margin_fallback(
    *,
    question: str,
    permitted_scope: Scope,
) -> MultiHopPlan:
    """Plan net profit margin as net income divided by revenue."""
    allowed_types = allowed_document_types(permitted_scope)
    if "10K" not in allowed_types or set(permitted_scope.required_doc_types) - {"10K"}:
        raise ValueError("Margin fallback requires an annual filing scope.")
    groups = tuple(
        EvidenceGroup(ticker=ticker, fiscal_year=year)
        for ticker, year in permitted_scope.groups
    )
    requirements: list[EvidenceRequirement] = [
        EvidenceRequirement(
            requirement_id="reported_revenue",
            question=(
                "Find the exact total revenue table row for every requested company "
                "and fiscal year, preserving period, currency, units, and scale."
            ),
            evidence_type=EvidenceType.NUMERIC,
            document_type="10K",
            groups=groups,
        ),
        EvidenceRequirement(
            requirement_id="reported_net_income",
            question=(
                "Find the exact net income table row for every requested company "
                "and fiscal year, preserving period, currency, units, and scale."
            ),
            evidence_type=EvidenceType.NUMERIC,
            document_type="10K",
            groups=groups,
        ),
    ]
    if re.search(r"\brisks?\b", question, re.IGNORECASE):
        requirements.append(
            EvidenceRequirement(
                requirement_id="risk_factor",
                question="Summarize one major risk factor for each requested company.",
                evidence_type=EvidenceType.NARRATIVE,
                document_type="10K",
                groups=groups,
            )
        )
    calculations = tuple(
        PlannedCalculation(
            calculation_id=f"{ticker.casefold()}_{year}_net_profit_margin",
            label=f"{ticker} {year} net profit margin",
            operation=CalculationOperation.RATIO,
            inputs=(
                FactReference(
                    requirement_id="reported_net_income",
                    ticker=ticker,
                    fiscal_year=year,
                    metric_hint="net income",
                ),
                FactReference(
                    requirement_id="reported_revenue",
                    ticker=ticker,
                    fiscal_year=year,
                    metric_hint="total revenue",
                ),
            ),
        )
        for ticker, year in permitted_scope.groups
    )
    return MultiHopPlan(
        original_question=question,
        requirements=tuple(requirements),
        calculations=calculations,
    )

def _comparison_fallback(
    *,
    question: str,
    permitted_scope: Scope,
) -> MultiHopPlan:
    """Build the common metric/change/explanation plan without model judgment."""
    search_question = question.replace("\u2019", "'")
    calculated_metric_match = re.search(
        r"\bcalculate\s+(?:each\s+(?:company|ticker)(?:'s|’s)?\s+)?"
        r"(?P<metric>.+?)\s+(?:(?:percentage|absolute)\s+)?change\s+from\b",
        search_question,
        re.IGNORECASE,
    )
    if calculated_metric_match:
        calculated_metric = calculated_metric_match.group("metric").casefold()
        if calculated_metric in {"a", "an", "the"} or re.search(
            r"\b(?:absolute|percentage)\s+change\b", calculated_metric
        ):
            calculated_metric_match = None
    metric_match = calculated_metric_match or re.search(
        r"\bcompare\s+(.+?)\s+for\s+(?=(?:19|20)\d{2})",
        search_question,
        re.IGNORECASE,
    )
    if metric_match is None:
        raise ValueError("Deterministic comparison fallback could not identify a metric.")
    metric = (
        metric_match.group("metric")
        if calculated_metric_match is not None
        else metric_match.group(1)
    )
    for value in (*permitted_scope.tickers, "Microsoft", "Tesla"):
        metric = re.sub(
            rf"\b{re.escape(value)}(?:['\u2019]s)?\b",
            " ",
            metric,
            flags=re.IGNORECASE,
        )
    metric = re.sub(
        r"^\s*(?:(?:and|versus|vs\.?)\s+)+",
        "",
        " ".join(metric.split()),
        flags=re.IGNORECASE,
    ).strip(" ,.-")
    if len(metric) < 3:
        raise ValueError("Deterministic comparison metric is ambiguous.")

    allowed_types = allowed_document_types(permitted_scope)
    numeric_type = "10K" if "10K" in allowed_types else next(iter(allowed_types))
    all_groups = tuple(
        EvidenceGroup(ticker=ticker, fiscal_year=year)
        for ticker, year in permitted_scope.groups
    )
    metric_id = re.sub(r"[^a-z0-9]+", "_", metric.casefold()).strip("_")
    metric_id = (metric_id or "financial_metric")[:40]
    numeric_id = f"reported_{metric_id}"
    requirements: list[EvidenceRequirement] = [
        EvidenceRequirement(
            requirement_id=numeric_id,
            question=(
                f"Find the exact reported {metric} table value for every requested "
                "company and source fiscal year, preserving row, period, units, and scale."
            ),
            evidence_type=EvidenceType.NUMERIC,
            document_type=numeric_type,
            groups=all_groups,
        )
    ]

    narrative_match = re.search(
        r"\b(?:explain|summarize)\s+(.+)$",
        question,
        re.IGNORECASE,
    )
    narrative_parts_before_calculation = re.split(
        r"\bthen\s+calculate\b", question, maxsplit=1, flags=re.IGNORECASE
    )
    narrative_prefix = (
        narrative_parts_before_calculation[0].strip(" ,.;")
        if len(narrative_parts_before_calculation) == 2
        else ""
    )
    if narrative_prefix and not re.search(
        r"\b(?:risks?|explain|summarize|discuss)\b",
        narrative_prefix,
        re.IGNORECASE,
    ):
        narrative_prefix = ""
    required_types = set(permitted_scope.required_doc_types)
    needs_narrative = (
        bool(narrative_prefix)
        or narrative_match is not None
        or bool(required_types - {numeric_type})
    )
    if needs_narrative:
        narrative_text = _focused_narrative_task(question, [metric])
        narrative_type = (
            "TRANSCRIPT"
            if "TRANSCRIPT" in required_types
            or "transcript" in narrative_text.casefold()
            else numeric_type
        )
        mentioned_years = set(YEAR_RE.findall(narrative_text))
        latest_year = max(year for _, year in permitted_scope.groups)
        narrative_groups = tuple(
            EvidenceGroup(ticker=ticker, fiscal_year=year)
            for ticker, year in permitted_scope.groups
            if (
                year in mentioned_years
                or (
                    not mentioned_years
                    and year == latest_year
                )
            )
        )
        requirements.append(
            EvidenceRequirement(
                requirement_id="management_explanation",
                question=narrative_text,
                evidence_type=EvidenceType.NARRATIVE,
                document_type=narrative_type,
                groups=narrative_groups,
            )
        )

    operations = _change_operations(question)

    calculations: list[PlannedCalculation] = []
    for ticker in dict.fromkeys(group.ticker for group in all_groups):
        ticker_groups = sorted(
            (group for group in all_groups if group.ticker == ticker),
            key=lambda group: group.fiscal_year,
        )
        if len(ticker_groups) < 2:
            raise ValueError(
                "Deterministic comparison fallback needs two source periods per company."
            )
        old_group, new_group = ticker_groups[0], ticker_groups[-1]
        for operation in operations:
            calculations.append(
                PlannedCalculation(
                    calculation_id=(
                        f"{ticker.casefold()}_{metric_id}_{operation.value}"
                    ),
                    label=(
                        f"{ticker} {metric} {operation.value.replace('_', ' ')} "
                        f"from {old_group.fiscal_year} to {new_group.fiscal_year}"
                    ),
                    operation=operation,
                    inputs=(
                        FactReference(
                            requirement_id=numeric_id,
                            ticker=ticker,
                            fiscal_year=old_group.fiscal_year,
                            metric_hint=metric,
                        ),
                        FactReference(
                            requirement_id=numeric_id,
                            ticker=ticker,
                            fiscal_year=new_group.fiscal_year,
                            metric_hint=metric,
                        ),
                    ),
                    periods=(
                        int(new_group.fiscal_year) - int(old_group.fiscal_year)
                        if operation == CalculationOperation.CAGR else None
                    ),
                )
            )
    return MultiHopPlan(
        original_question=question,
        requirements=tuple(requirements),
        calculations=tuple(calculations),
    )
