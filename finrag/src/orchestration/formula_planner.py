"""Compile declarative financial formulas into validated multi-hop plans."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from src.config import MULTIHOP_MAX_CALCULATIONS, MULTIHOP_MAX_SEARCHES
from src.financial.calculator import CalculationOperation
from src.financial.formula_registry import (
    FormulaDefinition,
    expression_metric_ids,
    load_formula_registry,
)
from src.orchestration.models import (
    CalculationReference,
    EvidenceGroup,
    EvidenceRequirement,
    EvidenceType,
    FactReference,
    MultiHopPlan,
    PlannedCalculation,
)
from src.orchestration.plan_builders import allowed_document_types
from src.schemas import Scope


NARRATIVE_TASK_RE = re.compile(
    r"\b(?:explain|why|risks?|summari[sz]e|drivers?|guidance|commentary)\b",
    re.IGNORECASE,
)


def _identifier(value: str, *, limit: int = 58) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return (cleaned or "formula")[:limit]


def _groups(scope: Scope) -> tuple[EvidenceGroup, ...]:
    return tuple(
        EvidenceGroup(ticker=ticker, fiscal_year=year)
        for ticker, year in scope.groups
    )


def _metric_batches(metric_ids: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """Keep precise one-metric searches, packing only catalog overflow."""
    if len(metric_ids) <= MULTIHOP_MAX_SEARCHES:
        return tuple((metric_id,) for metric_id in metric_ids)
    head = tuple((metric_id,) for metric_id in metric_ids[: MULTIHOP_MAX_SEARCHES - 1])
    return (*head, metric_ids[MULTIHOP_MAX_SEARCHES - 1 :])


def _requirements(
    formula: FormulaDefinition,
    metric_ids: tuple[str, ...],
    groups: tuple[EvidenceGroup, ...],
) -> tuple[tuple[EvidenceRequirement, ...], dict[str, str]]:
    registry = load_formula_registry()
    requirements: list[EvidenceRequirement] = []
    requirement_by_metric: dict[str, str] = {}
    for index, batch in enumerate(_metric_batches(metric_ids), start=1):
        requirement_id = (
            f"formula_{_identifier(batch[0], limit=48)}"
            if len(batch) == 1
            else f"formula_{_identifier(formula.formula_id, limit=43)}_inputs_{index}"
        )
        instructions = " ".join(
            registry.metrics[metric_id].search_instruction for metric_id in batch
        )
        requirements.append(
            EvidenceRequirement(
                requirement_id=requirement_id,
                question=instructions,
                evidence_type=EvidenceType.NUMERIC,
                document_type="10K",
                groups=groups,
            )
        )
        for metric_id in batch:
            requirement_by_metric[metric_id] = requirement_id
    return tuple(requirements), requirement_by_metric


def _formula_label(formula: FormulaDefinition) -> str:
    if not formula.assumption:
        return formula.name
    return f"{formula.name} — {formula.assumption}"


def _compile_expression_for_group(
    *,
    node: dict[str, Any],
    formula: FormulaDefinition,
    group: EvidenceGroup,
    requirement_by_metric: dict[str, str],
    calculations: list[PlannedCalculation],
    root: bool,
) -> FactReference | CalculationReference:
    registry = load_formula_registry()
    metric_id = node.get("metric")
    if metric_id is not None:
        metric = registry.metrics[str(metric_id)]
        return FactReference(
            requirement_id=requirement_by_metric[metric.metric_id],
            ticker=group.ticker,
            fiscal_year=group.fiscal_year,
            period=group.fiscal_year,
            metric_hint=metric.label,
        )

    inputs = tuple(
        _compile_expression_for_group(
            node=child,
            formula=formula,
            group=group,
            requirement_by_metric=requirement_by_metric,
            calculations=calculations,
            root=False,
        )
        for child in node["inputs"]
    )
    step = len(calculations) + 1
    suffix = "result" if root else f"step_{step}"
    calculation_id = _identifier(
        f"{group.ticker}_{group.fiscal_year}_{formula.formula_id}_{suffix}",
        limit=64,
    )
    label = (
        f"{group.ticker} {group.fiscal_year} {_formula_label(formula)}"
        if root
        else f"{group.ticker} {group.fiscal_year} {formula.name} intermediate step"
    )
    calculations.append(
        PlannedCalculation(
            calculation_id=calculation_id,
            label=label,
            operation=CalculationOperation(str(node["operation"])),
            inputs=inputs,
        )
    )
    return CalculationReference(calculation_id=calculation_id)


def _temporal_calculations(
    formula: FormulaDefinition,
    groups: Iterable[EvidenceGroup],
    requirement_id: str,
) -> tuple[PlannedCalculation, ...]:
    registry = load_formula_registry()
    metric = registry.metrics[str(formula.metric_id)]
    by_ticker: dict[str, list[EvidenceGroup]] = {}
    for group in groups:
        by_ticker.setdefault(group.ticker, []).append(group)
    calculations: list[PlannedCalculation] = []
    for ticker, ticker_groups in by_ticker.items():
        ordered = sorted(ticker_groups, key=lambda item: item.fiscal_year)
        if len(ordered) < 2:
            raise ValueError(
                f"{formula.name} requires at least two source fiscal years for {ticker}."
            )
        old, new = ordered[0], ordered[-1]
        operation = formula.temporal_operation or CalculationOperation.PERCENTAGE_CHANGE
        calculations.append(
            PlannedCalculation(
                calculation_id=_identifier(
                    f"{ticker}_{formula.formula_id}_{old.fiscal_year}_{new.fiscal_year}",
                    limit=64,
                ),
                label=(
                    f"{ticker} {formula.name} from {old.fiscal_year} "
                    f"to {new.fiscal_year}"
                ),
                operation=operation,
                inputs=(
                    FactReference(
                        requirement_id=requirement_id,
                        ticker=ticker,
                        fiscal_year=old.fiscal_year,
                        period=old.fiscal_year,
                        metric_hint=metric.label,
                    ),
                    FactReference(
                        requirement_id=requirement_id,
                        ticker=ticker,
                        fiscal_year=new.fiscal_year,
                        period=new.fiscal_year,
                        metric_hint=metric.label,
                    ),
                ),
                periods=(
                    int(new.fiscal_year) - int(old.fiscal_year)
                    if operation == CalculationOperation.CAGR
                    else None
                ),
            )
        )
    return tuple(calculations)


def formula_plan(*, question: str, permitted_scope: Scope) -> MultiHopPlan:
    """Match a catalog formula and compile it without formula-specific code."""
    formula = load_formula_registry().match(question)
    if formula is None:
        raise ValueError("Question does not match a registered financial formula.")
    if NARRATIVE_TASK_RE.search(question):
        raise ValueError("Compound narrative questions require the general planner.")
    allowed_types = allowed_document_types(permitted_scope)
    if "10K" not in allowed_types or set(permitted_scope.required_doc_types) - {"10K"}:
        raise ValueError("Registered financial formulas require a 10-K scope.")

    groups = _groups(permitted_scope)
    metric_ids = (
        expression_metric_ids(formula.expression or {})
        if formula.mode == "per_group"
        else (str(formula.metric_id),)
    )
    requirements, requirement_by_metric = _requirements(
        formula, metric_ids, groups
    )
    if formula.mode == "temporal":
        calculations = _temporal_calculations(
            formula,
            groups,
            requirement_by_metric[metric_ids[0]],
        )
    else:
        built: list[PlannedCalculation] = []
        for group in groups:
            _compile_expression_for_group(
                node=formula.expression or {},
                formula=formula,
                group=group,
                requirement_by_metric=requirement_by_metric,
                calculations=built,
                root=True,
            )
        calculations = tuple(built)
    if len(calculations) > MULTIHOP_MAX_CALCULATIONS:
        raise ValueError(
            f"{formula.name} needs {len(calculations)} calculation steps, exceeding "
            f"the configured limit of {MULTIHOP_MAX_CALCULATIONS}."
        )
    return MultiHopPlan(
        original_question=question,
        requirements=requirements,
        calculations=calculations,
    )

