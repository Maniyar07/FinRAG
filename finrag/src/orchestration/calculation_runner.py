"""Bind validated facts and execute calculations from a multi-hop plan."""

from __future__ import annotations

from dataclasses import dataclass

from src.financial.calculator import CalculationRequest, CalculationResult
from src.financial.models import ValidatedFinancialFact, ValueType
from src.financial.value_normalizer import SCALE_MULTIPLIERS
from src.orchestration.fact_selection import (
    _ordered_calculation_facts,
    _select_fact,
)
from src.orchestration.models import CalculationReference, PlannedCalculation
from src.tools.financial_calculator import FinancialCalculatorTool


@dataclass(frozen=True)
class ExecutedCalculation:
    calculation_id: str
    label: str
    result: CalculationResult


@dataclass(frozen=True)
class CalculationRun:
    completed: tuple[ExecutedCalculation, ...]
    issues: tuple[str, ...]
    trace: tuple[dict, ...]


def calculation_as_fact(calculation: ExecutedCalculation) -> ValidatedFinancialFact:
    """Convert a verified result into a provenance-preserving calculator operand."""
    result = calculation.result
    value_type = {
        "currency": ValueType.CURRENCY,
        "percent": ValueType.PERCENT,
        "percentage_points": ValueType.PERCENT,
        "basis_points": ValueType.BASIS_POINTS,
        "ratio": ValueType.RATIO,
    }.get(result.result_unit, ValueType.NUMBER)
    scale = result.scale if value_type == ValueType.CURRENCY else None
    return ValidatedFinancialFact(
        fact_id=calculation.calculation_id,
        ticker="CALC",
        metric=calculation.label,
        period="calculated",
        raw_value=str(result.result),
        numeric_value=result.result,
        base_value=result.result * SCALE_MULTIPLIERS.get(scale or "ones"),
        value_type=value_type,
        normalized_unit=(result.currency or "currency")
        if value_type == ValueType.CURRENCY else result.result_unit,
        currency=result.currency,
        scale=scale,
        source_id=next(iter(result.source_ids), "S0"),
        evidence_excerpt=f"Application-verified calculation: {calculation.label}",
        row_label=calculation.label,
        column_label="calculated",
        validation_checks=("application_verified_calculation",),
    )


def run_calculations(
    planned_calculations: tuple[PlannedCalculation, ...],
    *,
    facts_by_requirement: dict[str, tuple[ValidatedFinancialFact, ...]],
    source_map: dict[str, dict],
    available_facts: dict[str, ValidatedFinancialFact],
    calculator: FinancialCalculatorTool,
) -> CalculationRun:
    completed: list[ExecutedCalculation] = []
    issues: list[str] = []
    trace: list[dict] = []

    for planned in planned_calculations:
        try:
            input_sources: list[str] = []
            selected_items = []
            for reference in planned.inputs:
                if isinstance(reference, CalculationReference):
                    prior = next(
                        (
                            item for item in completed
                            if item.calculation_id == reference.calculation_id
                        ),
                        None,
                    )
                    if prior is None:
                        raise ValueError("referenced calculation is unavailable")
                    fact = calculation_as_fact(prior)
                    available_facts[fact.fact_id] = fact
                    input_sources.extend(prior.result.source_ids)
                else:
                    fact = _select_fact(
                        reference, facts_by_requirement, source_map
                    )
                    input_sources.append(fact.source_id)
                selected_items.append(fact)
            selected = tuple(selected_items)
            selected = _ordered_calculation_facts(planned, selected)
            # A binary formula must never silently use one evidence cell as
            # both operands. This catches an underspecified plan or an
            # extraction mix-up before it can produce a plausible but false
            # result such as revenue / revenue = 1.
            if len(selected) > 1 and len({fact.fact_id for fact in selected}) != len(selected):
                raise ValueError("calculation operands resolved to the same validated fact")
            request = CalculationRequest(
                operation=planned.operation,
                input_fact_ids=tuple(fact.fact_id for fact in selected),
                periods=planned.periods,
            )
            result = calculator.execute(request, available_facts=available_facts)
            result = result.model_copy(
                update={"source_ids": tuple(dict.fromkeys(input_sources))}
            )
        except ValueError as error:
            issues.append(f"{planned.calculation_id}:calculation_unavailable")
            trace.append(
                {
                    "calculation_id": planned.calculation_id,
                    "status": "unavailable",
                    "reason": str(error),
                }
            )
            continue

        completed.append(
            ExecutedCalculation(
                calculation_id=planned.calculation_id,
                label=planned.label,
                result=result,
            )
        )
        trace.append(
            {
                "calculation_id": planned.calculation_id,
                "status": "completed",
                "operation": result.operation.value,
                "input_fact_ids": list(result.input_fact_ids),
                "source_ids": list(result.source_ids),
                "result": str(result.result),
                "result_unit": result.result_unit,
            }
        )

    return CalculationRun(tuple(completed), tuple(issues), tuple(trace))
