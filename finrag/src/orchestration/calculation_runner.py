"""Bind validated facts and execute calculations from a multi-hop plan."""

from __future__ import annotations

from dataclasses import dataclass

from src.financial.calculator import CalculationRequest, CalculationResult
from src.financial.models import ValidatedFinancialFact
from src.orchestration.fact_selection import (
    _ordered_calculation_facts,
    _select_fact,
)
from src.orchestration.models import PlannedCalculation
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
            selected = tuple(
                _select_fact(reference, facts_by_requirement, source_map)
                for reference in planned.inputs
            )
            selected = _ordered_calculation_facts(planned, selected)
            request = CalculationRequest(
                operation=planned.operation,
                input_fact_ids=tuple(fact.fact_id for fact in selected),
                periods=planned.periods,
            )
            result = calculator.execute(request, available_facts=available_facts)
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
