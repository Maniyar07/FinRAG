"""Planner-safe tool wrapper for deterministic financial calculations."""

from __future__ import annotations

from collections.abc import Mapping

from src.financial.calculator import (
    CalculationRequest,
    CalculationResult,
    FinancialCalculator,
)
from src.financial.models import ValidatedFinancialFact


class FinancialCalculatorTool:
    name = "financial_calculator"
    description = (
        "Calculate changes, ratios, CAGR, and sums using only validated financial "
        "fact IDs while retaining their source provenance."
    )

    def __init__(self, calculator: FinancialCalculator | None = None) -> None:
        self.calculator = calculator or FinancialCalculator()

    def execute(
        self,
        request: CalculationRequest,
        *,
        available_facts: Mapping[str, ValidatedFinancialFact],
    ) -> CalculationResult:
        if not isinstance(request, CalculationRequest):
            raise TypeError("request must be a CalculationRequest instance.")
        return self.calculator.calculate(request, available_facts)
