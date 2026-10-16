"""Deterministic Decimal-based calculations over validated financial facts."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import (
    Decimal,
    DivisionByZero,
    InvalidOperation,
    ROUND_HALF_UP,
    localcontext,
)
from enum import Enum

from pydantic import Field, field_validator, model_validator

from src.financial.models import FinancialModel, ValidatedFinancialFact, ValueType
from src.financial.metric_terms import is_metric_match
from src.observability import trace_component
from src.financial.value_normalizer import SCALE_MULTIPLIERS


class CalculationOperation(str, Enum):
    ABSOLUTE_CHANGE = "absolute_change"
    PERCENTAGE_CHANGE = "percentage_change"
    PERCENTAGE_POINT_CHANGE = "percentage_point_change"
    BASIS_POINT_CHANGE = "basis_point_change"
    DIFFERENCE = "difference"
    RATIO = "ratio"
    PERCENTAGE_OF = "percentage_of"
    CAGR = "cagr"
    SUM = "sum"
    PRODUCT = "product"
    ABSOLUTE_VALUE = "absolute_value"
    PERCENT_COMPLEMENT = "percent_complement"


class CalculationRequest(FinancialModel):
    """Fact order matters: old/new for changes, numerator/denominator for ratio."""

    operation: CalculationOperation
    input_fact_ids: tuple[str, ...] = Field(min_length=1, max_length=20)
    periods: Decimal | None = Field(default=None, gt=0)
    decimal_places: int | None = Field(default=None, ge=0, le=12)

    @field_validator("input_fact_ids")
    @classmethod
    def unique_fact_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(values)) != len(values):
            raise ValueError("input_fact_ids must be unique")
        return values

    @model_validator(mode="after")
    def validate_operation_shape(self) -> "CalculationRequest":
        binary = {
            CalculationOperation.ABSOLUTE_CHANGE,
            CalculationOperation.PERCENTAGE_CHANGE,
            CalculationOperation.PERCENTAGE_POINT_CHANGE,
            CalculationOperation.BASIS_POINT_CHANGE,
            CalculationOperation.DIFFERENCE,
            CalculationOperation.RATIO,
            CalculationOperation.PERCENTAGE_OF,
            CalculationOperation.CAGR,
            CalculationOperation.PRODUCT,
        }
        if self.operation in binary and len(self.input_fact_ids) != 2:
            raise ValueError(f"{self.operation.value} requires exactly two facts")
        if self.operation == CalculationOperation.SUM and not self.input_fact_ids:
            raise ValueError("sum requires at least one fact")
        if self.operation in {
            CalculationOperation.ABSOLUTE_VALUE,
            CalculationOperation.PERCENT_COMPLEMENT,
        } and len(self.input_fact_ids) != 1:
            raise ValueError(f"{self.operation.value} requires exactly one fact")
        if self.operation == CalculationOperation.CAGR and self.periods is None:
            raise ValueError("cagr requires a positive periods value")
        if self.operation != CalculationOperation.CAGR and self.periods is not None:
            raise ValueError("periods is supported only for cagr")
        return self


class CalculationResult(FinancialModel):
    operation: CalculationOperation
    formula: str
    substituted_formula: str
    result: Decimal
    result_unit: str
    currency: str | None = None
    scale: str | None = None
    input_fact_ids: tuple[str, ...]
    source_ids: tuple[str, ...]


def _compatible(left: ValidatedFinancialFact, right: ValidatedFinancialFact) -> bool:
    if left.value_type != right.value_type:
        return False
    if left.normalized_unit != right.normalized_unit:
        return False
    if left.value_type in {ValueType.CURRENCY, ValueType.PER_SHARE} and (
        left.currency != right.currency
    ):
        return False
    if (
        left.accounting_basis
        and right.accounting_basis
        and left.accounting_basis.casefold() != right.accounting_basis.casefold()
    ):
        return False
    return True


def _require_compatible(facts: tuple[ValidatedFinancialFact, ...]) -> None:
    first = facts[0]
    if any(not _compatible(first, fact) for fact in facts[1:]):
        raise ValueError("Calculation inputs have incompatible units or currencies.")


def _require_same_series(facts: tuple[ValidatedFinancialFact, ...]) -> None:
    first = facts[0]
    if any(fact.ticker.casefold() != first.ticker.casefold() for fact in facts[1:]):
        raise ValueError("Temporal changes require facts for the same company.")
    if any(not is_metric_match(first.metric, fact.metric) for fact in facts[1:]):
        raise ValueError("Temporal changes require the same financial metric.")


def _display(value: Decimal) -> str:
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def _scale_result(base_value: Decimal, reference: ValidatedFinancialFact) -> Decimal:
    multiplier = SCALE_MULTIPLIERS.get(reference.scale or "ones", Decimal("1"))
    return base_value / multiplier


def _in_reference_scale(
    fact: ValidatedFinancialFact, reference: ValidatedFinancialFact
) -> Decimal:
    return _scale_result(fact.base_value, reference)


class FinancialCalculator:
    """Calculate only from validated fact objects with retained provenance."""

    @trace_component("finrag.financial.calculation", run_type="tool")
    def calculate(
        self,
        request: CalculationRequest,
        available_facts: Mapping[str, ValidatedFinancialFact],
    ) -> CalculationResult:
        missing = [fact_id for fact_id in request.input_fact_ids if fact_id not in available_facts]
        if missing:
            raise ValueError("Unknown or unvalidated fact IDs: " + ", ".join(missing))
        facts = tuple(available_facts[fact_id] for fact_id in request.input_fact_ids)
        if any(fact.value_qualifier and fact.value_qualifier.casefold() in {
                "over", "exceed", "exceeds", "exceeded", "below", "more than", "less than", "at least", "at most", "up to"} for fact in facts):
            raise ValueError("A bounded value cannot be used as an exact calculation input.")
        operation = request.operation

        if operation == CalculationOperation.SUM:
            _require_compatible(facts)
            total_base = sum((fact.base_value for fact in facts), Decimal("0"))
            result = _scale_result(total_base, facts[0])
            formula = "sum(values)"
            substituted = " + ".join(
                _display(_in_reference_scale(fact, facts[0])) for fact in facts
            )
            result_unit, currency, scale = self._same_unit_result(facts[0])
        elif operation in {
            CalculationOperation.ABSOLUTE_CHANGE,
            CalculationOperation.DIFFERENCE,
        }:
            _require_compatible(facts)
            left, right = facts
            if operation == CalculationOperation.ABSOLUTE_CHANGE:
                _require_same_series(facts)
                if left.value_type == ValueType.PERCENT:
                    raise ValueError(
                        "Use percentage_point_change for percent-valued facts."
                    )
                base_result = right.base_value - left.base_value
                reference = right
                formula = "new - old"
                substituted = (
                    f"{_display(_in_reference_scale(right, reference))} - "
                    f"{_display(_in_reference_scale(left, reference))}"
                )
            else:
                base_result = left.base_value - right.base_value
                reference = left
                formula = "minuend - subtrahend"
                substituted = (
                    f"{_display(_in_reference_scale(left, reference))} - "
                    f"{_display(_in_reference_scale(right, reference))}"
                )
            result = _scale_result(base_result, reference)
            result_unit, currency, scale = self._same_unit_result(reference)
        elif operation == CalculationOperation.PERCENTAGE_CHANGE:
            _require_compatible(facts)
            _require_same_series(facts)
            old, new = facts
            if old.base_value == 0:
                raise ValueError("Percentage change is undefined when the old value is zero.")
            result = ((new.base_value - old.base_value) / abs(old.base_value)) * Decimal("100")
            formula = "((new - old) / abs(old)) * 100"
            substituted = (
                f"(({_display(new.base_value)} - {_display(old.base_value)}) / "
                f"abs({_display(old.base_value)})) * 100"
            )
            result_unit, currency, scale = "percent", None, None
        elif operation in {
            CalculationOperation.PERCENTAGE_POINT_CHANGE,
            CalculationOperation.BASIS_POINT_CHANGE,
        }:
            old, new = facts
            _require_compatible(facts)
            _require_same_series(facts)
            if any(fact.value_type != ValueType.PERCENT for fact in facts):
                raise ValueError("Percentage-point changes require percent-valued facts.")
            difference = new.numeric_value - old.numeric_value
            if operation == CalculationOperation.BASIS_POINT_CHANGE:
                result = difference * Decimal("100")
                formula = "(new percent - old percent) * 100"
                substituted = (
                    f"({_display(new.numeric_value)} - {_display(old.numeric_value)}) * 100"
                )
                result_unit = "basis_points"
            else:
                result = difference
                formula = "new percent - old percent"
                substituted = f"{_display(new.numeric_value)} - {_display(old.numeric_value)}"
                result_unit = "percentage_points"
            currency, scale = None, None
        elif operation == CalculationOperation.RATIO:
            numerator, denominator = facts
            if denominator.base_value == 0:
                raise ValueError("Ratio is undefined when the denominator is zero.")
            result = numerator.base_value / denominator.base_value
            formula = "numerator / denominator"
            substituted = (
                f"{_display(numerator.base_value)} / {_display(denominator.base_value)}"
            )
            if _compatible(numerator, denominator):
                result_unit, currency, scale = "ratio", None, None
            elif (
                numerator.value_type == ValueType.CURRENCY
                and denominator.value_type == ValueType.COUNT
            ):
                result_unit = "per_share"
                currency = numerator.currency
                scale = None
            else:
                raise ValueError(
                    "Ratio inputs have incompatible units or currencies."
                )
        elif operation == CalculationOperation.PERCENTAGE_OF:
            numerator, denominator = facts
            _require_compatible(facts)
            if denominator.base_value == 0:
                raise ValueError(
                    "Percentage of is undefined when the denominator is zero."
                )
            result = (
                numerator.base_value / denominator.base_value
            ) * Decimal("100")
            formula = "(numerator / denominator) * 100"
            substituted = (
                f"({_display(numerator.base_value)} / "
                f"{_display(denominator.base_value)}) * 100"
            )
            result_unit, currency, scale = "percent", None, None
        elif operation == CalculationOperation.PRODUCT:
            left, right = facts
            dimensionless = {
                ValueType.PERCENT,
                ValueType.RATIO,
                ValueType.NUMBER,
            }
            dimensional = [fact for fact in facts if fact.value_type not in dimensionless]
            if len(dimensional) > 1:
                raise ValueError(
                    "Product supports at most one currency, count, or per-share input."
                )

            def factor(fact: ValidatedFinancialFact) -> Decimal:
                if fact.value_type == ValueType.PERCENT:
                    return fact.numeric_value / Decimal("100")
                if fact.value_type in dimensionless:
                    return fact.numeric_value
                return fact.base_value

            base_result = factor(left) * factor(right)
            reference = dimensional[0] if dimensional else None
            if reference is not None:
                result = _scale_result(base_result, reference)
                result_unit, currency, scale = self._same_unit_result(reference)
            else:
                result = base_result
                result_unit, currency, scale = "ratio", None, None
            formula = "left * right"
            substituted = f"{_display(factor(left))} * {_display(factor(right))}"
        elif operation == CalculationOperation.ABSOLUTE_VALUE:
            value = facts[0]
            result = abs(value.numeric_value)
            formula = "abs(value)"
            substituted = f"abs({_display(value.numeric_value)})"
            result_unit, currency, scale = self._same_unit_result(value)
        elif operation == CalculationOperation.PERCENT_COMPLEMENT:
            value = facts[0]
            if value.value_type != ValueType.PERCENT:
                raise ValueError("Percent complement requires a percent-valued fact.")
            result = Decimal("100") - value.numeric_value
            formula = "100 - percent"
            substituted = f"100 - {_display(value.numeric_value)}"
            result_unit, currency, scale = "percent", None, None
        elif operation == CalculationOperation.CAGR:
            _require_compatible(facts)
            _require_same_series(facts)
            start, end = facts
            if start.base_value <= 0 or end.base_value < 0:
                raise ValueError("CAGR requires a positive start and non-negative end value.")
            periods = request.periods or Decimal("0")
            try:
                with localcontext() as context:
                    context.prec = 28
                    result = (
                        (end.base_value / start.base_value)
                        ** (Decimal("1") / periods)
                        - Decimal("1")
                    ) * Decimal("100")
            except (InvalidOperation, DivisionByZero) as error:
                raise ValueError("CAGR could not be calculated from these inputs.") from error
            formula = "((end / start) ** (1 / periods) - 1) * 100"
            substituted = (
                f"(({_display(end.base_value)} / {_display(start.base_value)}) ** "
                f"(1 / {_display(periods)}) - 1) * 100"
            )
            result_unit, currency, scale = "percent", None, None
        else:  # pragma: no cover - enum validation makes this unreachable.
            raise ValueError(f"Unsupported calculation operation: {operation}.")

        if request.decimal_places is not None:
            quantum = Decimal(1).scaleb(-request.decimal_places)
            with localcontext() as context:
                context.prec = max(
                    context.prec,
                    len(result.as_tuple().digits) + request.decimal_places + 2,
                )
                result = result.quantize(quantum, rounding=ROUND_HALF_UP)

        return CalculationResult(
            operation=operation,
            formula=formula,
            substituted_formula=substituted,
            result=result,
            result_unit=result_unit,
            currency=currency,
            scale=scale,
            input_fact_ids=request.input_fact_ids,
            source_ids=tuple(dict.fromkeys(fact.source_id for fact in facts)),
        )

    @staticmethod
    def _same_unit_result(
        reference: ValidatedFinancialFact,
    ) -> tuple[str, str | None, str | None]:
        if reference.value_type == ValueType.CURRENCY:
            return "currency", reference.currency, reference.scale
        return reference.normalized_unit, None, reference.scale
