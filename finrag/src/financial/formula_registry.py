"""Load and match declarative financial formulas.

The JSON catalog contains business vocabulary and expression trees.  This module
contains only generic validation and matching, so adding a formula does not require
another planner branch.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.financial.calculator import CalculationOperation


CATALOG_PATH = Path(__file__).with_name("formula_definitions.json")
FORMULA_MODES = frozenset({"per_group", "temporal"})


def _normalized(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


@dataclass(frozen=True)
class MetricDefinition:
    metric_id: str
    label: str
    search_instruction: str


@dataclass(frozen=True)
class FormulaDefinition:
    formula_id: str
    name: str
    aliases: tuple[str, ...]
    mode: str
    expression: dict[str, Any] | None = None
    metric_id: str | None = None
    temporal_operation: CalculationOperation | None = None
    assumption: str | None = None


@dataclass(frozen=True)
class FormulaRegistry:
    metrics: dict[str, MetricDefinition]
    formulas: tuple[FormulaDefinition, ...]

    def match(self, question: str) -> FormulaDefinition | None:
        """Return the longest whole-phrase alias found in the question."""
        normalized = f" {_normalized(question)} "
        matches = (
            (len(_normalized(alias)), formula)
            for formula in self.formulas
            for alias in formula.aliases
            if f" {_normalized(alias)} " in normalized
        )
        return max(matches, key=lambda item: item[0], default=(0, None))[1]


def _validate_expression(
    node: object,
    *,
    metrics: dict[str, MetricDefinition],
) -> None:
    if not isinstance(node, dict):
        raise ValueError("Formula expression nodes must be objects.")
    if set(node) == {"metric"}:
        metric_id = str(node["metric"])
        if metric_id not in metrics:
            raise ValueError(f"Unknown formula metric: {metric_id}")
        return
    if set(node) != {"operation", "inputs"}:
        raise ValueError("Formula nodes require either metric or operation/inputs.")
    operation = CalculationOperation(str(node["operation"]))
    inputs = node["inputs"]
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("Formula operation inputs must be a non-empty list.")
    binary = {
        CalculationOperation.ABSOLUTE_CHANGE,
        CalculationOperation.PERCENTAGE_CHANGE,
        CalculationOperation.PERCENTAGE_POINT_CHANGE,
        CalculationOperation.BASIS_POINT_CHANGE,
        CalculationOperation.DIFFERENCE,
        CalculationOperation.RATIO,
        CalculationOperation.CAGR,
        CalculationOperation.PRODUCT,
    }
    unary = {
        CalculationOperation.ABSOLUTE_VALUE,
        CalculationOperation.PERCENT_COMPLEMENT,
    }
    if operation in binary and len(inputs) != 2:
        raise ValueError(f"Formula operation {operation.value} requires two inputs.")
    if operation in unary and len(inputs) != 1:
        raise ValueError(f"Formula operation {operation.value} requires one input.")
    for child in inputs:
        _validate_expression(child, metrics=metrics)


@lru_cache(maxsize=1)
def load_formula_registry(path: Path = CATALOG_PATH) -> FormulaRegistry:
    """Load and validate the immutable formula catalog once per process."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_metrics = payload.get("metrics")
    raw_formulas = payload.get("formulas")
    if not isinstance(raw_metrics, dict) or not isinstance(raw_formulas, list):
        raise ValueError("Formula catalog requires metrics and formulas collections.")

    metrics = {
        metric_id: MetricDefinition(
            metric_id=metric_id,
            label=str(item["label"]).strip(),
            search_instruction=str(item["search_instruction"]).strip(),
        )
        for metric_id, item in raw_metrics.items()
    }
    formulas: list[FormulaDefinition] = []
    aliases_seen: dict[str, str] = {}
    ids_seen: set[str] = set()
    for item in raw_formulas:
        formula_id = str(item["id"]).strip()
        if formula_id in ids_seen:
            raise ValueError(f"Duplicate formula ID: {formula_id}")
        ids_seen.add(formula_id)
        aliases = tuple(str(alias).strip() for alias in item["aliases"])
        if not aliases:
            raise ValueError(f"Formula {formula_id} requires aliases.")
        for alias in aliases:
            key = _normalized(alias)
            owner = aliases_seen.get(key)
            if owner is not None and owner != formula_id:
                raise ValueError(f"Formula alias {alias!r} is duplicated by {owner}.")
            aliases_seen[key] = formula_id
        mode = str(item.get("mode", "per_group"))
        if mode not in FORMULA_MODES:
            raise ValueError(f"Unsupported formula mode: {mode}")
        expression = item.get("expression")
        metric_id = item.get("metric")
        temporal_operation = item.get("operation")
        if mode == "per_group":
            _validate_expression(expression, metrics=metrics)
        else:
            if metric_id not in metrics:
                raise ValueError(f"Unknown temporal formula metric: {metric_id}")
            temporal_operation = CalculationOperation(str(temporal_operation))
            if temporal_operation not in {
                CalculationOperation.PERCENTAGE_CHANGE,
                CalculationOperation.CAGR,
            }:
                raise ValueError("Temporal formulas support percentage_change or cagr.")
        formulas.append(
            FormulaDefinition(
                formula_id=formula_id,
                name=str(item["name"]).strip(),
                aliases=aliases,
                mode=mode,
                expression=expression,
                metric_id=str(metric_id) if metric_id is not None else None,
                temporal_operation=temporal_operation,
                assumption=(
                    str(item["assumption"]).strip()
                    if item.get("assumption")
                    else None
                ),
            )
        )
    return FormulaRegistry(metrics=metrics, formulas=tuple(formulas))


def expression_metric_ids(node: dict[str, Any]) -> tuple[str, ...]:
    """Return unique expression leaves in stable traversal order."""
    found: list[str] = []

    def visit(current: dict[str, Any]) -> None:
        if "metric" in current:
            metric_id = str(current["metric"])
            if metric_id not in found:
                found.append(metric_id)
            return
        for child in current["inputs"]:
            visit(child)

    visit(node)
    return tuple(found)

