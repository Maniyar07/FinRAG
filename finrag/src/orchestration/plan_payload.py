"""Validate and conservatively repair structured planner output."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.config import MULTIHOP_MAX_CALCULATIONS, MULTIHOP_MAX_SEARCHES
from src.constants import COMPANY_NAMES, TICKER_ALIASES
from src.financial.calculator import CalculationOperation
from src.orchestration.models import (
    EvidenceRequirement,
    PlannedCalculation,
)


YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
CALCULATION_INTENT_RE = re.compile(
    r"\b(?:calculate|percentage[- ]point|percentage change|absolute change|"
    r"basis points?|cagr|ratio|sum|difference)\b",
    re.IGNORECASE,
)
DERIVED_CONCLUSION_RE = re.compile(
    r"\b(?:which|identify)\b.{0,80}\b(?:improved|increased|grew|changed)\b|"
    r"\bcompare\b.{0,80}\b(?:calculated|percentage changes?|results?)\b",
    re.IGNORECASE,
)
COMPANY_TERMS = tuple(
    sorted(
        {
            *(alias.casefold() for aliases in TICKER_ALIASES.values() for alias in aliases),
            *(name.casefold() for name in COMPANY_NAMES.values()),
        },
        key=len,
        reverse=True,
    )
)


def _safe_identifier(value: object, *, fallback: str) -> str:
    """Normalize harmless model casing/punctuation without changing plan meaning."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold()).strip("_")
    if not cleaned:
        cleaned = fallback
    if not cleaned[0].isalpha():
        cleaned = f"step_{cleaned}"
    return cleaned[:64].rstrip("_")


def _requirement_topic(requirement: EvidenceRequirement) -> tuple[str, str, str]:
    """Identify equivalent per-company searches so their groups can be combined."""
    text = requirement.question.casefold()
    for term in COMPANY_TERMS:
        text = re.sub(rf"\b{re.escape(term)}(?:'s|’s)?\b", " company ", text)
    text = YEAR_RE.sub(" year ", text)
    text = re.sub(r"\b(?:company|year)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = " ".join(text.split())
    return requirement.evidence_type.value, requirement.document_type, text


def _merge_equivalent_requirements(
    requirements: list[EvidenceRequirement],
) -> tuple[list[EvidenceRequirement], dict[str, str]]:
    """Merge the same search split by company and return the ID remapping."""
    merged: list[EvidenceRequirement] = []
    positions: dict[tuple[str, str, str], int] = {}
    remap: dict[str, str] = {}
    for requirement in requirements:
        key = _requirement_topic(requirement)
        position = positions.get(key)
        if position is None:
            positions[key] = len(merged)
            merged.append(requirement)
            remap[requirement.requirement_id] = requirement.requirement_id
            continue
        existing = merged[position]
        groups = tuple(dict.fromkeys((*existing.groups, *requirement.groups)))
        merged[position] = existing.model_copy(
            update={
                "groups": groups,
                "requires_complete_coverage": (
                    existing.requires_complete_coverage
                    or requirement.requires_complete_coverage
                ),
            }
        )
        remap[requirement.requirement_id] = existing.requirement_id
    return merged, remap


class PlanPayload(BaseModel):
    """Model-produced portion of a plan; the application supplies the question."""

    model_config = ConfigDict(extra="forbid")

    requirements: list[EvidenceRequirement] = Field(
        min_length=1,
        max_length=MULTIHOP_MAX_SEARCHES,
    )
    calculations: list[PlannedCalculation] = Field(
        default_factory=list,
        max_length=MULTIHOP_MAX_CALCULATIONS,
    )


def parse_plan_payload(payload: Any, *, question: str) -> PlanPayload:
    """Validate raw tool arguments while dropping only non-calculation noise."""
    if isinstance(payload, PlanPayload):
        return payload

    raw_payload = payload
    if isinstance(payload, dict) and "parsed" in payload and "raw" in payload:
        parsed = payload.get("parsed")
        if isinstance(parsed, PlanPayload):
            return parsed
        raw_message = payload.get("raw")
        tool_calls = getattr(raw_message, "tool_calls", None) or []
        if tool_calls:
            raw_payload = tool_calls[0].get("args", {})

    if not isinstance(raw_payload, dict):
        raise TypeError("Multi-hop planner returned an unsupported payload.")
    try:
        return PlanPayload.model_validate(raw_payload)
    except Exception as original_error:
        raw_requirements = raw_payload.get("requirements")
        if not isinstance(raw_requirements, list) or not raw_requirements:
            raise original_error
        requirement_fields = {
            "requirement_id",
            "question",
            "evidence_type",
            "document_type",
            "groups",
            "requires_complete_coverage",
        }
        requirements: list[EvidenceRequirement] = []
        raw_id_map: dict[str, str] = {}
        used_ids: set[str] = set()
        for index, item in enumerate(raw_requirements, start=1):
            if not isinstance(item, dict):
                raise original_error
            original_id = str(item.get("requirement_id", ""))
            normalized_id = _safe_identifier(
                original_id, fallback=f"requirement_{index}"
            )
            base_id = normalized_id
            suffix = 2
            while normalized_id in used_ids:
                tail = f"_{suffix}"
                normalized_id = f"{base_id[:64 - len(tail)]}{tail}"
                suffix += 1
            used_ids.add(normalized_id)
            cleaned = {
                key: value
                for key, value in item.items()
                if key in requirement_fields
            }
            cleaned["requirement_id"] = normalized_id
            requirement = EvidenceRequirement.model_validate(cleaned)
            requirements.append(requirement)
            raw_id_map[original_id] = normalized_id

        requirements, merged_id_map = _merge_equivalent_requirements(requirements)
        raw_id_map = {
            original: merged_id_map.get(normalized, normalized)
            for original, normalized in raw_id_map.items()
        }
        known_ids = {requirement.requirement_id for requirement in requirements}
        numeric_requirements_by_group: dict[
            tuple[str, str], list[EvidenceRequirement]
        ] = {}
        for requirement in requirements:
            if requirement.evidence_type.value != "numeric":
                continue
            for group in requirement.groups:
                numeric_requirements_by_group.setdefault(group.key, []).append(
                    requirement
                )

        calculations: list[PlannedCalculation] = []
        raw_calculations = raw_payload.get("calculations", [])
        if not isinstance(raw_calculations, list):
            raise original_error
        allowed_fields = {
            "calculation_id",
            "label",
            "operation",
            "inputs",
            "periods",
        }
        reference_fields = {
            "requirement_id",
            "ticker",
            "fiscal_year",
            "period",
            "metric_hint",
        }
        valid_operations = {operation.value for operation in CalculationOperation}
        used_calculation_ids: set[str] = set()
        for index, item in enumerate(raw_calculations, start=1):
            if not isinstance(item, dict):
                raise original_error
            operation = str(item.get("operation", "")).strip()
            # Some models place qualitative answer steps in the calculations
            # list. They are already represented by narrative requirements.
            if operation not in valid_operations:
                continue
            cleaned_item = {
                key: value for key, value in item.items() if key in allowed_fields
            }
            calculation_id = _safe_identifier(
                cleaned_item.get("calculation_id"),
                fallback=f"calculation_{index}",
            )
            base_id = calculation_id
            suffix = 2
            while calculation_id in used_calculation_ids:
                tail = f"_{suffix}"
                calculation_id = f"{base_id[:64 - len(tail)]}{tail}"
                suffix += 1
            used_calculation_ids.add(calculation_id)
            cleaned_item["calculation_id"] = calculation_id
            raw_inputs = cleaned_item.get("inputs")
            if not isinstance(raw_inputs, list):
                raise original_error
            normalized_inputs: list[dict] = []
            for raw_input in raw_inputs:
                if not isinstance(raw_input, dict):
                    raise original_error
                normalized = {
                    key: value
                    for key, value in raw_input.items()
                    if key in reference_fields
                }
                raw_requirement_id = str(normalized.get("requirement_id", ""))
                if raw_requirement_id in raw_id_map:
                    normalized["requirement_id"] = raw_id_map[raw_requirement_id]
                if normalized.get("requirement_id") not in known_ids:
                    ticker = str(normalized.get("ticker", "")).upper()
                    fiscal_year = str(normalized.get("fiscal_year", ""))
                    matches = numeric_requirements_by_group.get(
                        (ticker, fiscal_year), []
                    )
                    if len(matches) != 1:
                        raise original_error
                    normalized["requirement_id"] = matches[0].requirement_id
                normalized_inputs.append(normalized)
            cleaned_item["inputs"] = normalized_inputs
            calculations.append(PlannedCalculation.model_validate(cleaned_item))

        if CALCULATION_INTENT_RE.search(question) and not calculations:
            raise original_error
        if len(requirements) > MULTIHOP_MAX_SEARCHES and calculations:
            requirements = [
                requirement
                for requirement in requirements
                if not DERIVED_CONCLUSION_RE.search(requirement.question)
            ]
        if len(requirements) > MULTIHOP_MAX_SEARCHES:
            raise original_error
        return PlanPayload(
            requirements=requirements,
            calculations=calculations,
        )
