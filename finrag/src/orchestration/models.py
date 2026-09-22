"""Minimal orchestration contracts shared by planning and execution."""

from __future__ import annotations

import re
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.config import MULTIHOP_MAX_CALCULATIONS, MULTIHOP_MAX_SEARCHES
from src.financial.calculator import CalculationOperation
from src.schemas import Scope


REQUIREMENT_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
TICKER_RE = re.compile(r"^[A-Z0-9.]{1,10}$")
YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
DOCUMENT_TYPES = frozenset({"10K", "TRANSCRIPT"})
MAX_GROUPS_PER_REQUIREMENT = 18


class OrchestrationModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class EvidenceType(str, Enum):
    NUMERIC = "numeric"
    NARRATIVE = "narrative"


class EvidenceGroup(OrchestrationModel):
    """One exact company and source-fiscal-year pair."""

    ticker: str
    fiscal_year: str

    @field_validator("ticker", mode="before")
    @classmethod
    def normalize_ticker(cls, value: object) -> str:
        ticker = str(value).strip().upper()
        if not TICKER_RE.fullmatch(ticker):
            raise ValueError("ticker must be a valid symbol")
        return ticker

    @field_validator("fiscal_year", mode="before")
    @classmethod
    def validate_year(cls, value: object) -> str:
        year = str(value).strip()
        if not YEAR_RE.fullmatch(year):
            raise ValueError("fiscal_year must be a four-digit year")
        return year

    @property
    def key(self) -> tuple[str, str]:
        return self.ticker, self.fiscal_year


class EvidenceRequirement(OrchestrationModel):
    """One focused question to answer from one document type across exact groups."""

    requirement_id: str
    question: str = Field(min_length=3, max_length=1_000)
    evidence_type: EvidenceType
    document_type: str
    groups: tuple[EvidenceGroup, ...] = Field(
        min_length=1,
        max_length=MAX_GROUPS_PER_REQUIREMENT,
    )
    requires_complete_coverage: bool = True

    @field_validator("requirement_id")
    @classmethod
    def validate_requirement_id(cls, value: str) -> str:
        if not REQUIREMENT_ID_RE.fullmatch(value):
            raise ValueError(
                "requirement_id must start with a lowercase letter and contain only "
                "lowercase letters, numbers, and underscores"
            )
        return value

    @field_validator("document_type", mode="before")
    @classmethod
    def normalize_document_type(cls, value: object) -> str:
        document_type = str(value).strip().upper().replace("-", "")
        if document_type not in DOCUMENT_TYPES:
            raise ValueError("document_type must be 10K or TRANSCRIPT")
        return document_type

    @model_validator(mode="after")
    def validate_unique_groups(self) -> "EvidenceRequirement":
        keys = [group.key for group in self.groups]
        if len(keys) != len(set(keys)):
            raise ValueError("groups must not contain duplicate company/year pairs")
        return self

    def to_scope(self) -> Scope:
        """Build an exact search scope without adding unintended group combinations."""
        pairs = tuple(group.key for group in self.groups)
        return Scope(
            tickers=tuple(dict.fromkeys(ticker for ticker, _ in pairs)),
            years=tuple(dict.fromkeys(year for _, year in pairs)),
            doc_type=self.document_type,
            requested_groups=pairs,
        )


class RequirementResult(OrchestrationModel):
    """Compact provenance summary produced after one requirement is executed."""

    requirement_id: str
    covered_groups: tuple[EvidenceGroup, ...] = ()
    source_ids: tuple[str, ...] = ()
    fact_ids: tuple[str, ...] = ()

    @field_validator("requirement_id")
    @classmethod
    def validate_requirement_id(cls, value: str) -> str:
        if not REQUIREMENT_ID_RE.fullmatch(value):
            raise ValueError("requirement_id has an invalid format")
        return value

    @field_validator(
        "covered_groups",
        "source_ids",
        "fact_ids",
    )
    @classmethod
    def validate_unique_values(cls, values: tuple) -> tuple:
        if len(values) != len(set(values)):
            raise ValueError("result collections must not contain duplicates")
        return values


class FactReference(OrchestrationModel):
    """A deterministic selector resolved to one validated fact after retrieval."""

    requirement_id: str
    ticker: str
    # fiscal_year identifies the source document group. period identifies the
    # requested table column/fact period, which may be an earlier comparative
    # year inside that filing.
    fiscal_year: str
    period: str | None = Field(default=None, min_length=1, max_length=100)
    metric_hint: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("requirement_id")
    @classmethod
    def validate_requirement_id(cls, value: str) -> str:
        if not REQUIREMENT_ID_RE.fullmatch(value):
            raise ValueError("requirement_id has an invalid format")
        return value

    @field_validator("ticker", mode="before")
    @classmethod
    def normalize_ticker(cls, value: object) -> str:
        ticker = str(value).strip().upper()
        if not TICKER_RE.fullmatch(ticker):
            raise ValueError("ticker must be a valid symbol")
        return ticker

    @field_validator("fiscal_year", mode="before")
    @classmethod
    def validate_year(cls, value: object) -> str:
        year = str(value).strip()
        if not YEAR_RE.fullmatch(year):
            raise ValueError("fiscal_year must be a four-digit year")
        return year


class PlannedCalculation(OrchestrationModel):
    """A calculation whose ordered fact references are bound during execution."""

    calculation_id: str
    label: str = Field(min_length=3, max_length=300)
    operation: CalculationOperation
    inputs: tuple[FactReference, ...] = Field(min_length=1, max_length=20)
    periods: Decimal | None = Field(default=None, gt=0)

    @field_validator("calculation_id")
    @classmethod
    def validate_calculation_id(cls, value: str) -> str:
        if not REQUIREMENT_ID_RE.fullmatch(value):
            raise ValueError("calculation_id has an invalid format")
        return value

    @model_validator(mode="after")
    def validate_operation_shape(self) -> "PlannedCalculation":
        binary = {
            CalculationOperation.ABSOLUTE_CHANGE,
            CalculationOperation.PERCENTAGE_CHANGE,
            CalculationOperation.PERCENTAGE_POINT_CHANGE,
            CalculationOperation.BASIS_POINT_CHANGE,
            CalculationOperation.DIFFERENCE,
            CalculationOperation.RATIO,
            CalculationOperation.CAGR,
        }
        if self.operation in binary and len(self.inputs) != 2:
            raise ValueError(f"{self.operation.value} requires exactly two inputs")
        if len(self.inputs) != len(set(self.inputs)):
            raise ValueError("calculation inputs must be unique")
        if self.operation == CalculationOperation.CAGR and self.periods is None:
            raise ValueError("cagr requires a positive periods value")
        if self.operation != CalculationOperation.CAGR and self.periods is not None:
            raise ValueError("periods is supported only for cagr")
        return self


class MultiHopPlan(OrchestrationModel):
    """Bounded searches and calculations for one compound user question."""

    original_question: str = Field(min_length=3, max_length=6_000)
    requirements: tuple[EvidenceRequirement, ...] = Field(
        min_length=1,
        max_length=MULTIHOP_MAX_SEARCHES,
    )
    calculations: tuple[PlannedCalculation, ...] = Field(
        default=(),
        max_length=MULTIHOP_MAX_CALCULATIONS,
    )

    @model_validator(mode="after")
    def validate_references(self) -> "MultiHopPlan":
        requirement_map = {
            requirement.requirement_id: requirement
            for requirement in self.requirements
        }
        if len(requirement_map) != len(self.requirements):
            raise ValueError("requirement IDs must be unique")
        calculation_ids = [item.calculation_id for item in self.calculations]
        if len(calculation_ids) != len(set(calculation_ids)):
            raise ValueError("calculation IDs must be unique")

        for calculation in self.calculations:
            for reference in calculation.inputs:
                requirement = requirement_map.get(reference.requirement_id)
                if requirement is None:
                    raise ValueError(
                        "calculation references an unknown evidence requirement"
                    )
                if requirement.evidence_type != EvidenceType.NUMERIC:
                    raise ValueError(
                        "calculations may reference only numeric requirements"
                    )
                groups = {group.key for group in requirement.groups}
                if (reference.ticker, reference.fiscal_year) not in groups:
                    raise ValueError(
                        "calculation fact reference is outside its requirement groups"
                    )
        return self
