"""Shared contracts for evidence-grounded financial facts."""

from __future__ import annotations

import re
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


SOURCE_ID_RE = re.compile(r"^S\d+$", re.IGNORECASE)
TICKER_RE = re.compile(r"^[A-Z0-9.]{1,10}$")


class FinancialModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ValueType(str, Enum):
    CURRENCY = "currency"
    PERCENT = "percent"
    BASIS_POINTS = "basis_points"
    PER_SHARE = "per_share"
    RATIO = "ratio"
    COUNT = "count"
    NUMBER = "number"


class CandidateFinancialFact(FinancialModel):
    """One model-extracted claim that still requires deterministic validation."""

    ticker: str = Field(min_length=1, max_length=10)
    metric: str = Field(min_length=1, max_length=200)
    period: str = Field(min_length=1, max_length=100)
    raw_value: str = Field(min_length=1, max_length=100)
    unit: str | None = Field(default=None, max_length=100)
    scale: str | None = Field(default=None, max_length=30)
    currency: str | None = Field(default=None, max_length=10)
    accounting_basis: str | None = Field(default=None, max_length=50)
    source_id: str
    evidence_excerpt: str = Field(min_length=1, max_length=1_000)
    row_label: str | None = Field(default=None, max_length=300)
    column_label: str | None = Field(default=None, max_length=200)

    @field_validator("ticker", mode="before")
    @classmethod
    def normalize_ticker(cls, value: object) -> str:
        ticker = str(value).strip().upper()
        if not TICKER_RE.fullmatch(ticker):
            raise ValueError("ticker must be a valid symbol")
        return ticker

    @field_validator("source_id", mode="before")
    @classmethod
    def normalize_source_id(cls, value: object) -> str:
        source_id = str(value).strip().upper()
        # Structured models occasionally repeat the literal header prefix.
        # Normalize that harmless presentation variation while retaining the
        # strict S# identity checked against retrieved sources downstream.
        match = re.fullmatch(r"(?:SOURCE\s+)?(S\d+)", source_id)
        if match is None:
            raise ValueError("source_id must use the S# format")
        return match.group(1)

    @field_validator("currency", mode="before")
    @classmethod
    def normalize_currency(cls, value: object | None) -> str | None:
        if value is None or not str(value).strip():
            return None
        currency = str(value).strip().upper()
        return {
            "$": "USD",
            "US$": "USD",
            "€": "EUR",
            "£": "GBP",
            "¥": "JPY",
        }.get(currency, currency)


class FactExtractionPayload(FinancialModel):
    facts: list[CandidateFinancialFact] = Field(default_factory=list, max_length=24)


class NormalizedFinancialValue(FinancialModel):
    raw_value: str
    numeric_value: Decimal
    base_value: Decimal
    value_type: ValueType
    normalized_unit: str
    currency: str | None = None
    scale: str | None = None


class ValidatedFinancialFact(FinancialModel):
    fact_id: str
    ticker: str
    metric: str
    period: str
    raw_value: str
    numeric_value: Decimal
    base_value: Decimal
    value_type: ValueType
    normalized_unit: str
    currency: str | None = None
    scale: str | None = None
    accounting_basis: str | None = None
    source_id: str
    evidence_excerpt: str
    row_label: str | None = None
    column_label: str | None = None
    validation_checks: tuple[str, ...]


class RejectedFinancialFact(FinancialModel):
    source_id: str
    metric: str
    reason: str
    period: str | None = None
    raw_value: str | None = None
    row_label: str | None = None
    column_label: str | None = None


class FactValidationResult(FinancialModel):
    valid_facts: tuple[ValidatedFinancialFact, ...] = ()
    rejected_facts: tuple[RejectedFinancialFact, ...] = ()
