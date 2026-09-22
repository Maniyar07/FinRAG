from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


INDEX_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
TICKER_RE = re.compile(r"^[A-Z0-9.]{1,10}$")
YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
DOCUMENT_TYPES = frozenset({"10K", "TRANSCRIPT"})


class ApiModel(BaseModel):
    """Strict base model for the public API boundary."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


def _ticker(value: object) -> str:
    candidate = str(value).strip().upper()
    if not TICKER_RE.fullmatch(candidate):
        raise ValueError("must be a valid ticker symbol")
    return candidate


def _year(value: object) -> str:
    candidate = str(value).strip()
    if not YEAR_RE.fullmatch(candidate):
        raise ValueError("must be a four-digit fiscal year")
    return candidate


def _document_type(value: object | None) -> str | None:
    if value is None:
        return None
    candidate = str(value).strip().upper().replace("-", "")
    if not candidate:
        return None
    if candidate not in DOCUMENT_TYPES:
        raise ValueError("must be 10K or TRANSCRIPT")
    return candidate


class HistoryMessage(ApiModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8_000)


class ScopePayload(ApiModel):
    tickers: list[str] = Field(default_factory=list, max_length=12)
    years: list[str] = Field(default_factory=list, max_length=12)
    doc_type: str | None = None
    requested_groups: list[tuple[str, str]] = Field(default_factory=list, max_length=24)
    required_doc_types: list[str] = Field(default_factory=list, max_length=2)

    @field_validator("tickers", mode="before")
    @classmethod
    def validate_tickers(cls, values: object) -> list[str]:
        normalized = [_ticker(value) for value in (values or [])]
        if len(normalized) != len(set(normalized)):
            raise ValueError("tickers must not contain duplicates")
        return normalized

    @field_validator("years", mode="before")
    @classmethod
    def validate_years(cls, values: object) -> list[str]:
        normalized = [_year(value) for value in (values or [])]
        if len(normalized) != len(set(normalized)):
            raise ValueError("years must not contain duplicates")
        return normalized

    @field_validator("doc_type", mode="before")
    @classmethod
    def validate_doc_type(cls, value: object | None) -> str | None:
        return _document_type(value)

    @field_validator("required_doc_types", mode="before")
    @classmethod
    def validate_required_doc_types(cls, values: object) -> list[str]:
        normalized = [_document_type(value) for value in (values or [])]
        result = [value for value in normalized if value is not None]
        if len(result) != len(set(result)):
            raise ValueError("required_doc_types must not contain duplicates")
        return result

    @field_validator("requested_groups", mode="before")
    @classmethod
    def validate_requested_groups(cls, values: object) -> list[tuple[str, str]]:
        if values is None:
            return []
        normalized: list[tuple[str, str]] = []
        for value in values:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("each requested group must contain a ticker and year")
            normalized.append((_ticker(value[0]), _year(value[1])))
        if len(normalized) != len(set(normalized)):
            raise ValueError("requested_groups must not contain duplicates")
        return normalized

    @model_validator(mode="after")
    def validate_shape(self) -> ScopePayload:
        if self.doc_type and self.required_doc_types:
            raise ValueError("doc_type and required_doc_types cannot both be set")
        if self.requested_groups:
            if not self.tickers or not self.years:
                raise ValueError(
                    "requested_groups require the corresponding tickers and years"
                )
            ticker_set = set(self.tickers)
            year_set = set(self.years)
            if any(ticker not in ticker_set for ticker, _ in self.requested_groups):
                raise ValueError("requested group tickers must be present in tickers")
            if any(year not in year_set for _, year in self.requested_groups):
                raise ValueError("requested group years must be present in years")
        return self


class PendingClarificationPayload(ApiModel):
    original_question: str = Field(min_length=1, max_length=4_000)
    scope: ScopePayload = Field(default_factory=ScopePayload)
    candidate_tickers: list[str] = Field(default_factory=list, max_length=3)
    missing_fields: list[
        Literal["company", "year", "comparison years", "document type"]
    ] = Field(default_factory=list, max_length=4)
    query_expansions: list[str] = Field(default_factory=list, max_length=3)

    @field_validator("candidate_tickers", mode="before")
    @classmethod
    def validate_candidate_tickers(cls, values: object) -> list[str]:
        normalized = [_ticker(value) for value in (values or [])]
        if len(normalized) != len(set(normalized)):
            raise ValueError("candidate_tickers must not contain duplicates")
        return normalized

    @field_validator("missing_fields")
    @classmethod
    def validate_missing_fields(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("missing_fields must not contain duplicates")
        return values

    @field_validator("query_expansions", mode="before")
    @classmethod
    def validate_query_expansions(cls, values: object) -> list[str]:
        cleaned = [" ".join(str(value).split()) for value in (values or [])]
        if any(not value or len(value) > 100 for value in cleaned):
            raise ValueError("query expansions must contain 1-100 characters")
        if len({value.casefold() for value in cleaned}) != len(cleaned):
            raise ValueError("query_expansions must not contain duplicates")
        return cleaned


class UIFilters(ApiModel):
    company: str | None = None
    fiscal_year: str | None = None
    document_type: str | None = None

    @field_validator("company", mode="before")
    @classmethod
    def validate_company(cls, value: object | None) -> str | None:
        if value is None or not str(value).strip():
            return None
        return _ticker(value)

    @field_validator("fiscal_year", mode="before")
    @classmethod
    def validate_fiscal_year(cls, value: object | None) -> str | None:
        if value is None or not str(value).strip():
            return None
        return _year(value)

    @field_validator("document_type", mode="before")
    @classmethod
    def validate_document_type(cls, value: object | None) -> str | None:
        return _document_type(value)


class ChatRequest(ApiModel):
    index_version: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=4_000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=24)
    previous_scope: ScopePayload = Field(default_factory=ScopePayload)
    pending_clarification: PendingClarificationPayload | None = None
    filters: UIFilters = Field(default_factory=UIFilters)
    include_trace: bool = False

    @field_validator("index_version")
    @classmethod
    def validate_index_version(cls, value: str) -> str:
        if not INDEX_VERSION_RE.fullmatch(value):
            raise ValueError(
                "must start with a letter or number and contain only letters, "
                "numbers, dots, underscores, or hyphens"
            )
        return value

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()

    @model_validator(mode="after")
    def validate_history_size(self) -> ChatRequest:
        if sum(len(message.content) for message in self.history) > 64_000:
            raise ValueError("history is too large")
        return self


class ScopeView(ApiModel):
    tickers: list[str]
    years: list[str]
    doc_type: str | None = None
    requested_groups: list[tuple[str, str]]
    required_doc_types: list[str]
    label: str
    complete: bool


class SourceView(ApiModel):
    id: str
    ticker: str | None = None
    fiscal_year: str | None = None
    fiscal_period: str | None = None
    fiscal_year_end: str | None = None
    fiscal_q4_months: str | None = None
    doc_type: str | None = None
    source: str | None = None
    section: str | None = None
    pdf_page: str | None = None
    period_end_date: str | None = None
    call_date: str | None = None
    speaker: str | None = None
    speaker_role: str | None = None
    evidence_text: str


class ChatResponse(ApiModel):
    request_id: str
    trace_id: str
    decision: str
    answer: str
    scope: ScopeView
    inherited_fields: list[str]
    sources: list[SourceView]
    pending_clarification: PendingClarificationPayload | None = None
    trace: dict[str, Any] | None = None


class CompanyOption(ApiModel):
    ticker: str
    name: str


class AvailabilityEntry(ApiModel):
    company: str
    fiscal_year: str
    document_type: str


class IndexCatalogue(ApiModel):
    version: str
    companies: list[CompanyOption]
    fiscal_years: list[str]
    document_types: list[str]
    availability: list[AvailabilityEntry]


class CatalogueResponse(ApiModel):
    default_index: str
    trace_available: bool
    indexes: list[IndexCatalogue]


class HealthResponse(ApiModel):
    status: Literal["ok"]
    service_ready: bool
    default_index: str
    index_count: int


class ErrorItem(ApiModel):
    field: str
    message: str
    type: str


class ErrorBody(ApiModel):
    code: str
    message: str
    request_id: str
    trace_id: str
    details: list[ErrorItem] | None = None


class ErrorResponse(ApiModel):
    error: ErrorBody
