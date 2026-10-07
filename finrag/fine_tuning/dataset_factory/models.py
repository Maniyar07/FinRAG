from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


QueryType = Literal[
    "direct_extraction",
    "targeted_table",
    "complete_table",
    "narrative",
    "comparison",
    "year_over_year",
    "calculation",
    "transcript_reported",
    "transcript_explanation",
    "guidance",
    "cross_document_synthesis",
    "insufficient_evidence",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScopeRecord(StrictModel):
    tickers: list[str] = Field(min_length=1)
    years: list[str] = Field(min_length=1)
    doc_types: list[Literal["10K", "TRANSCRIPT"]] = Field(min_length=1)

    @field_validator("tickers", "years", "doc_types")
    @classmethod
    def unique_nonblank(cls, values: list[str]) -> list[str]:
        cleaned = [str(value).strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("scope values cannot be blank")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("scope values must be unique")
        return cleaned


class EvidenceBlock(StrictModel):
    source_id: str = Field(pattern=r"^S[1-9]\d*$")
    parent_id: str = Field(min_length=1)
    ticker: str = Field(min_length=1)
    fiscal_year: str = Field(pattern=r"^(?:19|20)\d{2}$")
    doc_type: Literal["10K", "TRANSCRIPT"]
    source: str = Field(min_length=1)
    section: str = Field(min_length=1)
    text: str = Field(min_length=1)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_hash: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")


class GroundedClaimTarget(StrictModel):
    text: str = Field(min_length=1, max_length=2_000)
    source_ids: list[str] = Field(min_length=1, max_length=12)

    @field_validator("source_ids")
    @classmethod
    def validate_source_ids(cls, values: list[str]) -> list[str]:
        normalized = [value.strip().upper() for value in values]
        if any(not value.startswith("S") or not value[1:].isdigit() for value in normalized):
            raise ValueError("claim source IDs must have the form S#")
        if len(normalized) != len(set(normalized)):
            raise ValueError("claim source IDs must be unique")
        return normalized


class GroundedAnswerTarget(StrictModel):
    answer: str = ""
    claims: list[GroundedClaimTarget] = Field(default_factory=list, max_length=24)
    source_ids: list[str] = Field(default_factory=list, max_length=24)

    @model_validator(mode="after")
    def exactly_one_output_mode(self) -> "GroundedAnswerTarget":
        if bool(self.answer.strip()) == bool(self.claims):
            raise ValueError("target must use either answer or claims, but not both")
        return self


class NumericFactRecord(StrictModel):
    label: str = Field(min_length=1)
    value: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    period: str = Field(min_length=1)
    qualifier: str = "reported"
    source_ids: list[str] = Field(min_length=1)


class CalculationRecord(StrictModel):
    operation: Literal["difference", "percentage_change", "percentage_of", "sum", "ratio"]
    operands: list[str] = Field(min_length=2, max_length=12)
    result: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    decimal_places: int = Field(ge=0, le=6)
    explanation: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)


class ProvenanceRecord(StrictModel):
    index_version: str = "phase1-v2"
    generator: str = Field(min_length=1)
    prompt_version: str = "finrag-runtime-v1"
    source_record_ids: list[str] = Field(min_length=1)
    source_hashes: list[str] = Field(default_factory=list)
    canonical_fact_signature: str = Field(min_length=16)


class QualityRecord(StrictModel):
    automated_status: Literal["pending", "passed", "failed"] = "pending"
    review_status: Literal["pending", "approved", "needs_edit", "rejected"] = "pending"
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    benchmark_overlap: bool = False
    reviewed_by: str = ""
    reviewed_at: str = ""
    reviewer_notes: str = ""


class MasterExample(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    example_id: str = Field(pattern=r"^finrag_[a-z0-9_\-]{8,96}$")
    group_id: str = Field(pattern=r"^grp_[0-9a-f]{16}$")
    query_type: QueryType
    difficulty: Literal["simple", "moderate", "complex"]
    usage: Literal["train_candidate", "prototype", "evaluation_reserved"]
    question: str = Field(min_length=10, max_length=4_000)
    scope: ScopeRecord
    evidence: list[EvidenceBlock] = Field(min_length=1, max_length=12)
    numeric_facts: list[NumericFactRecord] = Field(default_factory=list, max_length=24)
    calculations: list[CalculationRecord] = Field(default_factory=list, max_length=8)
    application_context: str = ""
    target: GroundedAnswerTarget
    provenance: ProvenanceRecord
    quality: QualityRecord = Field(default_factory=QualityRecord)
    split: Literal["train", "validation", "test"] | None = None

    @model_validator(mode="after")
    def scope_covers_evidence(self) -> "MasterExample":
        for block in self.evidence:
            if block.ticker not in self.scope.tickers:
                raise ValueError(f"evidence ticker {block.ticker} is outside scope")
            if block.fiscal_year not in self.scope.years:
                raise ValueError(f"evidence year {block.fiscal_year} is outside scope")
            if block.doc_type not in self.scope.doc_types:
                raise ValueError(f"evidence type {block.doc_type} is outside scope")
        return self
