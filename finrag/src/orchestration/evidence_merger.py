"""Merge retrieved evidence into one citation-safe generation bundle."""

from __future__ import annotations

import hashlib
from decimal import Decimal
from typing import Protocol

from src.financial.calculator import CalculationResult
from src.financial.models import ValidatedFinancialFact
from src.schemas import RetrievalBundle, Scope


class ExecutedCalculationLike(Protocol):
    calculation_id: str
    label: str
    result: CalculationResult


def normalize_doc_type(value: object) -> str:
    return str(value or "").strip().upper().replace("-", "")


def _source_identity(source: dict) -> tuple[str, ...]:
    parent_id = str(source.get("parent_id") or "").strip()
    source_hash = str(source.get("source_hash") or "").strip()
    if parent_id:
        return (
            "parent",
            source_hash,
            parent_id,
            str(source.get("ticker") or "").upper(),
            str(source.get("fiscal_year") or ""),
            normalize_doc_type(source.get("doc_type")),
        )
    evidence_hash = hashlib.sha256(
        str(source.get("evidence_text") or "").encode("utf-8")
    ).hexdigest()[:16]
    return (
        "fallback",
        str(source.get("ticker") or "").upper(),
        str(source.get("fiscal_year") or ""),
        normalize_doc_type(source.get("doc_type")),
        str(source.get("source") or ""),
        str(source.get("section") or ""),
        str(source.get("pdf_page") or ""),
        evidence_hash,
    )


def _merge_text(existing: object, incoming: object) -> str:
    left = str(existing or "").strip()
    right = str(incoming or "").strip()
    if not left:
        return right
    if not right or right in left:
        return left
    if left in right:
        return right
    return f"{left}\n\n{right}"


def _source_header(source: dict) -> str:
    fields = [
        f"SOURCE {source['id']}",
        f"TICKER: {source.get('ticker')}",
        f"FISCAL YEAR: {source.get('fiscal_year')}",
        f"DOCUMENT PERIOD ONLY: {source.get('fiscal_period')} (not the period of every fact)",
        f"TYPE: {source.get('doc_type')}",
        f"FILE: {source.get('source')}",
        f"SECTION: {source.get('section')}",
        f"PDF PAGE: {source.get('pdf_page', 'not available')}",
    ]
    for key, label in (
        ("period_end_date", "PERIOD END"),
        ("fiscal_q4_months", "FISCAL Q4 MONTHS"),
        ("call_date", "CALL DATE"),
        ("speaker", "SPEAKER"),
        ("speaker_role", "SPEAKER ROLE"),
    ):
        if source.get(key):
            fields.append(f"{label}: {source[key]}")
    roles = source.get("evidence_requirements") or []
    if roles:
        role_labels = ", ".join(
            f"{role.get('requirement_id')} ({role.get('evidence_type')})"
            for role in roles
        )
        fields.append(f"EVIDENCE ROLES: {role_labels}")
    return "[" + " | ".join(fields) + "]"


def _display(value: Decimal) -> str:
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def _result_unit_label(result: CalculationResult) -> str:
    if result.result_unit == "currency":
        return " ".join(
            value for value in (result.currency, result.scale) if value
        ) or "currency"
    return result.result_unit


class EvidenceMerger:
    """Deduplicate parent evidence and assign globally unique citation IDs."""

    def __init__(self) -> None:
        self.sources: list[dict] = []
        self._index_by_identity: dict[tuple[str, ...], int] = {}
        self.candidate_count = 0

    def add(self, bundle: RetrievalBundle) -> list[dict]:
        self.candidate_count += bundle.candidate_count
        remapped: list[dict] = []
        for original in bundle.sources:
            source = dict(original)
            identity = _source_identity(source)
            index = self._index_by_identity.get(identity)
            if index is None:
                source["id"] = f"S{len(self.sources) + 1}"
                self._index_by_identity[identity] = len(self.sources)
                self.sources.append(source)
                remapped.append(source)
                continue

            existing = self.sources[index]
            existing["evidence_text"] = _merge_text(
                existing.get("evidence_text"), source.get("evidence_text")
            )
            existing["full_evidence_text"] = _merge_text(
                existing.get("full_evidence_text"),
                source.get("full_evidence_text"),
            )
            source["id"] = existing["id"]
            source["evidence_text"] = existing.get("evidence_text", "")
            source["full_evidence_text"] = existing.get("full_evidence_text", "")
            remapped.append(source)
        return remapped

    def build(
        self,
        *,
        scope: Scope,
        facts: tuple[ValidatedFinancialFact, ...],
        calculations: tuple[ExecutedCalculationLike, ...],
    ) -> RetrievalBundle:
        selected_ids = {
            source_id
            for item in calculations
            for source_id in item.result.source_ids
        }
        if not selected_ids:
            selected_ids = {fact.source_id for fact in facts}

        narrative_counts: dict[tuple[str, str], int] = {}
        for source in self.sources:
            if not any(
                role.get("evidence_type") == "narrative"
                for role in source.get("evidence_requirements") or []
            ):
                continue
            key = (str(source.get("ticker")), str(source.get("fiscal_year")))
            if narrative_counts.get(key, 0) < 3:
                selected_ids.add(str(source["id"]))
                narrative_counts[key] = narrative_counts.get(key, 0) + 1

        selected_sources = (
            [source for source in self.sources if source["id"] in selected_ids]
            if selected_ids
            else list(self.sources)
        )
        context_parts = [
            f"{_source_header(source)}\n{str(source.get('evidence_text') or '').strip()}"
            for source in selected_sources
            if str(source.get("evidence_text") or "").strip()
        ]
        if facts:
            lines = ["[APPLICATION-VALIDATED FINANCIAL FACTS]"]
            for fact in facts:
                lines.append(
                    f"{fact.fact_id}: {fact.ticker} | {fact.metric} | {fact.period} | "
                    f"raw value {fact.raw_value} | source {fact.source_id}"
                )
            context_parts.append("\n".join(lines))
        if calculations:
            lines = ["[APPLICATION-VERIFIED CALCULATIONS]"]
            for item in calculations:
                result = item.result
                source_ids = ", ".join(result.source_ids)
                lines.append(
                    f"{item.calculation_id}: {item.label} | {result.formula} | "
                    f"{result.substituted_formula} = {_display(result.result)} "
                    f"{_result_unit_label(result)} | supporting sources: {source_ids}"
                )
            context_parts.append("\n".join(lines))
        covered = tuple(
            sorted(
                {
                    (str(source.get("ticker")), str(source.get("fiscal_year")))
                    for source in selected_sources
                }
            )
        )
        return RetrievalBundle(
            context="\n\n---\n\n".join(context_parts),
            sources=selected_sources,
            scope=scope,
            candidate_count=self.candidate_count,
            covered_groups=covered,
        )
