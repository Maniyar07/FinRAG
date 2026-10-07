from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .models import (
    CalculationRecord,
    EvidenceBlock,
    GroundedAnswerTarget,
    GroundedClaimTarget,
    MasterExample,
    NumericFactRecord,
    ProvenanceRecord,
    QualityRecord,
    ScopeRecord,
)
from .utils import read_json, sha256_text, stable_id


CATEGORY_TYPES = {
    "E": "transcript_reported",
    "F": "transcript_explanation",
    "G": "guidance",
    "H": "calculation",
    "I": "comparison",
    "J": "cross_document_synthesis",
}


def _review_statuses(path: Path) -> dict[int, str]:
    if not path.is_file():
        raise ValueError(f"Review CSV not found: {path}")
    statuses: dict[int, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            record_id = int(row["id"])
            status = str(row.get("status") or "pending").strip().casefold()
            if status not in {"approved", "pending", "needs_edit", "rejected"}:
                raise ValueError(f"Unknown review status {status!r} for Q{record_id}")
            statuses[record_id] = status
    return statuses


def _query_type(category: str) -> str:
    prefix = category.strip()[:1].upper()
    return CATEGORY_TYPES.get(prefix, "narrative")


def _application_context(
    facts: list[NumericFactRecord], calculations: list[CalculationRecord]
) -> str:
    blocks: list[str] = []
    if facts:
        lines = ["[APPLICATION-VALIDATED FINANCIAL FACTS]"]
        for index, fact in enumerate(facts, 1):
            lines.append(
                f"F{index}: {fact.label} | value={fact.value} | unit={fact.unit} | "
                f"period={fact.period} | qualifier={fact.qualifier} | "
                f"sources={','.join(fact.source_ids)}"
            )
        blocks.append("\n".join(lines))
    if calculations:
        lines = ["[APPLICATION-VERIFIED CALCULATIONS]"]
        for index, calculation in enumerate(calculations, 1):
            lines.append(
                f"C{index}: {calculation.operation}({', '.join(calculation.operands)}) "
                f"= {calculation.result} {calculation.unit}; "
                f"{calculation.explanation} sources={','.join(calculation.source_ids)}"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_approved_seed_examples(
    drafts_path: Path,
    review_csv: Path,
    *,
    index_version: str = "phase1-v2",
) -> list[MasterExample]:
    """Convert audited gold-builder records into evaluation-reserved prototypes.

    These records occur in the canonical benchmark, so they are deliberately marked
    ``evaluation_reserved`` and cannot be exported as training rows.  They provide
    exact, project-native examples for testing the dataset schema and generation
    contract without contaminating the held-out benchmark.
    """

    payload = read_json(drafts_path)
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError("Draft file must contain a non-empty records array")
    statuses = _review_statuses(review_csv)
    examples: list[MasterExample] = []
    for record in records:
        record_id = int(record["id"])
        if statuses.get(record_id) != "approved":
            continue
        if not bool((record.get("validation") or {}).get("passed")):
            raise ValueError(f"Q{record_id} is approved but deterministic validation failed")
        draft = record.get("draft") or {}
        if not draft.get("supported") or not draft.get("claims"):
            raise ValueError(f"Q{record_id} has no supported atomic claims")

        cited_old_ids = list(
            dict.fromkeys(
                evidence_id
                for claim in draft["claims"]
                for evidence_id in claim.get("evidence_ids") or []
            )
        )
        evidence_by_id = {item["id"]: item for item in record.get("evidence") or []}
        missing = set(cited_old_ids) - set(evidence_by_id)
        if missing:
            raise ValueError(f"Q{record_id} cites missing evidence IDs: {sorted(missing)}")
        id_map = {old: f"S{index}" for index, old in enumerate(cited_old_ids, 1)}
        evidence: list[EvidenceBlock] = []
        for old_id in cited_old_ids:
            item = evidence_by_id[old_id]
            text = str(item["text"]).strip()
            evidence.append(
                EvidenceBlock(
                    source_id=id_map[old_id],
                    parent_id=str(item["parent_id"]),
                    ticker=str(item["ticker"]),
                    fiscal_year=str(item["fiscal_year"]),
                    doc_type=str(item["doc_type"]).upper().replace("-", ""),
                    source=str(item["source"]),
                    section=str(item.get("section") or "Unknown section"),
                    text=text,
                    text_sha256=sha256_text(text),
                )
            )

        claims = [
            GroundedClaimTarget(
                text=str(claim["claim"]).strip(),
                source_ids=[id_map[value] for value in claim["evidence_ids"]],
            )
            for claim in draft["claims"]
        ]
        target_ids = list(
            dict.fromkeys(source_id for claim in claims for source_id in claim.source_ids)
        )
        numeric_facts = [
            NumericFactRecord(
                label=str(fact["label"]),
                value=str(fact["value"]),
                unit=str(fact["unit"]),
                period=str(fact["period"]),
                qualifier=str(fact.get("qualifier") or "reported"),
                source_ids=[id_map[value] for value in fact["evidence_ids"]],
            )
            for fact in draft.get("numeric_facts") or []
            if set(fact.get("evidence_ids") or []).issubset(id_map)
        ]
        calculation_sources = list(
            dict.fromkeys(source_id for fact in numeric_facts for source_id in fact.source_ids)
        ) or target_ids
        calculations = [
            CalculationRecord(
                operation=str(calculation["operation"]),
                operands=[str(value) for value in calculation["operands"]],
                result=str(calculation["result"]),
                unit=str(calculation["unit"]),
                decimal_places=int(calculation.get("decimal_places", 1)),
                explanation=str(calculation["explanation"]),
                source_ids=calculation_sources,
            )
            for calculation in draft.get("calculations") or []
        ]
        signature_parts = [
            str(record_id),
            *(block.parent_id for block in evidence),
            *(f"{fact.label}|{fact.value}|{fact.period}" for fact in numeric_facts),
            *(claim.text for claim in claims),
        ]
        fact_signature = sha256_text("\x1f".join(signature_parts))
        scope = ScopeRecord.model_validate(record["scope"])
        query_type = _query_type(str(record["category"]))
        examples.append(
            MasterExample(
                example_id=f"finrag_gold_{record_id:04d}",
                group_id=stable_id("grp_", fact_signature),
                query_type=query_type,
                difficulty=(
                    "complex" if query_type in {"calculation", "comparison", "cross_document_synthesis"}
                    else "moderate"
                ),
                usage="evaluation_reserved",
                question=str(record["question"]).strip(),
                scope=scope,
                evidence=evidence,
                numeric_facts=numeric_facts,
                calculations=calculations,
                application_context=_application_context(numeric_facts, calculations),
                target=GroundedAnswerTarget(
                    answer="", claims=claims, source_ids=target_ids
                ),
                provenance=ProvenanceRecord(
                    index_version=index_version,
                    generator="approved_gold_builder_seed_v1",
                    source_record_ids=[block.parent_id for block in evidence],
                    source_hashes=list(
                        dict.fromkeys(block.source_hash for block in evidence if block.source_hash)
                    ),
                    canonical_fact_signature=fact_signature,
                ),
                quality=QualityRecord(
                    automated_status="pending",
                    review_status="approved",
                    benchmark_overlap=True,
                    warnings=["evaluation_reserved_do_not_train"],
                ),
            )
        )
    if not examples:
        raise ValueError("No approved and validated seed records were found")
    return examples
