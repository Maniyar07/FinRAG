from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from .models import MasterExample


REVIEW_FIELDS = (
    "example_id",
    "query_type",
    "difficulty",
    "question",
    "target",
    "source",
    "section",
    "parent_ids",
    "evidence",
    "automated_status",
    "benchmark_overlap",
    "issues",
    "warnings",
    "status",
    "reviewed_by",
    "reviewed_at",
    "reviewer_notes",
)


def _valid_review_timestamp(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def write_review_csv(path: Path, examples: list[MasterExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for example in examples:
            writer.writerow(
                {
                    "example_id": example.example_id,
                    "query_type": example.query_type,
                    "difficulty": example.difficulty,
                    "question": example.question,
                    "target": json.dumps(example.target.model_dump(mode="json"), ensure_ascii=False),
                    "source": " | ".join(dict.fromkeys(block.source for block in example.evidence)),
                    "section": " | ".join(dict.fromkeys(block.section for block in example.evidence)),
                    "parent_ids": " | ".join(block.parent_id for block in example.evidence),
                    "evidence": "\n\n".join(
                        f"[{block.source_id} | {block.source} | {block.section} | "
                        f"parent_id={block.parent_id}]\n{block.text}"
                        for block in example.evidence
                    ),
                    "automated_status": example.quality.automated_status,
                    "benchmark_overlap": example.quality.benchmark_overlap,
                    "issues": " | ".join(example.quality.issues),
                    "warnings": " | ".join(example.quality.warnings),
                    "status": example.quality.review_status,
                    "reviewed_by": example.quality.reviewed_by,
                    "reviewed_at": example.quality.reviewed_at,
                    "reviewer_notes": example.quality.reviewer_notes,
                }
            )
    temporary.replace(path)


def apply_review_csv(
    examples: list[MasterExample], review_path: Path
) -> list[MasterExample]:
    by_id = {example.example_id: example for example in examples}
    seen: set[str] = set()
    updates: dict[str, tuple[str, str, str, str]] = {}
    with review_path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            example_id = str(row.get("example_id") or "").strip()
            if example_id not in by_id:
                raise ValueError(f"Review CSV contains unknown example {example_id!r}")
            if example_id in seen:
                raise ValueError(f"Review CSV contains duplicate example {example_id}")
            seen.add(example_id)
            status = str(row.get("status") or "pending").strip().casefold()
            if status not in {"pending", "approved", "needs_edit", "rejected"}:
                raise ValueError(f"Invalid review status {status!r} for {example_id}")
            updates[example_id] = (
                status,
                str(row.get("reviewed_by") or "").strip(),
                str(row.get("reviewed_at") or "").strip(),
                str(row.get("reviewer_notes") or "").strip(),
            )

    output: list[MasterExample] = []
    for example in examples:
        if example.example_id not in updates:
            output.append(example)
            continue
        status, reviewed_by, reviewed_at, notes = updates[example.example_id]
        if status == "approved":
            if example.quality.automated_status != "passed":
                raise ValueError(
                    f"Cannot approve {example.example_id}: automated validation did not pass"
                )
            if example.quality.benchmark_overlap and example.usage == "train_candidate":
                raise ValueError(
                    f"Cannot approve {example.example_id} for training: benchmark overlap"
                )
            if example.usage == "train_candidate" and not reviewed_by:
                raise ValueError(
                    f"Cannot approve {example.example_id}: reviewed_by is required"
                )
            if example.usage == "train_candidate" and not _valid_review_timestamp(reviewed_at):
                raise ValueError(
                    f"Cannot approve {example.example_id}: reviewed_at must be an "
                    "ISO-8601 timestamp with a timezone"
                )
        quality = example.quality.model_copy(
            update={
                "review_status": status,
                "reviewed_by": reviewed_by,
                "reviewed_at": reviewed_at,
                "reviewer_notes": notes,
            }
        )
        output.append(example.model_copy(update={"quality": quality}))
    return output
