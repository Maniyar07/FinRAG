from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from .models import MasterExample
from .policy import validate_training_policy
from .prompts import training_record
from .utils import read_json, sha256_text, write_json, write_jsonl
from .validators import validate_dataset


def eligible_for_training(example: MasterExample) -> bool:
    return (
        example.usage == "train_candidate"
        and example.quality.automated_status == "passed"
        and example.quality.review_status == "approved"
        and not example.quality.benchmark_overlap
    )


def assign_splits(
    examples: list[MasterExample],
    *,
    train_ratio: float = 0.84,
    validation_ratio: float = 0.08,
) -> list[MasterExample]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between zero and one")
    if not 0 < validation_ratio < 1 or train_ratio + validation_ratio >= 1:
        raise ValueError("validation_ratio must leave a positive test ratio")
    output: list[MasterExample] = []
    for example in examples:
        if example.usage == "evaluation_reserved":
            output.append(example.model_copy(update={"split": "test"}))
            continue
        if not eligible_for_training(example):
            output.append(example.model_copy(update={"split": None}))
            continue
        bucket = int(sha256_text(example.group_id)[:12], 16) / float(16**12)
        if bucket < train_ratio:
            split = "train"
        elif bucket < train_ratio + validation_ratio:
            split = "validation"
        else:
            split = "test"
        output.append(example.model_copy(update={"split": split}))
    return output


def balance_dominant_query_types(
    rows: list[MasterExample], policy_path: Path
) -> tuple[list[MasterExample], list[MasterExample]]:
    """Deterministically omit only the excess of a dominant type from an export.

    The reviewed master and split assignments remain unchanged. Omitted IDs are
    recorded in the export manifest, so balancing is reproducible and auditable.
    """
    policy = read_json(policy_path)
    maximum = float(policy.get("maximum_single_query_type_fraction", 1.0))
    if not 0 < maximum <= 1:
        raise ValueError("maximum_single_query_type_fraction must be in (0, 1]")
    selected = list(rows)
    omitted: list[MasterExample] = []
    while selected:
        counts = Counter(row.query_type for row in selected)
        dominant_type, dominant_count = counts.most_common(1)[0]
        if dominant_count / len(selected) <= maximum:
            break
        removal_count = math.ceil(
            (dominant_count - maximum * len(selected)) / (1 - maximum)
        )
        ranked = sorted(
            (row for row in selected if row.query_type == dominant_type),
            key=lambda row: sha256_text(
                f"finrag-export-balance-v1\x1f{row.group_id}\x1f{row.example_id}"
            ),
            reverse=True,
        )
        remove_ids = {row.example_id for row in ranked[:removal_count]}
        omitted.extend(row for row in selected if row.example_id in remove_ids)
        selected = [row for row in selected if row.example_id not in remove_ids]
    return selected, omitted


def cap_dataset_size(
    rows: list[MasterExample], maximum_examples: int
) -> tuple[list[MasterExample], list[MasterExample]]:
    """Deterministically cap an export by trimming its most common query type.

    Repeatedly taking from the current dominant type preserves rare capabilities
    such as calculations, transcripts, multi-source synthesis, and evidence
    refusals. The reviewed master remains unchanged and every omission is audited.
    """
    if maximum_examples <= 0:
        raise ValueError("maximum_examples must be positive")
    selected = list(rows)
    omitted: list[MasterExample] = []
    while len(selected) > maximum_examples:
        counts = Counter(row.query_type for row in selected)
        dominant_type = sorted(counts, key=lambda name: (-counts[name], name))[0]
        candidate = max(
            (row for row in selected if row.query_type == dominant_type),
            key=lambda row: sha256_text(
                f"finrag-export-cap-v1\x1f{row.group_id}\x1f{row.example_id}"
            ),
        )
        omitted.append(candidate)
        selected = [row for row in selected if row.example_id != candidate.example_id]
    return selected, omitted


def export_training_splits(
    examples: list[MasterExample],
    output_dir: Path,
    *,
    minimum_train_examples: int = 500,
    include_metadata: bool = False,
    policy_path: Path | None = None,
    balance_query_types: bool = False,
    maximum_examples: int | None = None,
) -> dict:
    validation = validate_dataset(examples)
    if not validation.passed:
        raise ValueError(
            "Export blocked because dataset validation failed:\n- "
            + "\n- ".join(validation.errors[:25])
        )
    eligible = [example for example in examples if eligible_for_training(example)]
    unassigned = [example.example_id for example in eligible if example.split is None]
    if unassigned:
        raise ValueError(
            "Approved examples are missing split assignments: " + ", ".join(unassigned[:10])
        )
    omitted_for_cap: list[MasterExample] = []
    if maximum_examples is not None:
        eligible, omitted_for_cap = cap_dataset_size(eligible, maximum_examples)
    by_split = {
        split: [example for example in eligible if example.split == split]
        for split in ("train", "validation", "test")
    }
    omitted_for_balance: list[MasterExample] = []
    if balance_query_types:
        if policy_path is None:
            raise ValueError("Query-type balancing requires a dataset policy")
        by_split["train"], omitted_for_balance = balance_dominant_query_types(
            by_split["train"], policy_path
        )
    if len(by_split["train"]) < minimum_train_examples:
        raise ValueError(
            f"Export blocked: only {len(by_split['train'])} approved training examples; "
            f"at least {minimum_train_examples} are required."
        )
    policy_result = None
    if policy_path is not None:
        policy_result = validate_training_policy(by_split["train"], policy_path)
        if not policy_result["passed"]:
            raise ValueError(
                "Export blocked by dataset diversity policy:\n- "
                + "\n- ".join(policy_result["errors"])
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    file_hashes: dict[str, str] = {}
    for split, rows in by_split.items():
        path = output_dir / f"{split}.jsonl"
        serialized = "\n".join(
            json.dumps(training_record(row, include_metadata=include_metadata), ensure_ascii=False)
            for row in rows
        ) + ("\n" if rows else "")
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(serialized, encoding="utf-8", newline="\n")
        temporary.replace(path)
        file_hashes[path.name] = sha256_text(serialized)
    manifest = {
        "schema_version": "1.0",
        "format": "trl_conversational_prompt_completion",
        "counts": {split: len(rows) for split, rows in by_split.items()},
        "query_types": dict(
            sorted(
                Counter(
                    row.query_type
                    for split_rows in by_split.values()
                    for row in split_rows
                ).items()
            )
        ),
        "files": file_hashes,
        "rules": {
            "approved_only": True,
            "automated_validation_required": True,
            "benchmark_overlap_excluded": True,
            "group_split_isolation": True,
            "deterministic_query_type_balancing": balance_query_types,
        },
        "balancing": {
            "enabled": balance_query_types,
            "selection_rule": "finrag-export-balance-v1",
            "omitted_count": len(omitted_for_balance),
            "omitted_by_query_type": dict(
                sorted(Counter(row.query_type for row in omitted_for_balance).items())
            ),
            "omitted_example_ids": sorted(
                row.example_id for row in omitted_for_balance
            ),
        },
        "capping": {
            "enabled": maximum_examples is not None,
            "maximum_examples": maximum_examples,
            "selection_rule": "finrag-export-cap-v1",
            "omitted_count": len(omitted_for_cap),
            "omitted_by_query_type": dict(
                sorted(Counter(row.query_type for row in omitted_for_cap).items())
            ),
            "omitted_example_ids": sorted(row.example_id for row in omitted_for_cap),
        },
        "diversity_policy": policy_result,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
