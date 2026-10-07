from __future__ import annotations

from collections import Counter
from pathlib import Path

from .models import MasterExample
from .utils import read_json


def validate_training_policy(rows: list[MasterExample], policy_path: Path) -> dict:
    policy = read_json(policy_path)
    if not isinstance(policy, dict):
        raise ValueError("Dataset policy must be a JSON object")
    errors: list[str] = []
    counts = Counter(row.query_type for row in rows)
    minimum = int(policy.get("minimum_train_examples", 0))
    if len(rows) < minimum:
        errors.append(f"training_count:{len(rows)}:minimum:{minimum}")
    max_fraction = float(policy.get("maximum_single_query_type_fraction", 1.0))
    if not 0 < max_fraction <= 1:
        raise ValueError("maximum_single_query_type_fraction must be in (0, 1]")
    if rows:
        largest_type, largest_count = counts.most_common(1)[0]
        actual_fraction = largest_count / len(rows)
        if actual_fraction > max_fraction:
            errors.append(
                f"query_type_dominance:{largest_type}:{actual_fraction:.3f}:"
                f"maximum:{max_fraction:.3f}"
            )
    group_counts: dict[str, int] = {}
    groups = policy.get("required_groups") or {}
    if not isinstance(groups, dict):
        raise ValueError("required_groups must be an object")
    for name, spec in groups.items():
        query_types = set(spec.get("query_types") or [])
        required = int(spec.get("minimum", 0))
        actual = sum(counts[value] for value in query_types)
        group_counts[str(name)] = actual
        if actual < required:
            errors.append(f"distribution_group:{name}:{actual}:minimum:{required}")
    return {
        "passed": not errors,
        "errors": errors,
        "counts": dict(sorted(counts.items())),
        "group_counts": group_counts,
        "policy": policy,
    }
