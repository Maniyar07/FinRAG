"""Compare two FinRAG smoke reports without making model or retrieval calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["id"]): item for item in payload.get("results", [])}


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _reduction(before: object, after: object) -> float | None:
    old, new = _number(before), _number(after)
    if old is None or new is None:
        return None
    if old == 0:
        return 0.0 if new == 0 else None
    return round((old - new) * 100 / old, 2)


def _display(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, list):
        return ", ".join(map(str, value)) or "N/A"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    before, after = _load(args.before), _load(args.after)
    shared = sorted(before.keys() & after.keys())
    if not shared:
        raise ValueError("The reports contain no matching case IDs.")

    rows = []
    for case_id in shared:
        old, new = before[case_id], after[case_id]
        old_actual, new_actual = old.get("actual", {}), new.get("actual", {})
        rows.append({
            "id": case_id,
            "candidate_before": old_actual.get("candidate_count"),
            "candidate_after": new_actual.get("candidate_count"),
            "candidate_reduction_pct": _reduction(
                old_actual.get("candidate_count"), new_actual.get("candidate_count")
            ),
            "latency_before_seconds": old_actual.get("elapsed_seconds"),
            "latency_after_seconds": new_actual.get("elapsed_seconds"),
            "latency_reduction_pct": _reduction(
                old_actual.get("elapsed_seconds"), new_actual.get("elapsed_seconds")
            ),
            "sources_before": old_actual.get("source_count"),
            "sources_after": new_actual.get("source_count"),
            "cohere_scored_sources_before": (old_actual.get("reranker") or {}).get("scored_sources"),
            "cohere_scored_sources_after": (new_actual.get("reranker") or {}).get("scored_sources"),
            "retrieval_modes_before": old_actual.get("retrieval_modes"),
            "retrieval_modes_after": new_actual.get("retrieval_modes"),
            "correct_before": bool(old.get("passed")),
            "correct_after": bool(new.get("passed")),
            "invalid_citations_after": new_actual.get("invalid_citation_ids", []),
            "citation_source_utilization_after_pct": new_actual.get("citation_source_utilization_pct"),
        })

    headers = (
        "ID", "Candidates B/A", "Candidate reduction", "Latency B/A (s)",
        "Latency reduction", "Sources B/A", "Cohere-scored B/A", "Modes B -> A",
        "Correct B/A", "Citation source use A", "Invalid citations A",
    )
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        cells = (
            row["id"],
            f'{_display(row["candidate_before"])}/{_display(row["candidate_after"])}',
            f'{_display(row["candidate_reduction_pct"])}%',
            f'{_display(row["latency_before_seconds"])}/{_display(row["latency_after_seconds"])}',
            f'{_display(row["latency_reduction_pct"])}%',
            f'{_display(row["sources_before"])}/{_display(row["sources_after"])}',
            f'{_display(row["cohere_scored_sources_before"])}/{_display(row["cohere_scored_sources_after"])}',
            f'{_display(row["retrieval_modes_before"])} -> {_display(row["retrieval_modes_after"])}',
            f'{row["correct_before"]}/{row["correct_after"]}',
            f'{_display(row["citation_source_utilization_after_pct"])}%',
            _display(row["invalid_citations_after"])
            if row["invalid_citations_after"] else "none",
        )
        lines.append("| " + " | ".join(cells) + " |")

    report = "\n".join(lines) + "\n"
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
