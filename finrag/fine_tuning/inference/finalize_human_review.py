"""Validate, unblind, and summarize a completed FinRAG A/B review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


METRICS = (
    "factual_correct",
    "complete",
    "numeric_correct",
    "evidence_supported",
    "hallucination",
    "overall_acceptable",
)
BINARY_VALUES = {"yes", "no"}
NUMERIC_VALUES = {"yes", "no", "na"}
PREFERENCES = {"A", "B", "Tie"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def apply_decisions(rows: list[dict[str, str]], decisions: dict) -> None:
    decision_rows = decisions.get("decisions")
    if not isinstance(decision_rows, dict):
        raise ValueError("Decision file must contain a decisions object")
    if set(decision_rows) != {str(index) for index in range(40)}:
        raise ValueError("Decision file must cover test indices 0 through 39")

    rows_by_index = {str(int(row["test_index"])): row for row in rows}
    if len(rows_by_index) != 40:
        raise ValueError("Review CSV must contain 40 unique test indices")

    for index, decision in decision_rows.items():
        row = rows_by_index[index]
        for label in ("a", "b"):
            ratings = decision.get(label)
            if not isinstance(ratings, dict):
                raise ValueError(f"Missing ratings for {label.upper()} at test {index}")
            for metric in METRICS:
                row[f"{label}_{metric}"] = str(ratings.get(metric, "")).lower()
        row["preference"] = str(decision.get("preference", ""))
        row["reviewer_notes"] = str(decision.get("notes", ""))


def validate_rows(rows: list[dict[str, str]], assignments: dict) -> None:
    if len(rows) != 40:
        raise ValueError(f"Expected 40 review rows; found {len(rows)}")
    indices = [str(int(row["test_index"])) for row in rows]
    if len(set(indices)) != 40 or set(indices) != {str(i) for i in range(40)}:
        raise ValueError("Review CSV must contain each test index 0 through 39 once")
    if set(assignments) != set(indices):
        raise ValueError("Review key assignments do not match review rows")

    for row in rows:
        index = str(int(row["test_index"]))
        for label in ("a", "b"):
            for metric in METRICS:
                value = row[f"{label}_{metric}"].strip().lower()
                allowed = NUMERIC_VALUES if metric == "numeric_correct" else BINARY_VALUES
                if value not in allowed:
                    raise ValueError(
                        f"Test {index}: {label}_{metric} must be one of {sorted(allowed)}"
                    )
        preference = row["preference"].strip().title()
        if preference not in PREFERENCES:
            raise ValueError(f"Test {index}: preference must be A, B, or Tie")
        if set(assignments[index]) != {"A", "B"}:
            raise ValueError(f"Test {index}: invalid A/B assignment")


def aggregate(rows: list[dict[str, str]], key: dict, decisions: dict) -> dict:
    assignments = key["assignments"]
    models = (key["base_model"], key["adapter_model"])
    counts = {
        model: {metric: Counter() for metric in METRICS}
        for model in models
    }
    preferences = Counter()
    flagged: list[dict[str, object]] = []

    for row in rows:
        index = str(int(row["test_index"]))
        assignment = assignments[index]
        for label in ("A", "B"):
            model = assignment[label]
            prefix = label.lower()
            for metric in METRICS:
                counts[model][metric][row[f"{prefix}_{metric}"].strip().lower()] += 1

        preference = row["preference"].strip().title()
        winner = "tie" if preference == "Tie" else assignment[preference]
        preferences[winner] += 1
        if (
            row["a_overall_acceptable"].strip().lower() == "no"
            or row["b_overall_acceptable"].strip().lower() == "no"
        ):
            flagged.append(
                {
                    "test_index": int(index),
                    "preference": preference,
                    "notes": row["reviewer_notes"],
                }
            )

    model_summary: dict[str, dict] = {}
    for model in models:
        model_summary[model] = {}
        for metric in METRICS:
            metric_counts = counts[model][metric]
            yes = metric_counts["yes"]
            no = metric_counts["no"]
            na = metric_counts["na"]
            applicable = yes + no
            model_summary[model][metric] = {
                "yes": yes,
                "no": no,
                "na": na,
                "applicable": applicable,
                "yes_rate_percent": round(100 * yes / applicable, 2) if applicable else None,
            }

    return {
        "schema_version": "finrag-ai-assisted-review-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer_type": decisions.get("reviewer_type", "ai_assisted_expert"),
        "reviewer_label": decisions.get("reviewer_label", "AI-assisted reviewer"),
        "reviewed_examples": 40,
        "dataset_sha256": key["dataset_sha256"],
        "prediction_sha256": key["prediction_sha256"],
        "models": model_summary,
        "preferences": dict(preferences),
        "flagged_examples": flagged,
        "limitations": [
            "This is an AI-assisted expert review, not an independent human review.",
            "A human owner should spot-check flagged examples before production acceptance.",
            "Format compliance was measured separately and was not used to score content.",
        ],
    }


def metric_cell(summary: dict, model: str, metric: str) -> str:
    value = summary["models"][model][metric]
    return f'{value["yes"]}/{value["applicable"]} ({value["yes_rate_percent"]:.1f}%)'


def build_report(summary: dict, key: dict) -> str:
    base = key["base_model"]
    adapter = key["adapter_model"]
    lines = [
        "# T26 AI-Assisted Content Review",
        "",
        "## Scope",
        "",
        "- Reviewed examples: 40 held-out test cases",
        f"- Base model: `{base}`",
        f"- Adapter: `{adapter}`",
        f"- Dataset SHA-256: `{summary['dataset_sha256']}`",
        "- Responses were randomized as A/B before content scoring.",
        "- JSON formatting was excluded from content scoring.",
        "",
        "## Results",
        "",
        "| Metric | Base | LoRA adapter |",
        "|---|---:|---:|",
    ]
    labels = (
        ("Factual correctness", "factual_correct"),
        ("Completeness", "complete"),
        ("Numerical correctness", "numeric_correct"),
        ("Evidence support", "evidence_supported"),
        ("Hallucination rate", "hallucination"),
        ("Overall acceptable", "overall_acceptable"),
    )
    for label, metric in labels:
        lines.append(
            f"| {label} | {metric_cell(summary, base, metric)} | "
            f"{metric_cell(summary, adapter, metric)} |"
        )
    preferences = summary["preferences"]
    lines.extend(
        [
            "",
            "## Pairwise preference",
            "",
            f"- Base preferred: {preferences.get(base, 0)}",
            f"- Adapter preferred: {preferences.get(adapter, 0)}",
            f"- Tie: {preferences.get('tie', 0)}",
            "",
            "## Material findings",
            "",
            "- The adapter consistently obeyed the output contract, but content review still found targeted omissions and row-selection errors.",
            "- Test 3 contains a flawed reference target: the source reports a dividend of USD 0.83 per share, not USD 0.83 million. The base answer is correct on this point and the adapter reproduces the bad target unit.",
            "- Test 13 is ambiguous in wording; the validated target expects the operating-lease discount rate, while both responses selected other lease rows.",
            "- The training/test data should be corrected before any future retraining or final production-quality claim.",
            "",
            "## Review status",
            "",
            "This AI-assisted review completes the content-analysis artifact for T26. It is not an independent human review. A human owner should spot-check the flagged cases and record sign-off before production acceptance.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--review-key", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = load_csv(args.review_csv)
    key = load_json(args.review_key)
    decisions = load_json(args.decisions)
    if sha256_file(args.review_csv) == sha256_file(args.review_key):
        raise ValueError("Review CSV and review key unexpectedly have the same content")
    apply_decisions(rows, decisions)
    validate_rows(rows, key["assignments"])
    summary = aggregate(rows, key, decisions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed_path = args.output_dir / "blind_review_completed_ai_assisted.csv"
    summary_path = args.output_dir / "human_review_summary_ai_assisted.json"
    report_path = args.output_dir / "T26_AI_ASSISTED_REVIEW_REPORT.md"

    with completed_path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path.write_text(build_report(summary, key), encoding="utf-8")

    print(f"Completed review: {completed_path}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
