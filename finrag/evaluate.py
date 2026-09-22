from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.retrieval.query_understanding import understand_query
from src.retrieval.scope_policy import resolve_scope
from src.schemas import Scope


def _values(value: str) -> tuple[str, ...]:
    return tuple(item.strip().upper() for item in value.split("|") if item.strip())


def _scope(row: dict, prefix: str) -> Scope:
    return Scope(
        tickers=_values(row.get(f"{prefix}_tickers", "")),
        years=_values(row.get(f"{prefix}_years", "")),
        doc_type=row.get(f"{prefix}_doc_type", "").strip().upper() or None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate deterministic FinRAG routing policy.")
    parser.add_argument("--suite", type=Path, default=Path("eval_questions.csv"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = []
    with args.suite.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            resolution = resolve_scope(
                understand_query(row["question"]),
                ui_scope=_scope(row, "ui"),
                previous_scope=_scope(row, "previous"),
            )
            expected = row["expected_decision"].strip()
            passed = resolution.decision.value == expected
            results.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "expected": expected,
                    "actual": resolution.decision.value,
                    "scope": resolution.scope.label(),
                    "passed": passed,
                }
            )

    passed = sum(result["passed"] for result in results)
    print(f"Routing evaluation: {passed}/{len(results)} passed")
    for result in results:
        if not result["passed"]:
            print(
                f"  FAIL {result['id']}: expected={result['expected']} "
                f"actual={result['actual']}"
            )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

