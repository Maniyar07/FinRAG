"""Run machine-checkable live regression cases against a FinRAG index."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = PROJECT_ROOT / "evals" / "regression_cases.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "evals" / "results" / "regression_latest.json"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.chat_service import ChatService
from src.config import ACTIVE_INDEX_VERSION, MULTIHOP_ENABLED
from src.schemas import ChatResult, Decision


def validate_cases(payload: Any) -> list[dict]:
    """Validate the small, data-driven regression schema without query-specific rules."""
    if not isinstance(payload, list) or not payload:
        raise ValueError("Regression cases must be a non-empty JSON array.")
    cases: list[dict] = []
    seen_ids: set[str] = set()
    valid_decisions = {item.value for item in Decision}
    for position, raw in enumerate(payload, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Case {position} must be a JSON object.")
        case_id = str(raw.get("id") or "").strip()
        question = " ".join(str(raw.get("question") or "").split())
        expected = raw.get("expected")
        if not case_id or case_id in seen_ids:
            raise ValueError(f"Case {position} has a missing or duplicate id.")
        if len(question) < 3:
            raise ValueError(f"Case {case_id} has an invalid question.")
        if not isinstance(expected, dict):
            raise ValueError(f"Case {case_id} must define expected checks.")
        decisions = expected.get("decision")
        decisions = [decisions] if isinstance(decisions, str) else decisions
        if not isinstance(decisions, list) or not decisions:
            raise ValueError(f"Case {case_id} must define an expected decision.")
        if any(value not in valid_decisions for value in decisions):
            raise ValueError(f"Case {case_id} contains an unsupported decision.")
        if "multihop" in expected and not isinstance(expected["multihop"], bool):
            raise ValueError(f"Case {case_id} multihop must be true or false.")
        for field in ("scope_groups", "source_groups", "answer_contains", "facts", "calculations"):
            if field in expected and not isinstance(expected[field], list):
                raise ValueError(f"Case {case_id} {field} must be a list.")
        seen_ids.add(case_id)
        cases.append({**raw, "id": case_id, "question": question, "expected": expected})
    return cases


def load_cases(path: Path) -> list[dict]:
    return validate_cases(json.loads(path.read_text(encoding="utf-8")))


def _decimal(value: object) -> Decimal | None:
    text = str(value or "").strip().replace(",", "").replace("$", "").replace("%", "")
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _execution(trace: dict) -> dict:
    return (trace.get("multihop") or {}).get("execution") or {}


def _facts(trace: dict) -> list[dict]:
    return [
        fact
        for requirement in _execution(trace).get("requirements", [])
        for fact in requirement.get("facts", [])
    ]


def _calculations(trace: dict) -> list[dict]:
    facts_by_id = {str(fact.get("fact_id")): fact for fact in _facts(trace)}
    calculations = []
    for item in _execution(trace).get("calculations", []):
        first_id = next(iter(item.get("input_fact_ids") or ()), None)
        fact = facts_by_id.get(str(first_id), {})
        calculations.append({**item, "ticker": fact.get("ticker")})
    return calculations


def _group(value: object, size: int) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        return ()
    return tuple(str(item or "").strip().upper().replace("-", "") for item in value)


def _fact_matches(actual: dict, expected: dict) -> bool:
    for field in ("ticker", "period"):
        if field in expected and str(actual.get(field) or "").casefold() != str(expected[field]).casefold():
            return False
    metric = str(expected.get("metric_contains") or "").casefold()
    if metric and metric not in str(actual.get("metric") or actual.get("row_label") or "").casefold():
        return False
    if "raw_value" in expected:
        left, right = _decimal(actual.get("raw_value")), _decimal(expected["raw_value"])
        if left is None or right is None or left != right:
            return False
    return True


def _calculation_matches(actual: dict, expected: dict) -> bool:
    for field in ("ticker", "operation", "status"):
        if field in expected and str(actual.get(field) or "").casefold() != str(expected[field]).casefold():
            return False
    if "result" not in expected:
        return True
    left, right = _decimal(actual.get("result")), _decimal(expected["result"])
    tolerance = _decimal(expected.get("tolerance", "0"))
    return bool(left is not None and right is not None and tolerance is not None and abs(left - right) <= tolerance)


def _check(name: str, expected: object, actual: object, passed: bool) -> dict:
    return {"name": name, "passed": passed, "expected": expected, "actual": actual}


def evaluate_result(case: dict, result: ChatResult, elapsed_seconds: float) -> dict:
    """Evaluate one live result using only expectations declared in the case data."""
    expected = case["expected"]
    trace = result.trace or {}
    multihop = trace.get("multihop") or {}
    multihop_selected = bool(multihop.get("selected"))
    facts = _facts(trace)
    calculations = _calculations(trace)
    source_groups = {
        _group((source.get("ticker"), source.get("fiscal_year"), source.get("doc_type")), 3)
        for source in result.sources
    }
    source_groups.discard(())
    scope_groups = {_group(group, 2) for group in result.scope.groups}
    checks: list[dict] = []

    expected_decisions = expected["decision"]
    expected_decisions = [expected_decisions] if isinstance(expected_decisions, str) else expected_decisions
    checks.append(_check("decision", expected_decisions, result.decision.value, result.decision.value in expected_decisions))

    if "multihop" in expected:
        checks.append(_check("multihop", expected["multihop"], multihop_selected, multihop_selected is expected["multihop"]))
    if "scope_groups" in expected:
        required = {_group(group, 2) for group in expected["scope_groups"]}
        checks.append(_check("scope_groups", sorted(required), sorted(scope_groups), required.issubset(scope_groups)))
    if "scope_doc_type" in expected:
        actual_type = str(result.scope.doc_type or "").upper().replace("-", "")
        wanted_type = str(expected["scope_doc_type"]).upper().replace("-", "")
        checks.append(_check("scope_doc_type", wanted_type, actual_type, wanted_type == actual_type))
    if "source_groups" in expected:
        required = {_group(group, 3) for group in expected["source_groups"]}
        checks.append(_check("source_groups", sorted(required), sorted(source_groups), required.issubset(source_groups)))

    folded_answer = " ".join(result.answer.replace("\\$", "$ ").casefold().split())
    for fragment in expected.get("answer_contains", []):
        wanted = " ".join(str(fragment).casefold().split())
        checks.append(_check(f"answer_contains:{fragment}", fragment, result.answer[:500], wanted in folded_answer))
    for item in expected.get("facts", []):
        matches = [fact for fact in facts if _fact_matches(fact, item)]
        checks.append(_check("fact", item, matches, bool(matches)))
    for item in expected.get("calculations", []):
        matches = [calculation for calculation in calculations if _calculation_matches(calculation, item)]
        checks.append(_check("calculation", item, matches, bool(matches)))
    if expected.get("no_issues"):
        issues = list(multihop.get("issues") or [])
        checks.append(_check("no_issues", [], issues, not issues))
    if "generation_mode" in expected:
        actual_mode = (trace.get("generation") or {}).get("mode")
        checks.append(_check("generation_mode", expected["generation_mode"], actual_mode, actual_mode == expected["generation_mode"]))
    if "max_elapsed_seconds" in expected:
        limit = float(expected["max_elapsed_seconds"])
        checks.append(_check("max_elapsed_seconds", limit, elapsed_seconds, elapsed_seconds <= limit))

    retrieval = trace.get("retrieval") or {}
    scored_sources = sum(source.get("rerank_score") is not None for source in retrieval.get("sources", []))
    source_ids = [str(source.get("id") or "") for source in result.sources]
    cited_ids = list(dict.fromkeys(re.findall(r"\[(S\d+)\]", result.answer)))
    invalid_citations = [source_id for source_id in cited_ids if source_id not in source_ids]
    requirement_modes = [
        str(requirement.get("retrieval_mode") or "unknown")
        for requirement in _execution(trace).get("requirements", [])
    ]
    retrieval_modes = requirement_modes
    if not retrieval_modes and retrieval.get("mode"):
        retrieval_modes = [str(retrieval["mode"])]
    if not retrieval_modes:
        retrieval_modes = ["hybrid" if "candidate_count" in retrieval else "none"]
    return {
        "id": case["id"],
        "category": case.get("category"),
        "question": case["question"],
        "passed": all(check["passed"] for check in checks),
        "failures": [check["name"] for check in checks if not check["passed"]],
        "checks": checks,
        "actual": {
            "decision": result.decision.value,
            "answer": result.answer,
            "elapsed_seconds": elapsed_seconds,
            "scope_groups": [list(group) for group in result.scope.groups],
            "scope_doc_type": result.scope.doc_type,
            "multihop_selected": multihop_selected,
            "issues": list(multihop.get("issues") or []),
            "facts": facts,
            "calculations": calculations,
            "source_groups": [list(group) for group in sorted(source_groups)],
            "source_count": len(result.sources),
            "source_ids": source_ids,
            "citation_ids": cited_ids,
            "invalid_citation_ids": invalid_citations,
            "citation_source_utilization_pct": round(
                100 * len(set(cited_ids) & set(source_ids)) / len(source_ids), 2
            ) if source_ids else 0.0,
            "retrieval_modes": retrieval_modes,
            "generation": trace.get("generation"),
            "reranker": {
                "configured": bool((trace.get("reranker") or {}).get("configured")),
                "applied": bool(retrieval.get("reranker_applied") or scored_sources),
                "scored_sources": scored_sources,
            },
            "candidate_count": retrieval.get("candidate_count"),
        },
    }


def run_cases(cases: list[dict], *, index_version: str) -> list[dict]:
    service = ChatService(index_version=index_version)
    results = []
    try:
        for number, case in enumerate(cases, start=1):
            started = perf_counter()
            result = service.ask(case["question"])
            elapsed = round(perf_counter() - started, 3)
            evaluated = evaluate_result(case, result, elapsed)
            results.append(evaluated)
            status = "PASS" if evaluated["passed"] else "FAIL"
            print(f"[{number:02d}/{len(cases):02d}] {status} {case['id']} ({elapsed:.2f}s)", flush=True)
            for failure in evaluated["failures"]:
                print(f"  - {failure}", flush=True)
    finally:
        service.close()
    return results


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--index-version", default=ACTIVE_INDEX_VERSION)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    start = max(args.start - 1, 0)
    cases = cases[start : start + args.limit if args.limit else None]
    if not cases:
        raise ValueError("The selected regression case range is empty.")
    if args.validate_only:
        print(f"Validated {len(cases)} regression cases from {args.cases}.")
        return 0
    if any(case["expected"].get("multihop") is True for case in cases) and not MULTIHOP_ENABLED:
        raise RuntimeError("Set FINRAG_MULTIHOP_ENABLED=true before running multi-hop cases.")

    results = run_cases(cases, index_version=args.index_version)
    passed = sum(item["passed"] for item in results)
    report = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "index_version": args.index_version,
        "multihop_enabled": MULTIHOP_ENABLED,
        "summary": {"passed": passed, "failed": len(results) - passed, "total": len(results)},
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Regression evaluation: {passed}/{len(results)} passed")
    print(f"Report: {args.output}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
