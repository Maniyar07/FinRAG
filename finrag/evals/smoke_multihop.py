"""Run a small live smoke suite against the active multi-hop FinRAG index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.chat_service import ChatService
from src.config import ACTIVE_INDEX_VERSION, MULTIHOP_ENABLED


QUESTIONS = (
    (
        "Compare MSFT and TSLA revenue for 2024 and 2025 using their 10-Ks. "
        "Calculate each company's percentage change, identify which improved more, "
        "and explain the reasons management gave in the 2025 earnings transcripts."
    ),
    (
        "Compare MSFT and TSLA operating income for 2024 and 2025 using their 10-Ks. "
        "Calculate each company's percentage change, identify which improved more, "
        "and explain the main drivers discussed in their 2025 earnings transcripts."
    ),
    (
        "Compare MSFT and TSLA cash and cash equivalents for 2024 and 2025 using "
        "their 10-Ks. Calculate the absolute change for each company, identify which "
        "increased more, and summarize the principal liquidity risks in the 2025 10-Ks."
    ),
    (
        "Compare MSFT and TSLA research and development expense for 2024 and 2025 "
        "using their 10-Ks. Calculate each company's percentage change, identify which "
        "grew R&D spending faster, and explain relevant management commentary from the "
        "2025 earnings transcripts."
    ),
    (
        "Compare MSFT and TSLA effective tax rates for 2024 and 2025 using their "
        "10-Ks. Calculate the percentage-point change for each company, identify the "
        "larger change, and explain the main tax reconciliation factors disclosed in "
        "the filings."
    ),
)


def _source_summary(source: dict) -> dict:
    return {
        "id": source.get("id"),
        "ticker": source.get("ticker"),
        "fiscal_year": source.get("fiscal_year"),
        "doc_type": source.get("doc_type"),
        "section": source.get("section"),
        "page": source.get("pdf_page"),
        "file": source.get("source"),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        choices=range(1, len(QUESTIONS) + 1),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=len(QUESTIONS),
        choices=range(1, len(QUESTIONS) + 1),
    )
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--verdict", action="store_true")
    args = parser.parse_args()
    questions = QUESTIONS[args.start - 1 : args.start - 1 + args.limit]
    print(
        json.dumps(
            {
                "index": ACTIVE_INDEX_VERSION,
                "multihop_enabled": MULTIHOP_ENABLED,
                "question_count": len(questions),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not MULTIHOP_ENABLED:
        raise RuntimeError("Set FINRAG_MULTIHOP_ENABLED=true before this smoke run.")

    service = ChatService(index_version=ACTIVE_INDEX_VERSION)
    try:
        for number, question in enumerate(questions, start=args.start):
            started = perf_counter()
            result = service.ask(question)
            elapsed = round(perf_counter() - started, 2)
            multihop = result.trace.get("multihop", {})
            payload = {
                "number": number,
                "question": question,
                "decision": result.decision.value,
                "elapsed_seconds": elapsed,
                "answer": result.answer,
                "scope": {
                    "groups": [list(group) for group in result.scope.groups],
                    "doc_type": result.scope.doc_type,
                    "required_doc_types": list(result.scope.required_doc_types),
                },
                "multihop_selected": multihop.get("selected"),
                "plan": multihop.get("plan"),
                "execution": multihop.get("execution"),
                "issues": multihop.get("issues", []),
                "generation": result.trace.get("generation"),
                "error_type": result.trace.get("error_type"),
                "error_detail": result.trace.get("error_detail"),
                "sources": [_source_summary(source) for source in result.sources],
            }
            if args.compact:
                plan = payload.get("plan") or {}
                payload["plan"] = {
                    "requirements": [
                        {
                            "id": item.get("requirement_id"),
                            "type": item.get("evidence_type"),
                            "document_type": item.get("document_type"),
                            "groups": [
                                [group.get("ticker"), group.get("fiscal_year")]
                                for group in item.get("groups", [])
                            ],
                        }
                        for item in plan.get("requirements", [])
                    ],
                    "calculations": [
                        {
                            "id": item.get("calculation_id"),
                            "operation": item.get("operation"),
                        }
                        for item in plan.get("calculations", [])
                    ],
                }
                generation = payload.get("generation") or {}
                generation.pop("raw_output_previews", None)
            if args.summary:
                execution = payload.get("execution") or {}
                payload = {
                    "number": number,
                    "decision": payload["decision"],
                    "elapsed_seconds": elapsed,
                    "answer": payload["answer"],
                    "issues": payload["issues"],
                    "facts": [
                        fact
                        for requirement in execution.get("requirements", [])
                        for fact in requirement.get("facts", [])
                    ],
                    "calculations": execution.get("calculations", []),
                    "generation": {
                        key: value
                        for key, value in (payload.get("generation") or {}).items()
                        if key != "raw_output_previews"
                    },
                    "source_groups": [
                        [source["id"], source["ticker"], source["fiscal_year"], source["doc_type"]]
                        for source in payload["sources"]
                    ],
                    "error_type": payload["error_type"],
                    "error_detail": payload["error_detail"],
                }
            if args.verdict:
                execution = payload.get("execution") or {}
                payload = {
                    "number": number,
                    "decision": result.decision.value,
                    "elapsed_seconds": elapsed,
                    "answer": result.answer,
                    "issues": multihop.get("issues", []),
                    "calculations": [
                        {
                            "id": item.get("calculation_id"),
                            "status": item.get("status"),
                            "result": item.get("result"),
                            "unit": item.get("result_unit"),
                        }
                        for item in execution.get("calculations", [])
                    ],
                    "generation": {
                        key: value
                        for key, value in (result.trace.get("generation") or {}).items()
                        if key != "raw_output_previews"
                    },
                }
            print(f"\n=== QUESTION {number} ===", flush=True)
            print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    finally:
        service.close()


if __name__ == "__main__":
    main()
