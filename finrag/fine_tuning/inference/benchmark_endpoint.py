"""Compare base and LoRA models served by one OpenAI-compatible endpoint.

Run from the finrag directory. The held-out JSONL rows must contain the
``prompt`` and ``completion`` message arrays used by the training export.
This measures serving performance and format only, not factual correctness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "fine_tuning/work/sft_final_800/test.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "fine_tuning/work/benchmarks"
SOURCE_RE = re.compile(r"\[SOURCE\s+(S[1-9]\d*)\b", re.IGNORECASE)
CITATION_RE = re.compile(r"\[(S[1-9]\d*)\]", re.IGNORECASE)
SOURCE_ID_RE = re.compile(r"^S[1-9]\d*$")
EXPECTED_OUTPUT_KEYS = {"answer", "claims", "source_ids"}
EXPECTED_CLAIM_KEYS = {"text", "source_ids"}
INSUFFICIENT_EVIDENCE_RESPONSE = (
    "I cannot find sufficient evidence in the retrieved financial documents "
    "to answer this question."
)
BENCHMARK_SCHEMA_VERSION = "finrag-serving-benchmark-v2"


def load_examples(path: Path, limit: int) -> list[dict]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    if not path.is_file():
        raise FileNotFoundError(f"Test split not found: {path}")
    examples = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"Record at line {line_number} must be an object")
            prompt = record.get("prompt")
            completion = record.get("completion")
            if not isinstance(prompt, list) or not isinstance(completion, list):
                raise ValueError(f"Invalid prompt/completion at line {line_number}")
            if len(prompt) != 2 or [item.get("role") if isinstance(item, dict) else None
                                    for item in prompt] != ["system", "user"]:
                raise ValueError(f"Invalid prompt messages at line {line_number}")
            if len(completion) != 1 or not isinstance(completion[0], dict) \
                    or completion[0].get("role") != "assistant":
                raise ValueError(f"Invalid completion messages at line {line_number}")
            if any(
                set(item) != {"role", "content"}
                or not isinstance(item.get("content"), str)
                or not item["content"].strip()
                for item in prompt + completion
            ):
                raise ValueError(f"Invalid message content at line {line_number}")
            examples.append({"index": line_number - 1, "prompt": prompt, "gold": completion})
            if len(examples) >= limit:
                break
    if not examples:
        raise ValueError("The selected test split is empty.")
    return examples


def check_output(content: str, prompt: list[dict], finish_reason: str | None) -> dict:
    source_ids = set(SOURCE_RE.findall("\n".join(item["content"] for item in prompt)))
    checks = {
        "json_valid": False,
        "schema_valid": False,
        "schema_errors": [],
        "source_ids_exact": None,
        "source_ids_normalized": None,
        "possibly_truncated": finish_reason == "length",
    }
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return checks
    checks["json_valid"] = True
    if not isinstance(payload, dict):
        checks["schema_errors"].append("root_not_object")
        return checks
    if set(payload) != EXPECTED_OUTPUT_KEYS:
        checks["schema_errors"].append("top_level_fields")

    answer = payload.get("answer")
    claims, declared = payload.get("claims"), payload.get("source_ids")
    if not isinstance(answer, str):
        checks["schema_errors"].append("answer_not_string")
    if not isinstance(claims, list):
        checks["schema_errors"].append("claims_not_list")
    if not isinstance(declared, list):
        checks["schema_errors"].append("source_ids_not_list")
    if not isinstance(answer, str) or not isinstance(claims, list) \
            or not isinstance(declared, list):
        return checks

    if len(claims) > 24:
        checks["schema_errors"].append("too_many_claims")
    if len(declared) > 24:
        checks["schema_errors"].append("too_many_source_ids")
    declared_strings = [item for item in declared if isinstance(item, str)]
    if len(declared_strings) != len(declared):
        checks["schema_errors"].append("source_id_not_string")
    if any(not SOURCE_ID_RE.fullmatch(item) for item in declared_strings):
        checks["schema_errors"].append("invalid_source_id_format")
    if len(declared_strings) != len(set(declared_strings)):
        checks["schema_errors"].append("duplicate_source_ids")

    claim_source_ids: list[str] = []
    for claim in claims:
        if not isinstance(claim, dict):
            checks["schema_errors"].append("claim_not_object")
            continue
        if set(claim) != EXPECTED_CLAIM_KEYS:
            checks["schema_errors"].append("claim_fields")
        text = claim.get("text")
        claim_ids = claim.get("source_ids")
        if not isinstance(text, str) or not text.strip() or len(text) > 2_000:
            checks["schema_errors"].append("invalid_claim_text")
        elif CITATION_RE.search(text):
            checks["schema_errors"].append("claim_contains_citation")
        if not isinstance(claim_ids, list) or not claim_ids or len(claim_ids) > 12:
            checks["schema_errors"].append("invalid_claim_source_ids")
            continue
        string_ids = [item for item in claim_ids if isinstance(item, str)]
        claim_source_ids.extend(string_ids)
        if len(string_ids) != len(claim_ids):
            checks["schema_errors"].append("claim_source_id_not_string")
        if any(not SOURCE_ID_RE.fullmatch(item) for item in string_ids):
            checks["schema_errors"].append("invalid_claim_source_id_format")
        if len(string_ids) != len(set(string_ids)):
            checks["schema_errors"].append("duplicate_claim_source_ids")

    answer_mode = bool(answer.strip())
    claims_mode = bool(claims)
    if answer_mode == claims_mode:
        checks["schema_errors"].append("exactly_one_output_mode_required")
    elif claims_mode:
        if set(declared_strings) != set(claim_source_ids):
            checks["schema_errors"].append("claim_source_ids_mismatch")
    else:
        cited = CITATION_RE.findall(answer)
        if set(declared_strings) != set(cited):
            checks["schema_errors"].append("answer_citations_mismatch")
        if not cited and answer.strip() != INSUFFICIENT_EVIDENCE_RESPONSE:
            checks["schema_errors"].append("uncited_answer")

    used = declared_strings + claim_source_ids
    if used:
        checks["source_ids_exact"] = all(item in source_ids for item in used)
        normalized = {item.strip().upper() for item in source_ids}
        checks["source_ids_normalized"] = all(
            item.strip().upper() in normalized for item in used
        )
    checks["schema_errors"] = list(dict.fromkeys(checks["schema_errors"]))
    checks["schema_valid"] = not checks["schema_errors"]
    return checks


def percentile(values: list[float], percentage: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 2)


def run_one(client: OpenAI, model: str, example: dict, max_tokens: int) -> dict:
    started = perf_counter()
    result = {"test_index": example["index"], "model": model,
              "question": example["prompt"][-1]["content"],
              "gold": example["gold"][0].get("content", "")}
    try:
        response = client.chat.completions.create(
            model=model, messages=example["prompt"], temperature=0,
            max_tokens=max_tokens, stream=False,
        )
        choice = response.choices[0]
        content = choice.message.content or ""
        result.update(
            output=content, finish_reason=choice.finish_reason,
            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
            completion_tokens=response.usage.completion_tokens if response.usage else None,
            checks=check_output(content, example["prompt"], choice.finish_reason),
            error=None,
        )
    except Exception as error:
        result.update(output="", finish_reason=None, prompt_tokens=None,
                      completion_tokens=None, checks=None,
                      error=f"{type(error).__name__}: {str(error)[:500]}")
    result["latency_ms"] = round((perf_counter() - started) * 1000, 2)
    return result


def summarize(results: list[dict], wall_seconds: float) -> dict:
    latencies = [row["latency_ms"] for row in results if row["error"] is None]
    successful = [row for row in results if row["error"] is None]
    token_counts = [row["completion_tokens"] for row in successful]
    error_counts: dict[str, int] = {}
    for row in successful:
        for error in (row["checks"] or {}).get("schema_errors", []):
            error_counts[error] = error_counts.get(error, 0) + 1
    return {
        "requests": len(results),
        "successful_requests": len(successful),
        "errors": len(results) - len(successful),
        "wall_seconds": round(wall_seconds, 3),
        "latency_p50_ms": round(median(latencies), 2) if latencies else None,
        "latency_p95_ms": percentile(latencies, 0.95),
        "requests_per_second": round(len(results) / wall_seconds, 3),
        "successful_requests_per_second": round(len(successful) / wall_seconds, 3),
        "output_tokens_per_second": (
            round(sum(token_counts) / wall_seconds, 3)
            if token_counts and all(count is not None for count in token_counts) else None
        ),
        "format_counts": {
            key: sum(row["checks"] is not None and row["checks"].get(key) is True
                     for row in results)
            for key in ("json_valid", "schema_valid", "source_ids_exact",
                        "source_ids_normalized", "possibly_truncated")
        },
        "source_id_checks_applicable": sum(
            row["checks"] is not None
            and row["checks"].get("source_ids_exact") is not None
            for row in results
        ),
        "schema_error_counts": dict(sorted(error_counts.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=os.getenv("FINRAG_LLM_API_KEY", "local"))
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-model", default="finrag-lora")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--server-max-num-seqs",
        type=int,
        required=True,
        help="The vLLM server's --max-num-seqs value; recorded and validated.",
    )
    parser.add_argument(
        "--allow-queued-load",
        action="store_true",
        help="Allow concurrency above server capacity and label it as queued load.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.limit, args.max_tokens, args.concurrency,
           args.server_max_num_seqs) <= 0 or args.warmup < 0 or args.timeout <= 0:
        parser.error(
            "limit, max-tokens, concurrency, server-max-num-seqs and timeout "
            "must be positive; warmup cannot be negative"
        )
    if args.concurrency > args.server_max_num_seqs and not args.allow_queued_load:
        parser.error(
            "concurrency exceeds server-max-num-seqs; increase the vLLM server "
            "capacity or pass --allow-queued-load for an explicitly queued test"
        )
    if args.base_model == args.adapter_model:
        parser.error("base-model and adapter-model must be different served model names")
    examples = load_examples(args.dataset, args.limit)
    output = args.output or DEFAULT_OUTPUT_DIR / (
        "base_vs_adapter_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".jsonl"
    )
    summary_path = output.with_suffix(".summary.json")
    existing_outputs = [path for path in (output, summary_path) if path.exists()]
    if existing_outputs:
        raise FileExistsError(
            "Refusing to overwrite benchmark output: "
            + ", ".join(str(path) for path in existing_outputs)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key,
                    timeout=args.timeout, max_retries=0)
    names = {str(item.id) for item in client.models.list().data}
    for name in (args.base_model, args.adapter_model):
        if name not in names:
            raise ValueError(f"Model {name!r} is absent from /v1/models: {sorted(names)}")
    report = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "base_url": args.base_url,
        "concurrency": args.concurrency,
        "server_max_num_seqs": args.server_max_num_seqs,
        "load_mode": (
            "parallel_capacity"
            if args.concurrency <= args.server_max_num_seqs
            else "queued"
        ),
        "max_tokens": args.max_tokens,
        "models": {},
    }
    for model in (args.base_model, args.adapter_model):
        for _ in range(args.warmup):
            warmup = run_one(client, model, examples[0], args.max_tokens)
            if warmup["error"]:
                raise RuntimeError(f"Warmup for {model} failed: {warmup['error']}")

    with output.open("x", encoding="utf-8") as stream:
        for model in (args.base_model, args.adapter_model):
            started = perf_counter()
            results = []
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = [pool.submit(run_one, client, model, example, args.max_tokens)
                           for example in examples]
                for future in as_completed(futures):
                    result = future.result()
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    stream.flush()
                    results.append(result)
            elapsed = perf_counter() - started
            report["models"][model] = summarize(results, elapsed)
            print(model, json.dumps(report["models"][model], indent=2))
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print("Predictions:", output)
    print("Summary:", summary_path)
    total_errors = sum(item["errors"] for item in report["models"].values())
    if total_errors:
        print(f"Benchmark completed with {total_errors} request error(s).")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
