"""Opt-in HTTP acceptance and separate fresh/cache benchmarks with one durable API cap."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from threading import Lock, Thread
from time import monotonic, sleep
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="evals/query_varieties_regression.json")
    parser.add_argument("--ids", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-version", required=True)
    parser.add_argument("--phase", choices=("plan", "acceptance", "performance"), default="plan")
    parser.add_argument("--approved-api-calls", type=int, default=0)
    parser.add_argument('--max-new-api-calls', type=int,
        help='Optional smaller per-run cap; preserves the shared ledger and remaining approval.')
    parser.add_argument("--additional-approved-api-calls", type=int, default=0,
        help="Explicitly approved extension to an existing ledger; never resets prior usage.")
    parser.add_argument("--fresh-repeats", type=int, default=3)
    parser.add_argument("--cache-repeats", type=int, default=3)
    parser.add_argument("--request-deadline", type=float)
    parser.add_argument("--provider-timeout", type=float)
    parser.add_argument("--quality-review", type=Path)
    return parser.parse_args()


def main(args):
    cases = [row for row in json.loads(Path(args.dataset).read_text(encoding="utf-8")) if row["id"] in args.ids]
    if {row["id"] for row in cases} != set(args.ids):
        raise ValueError("Every requested ID must exist in the dataset.")
    if args.fresh_repeats < 1 or args.cache_repeats < 1:
        raise ValueError("Repeat counts must be positive.")
    additional = getattr(args, "additional_approved_api_calls", 0)
    maximum_new = getattr(args, 'max_new_api_calls', None)
    if maximum_new is not None and maximum_new <= 0:
        raise ValueError('The per-run request cap must be positive.')
    if additional < 0 or additional and not args.output.exists():
        raise ValueError("Additional approval requires an existing ledger and a non-negative count.")
    if args.phase == "plan" or args.approved_api_calls <= 0:
        print(json.dumps({"provider_requests": 0, "cases": cases,
            "fresh_samples_per_case": args.fresh_repeats, "cache_repeats_per_case": args.cache_repeats,
            "live_requires": "Explicit approval for a shared hard cap; performance additionally requires a citation/coverage review."}, indent=2))
        return
    os.environ.update(LANGSMITH_TRACING="false", FINRAG_OPENAI_MAX_RETRIES="0", FINRAG_ANSWER_CACHE_MAX_ENTRIES="0")
    for name, value in (("FINRAG_REQUEST_DEADLINE_SECONDS", args.request_deadline),
                        ("FINRAG_OPENAI_REQUEST_TIMEOUT_SECONDS", args.provider_timeout)):
        if value is not None:
            if not 0 < value < float("inf"):
                raise ValueError(f"{name} must be finite and positive.")
            os.environ[name] = str(value)

    import httpx
    import requests
    import uvicorn
    from backend.main import create_app
    from evals.evaluate_ragas import _generate_row, summarize_generation
    from evals.http_client import HTTPChatService
    from src.app.answer_cache import AnswerCache
    from src.config import ANSWER_CACHE_TTL_SECONDS, EVIDENCE_MODEL, LLM_BASE_URL, LLM_MODEL, REQUEST_DEADLINE_SECONDS, runtime_signature
    from src.request_runtime import RequestAborted
    from src.telemetry import _ACTIVE_STAGE

    revision = hashlib.sha256(b"".join(Path(name).read_bytes() for name in (
        "src/agent/graph.py", "src/agent/tools.py", "src/generation/evidence_answer.py")) + json.dumps([LLM_MODEL, EVIDENCE_MODEL]).encode("utf-8")).hexdigest()
    ledger = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {
        "limit": args.approved_api_calls, "requests": [], "results": [], "index_version": args.index_version,
        "runtime": runtime_signature(), "settings": {name: os.environ.get(name) for name in
            ("FINRAG_OPENAI_MAX_RETRIES", "FINRAG_REQUEST_DEADLINE_SECONDS", "FINRAG_OPENAI_REQUEST_TIMEOUT_SECONDS")}}
    if ledger["limit"] != args.approved_api_calls or ledger["index_version"] != args.index_version:
        raise ValueError("An existing ledger's allowance and index cannot be reset or replaced.")
    lock = Lock()
    active = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, args.output)

    if args.phase == "performance":
        review = json.loads(args.quality_review.read_text(encoding="utf-8")) if args.quality_review else {}
        if not all(review.get(str(case["id"]), {}).get("passed") is True
                   and review[str(case["id"])].get("source_revision") == revision for case in cases):
            raise ValueError("Review actual citations and every comparison dimension before performance testing.")
        ledger["quality_review"] = review

    if additional:
        ledger.setdefault("allowance_extensions", []).append({"previous_limit": ledger["limit"],
            "additional_requests": additional, "used_at_extension": len(ledger["requests"])})
        ledger["limit"] += additional
        save()

    run_limit = min(ledger['limit'], len(ledger['requests']) + maximum_new) if maximum_new is not None else ledger['limit']
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    hosts = {httpx.URL(LLM_BASE_URL or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).host,
             httpx.URL(os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).host, "api.cohere.com"}
    send = httpx.Client.send
    def guarded_send(client, request, *values, **kwargs):
        if request.url.host == "127.0.0.1" and request.url.port == port and request.url.path == "/api/chat":
            return send(client, request, *values, **kwargs)
        if request.url.host not in hosts:
            raise RequestAborted("This acceptance allowance does not authorize another network destination.")
        with lock:
            if len(ledger["requests"]) >= run_limit:
                raise RequestAborted("The shared or per-run API request cap is exhausted.")
            record = {"number": len(ledger["requests"]) + 1, "host": request.url.host, "path": request.url.path,
                      "stage": _ACTIVE_STAGE.get(), **active}
            ledger["requests"].append(record)
            save()
        try:
            response = send(client, request, *values, **kwargs)
            record["status"] = response.status_code
            if response.status_code >= 400:
                error = response.json().get("error", {})
                record["provider_error"] = {key: str(error.get(key) or "")[:1000] for key in ("type", "code", "param", "message")}
            if request.url.path.endswith(("/chat/completions", "/embeddings")) and response.status_code < 400:
                body = response.json()
                record.update(model=body.get("model"), usage=body.get("usage"))
                if request.url.path.endswith("/chat/completions"):
                    capture = {"number": record["number"], **active, "request": json.loads(request.content),
                               "response": {key: body.get(key) for key in ("choices", "model", "usage")}}
                    with lock, args.output.with_suffix(".capture.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(capture, ensure_ascii=False) + "\n")
            return response
        except Exception as error:
            record["error_type"] = type(error).__name__
            raise
        finally:
            with lock:
                save()

    def blocked(*values, **kwargs):
        raise RequestAborted("Judging, tracing and additional transports are outside this run.")

    with patch.object(httpx.Client, "send", guarded_send), patch.object(httpx.AsyncClient, "send", blocked), \
         patch.object(requests.Session, "send", blocked):
        app = create_app(index_versions=[args.index_version], allow_traces=True, cors_origins=[])
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
        thread = Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
        thread.start()
        try:
            started = monotonic()
            while not server.started:
                if not thread.is_alive() or monotonic() - started > REQUEST_DEADLINE_SECONDS:
                    raise RuntimeError("The local acceptance API did not start.")
                sleep(0.05)
            adapter = HTTPChatService(base_url, args.index_version, timeout=REQUEST_DEADLINE_SECONDS)
            def measure(case, phase):
                if len(ledger["requests"]) >= run_limit and phase != "cache_repeat":
                    raise RequestAborted("No approved API allowance remains.")
                active.update(case=case["id"], phase=phase)
                row = _generate_row(adapter, case)
                row["phase"] = phase
                row["source_revision"] = revision
                row["evidence_model"] = EVIDENCE_MODEL
                ledger["results"].append(row)
                with lock:
                    ledger["summary_by_phase"] = {key: summarize_generation([r for r in ledger["results"] if r["phase"] == key])
                        for key in {r["phase"] for r in ledger["results"]}}
                    save()
                print(json.dumps({"id": row["id"], "phase": phase, "decision": row["decision"],
                    "requests_used": len(ledger["requests"]), "client_latency_ms": row.get("client_latency_ms")}), flush=True)
                return row

            try:
                for case in cases:
                    if args.phase == "acceptance":
                        measure(case, "acceptance")
                        continue
                    prior = [r for r in ledger["results"] if r["id"] == case["id"] and r["phase"] in {"acceptance", "fresh"}
                             and r["passed"] and r.get("source_revision") == revision]
                    for _ in range(max(0, args.fresh_repeats - len(prior))):
                        if not measure(case, "fresh")["passed"]:
                            raise RequestAborted("Fresh-query quality regressed; stop before cache benchmarking.")
                if args.phase == "performance":
                    for version in app.state.service_manager.index_versions:
                        service = app.state.service_manager.get(version)
                        service.answer_cache = AnswerCache(max(len(cases), 1), ANSWER_CACHE_TTL_SECONDS)
                    for case in cases:
                        population = measure(case, "cache_population")
                        if not population["passed"]:
                            raise RequestAborted("Cache population did not produce an accepted answer.")
                        for _ in range(args.cache_repeats):
                            row = measure(case, "cache_repeat")
                            if not row["passed"] or not row.get("trace", {}).get("answer_cache", {}).get("hit"):
                                raise RequestAborted("A cache repeat was not a validated cache hit.")
                            if any(row.get(key) != population.get(key) for key in ("answer", "retrieved_sources", "retrieved_contexts")):
                                raise RequestAborted("A cache repeat changed its answer or supporting sources.")
            finally:
                adapter.close()
        except RequestAborted as error:
            ledger["stopped"] = str(error)
            print(ledger["stopped"], flush=True)
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            listener.close()
            with lock:
                save()


if __name__ == "__main__":
    main(arguments())
