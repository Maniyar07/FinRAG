import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from time import monotonic, sleep
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from backend.main import create_app
from evals.evaluate_ragas import _fresh_score_row, _generate_row, run, summarize_generation
from evals.http_client import HTTPChatService
from src.schemas import ChatResult, Decision, Scope
from src.telemetry import UsageCallbackHandler


def record(key=1):
    return dict(id=key, category="numeric", question=f"Report ACME revenue for FY2025, case {key}.",
        reference="Revenue was 100.", scope=dict(tickers=["ACME"], years=["2025"], doc_types=["10K"]))


class FakeService:
    def __init__(self, **kwargs):
        self.available_keys = {("ACME", "2025", "10K")}
        self.manifest = {"status": "complete", "index_version": "test-index"}
        self.started, self.release = Event(), Event()
        self.release.set()

    def ask(self, question, **kwargs):
        self.started.set()
        assert self.release.wait(5)
        UsageCallbackHandler("gpt-4o-mini").on_llm_end(SimpleNamespace(llm_output={"model": "gpt-4o-mini",
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}}, generations=[]))
        return ChatResult(Decision.ANSWERED, "Revenue was 100. [S1]", Scope(("ACME",), ("2025",), "10K"),
            [{"id": "S1", "ticker": "ACME", "fiscal_year": "2025", "doc_type": "10K",
              "source": "filing.pdf", "evidence_text": "Revenue was 100."}],
            trace={"answer_validation": {"valid": True}, "agent": {"tool_results": [{"name": "search",
                "error": "An earlier source attempt was recovered.", "resolved_in_final_answer": True}]}})

    def close(self):
        pass


def app(service):
    return create_app(service_factory=lambda **kwargs: service, index_versions=["test-index"],
        allow_traces=True, cors_origins=[])


def test_http_evaluation_retains_server_usage_evidence_and_recovered_error_semantics():
    with TestClient(app(FakeService())) as client:
        service = HTTPChatService("http://testserver", "test-index", timeout=1, client=client)
        row = _generate_row(service, record())
    assert row["passed"] and row["http_status"] == 200
    assert row["retrieved_contexts"] == ["Revenue was 100."]
    assert row["llm_usage"]["llm_calls"] == 1 and row["llm_usage"]["total_tokens"] == 110
    assert row["llm_usage"]["total_token_cost_usd"] > 0
    assert row["client_latency_ms"] >= row["server_latency_ms"]
    assert row["queue_latency_ms"] >= 0
    assert row["errors"][0]["resolved_in_final_answer"]
    assert _fresh_score_row(row)["client_latency_ms"] == row["client_latency_ms"]


def test_concurrent_requests_measure_the_real_api_worker_queue():
    backend = FakeService()
    backend.release.clear()
    with patch("backend.main.MAX_CONCURRENT_REQUESTS", 1), TestClient(app(backend)) as client:
        service = HTTPChatService("http://testserver", "test-index", timeout=2, concurrency=2, client=client)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(_generate_row, service, record(1))
            assert backend.started.wait(3)
            second = pool.submit(_generate_row, service, record(2))
            try:
                deadline = monotonic() + 3
                while len(client.app.state.pending_workers) < 2 and monotonic() < deadline:
                    sleep(0.005)
                assert len(client.app.state.pending_workers) == 2
                sleep(0.02)
            finally:
                backend.release.set()
            rows = [first.result(5), second.result(5)]
    assert all(row["passed"] and row["llm_usage"]["llm_calls"] == 1 for row in rows)
    assert max(row["queue_latency_ms"] for row in rows) >= 10
    assert all(row["client_latency_ms"] >= row["server_latency_ms"] for row in rows)
    summary = summarize_generation(rows)
    assert summary["latency_ms"]["queue_latency_ms"]["p95"] >= 10
    assert summary["latency_ms"]["client_latency_ms"]["median"] is not None


def test_http_errors_preserve_latency_without_inventing_server_usage():
    with httpx.Client(base_url="http://testserver/", transport=httpx.MockTransport(
            lambda request: httpx.Response(503, json={"error": {"code": "unavailable"}}))) as client:
        service = HTTPChatService("http://testserver", "test-index", timeout=1, client=client)
        row = _generate_row(service, record())
    assert not row["passed"] and row["http_status"] == 503
    assert row["client_latency_ms"] >= 0 and row["llm_usage"] is None
    assert not summarize_generation([row])["token_cost"]["complete"]


def test_http_run_uses_the_existing_api_and_checkpoints_its_transport_and_costs(tmp_path):
    with TestClient(app(FakeService())) as client:
        service = HTTPChatService("http://testserver", "test-index", timeout=1, client=client)
        args = SimpleNamespace(dataset="unused.json", output=str(tmp_path / "run.json"), ids=None, limit=None,
            validate_only=False, generate_only=True, score_input=None, index_version="test-index",
            judge_model="unused", api_base_url="http://testserver", concurrency=2, http_timeout=2)
        with patch("evals.evaluate_ragas.load_gold", return_value=[record(1), record(2)]), \
             patch("evals.http_client.HTTPChatService", return_value=service), \
             patch("src.app.chat_service.ChatService", side_effect=AssertionError("HTTP evaluation must not start a second RAG service.")):
            assert asyncio.run(run(args)) == 0
    saved = json.loads(Path(args.output).read_text())
    assert saved["run"]["evaluation_path"] == "HTTP POST /api/chat"
    assert saved["summary"]["token_cost"]["complete"]
    assert saved["summary"]["latency_ms"]["client_latency_ms"]["count"] == 2
    assert [row["id"] for row in saved["results"]] == [1, 2]
