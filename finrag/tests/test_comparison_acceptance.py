"""Exercise the live harness with local HTTP and simulated provider responses only."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from backend.main import create_app
from evals.accept_comparisons import main
from src.schemas import ChatResult, Decision, Scope
from src.telemetry import UsageCallbackHandler


def args(tmp_path, phase="acceptance"):
    dataset = tmp_path / "cases.json"
    dataset.write_text(json.dumps([dict(id=i, category="comparison", question=f"Report ACME revenue for FY2025, case {i}.",
        reference="Revenue was 100.", scope=dict(tickers=["ACME"], years=["2025"], doc_types=["10K"])) for i in (1, 2)]))
    return SimpleNamespace(dataset=str(dataset), ids=[1, 2], output=tmp_path / "ledger.json", index_version="test-index",
        phase=phase, approved_api_calls=3, fresh_repeats=3, cache_repeats=3, request_deadline=None,
        provider_timeout=None, quality_review=None)


def test_acceptance_counts_requests_before_transport_and_keeps_a_single_durable_cap(tmp_path, monkeypatch):
    for name, value in (("LANGSMITH_TRACING", "false"), ("FINRAG_OPENAI_MAX_RETRIES", "0"), ("FINRAG_ANSWER_CACHE_MAX_ENTRIES", "0")):
        monkeypatch.setenv(name, value)
    sent = []
    def respond(request):
        sent.append(request)
        return httpx.Response(200, json={"model": "gpt-4o-mini", "choices": [],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}})
    class FakeService:
        def __init__(self, **kwargs):
            self.available_keys = {("ACME", "2025", "10K")}
            self.manifest = {"status": "complete", "index_version": "test-index"}
        def ask(self, question, **kwargs):
            with httpx.Client(transport=httpx.MockTransport(respond)) as client:
                for _ in range(2):
                    response = client.post("https://api.openai.com/v1/chat/completions", json={"model": "gpt-4o-mini", "messages": []})
                    UsageCallbackHandler("gpt-4o-mini").on_llm_end(SimpleNamespace(llm_output=response.json(), generations=[]))
            return ChatResult(Decision.ANSWERED, "Revenue was 100. [S1]", Scope(("ACME",), ("2025",), "10K"),
                [dict(id="S1", ticker="ACME", fiscal_year="2025", doc_type="10K", evidence_text="Revenue was 100.")])
        def close(self):
            pass
    settings = args(tmp_path)
    with patch("backend.main.create_app", side_effect=lambda **kwargs: create_app(service_factory=FakeService, **kwargs)):
        main(settings)
    ledger = json.loads(settings.output.read_text(encoding="utf-8"))
    assert len(sent) == len(ledger["requests"]) == ledger["limit"] == 3
    assert ledger["results"][0]["passed"] and not ledger["results"][1]["passed"]
    assert ledger["results"][0]["llm_usage"]["llm_calls"] == 2
    assert ledger["results"][1]["llm_usage"] is None
    settings.approved_api_calls = 4
    with pytest.raises(ValueError, match="cannot be reset"):
        main(settings)
    assert len(sent) == 3
    settings.approved_api_calls = 3
    settings.additional_approved_api_calls = 2
    with patch("backend.main.create_app", side_effect=lambda **kwargs: create_app(service_factory=FakeService, **kwargs)):
        main(settings)
    extended = json.loads(settings.output.read_text(encoding="utf-8"))
    assert len(sent) == len(extended["requests"]) == extended["limit"] == 5
    assert extended["allowance_extensions"] == [dict(previous_limit=3, additional_requests=2, used_at_extension=3)]
    assert extended["results"][:2] == ledger["results"]
    with pytest.raises(ValueError, match="cannot be reset"):
        main(settings)
    assert len(sent) == 5


def test_performance_phase_requires_citation_review_before_starting_any_service(tmp_path):
    settings = args(tmp_path, "performance")
    with patch("backend.main.create_app", side_effect=AssertionError("No service is needed before quality passes.")):
        with pytest.raises(ValueError, match="Review actual citations"):
            main(settings)
    assert not settings.output.exists()


def test_per_run_cap_preserves_unused_shared_allowance(tmp_path):
    settings = args(tmp_path)
    settings.max_new_api_calls = 1
    sent = []
    class Service:
        def __init__(self, **kwargs):
            self.available_keys = {('ACME', '2025', '10K')}
            self.manifest = {'status': 'complete', 'index_version': 'test-index'}
        def ask(self, question, **kwargs):
            def respond(request):
                sent.append(request)
                return httpx.Response(200, json={})
            with httpx.Client(transport=httpx.MockTransport(respond)) as client:
                for _ in range(2):
                    client.post('https://api.openai.com/v1/chat/completions', json={})
            raise AssertionError('The second request must have been blocked.')
        def close(self):
            pass
    with patch('backend.main.create_app', side_effect=lambda **kwargs: create_app(service_factory=Service, **kwargs)):
        main(settings)
    ledger = json.loads(settings.output.read_text(encoding='utf-8'))
    assert len(sent) == len(ledger['requests']) == 1
    assert ledger['limit'] == 3
