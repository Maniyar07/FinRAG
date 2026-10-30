from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.app.answer_cache import AnswerCache
from src.app.chat_service import ChatService
from src.request_runtime import RequestAborted, RequestControl, request_control
from src.schemas import ChatResult, Decision, Scope


def answer(decision=Decision.ANSWERED):
    return ChatResult(decision, "Revenue was 100 million USD [S1].", Scope(("ACME",), ("2024",), "10K"),
        [{"id": "S1", "ticker": "ACME", "fiscal_year": "2024", "doc_type": "10K", "evidence_text": "Revenue 100"}],
        trace={"trace_id": "origin", "answer_validation": {"valid": True}})


def test_cache_reuses_only_validated_answers_and_copies_mutable_evidence():
    cache, compute = AnswerCache(2, 60), Mock(side_effect=answer)
    first, hit = cache.get_or_compute("scope", compute)
    assert not hit
    first.sources[0]["evidence_text"] = "modified by caller"
    second, hit = cache.get_or_compute("scope", compute)
    assert hit and second.sources[0]["evidence_text"] == "Revenue 100"
    compute.assert_called_once()


@pytest.mark.parametrize("decision", [Decision.INSUFFICIENT_EVIDENCE, Decision.VALIDATION_FAILED, Decision.ERROR, Decision.CLARIFY])
def test_partial_or_failed_work_is_not_cached(decision):
    cache, compute = AnswerCache(2, 60), Mock(side_effect=lambda: answer(decision))
    cache.get_or_compute("scope", compute)
    assert not cache.get_or_compute("scope", compute)[1]
    assert compute.call_count == 2


def test_invalid_citations_are_not_cached_even_if_trace_claims_validation():
    cache = AnswerCache(2, 60)
    result = answer()
    result.sources.clear()
    compute = Mock(return_value=result)
    cache.get_or_compute("scope", compute)
    assert not cache.get_or_compute("scope", compute)[1]


def test_expired_and_evicted_answers_are_recomputed():
    cache, compute = AnswerCache(1, 10), Mock(side_effect=answer)
    with patch("src.app.answer_cache.monotonic", return_value=100) as clock:
        cache.get_or_compute("first", compute)
        cache.get_or_compute("second", compute)
        assert not cache.get_or_compute("first", compute)[1]
        clock.return_value = 111
        assert not cache.get_or_compute("first", compute)[1]
    assert compute.call_count == 4


def test_identical_concurrent_requests_share_work_without_blocking_other_scopes():
    cache, started, release = AnswerCache(2, 60), Event(), Event()
    def compute():
        started.set()
        assert release.wait(5)
        return answer()
    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(cache.get_or_compute, "same", compute)
        assert started.wait(5)
        second = pool.submit(cache.get_or_compute, "same", compute)
        assert not pool.submit(cache.get_or_compute, "different", answer).result(5)[1]
        release.set()
        assert not first.result(5)[1] and second.result(5)[1]


def test_owner_cancellation_does_not_poison_later_requests():
    cache = AnswerCache(1, 60)
    with pytest.raises(RequestAborted):
        cache.get_or_compute("same", Mock(side_effect=RequestAborted()))
    assert not cache.get_or_compute("same", answer)[1]
    control = RequestControl(float("inf"))
    with request_control(control):
        control.cancelled.set()
        with pytest.raises(RequestAborted):
            cache.get_or_compute("same", answer)


def test_cancelled_waiter_does_not_cancel_or_discard_the_shared_answer():
    cache, started, release = AnswerCache(1, 60), Event(), Event()
    control, waiting = RequestControl(float("inf")), Event()
    def compute():
        started.set()
        assert release.wait(5)
        return answer()
    def waiter():
        with request_control(control):
            waiting.set()
            return cache.get_or_compute("same", answer)
    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(cache.get_or_compute, "same", compute)
        assert started.wait(5)
        pending = pool.submit(waiter)
        assert waiting.wait(5)
        control.cancelled.set()
        with pytest.raises(RequestAborted):
            pending.result(5)
        release.set()
        assert not owner.result(5)[1]
    assert cache.get_or_compute("same", answer)[1]


def test_service_cache_preserves_request_ids_history_and_conversation_isolation():
    service = ChatService.__new__(ChatService)
    service.available_keys = {("MSFT", "2025", "10K")}
    service.retriever = SimpleNamespace(reranker=None)
    service.semantic_parser = None
    service.logger = SimpleNamespace(info=Mock())
    service.answer_cache = AnswerCache(8, 60)
    service.process_query = Mock(side_effect=lambda *args, **kwargs: answer())
    question = "What was Microsoft's revenue in its 2025 annual 10-K?"
    service.ask(question, thread_id="one", request_id="first", trace_id="trace1")
    hit = service.ask(question, thread_id="one", request_id="second", trace_id="trace2")
    assert hit.trace["request_id"] == "second" and hit.trace["trace_id"] == "trace2"
    assert hit.trace["answer_cache"] == {"hit": True, "source_trace_id": "origin"}
    assert service.process_query.call_count == 1
    service.ask(question, thread_id="two")
    service.ask(question, thread_id="one", history=[{"role": "user", "content": "Use GAAP figures."}])
    assert service.process_query.call_count == 3
    other_index = AnswerCache(8, 60)
    assert not other_index.get_or_compute("same", answer)[1]
