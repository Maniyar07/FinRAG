from __future__ import annotations

import asyncio
from dataclasses import asdict
from threading import Event
from time import monotonic
from types import SimpleNamespace

import httpx
import pytest

from backend.main import _measured_ask
from evals.evaluate_ragas import summarize_generation
from src.request_runtime import RequestAborted, RequestControl, check_request, request_control
from src.schemas import Scope
from src.telemetry import collect_usage, record_provider_response, usage_stage


def test_cancellation_stops_further_work_at_the_next_boundary():
    control = RequestControl(monotonic() + 10)
    with request_control(control):
        check_request()
        control.cancelled.set()
        with pytest.raises(RequestAborted):
            check_request()
    check_request()


def test_cancelled_async_waiter_keeps_slot_until_worker_finishes():
    async def scenario():
        started, release = Event(), Event()
        def ask(*args, **kwargs):
            started.set()
            release.wait(3)
            check_request()
            raise AssertionError("Cancelled work must not continue.")
        limiter, pending = asyncio.Semaphore(1), {}
        task = asyncio.create_task(_measured_ask(SimpleNamespace(ask=ask), "question",
            ui_scope=Scope(), previous_scope=Scope(), history=[], pending_clarification=None,
            limiter=limiter, pending_workers=pending))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert limiter.locked()
            assert len(pending) == 1
        finally:
            release.set()
            await asyncio.gather(*list(pending), return_exceptions=True)
        assert not limiter.locked()
        assert not pending
    asyncio.run(scenario())


def test_http_disconnect_cancels_further_worker_stages():
    async def scenario():
        started, release = Event(), Event()
        def ask(*args, **kwargs):
            started.set()
            release.wait(3)
            check_request()
            raise AssertionError("Disconnected work continued.")
        async def disconnected():
            return started.is_set()
        pending, limiter = {}, asyncio.Semaphore(1)
        task = asyncio.create_task(_measured_ask(SimpleNamespace(ask=ask), "question",
            ui_scope=Scope(), previous_scope=Scope(), history=[], pending_clarification=None,
            limiter=limiter, pending_workers=pending,
            request=SimpleNamespace(is_disconnected=disconnected)))
        try:
            with pytest.raises(asyncio.CancelledError):
                await task
            assert limiter.locked()
        finally:
            release.set()
            await asyncio.gather(*list(pending), return_exceptions=True)
        assert not pending
    asyncio.run(scenario())


def test_judge_usage_and_embedding_usage_are_recorded_separately():
    async def scenario():
        with collect_usage() as usage, usage_stage("ragas_judge"):
            response = httpx.Response(200, request=httpx.Request("POST", "https://example.test/chat/completions"),
                json={"model": "gpt-4o-mini", "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}})
            await record_provider_response(response)
            response = httpx.Response(200, request=httpx.Request("POST", "https://example.test/embeddings"),
                json={"usage": {"prompt_tokens": 7, "total_tokens": 7}})
            await record_provider_response(response)
        snapshot = usage.snapshot(latency_ms=1)
        assert snapshot.llm_calls == 1
        assert snapshot.input_tokens == 100
        assert snapshot.embedding_input_tokens == 7
        assert snapshot.embedding_calls == 1
        assert snapshot.stages["ragas_judge"]["total_tokens"] == 110
        summary = summarize_generation([{"decision": "answered", "passed": True,
                                         "llm_usage": asdict(snapshot)}])
        assert summary["llm_usage"]["stages"]["ragas_judge"]["total_tokens"] == 110
        assert summary["llm_usage"]["embedding_input_tokens"] == 7
    asyncio.run(scenario())
