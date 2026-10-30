"""Evaluate the real API without importing or running a second local RAG service."""
import json
import os
from threading import local

import httpx

from backend.models import ChatResponse
from src.schemas import ChatResult, Decision, Scope


class HTTPChatService:
    evaluation_transport = "http"

    def __init__(self, base_url: str, index_version: str, *, timeout: float, concurrency: int = 1, client=None):
        url = httpx.URL(base_url)
        if url.scheme not in {"http", "https"} or not url.host or url.username or url.password or url.query or url.fragment:
            raise ValueError("Evaluation API base URL must be HTTP(S), without credentials, query or fragment.")
        headers = json.loads(os.getenv("FINRAG_EVAL_HTTP_HEADERS", "{}"))
        if not isinstance(headers, dict) or any(not isinstance(value, str) for value in headers.values()):
            raise ValueError("FINRAG_EVAL_HTTP_HEADERS must be a JSON object of header strings.")
        self.client = client or httpx.Client(base_url=base_url.rstrip("/") + "/", timeout=timeout,
            headers=headers, follow_redirects=False, limits=httpx.Limits(max_connections=concurrency))
        self.index_version = index_version
        self._local = local()

    @property
    def request_metrics(self):
        return getattr(self._local, "metrics", None)

    @property
    def http_status(self):
        return getattr(self._local, "status", None)

    def ask(self, question: str) -> ChatResult:
        self._local.metrics = self._local.status = None
        response = self.client.post("api/chat", json={"index_version": self.index_version,
            "question": question, "include_trace": True})
        self._local.status = response.status_code
        response.raise_for_status()
        result = ChatResponse.model_validate(response.json())
        self._local.metrics = result.metrics.model_dump()
        scope = result.scope
        trace = {**(result.trace or {}), "request_id": result.request_id, "trace_id": result.trace_id,
                 "index_version": self.index_version}
        return ChatResult(decision=Decision(result.decision), answer=result.answer,
            scope=Scope(tuple(scope.tickers), tuple(scope.years), scope.doc_type,
                tuple(scope.requested_groups), tuple(scope.required_doc_types)),
            sources=[source.model_dump() for source in result.sources], trace=trace,
            inherited_fields=tuple(result.inherited_fields))

    def close(self):
        self.client.close()
