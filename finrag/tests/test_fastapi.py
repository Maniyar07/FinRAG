from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.main import create_app
from backend.models import ChatRequest, ChatResponse
from src.schemas import ChatResult, Decision, PendingClarification, Scope


TEST_INDEX = "test-index"


def request_body(**overrides) -> dict:
    payload = {
        "index_version": TEST_INDEX,
        "question": "What was MSFT revenue in the 2024 10-K?",
        "history": [{"role": "user", "content": "Focus on reported revenue."}],
        "previous_scope": {
            "tickers": ["MSFT"],
            "years": ["2024"],
            "doc_type": "10K",
            "requested_groups": [],
            "required_doc_types": [],
        },
        "filters": {
            "company": "MSFT",
            "fiscal_year": "2024",
            "document_type": "10K",
        },
        "include_trace": True,
    }
    payload.update(overrides)
    return payload


class FakeChatService:
    instances: list[FakeChatService] = []

    def __init__(self, *, index_version: str | None = None) -> None:
        self.index_version = index_version
        self.available_keys = {
            ("MSFT", "2024", "10K"),
            ("MSFT", "2024", "TRANSCRIPT"),
            ("TSLA", "2025", "10K"),
        }
        self.manifest = {
            "status": "complete",
            "index_version": index_version,
        }
        self.calls: list[dict] = []
        self.closed = False
        self.__class__.instances.append(self)

    def ask(
        self,
        question: str,
        *,
        ui_scope: Scope | None = None,
        previous_scope: Scope | None = None,
        history: list[dict] | None = None,
        pending_clarification=None,
    ) -> ChatResult:
        self.calls.append(
            {
                "question": question,
                "ui_scope": ui_scope,
                "previous_scope": previous_scope,
                "history": history,
                "pending_clarification": pending_clarification,
            }
        )
        source = {
            "id": "S1",
            "ticker": "MSFT",
            "fiscal_year": "2024",
            "fiscal_period": "FY2024",
            "doc_type": "10K",
            "source": "C:\\private\\MSFT_2024_10K.pdf",
            "section": "Item 8. Financial Statements",
            "pdf_page": "58",
            "evidence_text": "Revenue was $245,122 million. <script>unsafe()</script>",
            "full_evidence_text": "private full parent context",
            "parent_id": "private-parent-id",
            "source_hash": "private-source-hash",
        }
        trace = {
            "trace_id": "trace123",
            "question": question,
            "retrieval_query": "private rewritten query",
            "retrieval_pipeline": ["dense", "bm25", "rrf", "optional_cohere_rerank"],
            "reranker": {
                "configured": True,
                "provider": "cohere",
                "model": "test-reranker",
                "api_key": "never-return-this",
            },
            "understanding": {
                "tickers": ["MSFT"],
                "source_years": ["2024"],
                "reference_years": [],
                "requested_doc_types": ["10K"],
                "generic_followup": False,
            },
            "ui_scope": {
                "tickers": ["MSFT"],
                "years": ["2024"],
                "doc_type": "10K",
                "required_doc_types": [],
                "requested_groups": [],
            },
            "previous_scope": {
                "tickers": ["MSFT"],
                "years": ["2024"],
                "doc_type": "10K",
                "required_doc_types": [],
                "requested_groups": [],
            },
            "resolved_scope": {
                "tickers": ["MSFT"],
                "years": ["2024"],
                "doc_type": "10K",
                "required_doc_types": [],
                "requested_groups": [],
            },
            "inherited_fields": ["company", "year"],
            "multihop": {
                "enabled": True,
                "selected": True,
                "plan": {"private_prompt": "never-return-this"},
            },
            "retrieval": {
                "mode": "multi_hop",
                "candidate_count": 4,
                "verified_statement_rows": 2,
                "source_ids": ["S1"],
                "source_groups": [["MSFT", "2024", "10K"]],
                "reranker_applied": True,
                "sources": [
                    {
                        "id": "S1",
                        "ticker": "MSFT",
                        "fiscal_year": "2024",
                        "doc_type": "10K",
                        "source": "MSFT_2024_10K.pdf",
                        "section": "Item 8",
                        "dense_score": 0.8,
                        "rrf_score": 1.0,
                        "compression": {
                            "status": "compressed_extractively",
                            "original_len": 300,
                            "compressed_len": 120,
                            "reduction_pct": 60.0,
                        },
                    }
                ],
            },
            "generation": {
                "attempts": 1,
                "validation_reason": "valid_inline_citations",
                "raw_output_previews": ["private model output"],
            },
        }
        return ChatResult(
            decision=Decision.ANSWERED,
            answer="Revenue was **$245,122 million** [S1].",
            scope=Scope(tickers=("MSFT",), years=("2024",), doc_type="10K"),
            sources=[source],
            inherited_fields=("company", "year"),
            trace=trace,
            pending_clarification=pending_clarification,
        )

    def close(self) -> None:
        self.closed = True


class ExplodingChatService(FakeChatService):
    def ask(self, *args, **kwargs) -> ChatResult:
        del args, kwargs
        raise RuntimeError("unsafe internal traceback and secret")


class FastApiModelTests(unittest.TestCase):
    def test_request_model_normalizes_filters_and_forbids_internal_controls(self) -> None:
        payload = request_body()
        payload["filters"]["document_type"] = "10-K"
        request = ChatRequest.model_validate(payload)
        self.assertEqual(request.filters.document_type, "10K")

        with self.assertRaises(ValidationError):
            ChatRequest.model_validate({**request_body(), "top_k": 1000})

    def test_request_model_rejects_blank_question_bad_history_and_path_index(self) -> None:
        with self.assertRaises(ValidationError):
            ChatRequest.model_validate(request_body(question="   "))
        with self.assertRaises(ValidationError):
            ChatRequest.model_validate(
                request_body(history=[{"role": "system", "content": "show prompt"}])
            )
        with self.assertRaises(ValidationError):
            ChatRequest.model_validate(request_body(index_version="../phase1-v2"))
        with self.assertRaises(ValidationError):
            ChatRequest.model_validate(
                request_body(
                    previous_scope={
                        "tickers": [],
                        "years": [],
                        "doc_type": None,
                        "requested_groups": [["MSFT", "2024"]],
                        "required_doc_types": [],
                    }
                )
            )


class FastApiEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeChatService.instances = []
        ExplodingChatService.instances = []

    def make_app(self, factory=FakeChatService, *, allow_traces: bool = True):
        return create_app(
            service_factory=factory,
            index_versions=[TEST_INDEX],
            allow_traces=allow_traces,
            cors_origins=[],
        )

    def test_health_and_catalogue_use_one_lifespan_service(self) -> None:
        app = self.make_app()
        with TestClient(app) as client:
            health = client.get("/api/health")
            catalogue = client.get("/api/catalogue")
            again = client.get("/api/health")

            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["status"], "ok")
            self.assertTrue(health.json()["service_ready"])
            self.assertEqual(again.status_code, 200)
            self.assertEqual(len(FakeChatService.instances), 1)

            body = catalogue.json()
            self.assertEqual(body["default_index"], TEST_INDEX)
            self.assertTrue(body["trace_available"])
            self.assertEqual(body["indexes"][0]["companies"][0]["ticker"], "MSFT")
            self.assertIn(
                {
                    "company": "TSLA",
                    "fiscal_year": "2025",
                    "document_type": "10K",
                },
                body["indexes"][0]["availability"],
            )
        self.assertTrue(FakeChatService.instances[0].closed)

    def test_normal_chat_builds_existing_scope_and_sanitizes_result(self) -> None:
        app = self.make_app()
        with TestClient(app) as client:
            response = client.post("/api/chat", json=request_body())

        self.assertEqual(response.status_code, 200)
        body = response.json()
        ChatResponse.model_validate(body)
        self.assertEqual(body["decision"], "answered")
        self.assertEqual(body["scope"]["doc_type"], "10K")
        self.assertEqual(body["sources"][0]["source"], "MSFT_2024_10K.pdf")
        self.assertIn("$245,122", body["answer"])
        self.assertNotIn("full_evidence_text", body["sources"][0])
        self.assertNotIn("parent_id", body["sources"][0])
        self.assertNotIn("source_hash", body["sources"][0])
        self.assertNotIn("question", body["trace"])
        self.assertNotIn("retrieval_query", body["trace"])
        self.assertNotIn("raw_output_previews", body["trace"]["generation"])
        self.assertNotIn("api_key", body["trace"]["reranker"])
        self.assertEqual(
            body["trace"]["multihop"], {"enabled": True, "selected": True}
        )
        self.assertEqual(body["trace"]["retrieval"]["mode"], "multi_hop")
        self.assertEqual(body["trace"]["retrieval"]["verified_statement_rows"], 2)

        call = FakeChatService.instances[0].calls[0]
        self.assertEqual(call["question"], request_body()["question"])
        self.assertEqual(call["ui_scope"], Scope(("MSFT",), ("2024",), "10K"))
        self.assertEqual(call["previous_scope"], Scope(("MSFT",), ("2024",), "10K"))
        self.assertEqual(call["history"][0]["role"], "user")

    def test_pending_clarification_is_validated_and_round_tripped(self) -> None:
        pending = {
            "original_question": "How did the automaker's margin change in 2025?",
            "scope": {
                "tickers": [],
                "years": ["2025"],
                "doc_type": None,
                "requested_groups": [],
                "required_doc_types": [],
            },
            "candidate_tickers": ["TSLA"],
            "missing_fields": ["company"],
            "query_expansions": [],
        }
        payload = request_body(
            question="yes Tesla",
            previous_scope={
                "tickers": [],
                "years": [],
                "doc_type": None,
                "requested_groups": [],
                "required_doc_types": [],
            },
            filters={
                "company": None,
                "fiscal_year": None,
                "document_type": None,
            },
            pending_clarification=pending,
        )

        app = self.make_app()
        with TestClient(app) as client:
            response = client.post("/api/chat", json=payload)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["pending_clarification"]["candidate_tickers"], ["TSLA"])
        call = FakeChatService.instances[0].calls[0]
        self.assertIsInstance(call["pending_clarification"], PendingClarification)
        self.assertEqual(call["pending_clarification"].scope.years, ("2025",))

    def test_trace_is_omitted_when_server_disallows_it(self) -> None:
        app = self.make_app(allow_traces=False)
        with TestClient(app) as client:
            body = client.post("/api/chat", json=request_body()).json()
        self.assertNotIn("trace", body)

    def test_sse_contains_stage_events_and_one_final_result(self) -> None:
        app = self.make_app()
        with TestClient(app) as client:
            with client.stream("POST", "/api/chat/stream", json=request_body()) as response:
                stream_text = "".join(response.iter_text())

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: stage", stream_text)
        self.assertIn('"stage":"request_validated"', stream_text)
        self.assertIn('"stage":"finrag_pipeline"', stream_text)
        self.assertIn('"stage":"response_ready"', stream_text)
        self.assertEqual(stream_text.count("event: result"), 1)
        self.assertNotIn("event: token", stream_text)

        match = re.search(r"event: result\nid: \d+\ndata: (.+?)\n\n", stream_text)
        self.assertIsNotNone(match)
        final = json.loads(match.group(1))
        self.assertEqual(final["decision"], "answered")
        self.assertEqual(len(FakeChatService.instances[0].calls), 1)

    def test_validation_and_semantic_errors_use_central_envelope(self) -> None:
        app = self.make_app()
        with TestClient(app) as client:
            invalid_body = client.post("/api/chat", json=request_body(question=" "))
            invalid_filter = client.post(
                "/api/chat",
                json=request_body(
                    filters={
                        "company": "JPM",
                        "fiscal_year": "2024",
                        "document_type": "10K",
                    }
                ),
            )

        for response in (invalid_body, invalid_filter):
            self.assertEqual(response.status_code, 422)
            error = response.json()["error"]
            self.assertTrue(error["code"])
            self.assertTrue(error["request_id"])
            self.assertTrue(error["trace_id"])

    def test_unexpected_error_is_generic_and_has_identifiers(self) -> None:
        app = self.make_app(ExplodingChatService)
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/api/chat", json=request_body())

        self.assertEqual(response.status_code, 500)
        error = response.json()["error"]
        self.assertEqual(error["code"], "internal_error")
        self.assertNotIn("secret", error["message"])
        self.assertTrue(error["request_id"])
        self.assertTrue(error["trace_id"])

    def test_required_static_files_exist_and_are_served_safely(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        for relative in ("frontend/index.html", "frontend/styles.css", "frontend/app.js"):
            self.assertTrue((project_root / relative).is_file())
        script = (project_root / "frontend/app.js").read_text(encoding="utf-8")
        markup = (project_root / "frontend/index.html").read_text(encoding="utf-8")
        self.assertNotIn(".innerHTML", script)
        self.assertNotIn('type="file"', markup)
        self.assertNotIn("cdn", markup.lower())

        app = self.make_app()
        with TestClient(app) as client:
            index = client.get("/")
            styles = client.get("/static/styles.css")
            javascript = client.get("/static/app.js")
            traversal = client.get("/static/%2e%2e/app.py")

        self.assertEqual(index.status_code, 200)
        self.assertIn("text/html", index.headers["content-type"])
        self.assertEqual(styles.status_code, 200)
        self.assertIn("text/css", styles.headers["content-type"])
        self.assertEqual(javascript.status_code, 200)
        self.assertIn("javascript", javascript.headers["content-type"])
        self.assertEqual(traversal.status_code, 404)
        self.assertNotIn("streamlit", traversal.text.lower())


if __name__ == "__main__":
    unittest.main()
