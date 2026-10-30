r"""Observe FinRAG's real FastAPI routes with a small teaching service.

Run from finrag/: .\.venv\Scripts\python.exe fastapi_learning_lab.py
The service below does not open financial indexes or call model providers.
"""

from __future__ import annotations

import json
import os

# This setting applies only to this lab process, including imports that read .env.
os.environ["LANGSMITH_TRACING"] = "false"

from fastapi.testclient import TestClient

from backend.main import create_app
from src.schemas import ChatResult, Decision


class TeachingChatService:
    """Keep the same interface as ChatService; make its work visible."""

    def __init__(self, *, index_version: str) -> None:
        self.index_version = index_version
        self.available_keys = {("MSFT", "2024", "10K")}
        self.calls = 0
        print(f"  STARTUP: created one service for {index_version}")

    def ask(self, question: str, **kwargs) -> ChatResult:
        self.calls += 1
        print(f"  SERVICE CALL {self.calls}: question={question!r}")
        print(f"  SERVICE INPUT: ui_scope={kwargs['ui_scope']}")
        return ChatResult(
            decision=Decision.ANSWERED,
            answer="Lab response: the validated request reached service.ask().",
            scope=kwargs["ui_scope"],
            trace={"trace_id": kwargs["trace_id"]},
        )

    def close(self) -> None:
        print(f"  SHUTDOWN: closed the service after {self.calls} calls")


def main() -> None:
    app = create_app(
        service_factory=TeachingChatService,
        index_versions=["learning-index"],
        allow_traces=False,
        cors_origins=[],
    )

    # Entering the context runs lifespan startup; leaving it runs shutdown.
    with TestClient(app) as client:
        print("\n1. GET /api/health: uses the service created at startup")
        response = client.get("/api/health")
        print(f"  HTTP {response.status_code}; ready={response.json()['service_ready']}")

        print("\n2. GET /api/catalogue: discover the available index and filters")
        response = client.get("/api/catalogue")
        print(f"  HTTP {response.status_code}; default_index={response.json()['default_index']}")

        payload = {
            "index_version": "learning-index",
            "question": "Explain MSFT's 2024 10-K revenue.",
            "filters": {
                "company": " msft ",
                "fiscal_year": 2024,
                "document_type": "10-k",
            },
        }
        print("\n3. POST /api/chat: observe normalized filters at SERVICE INPUT")
        response = client.post("/api/chat", json=payload)
        result = response.json()
        print(f"  HTTP {response.status_code}; decision={result['decision']}")
        print(f"  PUBLIC ANSWER: {result['answer']}")
        print(f"  RESPONSE HAS X-Request-ID: {'x-request-id' in response.headers}")

        print("\n4. Blank question: model validation rejects it before the route runs")
        response = client.post("/api/chat", json={**payload, "question": " "})
        print(f"  HTTP {response.status_code}; code={response.json()['error']['code']}")
        print("  Notice: no SERVICE CALL was printed for this request.")

        print("\n5. Valid ticker format, unavailable company: route-level validation")
        response = client.post(
            "/api/chat",
            json={**payload, "filters": {"company": "JPM"}},
        )
        print(f"  HTTP {response.status_code}; code={response.json()['error']['code']}")
        print("  Notice: this request also never reached service.ask().")

        print("\n6. Unknown request field: extra='forbid' rejects it")
        response = client.post("/api/chat", json={**payload, "model": "anything"})
        print(f"  HTTP {response.status_code}; code={response.json()['error']['code']}")

        print("\n7. POST /api/chat/stream: read the SSE event frames")
        with client.stream("POST", "/api/chat/stream", json=payload) as response:
            frames = "".join(response.iter_text())
            print(f"  HTTP {response.status_code}; type={response.headers['content-type']}")
        for frame in frames.strip().split("\n\n"):
            event_line = next(
                (line for line in frame.splitlines() if line.startswith("event:")),
                "comment",
            )
            data_line = next(
                (line[6:] for line in frame.splitlines() if line.startswith("data: ")),
                None,
            )
            data = json.loads(data_line) if data_line else {}
            detail = data.get("stage", data.get("decision", ""))
            print(f"  {event_line} {detail}")

    print("\nPredict, then try: remove index_version; use a 4,001-character question;")
    print("change the available ticker; inspect result['scope'] and result['metrics'].")
    print("TestClient exercises the ASGI app in process; it does not start Uvicorn.")
    print("Its collected stream shows event contents, not real network timing.")


if __name__ == "__main__":
    main()
