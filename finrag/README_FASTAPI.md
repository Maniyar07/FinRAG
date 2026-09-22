# FinRAG FastAPI reference

The FastAPI application exposes the existing FinRAG `ChatService` through strict JSON
contracts and serves the vanilla frontend from the same origin. Retrieval, reranking,
compression, generation, guardrails, and persisted indexes remain in `src/`.

For installation, ingestion, architecture, configuration, and evaluation, see the
main [`README.md`](README.md).

## Start the server

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. The interactive Swagger and ReDoc pages are disabled;
the OpenAPI document is available at <http://127.0.0.1:8000/api/openapi.json>.

Minimal `.env` configuration:

```dotenv
OPENAI_API_KEY=your_openai_api_key
FINRAG_INDEX_VERSION=phase1-v2
FINRAG_API_INDEX_VERSIONS=phase1-v2,phase1-v1
```

`FINRAG_API_INDEX_VERSIONS` is a comma-separated allowlist. Every listed index is
initialized once during application lifespan, reused across requests, and closed at
shutdown. If omitted, only `FINRAG_INDEX_VERSION` is served.

## Endpoints

| Method | Path | Response |
|---|---|---|
| `GET` | `/` | Web application |
| `GET` | `/static/*` | Frontend assets |
| `GET` | `/api/health` | Service readiness and index count |
| `GET` | `/api/catalogue` | Allowed indexes and available scope combinations |
| `POST` | `/api/chat` | Complete JSON chat response |
| `POST` | `/api/chat/stream` | Server-sent stage events and final result |
| `GET` | `/api/openapi.json` | OpenAPI 3 schema |

### `GET /api/health`

```json
{
  "status": "ok",
  "service_ready": true,
  "default_index": "phase1-v2",
  "index_count": 2
}
```

### `GET /api/catalogue`

Returns the index allowlist and each manifest's companies, fiscal years, document
types, and valid company/year/type combinations. The frontend derives its filters
from this response instead of hard-coding corpus coverage.

### `POST /api/chat`

```json
{
  "index_version": "phase1-v2",
  "question": "Summarize Microsoft's 2024 risk factors from its 10-K.",
  "history": [],
  "previous_scope": {
    "tickers": [],
    "years": [],
    "doc_type": null,
    "requested_groups": [],
    "required_doc_types": []
  },
  "pending_clarification": null,
  "filters": {
    "company": "MSFT",
    "fiscal_year": "2024",
    "document_type": "10K"
  },
  "include_trace": false
}
```

All properties are validated strictly; unknown fields are rejected. Important request
limits include a 4,000-character question, 24 history messages, 8,000 characters per
history message, and 64,000 total history characters.

A successful response has this shape:

```json
{
  "request_id": "...",
  "trace_id": "...",
  "decision": "answered",
  "answer": "...",
  "scope": {
    "tickers": ["MSFT"],
    "years": ["2024"],
    "doc_type": "10K",
    "requested_groups": [],
    "required_doc_types": [],
    "label": "MSFT | 2024 | 10-K",
    "complete": true
  },
  "inherited_fields": [],
  "pending_clarification": null,
  "sources": [
    {
      "id": "S1",
      "ticker": "MSFT",
      "fiscal_year": "2024",
      "doc_type": "10K",
      "source": "MSFT_2024_10K.pdf",
      "section": "Risk Factors",
      "pdf_page": "...",
      "evidence_text": "..."
    }
  ]
}
```

The decision can be `answered`, `clarify`, `scope_conflict`, `data_unavailable`,
`out_of_scope`, `insufficient_evidence`, or `error`.

When `decision` is `clarify`, the response can contain a bounded
`pending_clarification` object. Send it unchanged with the next short scope reply
such as `yes Tesla`, `2025`, `10-K`, `transcript`, or `both`. It accumulates only
company, year, and document-type selections while preserving the original financial
question. Clear it when filters change or the user begins a new conversation.

### `POST /api/chat/stream`

This endpoint accepts the same body and responds as `text/event-stream`. It emits two
initial stages, heartbeat comments during long work, a response-ready stage, and one
final result:

```text
id: 1
event: stage
data: {"stage":"request_validated","message":"Request and filters validated"}

id: 2
event: stage
data: {"stage":"finrag_pipeline","message":"Resolving scope and retrieving grounded evidence"}

: heartbeat

id: 4
event: result
data: {"decision":"answered","answer":"..."}
```

These are processing-status events, not generated tokens. If a browser disconnects,
the server stops polling and writing the stream and cancels its asynchronous wait.
Python cannot forcibly terminate a synchronous `ChatService.ask()` already executing
in a worker thread.

## Errors

Validation and application errors use one envelope and include correlation IDs:

```json
{
  "error": {
    "code": "validation_error",
    "message": "The request payload is invalid.",
    "request_id": "...",
    "trace_id": "...",
    "details": [
      {
        "field": "body.question",
        "message": "String should have at least 1 character",
        "type": "string_too_short"
      }
    ]
  }
}
```

Every HTTP response includes `X-Request-ID` and `X-Trace-ID` headers.

## Traces

Retrieval traces are returned only when both conditions are true:

1. the request sends `"include_trace": true`; and
2. the server sets `FINRAG_DEBUG_TRACES=true`.

The API returns an explicit safe-field projection. It excludes API keys, prompts,
questions, retrieval queries, complete model context, generation previews, and
tracebacks. Local server-side query logs are written to `Data/logs/query_traces.jsonl`
on Windows (the configured path is `data/logs/`).

## CORS and response security

Same-origin use needs no CORS setting. To allow a separate trusted frontend, list
exact origins without paths or wildcards:

```dotenv
FINRAG_CORS_ORIGINS=https://finance.example.com,http://127.0.0.1:5173
```

The application rejects wildcard CORS and adds request IDs, MIME-sniffing protection,
a no-referrer policy, a restrictive permissions policy, and a same-origin Content
Security Policy to every response.

## Frontend state and rendering

Conversation history and theme preferences are stored only in browser `localStorage`.
There is no document upload, prompt editor, or model selector. Markdown and evidence
are rendered through DOM construction and text nodes rather than trusting
model-produced HTML.

## Verify the API layer

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_fastapi.py -q
.\.venv\Scripts\python.exe -m compileall -q backend frontend
```

The FastAPI tests inject a fake service and do not access cloud APIs or real indexes.
