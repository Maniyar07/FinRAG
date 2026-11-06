# FinRAG contributor instructions

## Orientation

FinRAG answers financial-research questions from a versioned local corpus of SEC
10-K filings and earnings-call transcripts. The application is in `finrag/`;
the outer directory is the Git repository root.

Start with `CURRENT_STATE.md` and the relevant sections of `finrag/README.md`.
Use `HANDOVER.md` for restoration and updating an existing clone. Read the source
and relevant tests before changing behavior. Older evaluation reports and repair
notes describe their own snapshots. `finrag/TECHNICAL_DESIGN_DOCUMENT_V2.md` is a
proposal with an implementation-status table, not a list of shipped features.

## Current code map

- `finrag/backend/`: FastAPI routes, request/response contracts, service lifecycle,
  streaming, admission limits, deadlines, and safe public diagnostics.
- `finrag/frontend/`: browser UI. `finrag/app.py` is the Streamlit entry point.
- `finrag/src/app/chat_service.py`: index validation, scope resolution,
  clarification, answer caching, graph execution, and result adaptation.
- `finrag/src/agent/graph.py`: LangGraph with deterministic tool routing.
  `src/agent/tools.py` owns planning, retrieval, statement lookup, calculation,
  and finalization. Routing itself does not call a model.
- `finrag/src/retrieval/`: scope policy, hybrid search, structured table lookup,
  parent context, compression, and evidence budgets.
- `finrag/src/financial/`: extraction, validation, units, coverage, and arithmetic
  over validated fact IDs. Formula-catalog execution is still a migration item.
- `finrag/src/generation/evidence_answer.py`: active evidence selection,
  synthesis, narrative audit, and rendering. `AnswerGenerator` is a separately
  tested legacy utility, not the active graph finalizer.
- `finrag/src/ingestion/`: parsing, chunking, stores, caches, and manifests.
- `finrag/tests/`, `finrag/evals/`: regression tests and evaluation tools.
- `finrag/fine_tuning/`: dataset factory, separate QLoRA trainer, serving
  benchmarks, and historical MLflow import. Saved adapters are not automatically
  selected by the application.

Paths beginning with `src/` above are relative to `finrag/`.

## Setup and commands

Use Python 3.11 or 3.12; the recorded verification used Python 3.12. Run application
commands from the inner `finrag/` directory. Prefer its `.venv`, but an existing
outer `.venv` can also be used without moving it:

```powershell
# From the repository root:
Set-Location finrag
if (Test-Path .\.venv\Scripts\python.exe) {
    $finragPython = (Resolve-Path .\.venv\Scripts\python.exe).Path
} elseif (Test-Path ..\.venv\Scripts\python.exe) {
    $finragPython = (Resolve-Path ..\.venv\Scripts\python.exe).Path
} else {
    py -3.12 -m venv .venv
    $finragPython = (Resolve-Path .\.venv\Scripts\python.exe).Path
}
& $finragPython -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
& $finragPython -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Configure `finrag/.env` before starting. Normal querying needs an authorized
`OPENAI_API_KEY` for embeddings; the default also uses it for chat generation.
The default restored index is `phase1-v2`. PDF ingestion additionally needs
LlamaParse credentials; Cohere and LangSmith are optional. A custom chat endpoint
does not change the embedding provider. On case-sensitive systems, follow the
`Data/` versus `data/` instructions in the application README.

Application requirements do not install pytest. The recorded test tool version
was pytest 9.1.1. Install it in the development environment if needed. From
`finrag/`, using the interpreter selected above:

```powershell
$env:LANGSMITH_TRACING = "false"
$env:LANGCHAIN_TRACING_V2 = "false"
$finragTestTemp = Join-Path ".test-temp" ("agent-check-" + [guid]::NewGuid().ToString("N"))
& $finragPython -m pytest tests fine_tuning/dataset_factory/tests -q -p no:cacheprovider --basetemp=$finragTestTemp
& $finragPython -m pip check
```

Choose checks appropriate to the change. Documentation-only changes need link,
path, command, and diff checks; do not rerun paid evaluations for them. Offline
unit/API tests use fixtures and controlled providers. Fresh end-to-end quality,
RAGAS, training, and serving benchmarks are separate work with separate resource
requirements. Report exactly which checks were run and their limits.

## Development rules

- Preserve company, source-year, reported-period, document-type, and comparison
  scope. Recovery must stay inside the validated request plan.
- Preserve fact provenance, row/column bindings, units, scale, arithmetic inputs,
  and citation validation. Missing evidence must remain visible.
- Fix generic contracts; do not hard-code benchmark question IDs, expected gold
  answers, or company-specific financial values into runtime behavior.
- Keep request budgets, cancellation checks, provider timeout/retry handling, and
  safe public errors intact. Cancellation is cooperative, not thread termination.
- Build changed corpora into a new immutable index version. Do not overwrite a
  working index or mix embeddings with incompatible models/dimensions. Embedded
  Qdrant stores must not be opened by competing server/evaluator processes.
- Keep application, evaluation, training, and tracking dependencies separate.
- Never print or commit real `.env` values, API keys, tokens, or private endpoints.
  Keep `.env.example` as placeholders. Generated models, indexes, experiment work,
  logs, caches, and virtual environments belong outside ordinary source commits.
- Preserve unrelated user work. `kadanes_algorithm.py` and `subarray_sum_k.py` are
  unrelated local practice files explicitly excluded from this project work.
- Update `CURRENT_STATE.md` and affected guides when implementation or operational
  behavior changes. Record verification dates and source snapshots; do not present
  historical scores or proposed features as current results.
