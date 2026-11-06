# FinRAG current state

Reviewed: **6 November 2026**. Application source baseline: **`35b8584`** on
`langgraph-experiment`. This documentation refresh does not change application
code. Update this page when the implementation changes.

## Start here

The outer `FinRAG/` directory is the Git root; `finrag/` contains the application.
The checkout can have another name, such as `FinRAG_Phase1_Corrected`.

| Need | Read |
| --- | --- |
| Contributor and Codex instructions | [AGENTS.md](AGENTS.md) |
| Setup, configuration, architecture, and commands | [Application README](finrag/README.md) |
| Restore saved artifacts or update an existing clone | [HANDOVER.md](HANDOVER.md) |
| Presentation and component walkthrough | [PROJECT_WALKTHROUGH.md](finrag/PROJECT_WALKTHROUGH.md) |
| Active and legacy dependencies | [DEPENDENCY_AUDIT.md](finrag/DEPENDENCY_AUDIT.md) |
| HTTP contracts | [README_FASTAPI.md](finrag/README_FASTAPI.md) |
| Traces and usage diagnostics | [OBSERVABILITY.md](finrag/OBSERVABILITY.md) |
| Evaluation datasets and execution | [Evaluation README](finrag/evals/README.md) |
| Dataset preparation, training, and tracking | [Fine-tuning README](finrag/fine_tuning/README.md), [training](finrag/fine_tuning/training/README.md), [tracking](finrag/fine_tuning/tracking/README.md) |
| Proposed Version 2 work and status | [Version 2 design](finrag/TECHNICAL_DESIGN_DOCUMENT_V2.md) |

Use current code and tests to resolve discrepancies. Dated repair notes and saved
evaluation reports are evidence for their stated snapshots. They are not a
current backlog or a guarantee about a later deployment.

## Implemented request path

1. FastAPI (`finrag/backend/main.py`) validates public requests and manages one
   `ChatService` per allowed index. Streamlit (`finrag/app.py`) uses the same
   application service. Startup checks manifests and stored record counts.
2. `src/app/chat_service.py` resolves company, filing year, document type, locked
   comparison groups, and pending clarifications. Optional semantic parsing cannot
   bypass the index catalogue. Complete validated answers can be reused from a
   bounded in-memory cache.
3. Uncached searchable requests enter `src/agent/graph.py`. A deterministic router
   selects tools within an action budget; routing itself makes no model call.
4. `src/agent/tools.py` exposes `plan_financial_request`,
   `search_financial_documents`, `get_financial_statement`, `financial_calculator`,
   and `finalize_financial_answer`. Model-assisted planning is validated before
   evidence tools can run.
5. Structured numeric/table lookup uses stored parent evidence. When that cannot
   satisfy the request, scoped dense and BM25 search, rank fusion, optional Cohere
   reranking, and budgeted parent context supply evidence. Facts must pass source,
   period, row/column, unit, and request-binding checks.
6. The calculator uses validated fact IDs. The finalizer uses
   `src/generation/evidence_answer.py` for evidence selection, deterministic numeric
   rendering, synthesis, narrative quote audits, and request coverage. Bounded
   recovery can search missing requirements without widening scope.
7. `ChatService` adapts the result and checks citations/output contracts. The API
   returns decisions, sources, request IDs, safe diagnostics, and usage metadata.
   Answers can be complete, explicitly partial, or withheld when validation fails.

Paths beginning with `src/` are relative to `finrag/`. See the source for exact
branch conditions; neither every request nor every tool needs every stage.

## What is implemented, and what remains proposed

| Area | Current implementation | Remaining boundary |
| --- | --- | --- |
| Runtime | LangGraph, deterministic routing, scoped tools, evidence/fact validation, calculator, citations, bounded recovery and caching | Formula-catalog-driven multi-step execution is not wired into the graph |
| Providers | OpenAI embeddings; chat factory supports a separate OpenAI-compatible URL/key and evidence-model selection | No complete role-aware provider/capability framework or automatic adapter loading |
| Fine-tuning | Dataset construction, review/splits, QLoRA trainer, training notebook, recorded reports, and backed-up adapter files | An available adapter is not proof of compatibility with the active finalization schema or approval for each model role |
| Tracking | Separate MLflow importer for reviewed historical training, quality, and serving summaries | Historical lineage is explicitly partial; this is not live MLflow instrumentation of every training run |
| Model serving | Endpoint benchmark tools and configurable compatible chat endpoints | No integrated Ollama packaging, model lifecycle, or readiness workflow |
| Runtime controls | Strict API contracts, scope/tool restrictions, security headers, trace sanitation, provider timeouts/retries, concurrency admission, request deadlines, cooperative cancellation, optional LangSmith tracing | No authentication/authorization or per-user isolation; no complete PII redaction/injection-detection enforcement layer |

`AnswerGenerator` and `answer_fact_validator.py` are retained, separately tested
legacy utilities. The active finalization path uses `evidence_answer.py`, financial
fact validation, and `answer_guardrails.py`. Structured lookup is active; it is
used by both agent tools and financial fact handling.

## Local configuration and restored data

- `.env` belongs at `finrag/.env`, next to `requirements.txt`. Recreate it from
  `.env.example`; real credentials are intentionally absent from Git and releases.
- Standard setup creates `finrag/.venv`. An existing outer `.venv` also works when
  commands use its interpreter. `AGENTS.md` shows how to detect both locations.
  Recreate virtual environments on a new machine instead of moving an old one.
- The default example selects `phase1-v2`, `text-embedding-3-small`, and 1,536
  dimensions. Embedding settings must match the selected manifest. The default
  chat model is `gpt-4o-mini`; a saved QLoRA adapter is not enabled automatically.
- Source documents are tracked. Saved indexes, adapter/tokenizer files, experiment
  work/MLflow data, gold-builder work, and additional evaluation results are in the
  five archives linked by `HANDOVER.md`. Extract them into the repository root.
- `phase1-v2` covers MSFT/TSLA, 2024/2025, 10-K/transcript. Read the selected index
  manifest and `/api/catalogue` for the actual restored coverage/counts. Other
  index versions can exist locally and are not implied by a plain clone.
- Keep the original ZIPs and checksum file as backups. `git pull` updates tracked
  source/documents; it does not download new release archives or recreate `.env`.
- The runtime uses lowercase `data/`, while the Windows-origin archives use
  `Data/`. On Linux/macOS case-sensitive filesystems, follow the symlink guidance
  in the application README before starting the server.
- Historical MLflow artifact URIs can contain old-machine paths. Use the separate
  restore/import instructions in `HANDOVER.md` to view summaries on a new machine.

## Verification record

Last recorded offline run: **6 November 2026**, application source `35b8584`:

- `tests` plus `fine_tuning/dataset_factory/tests`: **759 passed**,
  **70 subtests passed**, **49 dependency deprecation warnings**.
- Python syntax: 155 tracked Python files; JSON parsing: 26 JSON files and one
  notebook. `pip check`: no broken requirements.
- External network connections and tracing were disabled for that run; localhost
  test traffic was allowed. The sandbox run hit Windows temporary-folder access
  errors; the rerun outside the sandbox passed with the same network restrictions.

These results describe the source and environment tested here. A new PC still
needs its own dependency/configuration checks and a startup/query check. No fresh
provider-backed quality evaluation, RAGAS scoring, training, or inference benchmark
was performed for this documentation refresh. Do not combine scores from older
reports into a new current-quality claim.

## Next work

Choose improvements from an observed failure or an explicitly requested feature.
Keep financial grounding, request contracts, and immutable index compatibility
covered by relevant regression checks. Version 2 work needs explicit acceptance
criteria for provider roles, training/output-schema compatibility, model serving,
lineage, and security controls. Update this page and the affected guides as those
features are actually delivered.
