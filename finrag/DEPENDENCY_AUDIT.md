# FinRAG dependency audit

Date: 2026-09-27

## Result

The official LangChain agent still depends on FinRAG's planner, executor,
retrieval tools, financial validation, calculator, and answer composer.
Creating the agent did not make those modules obsolete.

The source tree contains 51 Python implementation files, excluding package
`__init__.py` files. Static import traversal from the API, Streamlit,
ingestion, and evaluation entry points reaches 50 of them, including the
currently selectable bounded backend. The remaining implementation file,
`src/ingestion/parse_cache.py`, is referenced only by its tests.

This is an import and source-code audit. Conditional imports are included;
reachability does not mean every file executes on every question. No application
code or files were removed, and no live queries or tests were run for this audit.

## Actual dependency chain

```text
backend/main.py -> backend/service_manager.py -> src/app/chat_service.py
app.py -------------------------------------> src/app/chat_service.py

ChatService
  |-> scope resolution, clarification, structured lookup, ordinary retrieval
  |-> LangChainReActAgent [selected backend: langchain]
  |     |-> MultiHopPlanner -> models, plan_builders, plan_payload
  |     |-> AgentResearchSession
  |     |     |-> MultiHopExecutor.execute (for each evidence requirement)
  |     |     |     |-> DocumentSearchTool -> HybridRetriever
  |     |     |     |     |-> dense vector search + BM25 + RRF
  |     |     |     |     |-> optional reranker
  |     |     |     |     `-> context builder + compression
  |     |     |     |-> structured financial lookup
  |     |     |     `-> financial extraction and validation
  |     |     |-> evidence merger
  |     |     `-> FinancialCalculatorTool + calculation_runner + fact_selection
  |     `-> response composer / answer generator + citations and validation
  `-> BoundedReActAgent [alternative backend: bounded]
        |-> same planner and executor
        `-> NarrativeQuoteSelector
```

Not every query uses the agent. `ChatService.ask` retains structured lookup and
an ordinary single-hop path; its agent route is selected using
`should_use_multihop`. Those paths also require their existing modules.

## Files to keep

| File or group | Why it is required |
|---|---|
| `src/app/chat_service.py` | Both interfaces use it; resolves scope and routes requests. |
| `src/app/langchain_react_agent.py` | Official LangChain controller. |
| `src/app/react_research_session.py` | Agent tool implementations, evidence state, typed completion checks. |
| `src/app/response_composer.py` | Formats validated facts and calculations, including intermediate results. |
| `src/orchestration/planner.py` | The LangChain agent calls its planner to create the typed checklist. |
| `src/orchestration/executor.py` | The session calls it for each retrieval requirement and verified facts. |
| `src/orchestration/models.py` | Shared typed evidence and calculation contracts. |
| `src/orchestration/plan_builders.py`, `plan_payload.py` | Plan construction and model-output normalization used by the planner. |
| `src/orchestration/fact_selection.py`, `calculation_runner.py`, `evidence_merger.py` | Operand selection, calculated-result reuse, and source identity/provenance. |
| All current `src/retrieval/*.py` implementation files | Called by retrieval, scope policy, compression, or structured lookup. |
| All current `src/financial/*.py` implementation files | Validated extraction, normalization, arithmetic, and provenance. |
| `src/tools/document_search.py`, `financial_calculator.py` | Validated capabilities used by agent execution. |
| Generation modules except `narrative_quotes.py` | Used by answer generation, citation handling, or validation. |
| Ingestion modules except `parse_cache.py` | Required by ingestion or by loading manifests, parent context, and lexical data. |
| `src/config.py`, `constants.py`, `schemas.py` | Shared configuration and contracts. |
| Package `__init__.py` files | Package structure; orchestration also exports shared types. |

## Cleanup candidates and required edits

| Candidate | Evidence | Required preparation |
|---|---|---|
| `src/app/multihop_coordinator.py` (497 lines) | Contains `BoundedReActAgent`, legacy event projection, and the compatibility alias. Production import is in `chat_service.py`; legacy event tests import it too. | Make routing LangChain-only, remove the bounded import/branch, update backend configuration validation and documentation, and retire bounded-specific tests. |
| `src/generation/narrative_quotes.py` (518 lines) | Production consumers are `chat_service.py` initialization and the bounded coordinator. The LangChain path does not call this selector. | Remove the coordinator and quote-selector initialization/import together. Retire its tests and update bounded-only cases in `test_chat_service.py`. Keep structured claim generation and citations. |
| `src/ingestion/parse_cache.py` (191 lines) | Repository search found only `tests/test_parse_cache.py` as a caller. The loader does not import or call it. | Remove or archive the unused utility and its dedicated test together if parse caching is not a planned feature. Existing index data is separate. |

Removing all three would remove 1,206 implementation lines, before subtracting
associated obsolete wiring. File counts alone do not justify removing other
modules that perform distinct, used responsibilities.

## Exact integration points for a LangChain-only cleanup

1. `src/app/chat_service.py`: remove the bounded agent import and routing branch;
   remove `NarrativeQuoteSelector` import and initialization; simplify backend
   trace labels and defaults. Keep planner and executor initialization.
2. `src/config.py`: retire the selectable `bounded` backend; decide whether to
   retain the existing environment key with only `langchain` accepted or remove
   the switch entirely. Preserve the multi-hop feature flag unless separately
   changing routing behavior.
3. `tests/test_react_agent.py`: legacy bounded event projection only; retire with
   the coordinator. Preserve `tests/test_langchain_react_agent.py`.
4. `tests/test_narrative_quotes.py`: retire with the legacy selector.
5. `tests/test_chat_service.py`: migrate or retire cases exercising the old
   quote composition path. Several fake services omit `agent_backend`, which
   currently makes `_run_multihop` choose its `bounded` default.
6. `tests/test_fastapi.py`: inspect old bounded trace fixtures; update them to
   reflect the supported backend while keeping API trace sanitization coverage.
7. `.env.example`, `README.md`, and presentation/walkthrough documentation:
   review the backend instructions and any claims about old orchestration.
8. If removing the unused parse cache, retire `tests/test_parse_cache.py` too.

The bounded backend is an explicitly selected alternative. The LangChain route
returns its own result; it does not automatically invoke the bounded coordinator
after an agent error. Typed pending-calculation recovery occurs inside the
LangChain workflow and does not require the bounded coordinator.

## Verification after an authorized cleanup

Check imports and startup, run the existing regression suite, and run the nine
demo questions once each in clean conversations. Confirm multi-hop traces select
`langchain`, all requested checklist items complete, intermediate calculations
retain citations, and ordinary single-hop questions still answer. No reingestion
is needed for retiring these backend modules.

## Other audit observations

- Source inventory, query interpretation, and scope policy still import company
  and year constants. Making corpus support dynamic is a separate change; deleting
  those modules would remove required functionality.
- `app.py` is a Streamlit entry point; `backend/` and `frontend/` support the web
  interface. Removing one interface is a product choice, not unused-code cleanup.
- Evaluation scripts and tests are runnable entry points even when the web app
  does not import them.
- These findings cover repository references and current entry points. External
  scripts that import these modules are outside the static scan.
