# FinRAG LangGraph dependency audit

Reviewed: 6 November 2026; application source baseline `35b8584`. This replaces
the 8 October audit's outdated description of structured lookup and routing.
See [current state](../CURRENT_STATE.md) for implementation boundaries.

## Active request path

```text
FastAPI / Streamlit
  -> ChatService
       -> deterministic query and scope resolution
       -> optional validated-answer cache
       -> LangGraph graph with deterministic routing
            -> plan_financial_request
                 -> model-assisted, validated requirements and calculations
            -> search_financial_documents
                 -> hybrid retrieval, optional reranking, fact validation
            -> get_financial_statement
                 -> indexed statement retrieval and fact validation
            -> financial_calculator
                 -> arithmetic over validated fact IDs
            -> finalize_financial_answer
                 -> evidence selection, numeric rendering, synthesis and audit
       -> ChatResult and sanitized tool/source trace
```

`ChatService` creates the graph during startup. Uncached requests that pass scope
policy invoke it. Routing is a Python policy over graph state, not a chat-model
tool-selection call; planning, extraction, selection, synthesis, and audit can make
model calls. There is no old backend selector or alternate production backend.

## Legacy runtime removed

The former bounded planner/executor and LangChain wrapper were connected to one
another and were also paired with a separate single-hop answer path. That wiring made
backend behavior difficult to reason about. The cleanup removes the old orchestration
package, old agent/coordinator/session/composer modules, and tool wrappers that only
served that stack. Environment validation, `.env.example`, evaluation instructions,
and the presentation guide now describe LangGraph as the only backend.

## Deliberately retained components

- `src/financial/calculator.py`, fact extraction/validation, and retrieval modules
  are used by the active graph.
- `src/financial/formula_definitions.json` and `formula_registry.py` remain as
  domain assets. The current graph does **not** execute their nested formula plans;
  wiring catalog formulas into the graph is a follow-up migration.
- `src/retrieval/structured_lookup.py` is active. `src/agent/tools.py` uses it for
  statement lookup and exact indexed numeric evidence; `src/financial/fact_pipeline.py`
  and `src/generation/evidence_answer.py` also use its source/table structures.
- `src/generation/evidence_answer.py`, `src/financial/extracted_fact_validator.py`,
  and `src/generation/answer_guardrails.py` participate in the active evidence,
  financial-fact, rendering, and citation checks.
- `src/generation/answer_generator.py` and `answer_fact_validator.py` remain
  separately tested legacy utilities. Their presence does not mean the graph
  finalizes through the legacy `GroundedAnswer` schema.
- Existing evaluation datasets and result artifacts are preserved.

## Providers and offline packages

`src/generation/llm_engine.py` supports a custom OpenAI-compatible chat endpoint
with separate credentials. Embeddings still use the configured OpenAI embedding
model. Evidence selection/synthesis/audit can select `FINRAG_EVIDENCE_MODEL`.
This is not a complete role-aware provider framework or automatic LoRA deployment.

Dataset construction, QLoRA training, benchmark tools, and historical MLflow import
live in `fine_tuning/`. Their dependencies and execution are separate from the
FastAPI runtime. The adapter and MLflow artifacts restored from the handover
release do not start a model server or instrument a new training run automatically.

## Verification boundary

This audit covers repository references and the checked-in application entry points.
It cannot identify external scripts that import removed modules. Search those scripts
before deleting any corresponding local backup or publishing the cleanup. No index
rebuild is required because these changes do not alter ingestion or stored data.
