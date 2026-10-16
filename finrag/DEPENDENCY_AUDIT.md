# FinRAG LangGraph dependency audit

Date: 2026-10-08

## Active request path

```text
FastAPI / Streamlit
  -> ChatService
       -> deterministic query and scope resolution
       -> LangGraph agent graph
            -> search_financial_documents
                 -> hybrid retrieval, optional reranking, fact validation
            -> get_financial_statement
                 -> indexed statement retrieval and fact validation
            -> financial_calculator
                 -> arithmetic over validated fact IDs
       -> ChatResult and sanitized tool/source trace
```

`ChatService` creates the graph during startup and invokes it for each request that
passes scope policy. There is no old backend selector or fallback route.

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
- `src/retrieval/structured_lookup.py` and the answer-generation validation modules
  remain separately tested utilities. They are not called by the current LangGraph
  request path, so do not treat them as active safeguards until they are integrated.
- Existing evaluation datasets and result artifacts are preserved.

## Verification boundary

This audit covers repository references and the checked-in application entry points.
It cannot identify external scripts that import removed modules. Search those scripts
before deleting any corresponding local backup or publishing the cleanup. No index
rebuild is required because these changes do not alter ingestion or stored data.
