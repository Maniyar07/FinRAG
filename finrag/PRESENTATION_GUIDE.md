# FinRAG manager presentation guide

## One-sentence project description

FinRAG answers questions grounded in indexed 10-K filings and earnings-call transcripts using hybrid retrieval, citation-aware financial evidence, and a scope-locked LangGraph agent.

## What is implemented

| Capability | Implementation | Source |
|---|---|---|
| Hybrid retrieval | Dense search, lexical search, reciprocal-rank fusion | `src/retrieval/hybrid_retriever.py` |
| Reranking | Optional Cohere reranker with fail-open RRF results | `src/retrieval/reranker.py` |
| Context building | Parent retrieval and extractive compression | `src/retrieval/context_builder.py`, `compression.py` |
| Agent | LangGraph ReAct loop with a recursion limit | `src/agent/graph.py` |
| Tools | Scope-bound filing search, calculator, statement lookup | `src/agent/tools.py` |
| Fact checks | Normalization and validation of extracted financial facts | `src/financial/` |
| Application policy | Company/year/doc-type resolution and clarification | `src/app/chat_service.py`, `src/retrieval/scope_policy.py` |

## Architecture

```text
Question
   |
   v
Scope resolution and corpus availability checks
   |
   +--> Clarify, reject, or report unavailable scope
   |
   v
LangGraph agent (scope locked)
   |-- search_financial_documents
   |      `-- dense + lexical -> RRF -> optional reranking -> parent context
   |-- get_financial_statement
   |      `-- retrieve indexed statement evidence
   `-- financial_calculator
          `-- calculate only from validated fact IDs
   |
   v
Answer with evidence citations and tool/source trace
```

Every in-scope question uses the LangGraph backend. The application resolves the
allowed company, filing year, and document type before the graph can search. Tool
schemas are strict, and search results carry source IDs and validated financial
facts into later calculator calls.

## Start the demo

Run from the `finrag` directory:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` and select the `phase1-v2` index. Set
`FINRAG_DEBUG_TRACES=true` if you want to show tool and source diagnostics. Do not
display `.env`; it may contain API keys. There is no backend-selection or multihop
environment flag.

The bundled Qdrant index is an embedded local database and permits only one process
to open it. Check `/api/health` before starting the server, and stop an existing
server cleanly before restarting it after code changes.

## Demo questions

1. **Narrative retrieval:** Summarize Tesla's AI-related resource requirements and
   supply-chain challenges in its 2025 10-K. Show the retrieved source and relevant
   section.
2. **Calculation:** Calculate Microsoft's total revenue percentage change from
   fiscal 2024 to 2025 using its annual 10-Ks. Show the cited values and calculator
   result.
3. **Cross-company and cross-document:** Compare Microsoft and Tesla revenue across
   fiscal 2024 and 2025, then summarize management's explanation from the 2025 Q4
   transcripts. Check that every requested group is supported and cited.

For each demo, inspect the trace's backend, tool calls, calculator results, source
IDs, and errors. The model should not perform financial arithmetic itself.

## Honest limitations

- The active index covers Microsoft and Tesla, fiscal years 2024 and 2025, 10-Ks,
  and Q4 transcripts. Other companies and years require indexed data.
- The formula catalog in `src/financial/formula_definitions.json` is retained but
  is not yet compiled into the LangGraph tools. The calculator currently exposes
  its declared validated-fact operations.
- Answer quality depends on retrieved evidence and the configured model and reranker.
- The graph is scope-locked and does not access the public web or live financial feeds.

## Answers to likely manager questions

**Why combine dense and lexical search?**

Dense retrieval finds semantically related passages. Lexical search helps with exact
financial terms, table labels, abbreviations, and values. Reciprocal-rank fusion
combines the ranked results before optional reranking.

**What happens if Cohere reranking fails?**

The system continues with the fused retrieval order.

**How are calculations grounded?**

The search tool validates numeric facts and returns their IDs. The calculator accepts
only those IDs, checks units and compatible periods, and returns the formula,
inputs, result, and source IDs.

**Is the project corpus agnostic?**

The retrieval and agent contracts are reusable, but the current index and parts of
scope vocabulary are tailored to the Phase 1 Microsoft and Tesla corpus.
