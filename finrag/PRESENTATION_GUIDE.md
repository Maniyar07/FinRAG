# FinRAG manager presentation guide

## One-sentence project description

FinRAG is a citation-grounded assistant for indexed 10-K filings and earnings-call
transcripts that combines hybrid retrieval, reranking, context compression, bounded
ReAct orchestration, multi-hop evidence gathering, and deterministic financial tools.

## What is implemented

| Requirement | Implementation | Evidence in the project |
|---|---|---|
| Hybrid retrieval | Dense search and BM25-style lexical search | `src/retrieval/hybrid_retriever.py` |
| Rank fusion | Reciprocal-rank fusion before reranking | `src/retrieval/hybrid_retriever.py` |
| Reranking | Optional Cohere reranker; failure keeps the RRF order | `src/retrieval/reranker.py` |
| Context compression | Extractive block selection after parent retrieval | `src/retrieval/compression.py`, `context_builder.py` |
| Tools | Scope-safe document search and deterministic calculator | `src/tools/` |
| ReAct agent | Official LangChain model/tool loop with a hard action budget | `src/app/langchain_react_agent.py` |
| Multi-hop RAG | Bounded evidence plan across companies, years, and document types | `src/orchestration/` |
| Grounding | Scope, coverage, fact, calculation, and citation validation | `src/generation/`, `src/financial/` |

## Architecture to present

```text
User question
    |
    v
Scope resolution and corpus availability check
    |
    +-- simple/table query --> optimized single-hop or structured lookup
    |
    +-- compound/calculation query --> official LangChain ReAct agent
                                      |
                                      +-- document_search
                                      |      |
                                      |      +-- dense + BM25
                                      |      +-- RRF
                                      |      +-- Cohere rerank or RRF fallback
                                      |      +-- parent context + compression
                                      |
                                      +-- financial_calculator
                                      |
                                      +-- complete_research (safe terminal action)
                                      |
                                      +-- observe coverage and stop safely
    |
    v
Grounded answer, citations, and validation
```

Simple questions intentionally avoid the agent loop because one retrieval is faster
and cheaper. The official LangChain agent is selected for arithmetic and compound
questions where multiple evidence or calculation actions are useful. The older
bounded workflow remains available through a feature flag until evaluation is done.

## Start the Monday demo

Run these commands from the `finrag` directory:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`, enable the workflow trace, and use the `phase1-v2`
index. The current `.env` enables reranking, compression, multihop, and workflow
traces. Do not display `.env` because it can contain API keys.

For the official agent demo, confirm `FINRAG_AGENT_BACKEND=langchain` before starting
the server. Use `bounded` as the rollback value if a regression is discovered.

The bundled Qdrant index is an embedded local database and permits only one process
to open it. Check `http://127.0.0.1:8000/api/health` first. After code changes, stop
the existing server cleanly and start it once so it loads the new code; never start a
second process against the same index.

## Demo questions

### 1. Narrative retrieval

> According to Tesla's 2025 10-K, summarize the AI-related resource requirements and supply-chain challenges described in Item 1A.

Explain the dense score, lexical score, RRF score, reranker result, compression
statistics, and cited source. This demonstrates hybrid retrieval, reranking, and
compression.

### 2. Calculation tool

> Using Microsoft's 2024 and 2025 10-K filings, calculate the percentage change in total revenue from fiscal 2024 to fiscal 2025. Show the formula and inputs.

Explain that values are retrieved and validated first. The language model is not
allowed to perform the arithmetic; the calculator receives validated fact IDs.

### 3. ReAct and multi-hop RAG

> Compare Microsoft and Tesla revenue for 2024 and 2025 using their 10-Ks. Calculate each company's percentage change, identify which improved more, and explain the reasons management gave in the 2025 earnings transcripts.

In the workflow trace, show:

1. `agent.selected = true`.
2. `agent.framework = langchain.create_agent` and the hard action budget.
3. Document-search actions and their source/fact observations.
4. Calculator actions and verified results.
5. 10-K and transcript evidence coverage.
6. Final citations or an explicit evidence limitation.

## Two-minute explanation

1. The query-understanding layer resolves company, fiscal year, and document type
   before retrieval. This prevents the agent from silently widening scope.
2. Narrative evidence uses dense and lexical retrieval. RRF fuses both rankings,
   and Cohere optionally reranks the fused candidates. A reranker failure returns the
   original RRF candidates.
3. Child chunks improve retrieval precision. Parent chunks restore context, and the
   compressor retains relevant blocks and complete table evidence.
4. Compound questions enter the official LangChain model/tool loop. The request-local
   agent can call only document search, the financial calculator, and a control-only
   completion action, and has a fixed action budget.
5. Numeric facts preserve source, period, currency, scale, and row identity.
   Calculations accept only validated fact IDs.
6. The answer generator can cite only retrieved source IDs. Missing comparison groups
   or invalid evidence produce an incomplete/abstention response instead of a guess.

## Honest limitations

- The active index contains eight source documents for Microsoft and Tesla covering
  2024 and 2025 10-Ks and transcripts.
- Corpus onboarding and parts of query routing are still specialized to the Phase 1
  corpus. Manifest-driven dynamic company support is the next milestone.
- Some deterministic recovery rules were added for financial tables. They improve
  safety on the prototype corpus but should later be replaced by a normalized fact
  store and generic schema-driven routing.
- The official agent is deliberately scope-locked and is not an open-ended web agent.
- The previous bounded workflow remains as a migration fallback; remove it only after
  the official backend passes the project evaluation set.
- Live answer quality still depends on the configured OpenAI and Cohere services.

## Answers to likely manager questions

**Why use both dense and BM25 retrieval?**

Dense retrieval captures semantic similarity. BM25 is strong for exact financial
terms, table labels, abbreviations, and numbers. RRF combines both without assuming
their scores are directly comparable.

**What happens when Cohere fails?**

Reranking is fail-open. The system retains the pre-rerank RRF candidates and continues
instead of returning an empty result.

**Why not run every question through the agent?**

Simple questions need only one retrieval and answer-generation pass. The direct path
reduces latency, cost, and failure surface. Agentic execution is reserved for tasks
that benefit from multiple evidence or calculation steps.

**How is this ReAct?**

LangChain `create_agent` runs the model/tool/observation loop. Observations include
retrieved sources, validated facts, missing groups, and calculator results. Final
answers are still composed and checked by the existing grounded generation layer;
hidden model reasoning and private prompts are not exposed.

**Is it completely generic?**

The retrieval, reranking, compression, tools, calculator, and agent contracts are
reusable. Company discovery and some query/fact routing still require the planned
dynamic-catalogue and normalized-fact-store refactor.

## Demo safety checklist

- Confirm `/api/health` reports ready before the meeting.
- Keep `phase1-v2` selected.
- Confirm `FINRAG_DEBUG_TRACES=true`, but never display `.env`.
- Run the three demo questions once before the meeting and save screenshots.
- Keep one previously successful response available in case an external API is slow.
- Do not re-ingest or refactor the corpus immediately before the presentation.
