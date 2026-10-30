# FinRAG evaluation

This directory contains the canonical evaluation dataset, runners, audit notes,
and regression guidance for FinRAG. Generated run outputs belong under
`evals/results/`, which is intentionally ignored by Git.

## Canonical files

- `ragas_gold_90.json`: the 90-question gold set used for current LangGraph regression runs.
- `ragas_gold.json`: the smaller default source-grounded evaluation set.
- `evaluate_ragas.py`: end-to-end retrieval and generation evaluation.
- `MANUAL_REGRESSION_QUESTIONS.md`: 30-question manual end-to-end checklist.
- `gold_audit.md`: provenance and corrections for the gold set.
- `REGRESSION_CHECKLIST.md`: checks required before accepting pipeline changes.
- `../requirements-eval.txt`: evaluation-only dependencies.

The legacy 30-question JSONL dataset is retained as `ragas_gold_old.jsonl` only
for historical reproducibility; it is not the default evaluation input.

## Validate without model calls

From the `finrag/` directory:

```powershell
.\.venv\Scripts\python.exe evals\evaluate_ragas.py --validate-only
```

## Run the current evaluation

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-eval.txt
.\.venv\Scripts\python.exe evals\evaluate_ragas.py --index-version phase1-v2
```

The default detailed output is `evals/results/ragas_results.json`. Treat scores
as regression signals rather than universal accuracy: review low-scoring rows,
a sample of high-scoring rows, citation validity, and comparison-group coverage.

## Run the 90-question LangGraph evaluation

The checked-in 90-question set is `ragas_gold_90.json`; pass it explicitly
because the evaluator's default dataset is `ragas_gold.json`.

Generate answers and save one checkpoint after each question, without running
the separate RAGAS judge calls:

```powershell
.\.venv\Scripts\python.exe evals\evaluate_ragas.py `
  --dataset evals\ragas_gold_90.json `
  --generate-only `
  --output evals\results\langgraph_90_answers.json
```

Each result records the gold outcome and expected decision, actual decision and
answer, per-question generation latency, retrieved contexts and source
metadata, extracted citations with source-ID validation, compact LangGraph
tool calls and results, calculator inputs/results, runtime and tool errors, and
scope matching. The report summary counts expected decision matches and
expected clarifications separately; an actual `clarify` decision passes when
the row declares `"expected_decision": "clarify"`.

Existing answerable gold rows without `expected_decision` are treated as
`answered` because they provide a reference answer. For a clarification case,
add an explicit decision and a concise description of the expected behavior:

```json
{
  "expected_decision": "clarify",
  "expected_outcome": "Ask the user to choose the fiscal year before searching."
}
```

`reference` may be omitted for a non-answer expectation. Its `scope` can leave
unknown fields as empty lists. Such rows are not sent to RAGAS answer-quality
scoring. To score a saved answer report later, use a different output path:

```powershell
.\.venv\Scripts\python.exe evals\evaluate_ragas.py `
  --dataset evals\ragas_gold_90.json `
  --score-input evals\results\langgraph_90_answers.json `
  --output evals\results\langgraph_90_scored.json
```

Generated reports remain under `evals/results/` and should not be committed.

## Measure the running application

The default transport calls `ChatService.ask` directly. To include the HTTP path,
start the application separately (see [`README_FASTAPI.md`](../README_FASTAPI.md))
and pass its base URL. This reuses the running service and its real request queue;
it does not initialize a second local index or change retrieval or generation.

```powershell
.\.venv\Scripts\python.exe evals\evaluate_ragas.py `
  --dataset evals\ragas_gold_90.json `
  --api-base-url http://127.0.0.1:8000 `
  --concurrency 1 `
  --generate-only `
  --output evals\results\http_answers.json
```

`FINRAG_EVAL_API_BASE_URL` also selects the HTTP transport. Use the deployed URL
from a representative client machine to measure that network path. Set
`--concurrency` to the workload being tested; values greater than one require HTTP.
`--http-timeout` is configurable in seconds and defaults to the configured request
deadline. Authentication headers, when needed, come from the JSON object in
`FINRAG_EVAL_HTTP_HEADERS`; they are not written to the report. The server must
serve the selected `--index-version`. Enable `FINRAG_DEBUG_TRACES=true` on a test
server to retain tool and validation diagnostics; evidence and usage are returned
regardless of that setting.

Rows record client-observed time through response processing, server work time,
and API worker-queue wait as `client_latency_ms`, `server_latency_ms`, and
`queue_latency_ms`. Summaries give count, mean, median, p95 and maximum for each.
HTTP failures retain their observed latency and status; missing server usage is
marked incomplete instead of recorded as zero cost. The HTTP client does not
automatically retry requests. The browser separately displays its actual request
duration through stream parsing, including transfer. Neither measure includes DOM
paint or browser rendering; those require a browser test.

Use `--generate-only` for latency/load runs so judging does not compete with the
application. Score the saved report later with `--score-input` and a different
output path. These commands make paid application/judge calls when backed by paid
providers. For a fresh-query baseline, disable answer caching on the **server**
with `FINRAG_ANSWER_CACHE_MAX_ENTRIES=0` before startup; measure cache-enabled
workloads separately. Concurrent HTTP evaluation measures request completion and
queueing, but does not by itself verify cancellation, cache isolation, memory
limits or rollback.

## Configure token prices and read cost reports

Rates live in [`config/model_pricing.json`](../config/model_pricing.json), outside
calculation logic. The checked-in OpenAI rates were verified on 26 October 2026
against [OpenAI pricing](https://developers.openai.com/api/docs/pricing).
Set `FINRAG_PRICING_FILE` to another JSON file to use your provider's rates;
relative paths resolve from `finrag/`. Its format is:

```json
{
  "currency": "USD",
  "rate_unit": "per_million_tokens",
  "providers": {
    "your-provider": {
      "your-model": {
        "input": "1.00",
        "cached_input": "0.50",
        "output": "2.00",
        "aliases": ["your-model-snapshot"]
      }
    }
  }
}
```

The example rates are placeholders. Add embeddings as models with an `input`
rate. Only explicit aliases share prices; no snapshot-name guessing is used.
Provider/model selection follows the actual response model and these settings:

| Setting | Purpose |
|---|---|
| `FINRAG_LLM_PRICING_PROVIDER` | Chat provider; defaults to `openai`, or the custom endpoint host when configured |
| `FINRAG_LLM_PRICING_MODEL` | Model whose rates can receive legacy environment overrides; defaults to the configured chat model |
| `FINRAG_EMBEDDING_PRICING_PROVIDER` | Application embedding provider; defaults to `openai` |
| `FINRAG_JUDGE_PRICING_PROVIDER` | RAGAS chat/embedding provider; defaults to `openai` |

Existing `FINRAG_LLM_INPUT_USD_PER_1M`, `FINRAG_LLM_CACHED_INPUT_USD_PER_1M`,
`FINRAG_LLM_OUTPUT_USD_PER_1M` and
`FINRAG_EMBEDDING_INPUT_USD_PER_MILLION` overrides remain supported. A zero rate
means free; an absent rate remains unknown. Invalid or negative rates fail at
startup, before spending evaluation requests. Restart the server and evaluator
after changing rates. Pricing is loaded once; no pricing network call is added to
queries. Reports fingerprint the catalog and effective overrides with
`pricing_revision` so different runs can be distinguished. Update the rate data
when provider prices or your contract change.

Each request's `estimated_cost_usd` remains chat-only for compatibility;
`embedding_cost_usd` is separate, and `total_token_cost_usd` combines both.
`known_token_cost_usd` retains the priced subtotal when rates or usage are missing.
`cost_complete` and `cost_issues` explain whether the request total is available.

The report's `summary.token_cost` separates `application_usd` and `judging_usd`
and provides `total_usd`, `known_subtotal_usd`, completeness and missing-usage row
counts. Generation-only reports have zero judging cost. Scoring saved answers
retains their historical generation usage without generating them again. No token
cost is counted twice. These are token-cost estimates; reranking fees, other
tools, hosting, taxes and provider billing adjustments are outside this total.

## Comparison quality acceptance

See the [comparison diagnosis and acceptance protocol](COMPARISON_ACCEPTANCE_2026_10_26.md)
for the saved failure analysis and recovery rules. `accept_comparisons.py` prepares
a zero-request plan by default:

```powershell
.\.venv\Scripts\python.exe evals\accept_comparisons.py `
  --ids 7 12 --index-version phase1-v2-repair-20261024 `
  --output evals\results\comparison_acceptance_20261026.json
```

Live phases require an explicit approved API allowance. One durable ledger counts
all provider requests, including failures and recovery, across acceptance and
performance. Actual citation/coverage review must pass for the same source
revision before performance can run. Fresh runs have answer caching disabled;
cache population and repeats are measured separately through the real local API.
The three-sample default is a smoke benchmark, not a reliable production p95.
If the user explicitly approves more requests, pass the current ledger cap as
`--approved-api-calls` and the new allowance as `--additional-approved-api-calls`
once. The ledger records the extension and retains every earlier request. Later
runs use the new total cap without the extension flag; repeating the old command
fails before transport. This flag itself is not permission to spend more.

## Run targeted gold questions

Use the canonical gold dataset to rerun selected LangGraph questions after a fix:

```powershell
.\.venv\Scripts\python.exe evals\evaluate_ragas.py `
  --dataset evals\ragas_gold_90.json `
  --ids 81 `
  --generate-only `
  --output evals\results\q081_after_fix.json
```

Use a new output filename for each comparison. The evaluator records expected and
actual decisions, citations, tool calls/results, calculator inputs/results, errors,
and per-question latency. Rows and summaries also include per-stage model usage
and separate embedding usage; RAGAS scoring records its judge usage separately.
Recovered tool errors remain in the trace without failing a fully validated
answer. Unknown provider prices remain explicit; the combined token-cost summary
includes chat and embeddings, with judging shown separately. Reranking charges
are outside this total. No backend-selection or multi-hop flag is used; all
in-scope requests run through LangGraph. Keep generated reports under `evals/results`
and out of commits.

The twelve-case filing/transcript regression dataset is
[`query_varieties_regression.json`](query_varieties_regression.json). Read the
[`24 October repair review`](FIX_REVIEW_2026_10_24.md) for verified changes, the
active repaired transcript index, both authorized 12-request ledgers and
outstanding live acceptance checks.
