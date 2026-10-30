# FinRAG optimization review - 25 October 2026

The local optimizations are implemented and the regression suite passes: **560 tests and 70 subtests**. Live validation used exactly **16 of the 16 authorized OpenAI requests**. The cache repeat demonstrated a large saving; full answer-quality acceptance remains incomplete. In particular, the business comparison returned an unsupported claim despite its model audit passing. No project-wide percentage saving or failure-free generation is established.

## Implemented changes

| Change | Behavior and benefit | Tradeoff and protection |
| --- | --- | --- |
| Bounded answer reuse and concurrent request coalescing | Identical scoped messages, bounded history and conversation IDs share work within one index service. Complete validated answers have a configurable TTL and entry limit. | Conservative keys reduce hit rate. Deep copies isolate mutable sources; request IDs remain fresh. Failures, partial answers and invalid citations are not retained. Cancellation of a waiter does not cancel the original work. Model-audit false positives can still be cached; caching does not improve answer correctness. |
| Exact numeric lookup before general retrieval | Reuse existing indexed table rows and the existing financial validator. Skip search and extraction only when every numeric requirement is covered. | Mixed narrative/numeric requests, ambiguity, missing cells or exceeded evidence budgets keep the existing retrieval/extraction path. No company, question or answer values are hardcoded in production. |
| Focused recovery retaining accepted narrative selections | Select missing observations rather than regenerating accepted selections. Render and audit the complete answer again, including retained claims and any interpretation. | Re-auditing still costs tokens and may reject an earlier claim. Reuse never overrides a failed audit. Existing action and evidence budgets remain enforced. |
| Separate evidence recovery from validation attempts | A pre-generation request for missing numeric evidence no longer consumes an actual answer-validation attempt. Remove the streaming cutoff that counted all finalization messages as failed validations. | Difficult requests may use a repair that was previously cut off early. Actual generation/validation attempts remain bounded at two, with at most one evidence-recovery round. |

Routing already used a deterministic function rather than a routing-model call before this change. The existing LangGraph tool workflow remains in place; replacing its wrapper would not establish an additional token saving. The routing change here keeps recovery termination consistent with actual validation attempts.

The production model, generation prompts, passage selection schemas, citation checks, financial validation and normal retrieval fallback are retained. A redundant serialization/parsing step in synthesis preparation was removed without changing its model input.

## Verification

- Baseline before these edits: 542 tests and 70 subtests passed. Final suite: **560 tests and 70 subtests passed**; the added cases cover cache isolation/expiry/concurrency/cancellation, numeric fallback, focused recovery and validation attempt accounting. External network access was blocked during these checks.
- Offline before/after checks on saved evidence for cases 5, 6, 7, 11 and 12 found identical passage catalogs, selection prompts and schemas.
- Known-plan checks against the real index produced MSFT FY2025 revenue of 281,724 million USD and TSLA FY2025 revenue of 94,827 million USD without model, embedding or general-retrieval calls. These checks supplied the plans; they do not measure fresh planning or full-query latency.
- A recorded-response replay for case 12 preserved the previous model inputs and rendered output. That recorded case remained incomplete; replay is not a new answer-quality pass.
- Whitespace checks on the touched files passed. Existing unrelated workspace changes were preserved.

## Authorized live results

The run used the configured `gpt-4o-mini` model and index `phase1-v2-repair-20261024`. A persistent transport-level ledger counted requests before sending, across planning, embeddings, generation, audit and recovery. SDK retries, tracing and RAGAS judging were disabled. The limit was reached with **11 chat requests and 5 embedding requests**, all HTTP 200; a further send was prevented. The earlier 24 October allowances were not reused.

| Fresh workflow / repeat | Outcome | Service latency | Model tokens | Model requests / embedding requests | Estimated LLM cost |
| --- | --- | ---: | ---: | ---: | ---: |
| MSFT FY2025 total revenue | Answered; reported amount, currency and scale match the cited income statement | 17,691.21 ms | 18,769 | 3 / 1 | $0.0029967 |
| Identical numeric request, same conversation and history | Cache hit; identical answer and sources, fresh request IDs | 8.14 ms | 0 | 0 / 0 | $0 |
| MSFT/TSLA 2024 business-model comparison | Returned `answered` and passed its model audit; manual citation review found an unsupported capital-expenditure claim | 32,239.81 ms | 35,796 | 4 / 2 | $0.0063744 |
| TSLA filing/transcript comparison | Started; stopped when the remaining six requests exhausted the allowance. No completed result or quality verdict | Not established | Not established | 4 / 2 before stop | Not established |

Embedding cost is unconfigured in local telemetry and is excluded from the LLM cost column. Latency measures `ChatService.ask`, excluding HTTP queue/network/browser time. The first numeric query may also include service warmup; these are single observations, not percentile benchmarks.

For this exact cache repeat, measured service latency fell by approximately **99.95%** and application model/embedding requests and model tokens fell by **100%**. This is a warm-cache comparison only. It cannot be applied to unique questions, changed history or the whole project. No fresh before/after cold-query benchmark was run.

Two concrete limits remain visible:

1. The fresh numeric planner emitted both a fact requirement and a duplicate narrative requirement for the same revenue question. The answer consequently repeated the value, and the conservative numeric-only shortcut correctly stayed off. Removing arbitrary narrative requirements would risk omitting requested explanations; the optimization does not make that assumption.
2. The comparison stated that Microsoft anticipates significant capital expenditures while citing `S9868982889484482596`. That returned evidence discusses derivative credit-risk provisions and minimum liquidity, rather than capital-expenditure plans. The audit accepted this unsupported point. This is consistent with the previously documented claim-binding problem in [the 24 October review](FIX_REVIEW_2026_10_24.md); a fresh baseline comparison was not run to attribute it conclusively. Returning `answered` is therefore not counted as a quality pass.

The retrieval evidence and validators were not weakened to make these cases pass. Full live acceptance needs a reviewed fix for claim-to-evidence binding and a completed filing/transcript comparison. More live calls require a new explicit allowance; none remain in this run.

## Configuration and artifacts

`FINRAG_ANSWER_CACHE_MAX_ENTRIES` defaults to 32; set it to `0` to disable answer caching. `FINRAG_ANSWER_CACHE_TTL_SECONDS` defaults to 300. Cache lifetime is limited to one process/service and immutable index; a restart clears entries. Scope resolution runs before cache lookup, so a request needing semantic scope parsing can still incur that separate model call. Cold-generation benchmarks should disable the cache and report repeats separately.

Local generated artifacts are under `evals/results/` and ignored by Git:

- `optimization_live_20261025.json`: persistent 16-request ledger, completed answers, sources and per-stage telemetry. Do not reset it to bypass the exhausted allowance.
- `optimization_live_20261025.py`: opt-in capped live harness. Its default invocation makes zero provider requests; do not rerun its approved mode on this exhausted ledger.
- `optimization_review_20261025.json` and `.py`: zero-request source/selection comparisons, known-plan numeric checks and recorded-response replay.
- `optimization_offline_20261025.py`: network-blocked regression runner with isolated temporary directories for this managed Windows environment.
- `optimization_before_117f03ac056c44b58f8a382d9a6d2e93/`: snapshots of the relevant already-modified working-tree files before this optimization pass.

Run local checks from `finrag` without provider calls:

```powershell
.\.venv\Scripts\python.exe -X utf8 -B evals/results/optimization_offline_20261025.py tests
.\.venv\Scripts\python.exe -X utf8 -B evals/results/optimization_review_20261025.py
```
