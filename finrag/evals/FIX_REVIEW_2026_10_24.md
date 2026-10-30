# FinRAG repair status - 24 October 2026

The shared pipeline repairs and corrected transcript index are applied locally. **542 tests and 70 subtests pass. Full live answer-quality acceptance across the twelve questions is still incomplete.** Passing contract tests or returning `answer_validated=true` does not establish answer completeness or financial correctness.

The first authorized run used 12 provider requests. The user subsequently authorized **12 additional requests**; that separate allowance is now exhausted: **one embedding request and eleven chat requests**, all HTTP 200. Retries and tracing were disabled in the bounded continuation. No new RAGAS score was calculated.

## Shared-path simplification

The subsequent optimization review reduced production code by **17 lines** in `src/generation/evidence_answer.py` and `src/agent/graph.py`, without adding helper functions, query-specific branches or validation gates:

- Use standard word wrapping for stored paragraph chunks and process each source's own passages once.
- Build the passage ranking corpus once per company/year/document group and reuse it across independent observations. The cache lasts only for the current request.
- Build the audit payload directly instead of serializing and immediately parsing it again.
- Remove duplicate object-ID checks and calculation-operand normalization already performed by the typed plan validator.

All **542 tests and 70 subtests** still pass. An offline before/after comparison on saved evidence for cases 5, 6, 7, 11 and 12 confirms identical passage catalogs, selection inputs and selection schemas. This review made **zero provider requests**. It preserves behavior and does not resolve or establish the outstanding live comparison-quality acceptance checks.

## What was wrong and what changed

| Problem | Shared repair | Evidence |
| --- | --- | --- |
| Plans invented restrictions, lost requested table dimensions, or confused search wording with requested outputs | Flat typed requirements; original-question interpretation; grounded subjects/qualifiers; separate calculation operands | Planning, scope and workflow regression tests |
| Combined queries displaced independent financial observations | Embed the baseline and independent content queries together; run scoped dense and BM25 searches; fuse candidates and allocate parent context by observation | One embedding batch reused across source groups; retrieval/context tests; real comparison evidence |
| Final passage ranking discarded retrieved explanations | Preserve term frequencies; use BM25Plus to avoid small-corpus zero/negative-IDF degeneracy; rank actual excerpts instead of neighboring context; remove unconditional table promotion | Constrained-budget regressions; actual cost definitions, cloud/AI investments and capex disclosures now reach selection |
| Transcript parsing treated editorial definitions, image credits and participant rosters as speakers | Recognize the actual transcript boundary and role-qualified turns; preserve dialogue and quoted attributions; exclude editorial prelude and trailing rosters | Four transcripts reparsed; parser regressions; new index integrity checks |
| Generated claims and interpretations cited related but non-supporting passages | Scope-restricted stored IDs; literal evidence beside each reviewed point; independently review interpretation; withhold interpretations dependent on discarded claims; explicit company/document grounding before drafting | Source isolation, entailment and dependency tests; live responses still exposed weak model judgments and are not counted as quality passes |
| Validation rejected legitimate numeric forms and attribution | Decimal-normalized values; fiscal-year labels; compare synthesis numbers with its actual input observations; preserve a business explicitly named in the quote even when the reviewer labels its subject as a metric | Positive/negative numeric and attribution tests; recorded-response replay restores an explicitly attributed segment margin |
| Repeated routing and narrative review spent model calls | Deterministic routing; shared passage catalog; batched selection/review within token budgets; one bounded focused recovery | Complete-graph and budget tests; no routing-model calls |
| Quota/authentication failures appeared as missing financial evidence | Preserve provider codes; terminate non-retryable operations; disable the locally unavailable Cohere reranker | Runtime/streaming tests; dense/BM25/RRF remain active |

The changes operate on source identity, document structure, query observations and budgets. Production code has no company-specific answer branches or stored gold answers. The production model remains `gpt-4o-mini`; a `gpt-5-mini` benchmark alone did not resolve claim-to-passage binding, so no model migration was made.

## Active local configuration

The local `.env` selects **`phase1-v2-repair-20261024`** and sets `FINRAG_RERANK_ENABLED=false`. The original `phase1-v2` index is retained. A running application must restart to load the updated configuration.

The repaired index contains **375 parents and 1,491 child vectors**, including 32 parents and 181 children rebuilt from four transcripts. All **343 non-transcript parent files are byte-for-byte identical** to the previous index. Manifest, vector, lexical and parent counts agree. The application initializes against the repaired index with network calls blocked. Original source files were not edited.

Rebuilding transcript children and embedding eighteen validation queries used one batch of 199 embedding inputs. Subsequent local dense searches reused the saved query vectors. This exercised the real dense/BM25/fusion/context path without another embedding request per query. Successful Cohere reranking was not validated.

## Additional twelve-request results

| Requests | Work | Actual outcome |
| --- | --- | --- |
| 1 | Embed repaired transcript children and validation queries; build an isolated index; retrieve five narrative cases | Completed; source integrity verified |
| 2-3 | Configured-model business comparison, before final ranking repairs | Numeric validation interrupted the response; company claims were also mixed |
| 4-7 | Stronger-model comparison benchmark on the same retrieved evidence | Better company separation, but wrong supporting IDs for a capex claim; partial response, not a pass |
| 8-9 | Configured-model comparison after passage ranking repair | Correct cost/capex paragraphs reached drafting, but company mixing persisted and numeric validation interrupted synthesis; final observation-number and scope-grounding changes followed this run |
| 10-12 | Configured-model filing/transcript comparison on corrected evidence | Filing observations and actual management outlook survive; dependent interpretation withheld; `complete=false`, missing `synthesis` |

The last mixed-source output distinguishes historical filing disclosures from management's 2026 deployment/margin outlook. Its draft also included irrelevant historical observations and company-wide capex under a business-specific heading. Some model review verdicts were too permissive. Deterministic source checks removed those claims and invalidated their dependent interpretation.

A zero-request replay after the final literal-attribution fix additionally preserves the explicitly attributed energy-segment gross margin. The interpretation remains withheld. This replay uses recorded outputs and is **not fresh generation, a regenerated audit, or a complete answer-quality pass**.

The remaining issue is not simply absent documents: responsive paragraphs exist and now reach selection. Reliable claim-to-passage binding and complete comparison synthesis remain unaccepted under the configured model. Further live evaluation should measure these issues using the final pipeline; adding broad rejection gates or changing a model without reviewing its actual answers is not an established solution.

## Twelve-case acceptance position

| Cases | Established | Still required for full acceptance |
| --- | --- | --- |
| 1-4, 8-10: tables, growth, margins, geography, trends, EPS, metrics | Seven previous offline numerical/table checks passed against real evidence; filing parents are unchanged | Fresh full workflow; case 8 also needs trend interpretation |
| 5: tax reconciliation | Indexed 20% rate/reconciliation evidence; amount-versus-rate handling tested | Fresh full explanation |
| 6: concentration risks | Supplier/component evidence and a receivables concentration note are available | Fresh scoped answer; receivables concentration cannot establish customer revenue concentration |
| 7: Microsoft/Tesla business models | Operating-cost and investment/capex explanations retrieved and included in the final catalog | Complete, correctly bound comparison under the final generation code |
| 11: Azure quarterly growth/capacity | Actual dialogue reparsed; quarterly 39% growth/capacity preserved separately from annual figures | Fresh answer preserving quarterly/annual/guidance basis |
| 12: filing versus call outlook | Fresh source-separated observations; final recorded-output replay preserves supported margin and outlook | Complete interpretation and full-graph recovery under fresh live validation |

## Reproduction and artifacts

From `finrag`, local tests:

```powershell
$env:LANGSMITH_TRACING='false'
.\.venv\Scripts\python.exe -m pytest tests -q --disable-warnings
```

The dataset is [query_varieties_regression.json](query_varieties_regression.json). A normal `evals/evaluate_ragas.py` run is **not automatically limited to twelve provider requests**. Planning, embeddings, extraction, generation, audits, retries, reranking and optional scoring can all consume requests. The one-off continuation harness uses a persistent HTTP-request ledger; do not reset it to bypass the exhausted allowance.

Generated artifacts under `evals/results/` are ignored by Git:

- `pytest_continued_release_20261024.txt`: 542 tests and 70 subtests passed.
- `pytest_shared_cleanup_20261024.txt`: all tests still pass after shared-path simplification.
- `cleanup_review_20261024.json`: five real-evidence before/after comparisons; zero provider requests. Single timing samples are not an end-to-end performance benchmark.
- `continued_validation_20261024.json`: additional 12-request ledger, live results and active-index status.
- `continued_capture_20261024.jsonl`: literal continuation inputs, schemas and outputs.
- `continued_evidence_20261024.json`: scoped plans, retrieved sources and diagnostics; plans were reused, not freshly accepted.
- `transcript_parser_review_20261024.json`: four transcript parser reviews and staged chunk counts.
- `continued_local_review_20261024.json`: zero-request index/service verification and recorded-response replay.
- `continued_startup_20261024.json`: zero-request default-configuration startup verification.
- `query_varieties_20261023_offline_evidence.json`: seven earlier real-evidence numerical/table checks.
- `bounded_narrative_review_20261024.json`: the earlier, separate twelve-request run.

The earlier review is [FIX_REVIEW_2026_10_23.md](FIX_REVIEW_2026_10_23.md). This review supersedes its quota statements and the earlier 529-test status from today.
