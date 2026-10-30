# Comparison evidence and recovery acceptance — 26 October 2026

Final status (27 October): **both comparison answers pass manual review**.
The completed fixes, validation methods, measured costs and limits are in
[the final evidence review](EVIDENCE_FIXES_2026_10_27.md).
The shared ledger is at 388 of 400 approved requests, and paid testing has stopped.
The [offline acceptance review](OFFLINE_ACCEPTANCE_REVIEW_2026_10_27.md) records
the pre-fix failures. Implementation history follows.

## Offline diagnosis

Reviewed the saved business comparison in `optimization_live_20261025.json`, the
selection/audit captures in `quality_planning_capture_20261025.jsonl` and
`quality_planning_capture_20261026.jsonl`, and the filing/transcript failures in
`query_varieties_20261022_final.json` and `query_varieties_20261023_claims.json`.
These generated artifacts are under ignored `evals/results/`.

| Requested detail | Evidence and failure location |
|---|---|
| MSFT revenue sources, FY2024 10-K | Product/service disclosures were available. Later selection converted source amounts in millions into new billion-valued prose; deterministic checks removed those points. Coverage must be reassessed after removal, with literal source wording available as fallback. |
| MSFT cost structure, FY2024 10-K | The cost-of-revenue disclosure and expense-allocation discussion were available. Valid operating-cost observations must survive rejection of an unrelated capital claim; allocation policy alone does not explain all cost drivers. |
| MSFT capital requirements, FY2024 10-K | The first saved answer's four MSFT source excerpts contain no capital-expenditure discussion. The indexed 10-K does contain an explicit statement linking expected increases to cloud offerings and AI infrastructure in parent `af821a78-9dc4-5aef-826d-d72da78c5fea`. This was a retrieval/evidence-input gap, followed by a validation false positive: source `S9868982889484482596` concerned credit/liquidity provisions. The later capture selected contract-cost capitalization and receivables instead, and its audit accepted the unrelated relationship. The later capture lacks the entire initial retrieval payload, so it cannot distinguish retrieval exclusion from passage-budget exclusion for every omitted passage. |
| TSLA revenue sources, FY2024 10-K | Automotive, energy and service disclosures were present; retain supported source/product distinctions without adding unquoted claims about strength or revenue significance. |
| TSLA cost structure, FY2024 10-K | Selected passages mainly concerned leasing and revenue-recognition policies. These can explain their disclosed accounting costs; they cannot alone establish a broad manufacturing-cost-driver claim. This combines a selection/coverage gap with overly permissive semantic validation. |
| TSLA capital requirements, FY2024 10-K | The selection catalog contained explicit capital-expenditure outlook and investment-use passages (`P115`, `P128`), including factories/equipment. Preserve these positive cases and their future-period qualifiers. Reject ungrounded conclusions about financial adequacy or management effectiveness. |
| TSLA energy-storage filing observations, FY2025 10-K | The saved run contained relevant filing evidence and supported observations. Do not discard that section because interpretation or transcript verification fails. |
| TSLA energy-storage outlook, Q4 FY2025 transcript | Saved evidence included management discussion of backlog, competition, policy/tariff exposure and margin outlook. Historical results are context, not a substitute for future outlook. Saved failures include rejection of synthesis numbers and inconsistent completeness judgments; source-specific business attribution and final interpretation still need live citation review. |

## General implementation

- The existing audit now states the source relationship before judging the claim,
  identifies any missing premise, and explicitly decides responsiveness. A related
  financial topic cannot override an identified missing premise. Funding and cost
  accounting remain valid when those are what the user requested.
- Every requested detail needs references to points that survive rejection, or to
  responsive literal quotations when those are actually rendered. A positive
  aggregate verdict cannot override missing details or discarded support.
- Supported points inside an incomplete requirement are retained separately from
  completely verified requirements. Recovery adds missing observations, merges
  their source references, and audits the resulting answer again.
- Focused recovery searches use the audited gaps. The search tool no longer adds
  the entire umbrella metric back to a narrative-recovery query. Original source,
  company, year, subject and qualifier constraints remain enforced.
  A completely unavailable selection now exposes its gap to the router too.
  Narrative recovery passes the focused query into per-requirement ranking,
  instead of reintroducing the original broad ranking queries.
- Source observations are selected by scoped sentence IDs and copied from storage.
  Audit proofs use bounded local indices, avoiding repeated source text in JSON
  schema enums. An explicit coverage gap survives partial selection and recovery.
  Passage chunks preserve ordinary sentence boundaries; a paragraph-size boundary
  no longer cuts an investment statement into an artificial sentence fragment.
- Synthesis tracks individual observation IDs. Rejecting an unrelated point from
  a shared source no longer invalidates an interpretation that uses only retained
  observations. Unknown or rejected dependencies still prevent acceptance.
- Initial narrative retrieval reserves one fair parent share for focused recovery
  within the existing total parent, character and token limits.
- The evidence model is configurable through `FINRAG_EVIDENCE_MODEL` and defaults
  to the existing model. Live model overrides remain experimental until reviewed.
  Audit batches assess their own evidence; the last batch checks original-request
  assignment across the entire plan. Intermediate batches cannot reject a clause
  merely because its evidence belongs to another batch.
- Business ownership uses bounded indices into literal source-word catalogs,
  with source-context binding retained. Empty or invented owners cannot establish
  business scope. Generic observation labels cannot stand in for business owners.
- Unused schema definitions are removed while retaining transitive references.
  Full original paragraphs are supplied once instead of copying their growing
  prefixes for each selected sentence. Exact stored sentences with empty headings
  inherit the same quote's semantic verdict, avoiding duplicate entailment review;
  generated prose and additional headings still require point-level proof.
- No new model stage, company-specific branch, stored answer, relaxed citation
  check or relaxed numeric check was introduced. Semantic judgments still require
  model review; structural proof checks do not guarantee semantic accuracy.

## Offline verification

The latest full run passed **680 tests and 70 subtests**, including relation,
coverage, dependency and recovery cases. Positive cases cover funding, cost capitalization,
investment requirements, supported negative answers and removal of redundant bad
points without rejecting a complete answer. A real-graph recovery case preserves
two valid observations, retrieves only the third missing dimension, merges the
new observation and re-audits all three. Two additional harness checks pass using
local HTTP and simulated provider responses: the durable cap is enforced before
transport, explicit cap extensions preserve prior usage and cannot be replayed,
and performance cannot start without a citation review. The final harness checks
were rerun after adding the extension mechanism: both passed with zero live calls.

A controlled replay of saved provider selection 188 reproduces the budget failure
before schema/context compaction (12,860 payload plus schema tokens for its first
requirement). Compaction reduces that input to 10,508 tokens, an 18.3% reduction,
and reaches verification with the same quotations and original paragraphs. This
isolates those two compaction changes with literal-verdict reuse enabled in both
conditions; it is not a whole-query cost or latency saving. The replay uses zero
provider requests; its artifact is `evals/results/comparison_budget_replay_188_20261026.json`.

## Live protocol and shared allowance

`accept_comparisons.py` defaults to a zero-request plan. Live execution requires
an explicitly approved request count and a persistent ledger; the count cannot be
reset. It records actual HTTP latency, server/queue timing, model usage, prices,
answer evidence, sanitized traces and provider-request stages. Provider headers
and credentials are never captured. Requests count before sending; timeouts and
failed attempts consume slots. OpenAI SDK retries, tracing and RAGAS are disabled.

Approved maximum: **400 provider requests shared across all phases**, including
planning, embeddings, selection, synthesis, all audit batches, complete bounded
recovery, failed attempts, scope parsing, any optional reranking and acceptance
reruns. The original 192-request allowance used 191 requests during diagnosis and
earlier acceptance attempts. The user explicitly approved a further 96-request
extension; the same ledger records the old limit and usage and raises the cap to
288 without resetting any requests. At 286 used, the user approved another 112,
raising the same cap to 400. The latest two acceptance runs used 24 requests,
bringing usage to 310. Subsequent implementation and acceptance used 78 further
requests; final usage is 388. Paid testing is stopped; the remaining allowance
is not a spending target. Stop at the cap;
never overwrite an exhausted ledger or silently enlarge the allowance. Only use
`--additional-approved-api-calls` after explicit new approval; the supplied base
limit must match the ledger, which prevents replaying an old extension command.

1. Run cases 7 and 12 with answer caching disabled. Review each requested
   dimension and its actual citation passages, including numbers, period, business
   ownership, caveats, inference and omissions. `answered` alone does not pass.
2. Save an explicit per-case citation/coverage review. Performance execution
   refuses to start unless both cases have a passing review for the same source
   revision. Failed attempts and superseded code do not fill fresh sample slots.
3. Count the accepted cache-disabled run as the first fresh sample; collect two
   more per case. Keep every additional answer subject to citation/completeness
   review. Report tokens, combined chat/embedding cost, completion rate, median,
   p95 and maximum HTTP latency. Three samples per case are descriptive only;
   p95 is effectively the slowest observation, not a production tail estimate.
4. Populate an empty answer cache separately for each case, then measure three
   identical repeats. Verify actual cache-hit traces and answer/source identity;
   retain population cost separately. Never combine repeats into fresh savings.
5. Compare to saved single-run baselines only with their limits: earlier costs
   excluded embeddings and latency was service-only. A historically unsupported
   answer is not a quality-equivalent performance baseline. Report any timing
   configuration differences and any unpriced non-token provider charges.

The local harness uses the real API on a dynamically assigned loopback port.
Those results include local HTTP and worker queueing, not a remote deployment's
network or browser rendering. Request/provider timeouts can be set explicitly for
acceptance without changing `.env`; record them when interpreting measurements.

Live work uses `evals/results/comparison_acceptance_20261026.json` as its one
durable ledger. Earlier revisions failed manual acceptance: unrelated liquidity
and accounting observations were accepted as investment evidence, principal
revenue activities were omitted, or valid evidence was rejected during recovery.
Those failed runs cannot be used as quality-equivalent performance samples.
The pre-fix revision returned `answered` for both cases, but failed the 27 October
offline citation and completeness review. The final manual reviews now pass;
the business case has live acceptance plus an unchanged-answer compatibility
replay after the scoped patch, and the transcript case has final-revision live
acceptance. Repeated performance sampling was not run. Exact usage and timings
are in the final report; original provider results retain their actual revisions.

## Model and configuration tradeoffs under validation

The original model configuration remains in `.env`; final comparison acceptance
uses the explicit test overrides described in the final report.
Test runs explicitly override planning with `gpt-4.1-mini`; evidence selection,
interpretation and semantic review have been tested with both `gpt-4.1-mini` and
`gpt-4.1`. The compact mini-only run returned `answered` for both cases, but manual
review rejected the business answer because its principal automotive revenue
activity was missing and depreciation policies were offered as investment needs.
That result did not unlock performance testing. A stronger model also exposed
missing retrieval evidence; the final router/ranking and scope fixes passed the
review documented above.

These are configurable model choices, not company-specific branches. Rates come
from the external pricing catalog: the tested `gpt-4.1` evidence role has higher
token prices than `gpt-4.1-mini`. Report measured request totals rather than
presenting that model change as a saving. Rates were checked against the official
[GPT-4.1 model page](https://developers.openai.com/api/docs/models/gpt-4.1) and
[GPT-4.1 mini model page](https://developers.openai.com/api/docs/models/gpt-4.1-mini).
The provider currently labels these models deprecated; their availability in the
validation runs does not establish long-term availability. Keep model selection
configurable and repeat acceptance before migrating it.

Acceptance uses a 300-second request deadline and 90-second provider timeout;
these test overrides do not change the existing application defaults. Do not
interpret successful tests as proof that every future query will fit a shorter
production deadline or retain the same quality. A full representative evaluation
and separate concurrent application load test remain release checks.
