# Evidence, recovery and cost review — 27 October 2026

The remaining comparison evidence/recovery fixes are implemented. Both target
answers pass manual citation and completeness review. **680 offline tests and
70 subtests pass. Paid testing has stopped at 388 of 400 approved requests.**

## Final behavior

- Generated interpretation sentences have separate, bounded evidence bindings.
  A positive whole-paragraph verdict cannot replace missing sentence support.
  Coordinated reporting scopes survive summarization; isolated component results
  cannot be inferred from a combined measure.
- Scoped candidate selection excludes explicit broader-segment financial trends,
  comparative business judgments and shared multi-product spending that do not
  isolate the requested business. Reporting labels come from the retrieved
  company/year documents. Short aliases are not rejected solely because they
  share a word with the requested subject. Company-wide requests remain valid.
- Asset accounting and useful-life schedules cannot substitute for productive
  investment needs. Cost totals and allocation policies alone leave a cost-driver
  gap; genuine accounting-policy requests still accept that evidence.
- Bounded recovery preserves valid observations, reserves parent capacity for
  independent missing dimensions, and retains positive planned source terminology
  alongside the actual gap. Definitional cost queries derive their financial
  label from the request plan. The existing embedding batch carries the facets;
  no new provider or recovery round was added.
- Explicit investment-purpose sentences can be selected from the scoped stored
  catalog without another selection-model call. The semantic auditor still has
  to accept their relationship, coverage and source. Mixed or ambiguous narrative
  requests retain model selection.
- Unchanged requirement audits are reused only inside the same request, bound to
  question, plan, claim, used source contents, model and audit policy. Changed
  bindings are re-audited. New claims and interpretation remain checked; adding
  unrelated evidence does not randomly invalidate a previously verified section.
- Identical quote schema definitions are shared through references, preserving
  their constraints and instructions. Sentence-proof unions are checked against
  local evidence instead of an incompatible old aggregate item limit. Recovery
  selection receives concise gaps rather than resending full audit proof bodies.
- Heading-only sources do not consume evidence slots. Stored sentence boundaries
  are preserved across supported page-break serialization, and literal table rows
  retain their column, group and unit context.

No company, year, amount, product-specific answer or stored answer was introduced
into production logic. Numeric calculation/source checks, source isolation and
the existing answer cache remain covered by the regression suite. No new package
or model stage was added.

## Accepted answers and source checks

| Case | Reviewed coverage | Result |
|---|---|---|
| 7: Microsoft/Tesla 2024 business models | Both companies' principal revenue activities, real operating-cost inputs and productive investments, plus supported qualitative interpretation | Pass |
| 12: Tesla 2025 filing versus Q4 2025 call | Filing product/customer/technology observations; call deployment, demand, backlog and 2026 product outlook; competition/policy/tariff margin caveats; separate source sections | Pass |

Case 7 cites Microsoft productivity/cloud/personal-computing disclosures (S1),
actual cost categories (S2), and cloud/AI investment outlook (S3). Tesla's vehicle,
service and energy activities (S4), materials/labor/manufacturing/logistics costs
(S5), and actual AI/factory/equipment capital expenditures (S6) are supported.
The actual 2024/2023 expenditures and separately identified 2025 forecast are not
conflated. There is no comparative numerical capital-intensity claim.

Case 12's final filing source describes Powerwall, Megapack, customers, software,
grid efficiency and technology integration (S1). The call provides record storage
deployments, demand, backlog, product launches and margin warnings (S2). The
combined-energy revenue figure and company-wide AI/new-product spending list are
absent. Both answers have valid citation IDs and company/year/document scope.

## Verification method and revisions

The final tested profile is planning `gpt-4.1-mini`, evidence `gpt-4.1`, and the
existing semantic parser/embedding settings. The index is
`phase1-v2-repair-20261024`. Production `.env` model settings were not changed;
the stronger evidence profile has higher unit prices and remains configurable.
These results must not be represented as live acceptance of another model profile.

Current profile/core-source revision:
`fdbc06e79e961022c15ae7871894a298b17a2015fff3a8af7f4f5a5018d3d04e`.

- Case 7 passed live HTTP acceptance at requests 360–371 on revision
  `1b1cfbcbe395c6d057957b86a809b52c57244b6d604bad4059c0baa4b6f79d9d`.
  After the final named-business scope patch, its actual saved source excerpts
  and provider audit proofs were replayed through the current renderer with
  networking blocked. Answer text and all six cited source identities remain
  unchanged. That company-wide path was not paid to run again solely for a new
  revision hash. This is live acceptance plus offline compatibility validation.
- Case 12 passed live HTTP acceptance on the final revision at requests 380–388.
  Its final source/citation review checks the actual generated answer, not just
  the `answered` status.
- The final full offline run passed 680 tests and 70 subtests. Positive cases
  cover valid narrower/combined/company scope, legitimate accounting and negative
  answers, multi-dimension recovery, local proof bounds and request-local audit
  reuse. The accepted business replay and scope-exclusion replay used zero APIs.
- OpenAI SDK retries, tracing, RAGAS judging and answer caching were disabled for
  acceptance. Test request/provider deadlines were 300/90 seconds. Existing
  application defaults were preserved.

## Recorded performance and spending

| Accepted case | Chat + embedding requests | Tokens including embeddings | Estimated token cost, USD | Local HTTP time |
|---|---:|---:|---:|---:|
| Business comparison | 8 + 4 = 12 | 58,030 | 0.13427608 | 49.22 s |
| Filing/transcript comparison | 7 + 2 = 9 | 43,745 | 0.09528007 | 34.71 s |
| Both | 21 | 101,775 | 0.22955615 | — |

Business evidence selection ran twice but made only one selection-model call:
the investment-only recovery used stored purpose disclosures and the same final
semantic checks. Source-bound audit reuse avoids repeating unchanged proofs.

A saved eight-quote audit previously required 12,304 payload/schema tokens and
could not run inside the configured 12,000-token budget. Sharing definitions
allowed that same saved evidence to reach auditing below the existing limit.
The replay artifact records its exact budget checks. This is a controlled
component improvement, not a project-wide cost or latency percentage.

The continuation from request 310 used **78 requests, estimated USD 0.92908095**.
Of these, 57 requests costing USD 0.69952480 belong to superseded or failed-quality
attempts; the 21 requests above produced the accepted answers. These development
costs are included, not hidden in the successful-run totals.

The whole shared ledger estimates **USD 2.61437098 at 388/400 requests**. One
earlier HTTP 400 has no token-usage record; all recorded costs use the configured
price catalog and provider cached-input counts. This is an estimate from this
ledger, not an invoice or a total of every earlier project/ingestion expense.
The 12 remaining approved requests were left unused.

These are one accepted live run per case. The HTTP measurements include local
server execution/queueing, not browser rendering or a remote network. Repeated
fresh-query sampling, separate cache-population benchmarks and concurrent load
tests were not run in this continuation. No median/p95, universal accuracy,
production-readiness or fresh-query percentage-saving claim follows from two
examples. Those broader benchmarks remain separate work.

## Review artifacts

- `results/comparison_acceptance_20261026.json`: original cumulative provider
  ledger, actual generated answers, stage usage and timings.
- `results/comparison_acceptance_20261026.review.json`: final manual reviews,
  including the business compatibility validation.
- `results/final_scope_replay_20261027.json`: unchanged accepted business text
  and the three excluded broader-scope observations, with valid outlook retained.
- `results/cost_retrieval_replay_20261027.json`: local lexical retrieval/context
  replay finds real manufacturing-cost evidence within the same two-parent budget.
  Dense retrieval was mocked out; the replay made zero provider requests.
- `results/new_acceptance_replay_20261027.json`: the 26-span proof decodes and
  investment disclosures survive a recorded false-unavailable selector response.
- `results/comparison_budget_replay_328_20261026.json`: exact audit budget replay.
- `results/evidence_fixes_full_20261027.log`: final offline suite output.

No deployment or commit was made. Production model settings were preserved. All
changes are available in the workspace for review.
