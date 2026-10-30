# Offline comparison acceptance review — 27 October 2026

This is the pre-fix diagnosis. The completed implementation, final passing
reviews and spending are documented in [the final evidence review](EVIDENCE_FIXES_2026_10_27.md).

**Decision: both latest comparison answers fail manual acceptance.** They returned
`answered` and valid citation IDs, but those checks did not establish complete,
correctly scoped support. Performance testing remains gated. This review made
**zero provider requests** and changed no application code or model settings.

## Review scope and evidence

Reviewed the latest cases 7 and 12 in
`results/comparison_acceptance_20261026.json`, their source excerpts, plans,
tool traces and captured provider requests 287–310 in the matching
`.capture.jsonl`. Inspected the current selection, synthesis and validation code,
relevant offline tests and the saved full-suite result. Earlier saved retrievals
were used only to locate positive evidence, not as performance baselines.

The current core-source hash with the tested planning/evidence models matches
both saved answers:
`015dcbaf58a94d17cf932829d1694665f3cf6f233561f6740b4f23aeae88f741`.
Models were `gpt-4.1-mini` for planning and `gpt-4.1` for evidence work;
case 12 also used `gpt-4o-mini` for semantic parsing. The test index was
`phase1-v2-repair-20261024`. Production model settings have not been changed.

## Per-dimension review

| Case / requested dimension | Finding |
|---|---|
| 7: Microsoft revenue | Pass for a qualitative overview: productivity, cloud/server and personal-computing segments are described and cited to FY2024 source S1. |
| 7: Tesla revenue | Pass: actual vehicle sales/leasing, credits, services and energy products are retained from FY2024 S4. |
| 7: Microsoft costs | Pass: S2 supplies cost-of-revenue, development and sales/marketing categories with actual drivers. |
| 7: Tesla costs | Partial: S4/S5 supply segment totals, depreciation and lease allocation, but not the principal manufacturing inputs used in the interpretation. The two displayed totals also omit their segment labels, years and millions unit. |
| 7: Microsoft capital needs | Supported by S3's explicit cloud/AI capital-expenditure outlook. One additional bullet starts mid-sentence after a page break and should be repaired or omitted. |
| 7: Tesla capital needs | Fail: S6 supplies an asset definition and useful-life schedule. These establish accounting treatment, not capital spending needs or their investment drivers. |
| 7: Interpretation | Fail: manufacturing/energy investment needs and manufacturing cost emphasis exceed the selected evidence. Correct source IDs do not cure the missing relationship. |
| 12: Filing observations | Relevant storage-product deployment and cost drivers survive, but the reported revenue and margin figures belong to the combined energy generation-and-storage segment. They must retain that scope. |
| 12: Call outlook | The S4 backlog, Megapack 3/Mega Block deployment outlook and competition/policy/tariff caveats are supported. The additional S3 growth/capacity remarks concern broader energy operations, with solar discussion in the same paragraph; storage-only attribution is not established. |
| 12: Interpretation | Fail: it turns combined-segment revenue and gross-margin results into storage-only results. Filing and transcript sections are separated correctly, but their business scope is not preserved throughout. |

All cited IDs exist in the saved source lists and point to the intended company,
filing year and document type. Citation identity passes; semantic support and
completeness do not. These conclusions concern the saved outputs, not every
query the project can answer.

## Failure locations confirmed offline

1. **Retrieval missed useful Tesla cost and investment evidence.** Case 7's
   three Tesla sources are segment reporting, leases and accounting policies.
   Selection request 290 contains no explicit Tesla capital-spending passage or
   principal automotive cost-input disclosure. Earlier saved retrievals from the
   same index contain both: FY2024 10-K pages 61–64 enumerate materials, labor,
   manufacturing overhead, logistics and warranty; pages 45–46 link capital
   expenditures to AI, global factory expansion, machinery and equipment.
   These are available evidence, not facts to hardcode into the application.

2. **Validation substituted a related topic for the requested relationship.**
   Request 297 accepts Tesla's owned-datacenter definition and depreciation
   useful-life table as complete capital-requirements evidence, setting no
   missing premise. The source context explicitly identifies depreciation.
   Literal sentence matching correctly establishes quotation fidelity, but the
   model's positive relevance verdict is wrong.

3. **Business ownership is represented too weakly.** Request 309 chooses the
   single word `Energy` as source owner and sets `source_is_aggregate=false` for
   combined-segment revenue and margin observations. Request 310 similarly
   accepts broader energy growth remarks for storage outlook. The aggregate
   guard works when the flag is correct; it cannot catch a falsely negative
   classification. A matching word does not establish the full economic scope.

4. **One proof can approve several independent interpretation claims.** The
   final audit in request 310 selects just one generic growth sentence as proof
   for the entire three-sentence synthesis. An offline replay of the captured
   proof through the current `_point_supported` helper returns `True` with all
   network connections blocked. The helper checks literal proof membership and
   model verdicts; it does not require evidence for each independent assertion.
   Source: `src/generation/evidence_answer.py`, `_point_supported` and the
   synthesis draft-point construction.

5. **Repair repeats work without fixing these gaps.** Both latest traces have
   two initial searches followed by two finalization attempts and no recovery
   search. The first pass removes some extras; synthesis and audits run again.
   Because the real missing relationships were incorrectly marked complete,
   the focused-retrieval path never receives those gaps. The router cannot
   recover evidence from an incorrect completeness signal.

6. **Some retrieved/displayed context is unusable.** Case 12's extra filing
   source S5 contains only the heading `Automotive Regulatory Credits`, despite
   broad Business-section metadata. It is not a missed storage passage. Case 7
   also exposes a page-break sentence fragment and flattened table values whose
   labels/units were available in context but absent in the displayed bullets.

## What the existing tests establish

The saved final offline log reports **624 tests and 70 subtests passed**, with
43 warnings. That suite was not rerun merely to change review documents.
The source hash above confirms the reviewed core implementation matches the
latest live outputs. The additional helper replay reproduced the false positive
without an SDK call; socket connections were explicitly blocked.

Existing aggregate tests check that an explicitly flagged aggregate is rejected
for a narrower business and that valid scoped/company-level statements survive.
They do not prove that a model reliably identifies aggregates. Similar mocked
relationship tests cannot establish live semantic accuracy when the mock itself
supplies the correct verdict. Preserve those positive tests and add captured
false-positive cases when implementing a correction.

## Cost and latency already recorded

| Latest saved run | Provider requests | Recorded tokens, including embeddings | Estimated token cost (USD) | Local HTTP latency | Manual acceptance |
|---|---:|---:|---:|---:|---|
| Case 7 | 12 | 74,986 | 0.161770 | 56.05 s | Fail |
| Case 12 | 12 | 57,736 | 0.115652 | 54.31 s | Fail |
| Total | 24 | 132,722 | 0.277422 | — | 0/2 |

The synthesis/audit telemetry stage accounts for 15 of these 24 requests and
about 79% of their token cost. Its reported time is about 41 seconds per query.
These are two failed-quality samples, not proof of fresh-query savings, a median
or p95 benchmark, or production concurrency performance. Local HTTP timing does
not include a remote network or browser rendering. Test request/provider
deadlines were 300/90 seconds.

The shared ledger remains **310 requests used / 400 approved**. Recorded usage
across that ledger estimates **USD 1.68529003** using the configured catalog and
provider cached-input counts. Of 310 requests, 309 have usage and one HTTP 400
has none; the figure is a usage-based estimate, not an invoice or a total for all
earlier project work. The remaining allowance is not a target to consume.

## Smallest next implementation scope

1. Bind each generated assertion to its supporting observation(s), full source
   business scope, period and relationship. Preserve combined-segment wording;
   product-specific drivers may be stated without reallocating a combined total.
   Reuse the existing audit stage rather than add another model call.
2. Distinguish requested investment/cost-driver evidence from accounting-policy
   evidence using general evidence contracts. Preserve genuine accounting-policy
   answers when that is what the user asks. Send only actual missing dimensions
   to bounded retrieval; retain the valid comparison observations.
3. Filter heading-only retrieval results and preserve table labels, units and
   sentence context in presentation. Recover complete source sentences across
   page breaks without inventing text.
4. Replay the saved failures and positive cases offline first. Once correctness
   is established, reuse unchanged verified observations within the request and
   recheck only changed claims plus their dependent synthesis, keyed to evidence,
   requirement, model and audit-policy identity. Never reuse a stale verdict
   across a changed scope or policy merely to reduce calls.
5. Reassess whether one bounded live check is necessary only after these offline
   checks pass. Keep performance/cache-repeat runs deferred until both final
   answers pass semantic review. No additional live run was launched here.

These are recommended fixes, not changes made by this review. The machine-readable
manual review in `results/comparison_acceptance_20261026.review.json` now records
both cases as failed for the current tested revision, preserving the performance
gate. The provider ledger and captured responses remain unchanged.
