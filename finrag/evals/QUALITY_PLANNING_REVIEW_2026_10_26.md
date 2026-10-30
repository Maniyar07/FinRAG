# Claim verification and planning review - 26 October 2026

This follow-up implements steps 1 and 2 from the optimization review: bind narrative claims to actual evidence and remove duplicate scalar/narrative planning work. **574 tests and 70 subtests pass.** Fresh numeric validation passed. Full live comparison acceptance remains incomplete. Both newly authorized allowances, twelve requests and then eight requests, are exhausted.

## Changes

- Reuse the existing requested-output inventory in the production planning schema. Classify each output once, then inherit its type in the evidence requirements. Remove the redundant requirement-level type decision and the duplicate formula field from the model schema. Keep explicit explanations, table dimensions, source groups and calculation operands.
- Normalize repeated scalar requirements against their requested metric before existing deduplication. Include output and business identity in deduplication so distinct requested outputs or businesses are preserved.
- Replace narrative point Boolean verdicts with stored evidence spans and explicit unsupported details in the existing audit call. The application independently checks that every returned span belongs to that point's attached quotes. A positive overall coverage verdict cannot override an empty, fabricated or misplaced point proof.
- Constrain each point's proof schema to literal sentence spans from its own stored quotations, and put source evidence before the proposed claim in the audit input. This prevents the model from repeating its generated prose as a purported quotation. Full source quotations, attribution context and metadata remain available. Sentence boundaries constrain the proof format; they do not remove retrieval evidence or replace source attribution checks.
- Move the existing deterministic numeric/citation checks before interpretation and coverage audit. Those stages now assess the points that can actually be rendered, rather than approving claims that are discarded afterward. Literal source quotations remain available when a generated paraphrase fails those checks.
- Repair an interpretation-only failure once using retained observations, within the existing two-attempt limit. Remove the automatic expansion of a missing interpretation into new searches for every narrative requirement. Genuine source-observation gaps retain focused retrieval recovery.

The retrieval algorithms, index, source filters, numeric validator, calculation engine and normal search fallback were retained. There are no company-specific answer branches, keyword rejection thresholds or stored gold answers in production. No additional model stage was added. The proof schema uses additional source text in its constraints; its cost must be evaluated against avoided failures and recovery, rather than assumed to be free.

## What live validation established

The 25 October follow-up used exactly 12 newly authorized OpenAI requests, counted before transport sends in a persistent ledger. It included ten chat requests and two embedding requests. Eleven requests returned HTTP 200; one chat request had a read timeout and still counts against the allowance. Answer caching, SDK retries, tracing and RAGAS judging were disabled.

| Check | Result |
| --- | --- |
| Fresh MSFT FY2025 total revenue | Correct cited value: 281,724 million USD. One scalar requirement after deduplication; exact lookup; one planning request, zero embeddings or narrative generation. |
| First comparison planning attempt | Two requests were spent on a brittle literal-phrase check that rejected a paraphrased shared clause. That check was removed, and a regression preserves valid paraphrases. It is not part of the final implementation. |
| Comparison continuation | Planning succeeded with all six company/observation requirements. The auditor copied generated claims into unrestricted quotation fields. Literal binding rejected those fabricated proofs, including the unsupported Microsoft capital-expenditure claim, but also rejected valid points. A later read timeout and recovery exhausted the allowance before a completed answer. |
| Constrained-span audit, separate eight-request run | Returned stored quotations rather than fabricated prose. Planning, initial retrieval, selection, interpretation and audit completed. The run then retrieved during recovery and regenerated interpretation; its final audit was prevented by the eight-request cap. No completed business comparison was recorded. |

For the numeric query, the earlier cold sample used 18,769 model tokens, 3 chat requests plus 1 embedding request, 17.69 seconds, and $0.0029967 estimated LLM cost. The follow-up used 2,653 model tokens, 1 chat request, no embeddings, 6.09 seconds, and $0.0004983 estimated LLM cost: approximately **85.9% fewer model tokens, 83.4% lower estimated LLM cost, and 65.6% lower observed service latency for this query**. These are single observations on the same indexed filing, with different planning behavior; they do not establish project-wide savings or latency percentiles. Embedding prices were unconfigured in the earlier cost telemetry and are excluded from the cost comparison.

The separate eight-request run used five chat requests and three embedding requests, all HTTP 200. Its audit still accepted some weak semantic judgments, including capitalized contract costs under a capital-requirements heading. The evidence-span constraint establishes quotation identity, not entailment. Deterministic checks subsequently removed generated numeric assertions absent from their attached evidence, which also invalidated the dependent interpretation. Reviewing this run prompted the final two corrections above: check renderable claims before audit/synthesis, and repair interpretation without searching again for already-covered observations. Those corrections have offline regression coverage; no further live requests were made to validate them.

## Offline verification and limits

The full offline suite reports **574 tests and 70 subtests passed**. External network access was blocked. The regression checks cover scalar planning through the real graph/table lookup, preserved explicit explanations, separate output/business identities, paraphrased comparison clauses, empty or fabricated proofs, evidence borrowed from another point/source, unsupported assertion details, whitespace normalization, and the strict source-span choices offered to the auditor. Additional graph checks verify interpretation repair without another search, reuse and re-audit of accepted observations, and rejection of invalid numeric prose before interpretation/coverage review. Existing tests continue to cover scope, citations, calculations, tables, partial answers, recovery, caching and cancellation.

Before/after checks on five saved evidence cases found unchanged passage catalogs and narrative selection prompts/schemas for the same scoped plans. Known-plan checks against the real index still return the correct MSFT and TSLA FY2025 revenue values without provider or general-retrieval calls. These checks do not establish fresh narrative quality.

Literal evidence binding prevents fabricated quotations; semantic entailment and complete coverage still require model judgment and review. A stored sentence can be real yet fail to support a proposed claim. The final comparison must therefore be inspected against its actual cited passages before being counted as accepted.

## Artifacts and remaining acceptance

Generated artifacts under `evals/results/` are ignored by Git:

- `quality_planning_live_20261025.json`: exhausted 12-request ledger and the completed numeric result. The failed first comparison attempt remains in the ledger; it is not a result from the final code.
- `quality_planning_capture_20261025.jsonl`: model inputs, schemas and completed responses for reviewing the copied-proof failure. Request headers and credentials are not captured.
- `quality_planning_offline_20261025.json`: source-preservation and known-plan numeric checks; zero provider requests.
- `quality_planning_before_20261025/`: working-tree source snapshots before this follow-up.
- `quality_planning_live_20261026.json`: exhausted eight-request ledger, including per-request stages. Its incomplete run is not an answer-quality pass.
- `quality_planning_capture_20261026.jsonl`: captured model inputs and responses showing genuine source-span selections and the remaining semantic/coverage limits.
- `quality_planning_live_20261026.py`: capped single-comparison harness. Its default invocation makes zero provider requests. Do not reset or bypass an exhausted ledger.
- `quality_planning_pytest_20261026.txt`: final offline suite output, 574 tests and 70 subtests passed.

Local implementation and regression verification are complete. Fresh live acceptance of the final recovery/check-order changes, complete comparison coverage, and semantic review of capital requirements remain outstanding. Additional live work requires a new explicit request allowance. The filing/transcript comparison remains a separate outstanding acceptance item from [the earlier optimization review](OPTIMIZATION_REVIEW_2026_10_25.md).
