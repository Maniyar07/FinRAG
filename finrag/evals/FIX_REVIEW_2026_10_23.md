# FinRAG repair review — 23 October 2026

Historical review. See [24 October status](FIX_REVIEW_2026_10_24.md) for subsequent fixes, the 12-request evaluation and remaining acceptance gaps.

The shared pipeline changes are implemented and locally verified. **Final live acceptance is incomplete:** the configured provider returned `provider_quota_exhausted` for model and embedding calls during the final 12-case run. That run is blocked infrastructure validation, not a measured answer-accuracy score. No new RAGAS score was calculated.

## What caused the failures

1. Planning mixed requested outputs with search instructions. It sometimes invented industry restrictions, lost grouping dimensions, added overlapping requirements, or treated expanded research questions as extra user requirements.
2. Context packing retained large parents and displaced evidence for other facets. Company/year wording also competed with the financial observation in passage ranking. Relevant cost and capital disclosures exist in the index; the original examples do not establish missing source documents.
3. Sentence selection separated figures, business attribution and caveats. Generated point citations required the model to count indices in a second list. Some related quotations were accepted for the wrong relationship; other valid fragments were rejected for not individually answering the whole question.
4. Verification lacked the actual numerical outputs already rendered elsewhere. It could demand the same values/formulas again or enforce details introduced by planning rather than requested by the user.
5. A model chose routine tool-routing steps even after a complete plan existed. It could repeat exhausted searches. Provider billing failures could then appear as a bad question or missing evidence.

## What changed

- The planner emits one evidence list, explicit calculations, and a boolean indicating whether qualitative interpretation is requested. Interpretation always refers back to the original question. Numeric inputs retain separate reported periods and source years. Business scope and qualifiers are grounded in the question; table dimensions are retained without rewriting an already specific title.
- Retrieval ranks scoped content and reserves context for independent observations. Narrative compression shares the existing budget; explicitly requested full tables retain their structure. Whole source paragraphs keep observations and caveats together, and compact reconciliation tables remain selectable.
- Points cite stored passage IDs directly. The application resolves citations to immutable source excerpts. Candidate selection uses the token budget instead of a fixed 18-passage cutoff. Numerical values, currency, scale, period, accounting basis and calculator inputs retain deterministic checks.
- Claim review sees each point beside its evidence and the already rendered numerical outputs. Search wording cannot expand the answer contract. Supported portions can survive an evidence gap; rejected claims and unsupported interpretations are removed.
- LangGraph follows the validated plan without routing-model calls: retrieve each source group, run requested calculations in finalization, allow one focused recovery and at most two finalization calls, then stop. Non-retryable provider failures terminate without repeated searches.
- Tool-level provider failures now reach the UI as provider errors with billing/connection guidance. Query traces now record timestamps and returned answers for future review.

These changes use financial types, source scope, evidence identity and budgets. There are no MSFT/TSLA question-specific answer branches. The configured production model and credentials were not changed.

## Verification

- Full local suite: **520 passed**, plus **70 passing subtests**. This includes numerical/source binding, table preservation, unsupported headings, source isolation across companies, bounded missing-evidence recovery, and quota handling in streaming and invocation paths.
- Seven additional offline checks passed using actual saved evidence/current indexed tables: Tesla cash flow, revenue growth, geographic revenue; Microsoft margins, historical values and financial snapshot; and the four company/year EPS observations. These verify numerical/table stages, not fresh narrative generation.
- Final live run: all 12 cases encountered provider quota errors. The final UI error handling was then verified locally using the recorded failure shape. Fresh end-to-end answer quality remains unverified after the last changes.
- Cohere returned HTTP 403 during earlier live runs. Dense retrieval, BM25 and reciprocal-rank fusion provided the existing fallback. Cohere reranking itself was not successfully validated.

| Case | Acceptance check requiring a fresh live run |
| --- | --- |
| 1 | Full cash-flow table, all rows, columns, units and negative values |
| 2 | Revenue inputs and 0.95% growth, with calculator formula |
| 3 | Gross/operating/net margins: 69.76%, 44.64%, 35.96% |
| 4 | US 47,725; China 20,944; other international 29,021 — million USD |
| 5 | 20% effective tax rate and the statutory reconciliation items; no amount/rate confusion |
| 6 | Supplier/component risks; distinguish receivables concentration from customer revenue concentration |
| 7 | All three business-model dimensions for both companies, plus a supported comparison |
| 8 | All six historical observations and the requested trend explanation |
| 9 | Diluted EPS with the correct company, year and per-share basis |
| 10 | Revenue, operating income, net income, assets and operating cash flow |
| 11 | Management's Azure growth with the correct annual/quarterly basis and capacity outlook |
| 12 | Energy-storage filing observations and transcript outlook kept separate, with a supported comparison |

For case 6, the indexed credit-risk note states that no entity represented 10% or more of total receivables at December 31, 2024 and 2023. Receivables include financing institutions, regulatory-credit customers and other balances. This does **not** establish the absence of customer revenue concentration. The regression reference records that distinction; an explicit broader evidence gap is acceptable.

## Reproduce after provider quota is restored

From `finrag` in PowerShell:

```powershell
$env:LANGSMITH_TRACING='false'
.\.venv\Scripts\python.exe -m pytest tests -q --disable-warnings
.\.venv\Scripts\python.exe evals/evaluate_ragas.py --dataset evals/query_varieties_regression.json --generate-only --output evals/results/query_varieties_after_quota_restore.json --index-version phase1-v2
```

Read the actual answers, source/period attribution and evidence gaps as well as the decision labels. `answered` alone is not a correctness measurement. The model/embedding quota must be restored before the live command can establish acceptance.

Local artifacts (generated and ignored by Git):

- `results/query_varieties_20261023_workflow.json`: final live attempt and provider failure diagnostics.
- `results/query_varieties_20261023_offline_evidence.json`: seven numerical/table checks and rendered results.
- `results/pytest_20261023_final.txt`: final local suite output.
- Earlier `query_varieties_20261023_*.json` files: intermediate debugging runs, **not** final acceptance results.
