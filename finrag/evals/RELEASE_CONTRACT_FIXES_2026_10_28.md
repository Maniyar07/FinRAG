# Shared planning, evidence, recovery and evaluation fixes

Implemented on 28 October 2026 using the saved 90-question results and synthetic
positive/negative controls. **736 tests and 70 subtests pass. No paid provider
requests were made.** The shared acceptance ledger remains 400/400.

These changes replace weak logic in existing components. No new production
helper functions, model stages, packages, company-specific conditions, query-ID
branches or stored gold answers were added. Models, local `.env`, source
documents, indexes and the original 90-question evaluation report are unchanged.
The original evaluation remains the baseline; it has not been rerun or relabelled
as passing. This is implementation plus offline verification, not fresh 85+/90
answer-quality acceptance or approval to deploy.

## 1. Preserve request meaning in planning

In `src/agent/graph.py`, numeric requirements retain their actual financial
metric. The previous assignment overwrote it with the requested object's subject,
which could be a company or business name. Narrative reported periods are also
retained, and business names tolerate ordinary hyphen/space variations while
remaining grounded in the user's wording.

The structured planning response now inventories business subjects before
outputs and declares requirements before formulas, so calculation operands are
defined before their IDs are referenced. This also follows the documented
relationship between schema key order and structured response order.
[Official OpenAI documentation](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat).

`RequestPlanInput` checks actual output references, formula ownership and
self-dependencies. A derived result cannot masquerade as an extra reported fact.
Duplicate scalar requirements for the same output/source/period remain merged;
independent outputs and calculation operands remain distinct. Invalid plans use
the existing bounded planning repair rather than silently entering retrieval.

## 2. Strengthen claim and source binding

`src/financial/coverage.py` no longer treats a matching requirement ID as
sufficient metric proof. Semantic aliases remain supported when the extraction
validator recorded a source-backed request binding. The shared financial matcher
also distinguishes a reported level from a reported movement; ordinary movement
wording such as up/down remains valid.

Table-backed narrative observations carry actual column years, cell values and
source context into synthesis. Each numerical interpretation sentence is checked
against its selected observations and reported columns. Explicit historical
years remain valid; a prior-year value cannot silently become a current result.
Synthesis evidence is narrowed to its selected source observations, with column
context retained for auditing. Existing business/source and semantic checks remain.

The selection schema also had a concrete scalability defect. Reconstructing Q79
from its saved sources produced **1,045 enum values**, beyond the documented
1,000-value schema limit. Selection now uses exact, escaped allowed-ID patterns,
sharing identical definitions through references. This keeps the same strict
per-requirement membership constraint without large repeated enums. Application
checks independently reject unknown, wrong-source or ineligible IDs.
[OpenAI schema constraints](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat).

For that reconstructed schema, size fell from **12,246 to 4,434 bytes**, and from
**7,128 to 1,976 schema tokens**, with zero enum values. These are component/schema
measurements, not a project-wide cost or latency claim. The historical provider
response body was not retained, so this establishes a schema-limit violation
consistent with Q79's rejection, not proof that it was the only rejection cause.

## 3. Preserve supported content during bounded repair

Evidence selection and auditing receive the original requested output clause,
not only a shortened metric label or expanded search wording. Selection also
receives already validated observations from the same source group, so reasons
can be related to the actual reported result. Other companies' facts are excluded.

The audit no longer restricts a whole requested output to one metric-labelled
coverage entry. Its instructions distinguish descriptions, status, lists, causes
and qualitative outlook from requests for financial quantities, and require a
cause for a why question. Unrequested amounts or extra categories are not part
of the answer contract.

If interpretation fails after source selection, the finalization checkpoint keeps
those selections under a plan/source fingerprint. With unchanged inputs, repair
reuses selection and still audits the evidence; selections are not relabelled as
verified. Changed inputs invalidate reuse. The regression control completes a
repair with one retrieval, one selection, two synthesis attempts and one audit.
No new recovery round or model stage was introduced.

## 4. Bound execution and retain accurate failure causes

The deterministic router applies the configured action budget across source
groups and recovery routes. Explicit requirement IDs from failed calls are marked
attempted/refined, so an incompatible statement call cannot remain perpetually
pending. A transcript observation cannot use the filing-only statement capability;
the planner receives a bounded capability error instead of repeated tool calls.

Runtime classification preserves `RequestAborted`, including when the SDK wraps
it as a connection error. Local budget/cancellation stops are non-retryable and
are not presented as a provider outage. Provider request failures retain bounded
HTTP status, error code and parameter metadata without storing raw prompts,
credentials or entire response bodies in the diagnostic detail.

## 5. Make evaluation claims explicit

`evals/evaluate_ragas.py` separates generation-contract status from quality
acceptance. `generation_passed` describes the former; `quality_passed` stays
unassessed rather than being inferred from a returned answer. Report metadata
explains the legacy `passed`/`decision_match` fields; no arbitrary quality
threshold or gold-answer comparison was inserted into runtime logic.

Scoring distinguishes fully scored, partially scored and failed-scoring rows.
Summary counts expose complete metric coverage. When judging fails before any
provider usage is returned, total cost is unavailable rather than asserted to be
zero. Known subtotals remain available. These changes do not convert automated
metric averages into human source/completeness approval.

## Verification and limits

- **Full offline suite:** 736 tests plus 70 subtests passed, with 49 warnings.
  External connections were blocked by the existing offline runner.
- **31 new regression cases** cover metric/subject separation, periods,
  calculation references, source-scoped selection, strict sentence IDs, quantity
  binding, valid movement terminology, aliases, table-year proof, unsupported
  tool routes, shared budgets, recovery reuse, errors and incomplete judging.
- Three existing schema tests now check the allowed and forbidden sentence IDs
  through the equivalent pattern constraint rather than assuming enum storage.
  Their cross-company and broader-scope exclusions remain enforced.
- The saved-data replay checked **77 distinct fact records within their query
  rows**, comparing **76 previous fact-to-requirement matches**. It preserved 75
  and rejected only Q83's false 141,218-million liabilities-change binding.
- Q89's saved wrong-year interpretation is rejected using the actual saved
  source table. A corrected-period control reaches the semantic auditor. This
  control is not a new live answer or an independently accepted interpretation.
- Q79's before/after schemas pass local JSON Schema validation; the oversized
  enum construction is removed. No request was sent to verify provider acceptance.
- Original evaluation SHA-256 remains
  `b6b92c02d2eff471cfa4f7e8f02ea1b2db6e50cdbb956262c0cdb742129d1693`.

The changes span six existing runtime files and the evaluator, with **87 net
additional implementation lines** against the snapshot taken at the start of
this task. Tests, reports and ignored replay artifacts are separate. Git HEAD
and the earlier frozen rollback archive were not changed.

Free-form planning, selection and semantic judgments still require targeted live
acceptance under the deployed model profile. The saved 90-question report cannot
establish their new success rate. Stricter checks can invoke the existing bounded
repair or disclose a remaining gap; no universal correctness, 85+ completion,
latency improvement or deployment-readiness claim is made from offline tests.
Any paid follow-up needs a separate approved hard cap. Restart the application
or evaluator process before testing the updated code.

Artifacts:

- [New regression module](../tests/test_release_contract_regressions.py).
- [Full-suite log](results/release_contracts_full_20261028.log) and [JUnit XML](results/release_contracts_full_20261028.junit.xml).
- [Saved-data replay results](results/release_contract_replay_20261028.json) and [replay script](results/replay_release_contracts_20261028.py).
- [Before-change snapshot manifest](results/priorities_before_20261028/manifest.json).
- [Implementation verification summary](results/release_contracts_20261028.summary.json).
