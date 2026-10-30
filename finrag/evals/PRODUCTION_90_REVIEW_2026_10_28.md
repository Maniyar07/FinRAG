# Production 90-question evaluation review

Reviewed on 28 October 2026. The saved evaluation completed on 27 October 2026
at 22:12:40 IST. **Recommendation: hold deployment until the demonstrated answer,
planning and execution failures are corrected.** Completion of the run and a
high average RAGAS score do not establish release acceptance.

This review made **zero provider requests** and changed no application code,
configuration, index or original evaluation results. It reviews all 90 saved
answers against their supplied references and examines failure traces and
selected retrieved evidence. This is not fresh generation, independent model
judging, or a guarantee that every remaining claim is correct.

## Integrity and outcome

Input: [production_90_full.json](results/production_90_full.json).
SHA-256: `b6b92c02d2eff471cfa4f7e8f02ea1b2db6e50cdbb956262c0cdb742129d1693`.
All 90 unique gold IDs are present, with no missing/extra IDs or changed
questions/references. Recomputing the summary from the saved rows reproduced
the stored summary exactly.

The index is `phase1-v2-repair-20261024`; recorded application usage identifies
GPT-4o mini, and the configured judge is GPT-4o mini. The evaluation path calls
`ChatService.ask` directly. IDs throughout this review refer to
`ragas_gold_90.json`; they are not IDs from `query_varieties_regression.json`.

| Recorded outcome | Questions |
| --- | ---: |
| Answered | 67 |
| Insufficient evidence | 19 |
| Plan validation failed | 3 |
| Provider request error | 1 |
| Total | 90 |

All gold rows expect `answered`. The reported decision-match rate is **74.44%**.
It is a status/operational measure, not an accuracy rate: `passed` does not
incorporate RAGAS thresholds or this manual completeness review.

All 67 answered rows contain citations. All 103 recorded citation-ID entries
resolve; there are zero invalid IDs. That checks source identity, not whether
each citation establishes the claim. Sixteen rows with no citations are among
the unsuccessful outcomes, not among the 67 answered rows.

The sole scope mismatch is Q42: resolved scope leaves document type unset,
while the gold scope requires 10-K. Saved sources are all 10-Ks for the intended
companies/year, and its figures match the reference. This diagnostic alone does
not demonstrate wrong-source retrieval.

## RAGAS coverage and interpretation

| Metric | Mean | Available scores / 67 answered |
| --- | ---: | ---: |
| Faithfulness | 0.8808 | 65 |
| Context precision | 0.8710 | 66 |
| Context recall | 0.9577 | 67 |
| Response relevancy | 0.7629 | 67 |
| Factual correctness | 0.7806 | 66 |

The reported aggregate is **0.8506**, the mean of these metric means over
answered rows. The other 23 questions receive no answer-quality score.
The report's coverage-adjusted composite is **0.6332**; this is also not an
independent accuracy estimate.

Only **63 questions have all five scores**. Four answered rows are partially
scored: Q33 context precision had a connection error; Q47 faithfulness and Q76
factual correctness timed out; Q89 faithfulness reached an output-token limit.
These are not the earlier systematic TLS failure: scoring succeeded across the
run, with four individual missing metric results.

Some scores need human interpretation. Q23's combined cash/investment amount
94,565 million matches the gold reference and cited table despite factual
correctness 0. Q24 and Q51 likewise match their reference figures despite
faithfulness 0. Q74 supplies the requested tariff explanation despite relevancy
0. Conversely, Q83 receives faithfulness 1 while adding an incorrect financial
label. Neither low scores nor high scores should decide acceptance alone.

## Confirmed problems in answers marked successful

| Question | Finding | Why it matters |
| --- | --- | --- |
| Q14: AI resource requirements | Returns generic manufacturing/supply risks instead of compute, memory, energy and thermal requirements, which are in retrieved source S6007127036082993332 | Related text is accepted without covering the requested relationship |
| Q76: total gross-margin headwinds | Replaces the historical lower-fixed-cost-absorption/tariff explanation with a future energy-storage margin-compression quote; both appear in S18225565360489512408 | Wrong business/temporal relationship despite a valid source ID |
| Q77: why robotaxi metrics were not meaningful | Repeats the premise without the early-deployment/testing explanation present in the same source | An answer to a why question must include the cause |
| Q80: why AI5 was critical | Quotes that it is critical but omits the Optimus/AI-chip dependency in the reference; coverage of that rationale requires further evidence review | Quoting the priority does not explain its technical reason |
| Q83: liabilities change | Correctly calculates 15,932 million and 12.72%, but also labels 141,218 million as total-current-liabilities change | A balance is falsely presented as a change |
| Q89: quarterly versus annual automotive margins | Interpretation uses 18.4% as the annual comparison figure, while its own table shows 17.8% for 2025 and 18.4% for 2024 | Wrong-year synthesis passes application validation; quarterly excluding-credit and annual measures also need their different bases kept explicit |

Additional completion issues include Q12, which lists both revenue values
without stating the requested decrease; Q54, which omits the 23% increase;
and Q66, which omits the reference's constant-currency growth. Several other
answers duplicate quotations or add unrelated context. The appendix distinguishes
substantive problems, reference-detail gaps, presentation issues and judge
discrepancies instead of counting them all as equivalent failures.

## Unsuccessful outcomes and diagnosis

Unsuccessful IDs: **2, 4, 7, 8, 13, 22, 31, 36, 37, 38, 39, 43, 44, 62, 67,
75, 79, 82, 84, 85, 87, 88, 90**.

- **Semantic metric binding:** Q22 plans only `Intelligent Cloud` instead of
  operating income. Q31/Q37/Q38/Q39/Q43 use company names as metrics even though
  the evidence questions ask for specific financial quantities. Requested values
  are present in the saved contexts. Q44 retrieves Microsoft's 171,008 million
  gross margin but still marks it unavailable. These are not established corpus
  gaps. Q7 also collapses total/commercial RPO into the same unqualified metric.
- **Fact/narrative and calculation coverage:** Q4 retrieves the 50% growth and
  44-point acquisition contribution but emits an unavailable fact. Q13 marks an
  increase unavailable while quoting its exact 684 million/13% change. Q75 makes
  an unnecessary fact prerequisite for the supported Boolean statement that
  Optimus is not design-locked. Q82 verifies both ratio inputs but adds a third
  reported-ratio requirement that prevents calculation. Q62 adds a missing
  prior-year fact despite quoting the requested margin, decline and explanation.
- **Selection and excessive rejection:** Q2's requested segment description and
  Q8's five-milestone table are retrieved. Q2's audit demands additional product
  and revenue details; Q8 does not select the actual list. Q87 retrieves the 67%
  guidance but fails to retain it. Q88 rejects physical dependencies for lacking
  financial details, although its question explicitly asks for dependencies;
  the saved retrieval also lacks the expected capacity-constraint wording.
- **Plan validation:** Q36 omits a requested difference output. Q84/Q85 contain
  invalid requested-object references. All three stop before retrieval.
- **Wrong tool route and repeated failures:** Q67 classifies transcript RPO
  recognition as a financial statement. Thirteen incompatible statement-tool
  calls repeat a non-retryable document-scope error until `GraphRecursionError`.
  Its final insufficient-evidence label conceals this execution failure.
- **Provider rejection:** Q79 fails with `OpenAIInvalidRequestError`. The saved
  report does not retain the underlying provider rejection body; its exact
  reason is unresolved. Do not infer bad credentials, exhausted credits, schema
  incompatibility or context size from the generic message alone.
- **Lost business subject:** Q90 retains the energy-storage subject for the
  2024 clause but loses it in the 2025 requirements. It accepts automotive
  backlog and shared product spending, then demands unrequested numeric figures
  while rejecting retrieved storage deployment/margin outlook. Source
  S18225565360489512408 already contains the requested 2026 storage outlook.

## Cost, token usage and latency

| Recorded usage | Application | RAGAS judging |
| --- | ---: | ---: |
| LLM call events | 251 | 971 |
| Embedding call events | 77 | 134 |
| Chat tokens | 1,503,827 | 2,206,404 |
| Embedding input tokens | 2,871 | 5,371 |
| Known estimated token-cost subtotal, USD | 0.23188677 | 0.35779937 |

**Known combined estimated subtotal: USD 0.58968614. Exact run total: unavailable.**
Q33 judge usage and Q79 application usage contain unavailable token records.
These are configured token-price estimates, not an invoice or all historical
development spending. Telemetry call events are not an independently audited
HTTP request ledger; failed or locally rejected invocations can differ from paid
provider requests. Judging is a substantial separate workload, not an ordinary
user-query cost.

| Timing | Median | Mean | p95 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Application generation, 90 rows | 15.48 s | 18.46 s | 42.48 s | 111.56 s |
| RAGAS scoring, 67 rows | 34.88 s | 34.69 s | 48.56 s | 63.32 s |
| Generation plus scoring, 90 rows | 41.66 s | 44.29 s | 74.39 s | 111.56 s |

Use application generation time when discussing query responsiveness. The total
includes evaluation-only scoring, with no scoring for unsuccessful questions.
It is not production response latency. The slowest application outcomes include
Q88 at 111.56 s and Q90 at 92.77 s, both incomplete. Planning is a major measured
stage: 675.49 cumulative seconds and 109 LLM events across 90 rows. Stage spans
can be nested or overlapping and should not simply be added together.

This direct-service run does not measure HTTP/network queueing, browser paint,
concurrent-load limits, cancellation, memory ceilings or rollback. Cache-hit
state is absent from the saved traces; the supplied command disabled caching,
and every row records at least one model call, but the report does not retain
enough configuration to independently attest the complete runtime/cache profile.

## Bounded path to release acceptance

1. Correct the demonstrated wrong-period, wrong-business, wrong-quantity and
   missing-relationship acceptances, using saved examples as negative cases and
   keeping positive cases to detect excessive rejection.
2. Preserve requested financial metrics, qualifiers, Boolean/narrative outputs
   and calculation operands in plans. Avoid inventing extra prerequisites.
3. Stop incompatible/non-retryable tool routes after the bounded failure;
   preserve the real failure category. Investigate Q79 from its actual provider
   rejection details rather than changing credentials or models speculatively.
4. Verify fixes offline with these saved contexts and responses first. A later
   paid check should rerun the affected subset plus representative positive
   controls under one approved cap, not repeat the whole suite for diagnosis.
5. After quality acceptance, verify the HTTP application and expected workload
   before deployment. Keep the frozen rollback artifact available.

This review does not recommend a model upgrade as a substitute for the measured
logic issues. No new provider-call allowance was requested or consumed.

The [machine-readable review](results/production_90_review_20261028.json) retains
all 90 row IDs, metric values and errors, diagnostic labels and review notes.

## Per-question review

The table below is a review index, not a newly measured accuracy score. Core
covered means no specific discrepancy was identified against the supplied
reference in this review; it does not certify every source claim.

| ID | Recorded outcome | Review | Finding |
| --- | --- | --- | --- |
| Q001 | answered | Core covered | The core answer matches the supplied reference in this manual review; not an independent accuracy certification. |
| Q002 | insufficient_evidence | Evidence rejected | The exact requested segment description is present in retrieved evidence; the audit demands extra product/revenue detail. |
| Q003 | answered | Presentation | Correct core dependency list is duplicated across sources. |
| Q004 | insufficient_evidence | Planning/fact coverage | The retrieved text gives 50% growth and 44-point acquisition impact; the fact-only output still returns unavailable. |
| Q005 | answered | Change presentation | Returns 14 percent without clearly saying increase; the reference also gives 15% Commercial cloud growth. |
| Q006 | answered | Presentation | The same useful-life explanation is repeated verbatim. |
| Q007 | insufficient_evidence | Metric qualification/date binding | Both 375 and 368 billion are retrieved; total/commercial requirements share the same metric and omit distinguishing qualifiers. |
| Q008 | insufficient_evidence | Table selection/coverage | The five milestones are in a retrieved HTML table; the answer retains introductory text rather than the list. |
| Q009 | answered | Presentation | Core xAI agreement is present twice; the low factual score needs interpretation rather than automatic rejection. |
| Q010 | answered | Relevance | Core regulatory-credit explanation is present, followed by substantial unrequested accounting/revenue detail. |
| Q011 | answered | Relevance | Core explanation is present, followed by substantial extra detail; verify the requested Note 2 attribution when tightening citations. |
| Q012 | answered | Missing requested change | Both annual revenue values are shown, but the requested decrease of approximately 2.86 billion is not stated. |
| Q013 | insufficient_evidence | Contradictory missing fact | The answer marks the SG&A increase unavailable while quoting the exact 684 million/13% increase and drivers. |
| Q014 | answered | Substantive answer gap | AI resource requirements are replaced with generic manufacturing/supply risks despite retrieved compute/memory/energy/thermal wording. |
| Q015 | answered | Presentation | Correct Bitcoin values appear in a raw two-year row; the cost-basis and fair-value labels should be explicit. |
| Q016 | answered | Core covered | The core answer matches the supplied reference in this manual review; not an independent accuracy certification. |
| Q017 | answered | Core covered | The core answer matches the supplied reference in this manual review; not an independent accuracy certification. |
| Q018 | answered | Core covered | The core answer matches the supplied reference in this manual review; not an independent accuracy certification. |
| Q019 | answered | Core covered | The core answer matches the supplied reference in this manual review; not an independent accuracy certification. |
| Q020 | answered | Core covered | The core answer matches the supplied reference in this manual review; not an independent accuracy certification. |
| Q021 | answered | Core covered | The core answer matches the supplied reference in this manual review; not an independent accuracy certification. |
| Q022 | insufficient_evidence | Metric binding | The requested metric is operating income, but the plan metric is only Intelligent Cloud; 44,589 is retrieved. |
| Q023 | answered | Judge discrepancy | Combined cash/investment amount 94,565 matches the reference and cited table despite factual_correctness=0. |
| Q024 | answered | Judge discrepancy | Net income 3,855 matches the reference and retrieved cash-flow row despite faithfulness=0. |
| Q025 | answered | Core covered | The core answer matches the supplied reference in this manual review; not an independent accuracy certification. |
| Q026 | answered | Core covered | The core answer matches the supplied reference in this manual review; not an independent accuracy certification. |
| Q027 | answered | Core covered | The core answer matches the supplied reference in this manual review; not an independent accuracy certification. |
| Q028 | answered | Core covered | The core answer matches the supplied reference in this manual review; not an independent accuracy certification. |
| Q029 | answered | Core covered | The core answer matches the supplied reference in this manual review; not an independent accuracy certification. |
| Q030 | answered | Core covered | The core answer matches the supplied reference in this manual review; not an independent accuracy certification. |
| Q031 | insufficient_evidence | Metric binding | Plan metrics are Microsoft and Tesla rather than net income; both requested amounts are retrieved. |
| Q032 | answered | Core covered | The core answer matches the supplied reference in this manual review; not an independent accuracy certification. |
| Q033 | answered | Partial metric failure | The amounts and 2,176 difference match the reference; context_precision failed with a connection error. |
| Q034 | answered | Core covered | The core answer matches the supplied reference in this manual review; not an independent accuracy certification. |
| Q035 | answered | Core covered | The core answer matches the supplied reference in this manual review; not an independent accuracy certification. |
| Q036 | validation_failed | Invalid plan | The difference output is missing from the plan; rejected before retrieval. |
| Q037 | insufficient_evidence | Metric binding | Plan metrics are company names rather than R&D expense; both requested amounts are retrieved. |
| Q038 | insufficient_evidence | Metric binding | Plan metrics are company names rather than revenue; both requested amounts are retrieved. |
| Q039 | insufficient_evidence | Metric/scope binding | Plan metrics are company names, losing geographic revenue distinctions even though both amounts are retrieved. |
| Q040 | answered | Judge discrepancy | Amounts and 176,799 difference match the reference despite faithfulness=0.25. |
| Q041 | answered | Core covered | The core answer matches the supplied reference in this manual review; not an independent accuracy certification. |
| Q042 | answered | Scope diagnostic | Values and calculation match; scope_match is false because resolved document type is unset, although saved sources are all 10-Ks. |
| Q043 | insufficient_evidence | Metric binding | Plan metrics are company names rather than net income; both amounts are retrieved. |
| Q044 | insufficient_evidence | Metric binding | Microsoft amount 171,008 is retrieved but remains unavailable under a company-prefixed metric. |
| Q045 | answered | Core covered | The core answer matches the supplied reference in this manual review; not an independent accuracy certification. |
| Q046 | answered | Core covered | The core answer matches the supplied reference in this manual review; not an independent accuracy certification. |
| Q047 | answered | Partial metric failure | Values and 36,602 difference match; faithfulness judging timed out. |
| Q048 | answered | Core covered | The core answer matches the supplied reference in this manual review; not an independent accuracy certification. |
| Q049 | answered | Core covered | The core answer matches the supplied reference in this manual review; not an independent accuracy certification. |
| Q050 | answered | Core covered | The core answer matches the supplied reference in this manual review; not an independent accuracy certification. |
| Q051 | answered | Judge discrepancy | Assets 512,163 match the reference and indexed table despite faithfulness=0. |
| Q052 | answered | Core covered | The core answer matches the supplied reference in this manual review; not an independent accuracy certification. |
| Q053 | answered | Core covered | The core answer matches the supplied reference in this manual review; not an independent accuracy certification. |
| Q054 | answered | Missing requested change | Returns 168.9 billion but omits the requested 23% revenue increase. |
| Q055 | answered | Core covered | The core answer matches the supplied reference in this manual review; not an independent accuracy certification. |
| Q056 | answered | Judge discrepancy | 2023 net income 14,974 matches the reference despite faithfulness=0. |
| Q057 | answered | Change presentation | Returns 16 percent without explicitly identifying it as revenue growth. |
| Q058 | answered | Core covered | The core answer matches the supplied reference in this manual review; not an independent accuracy certification. |
| Q059 | answered | Judge discrepancy | Current liabilities 141,218 match the reference despite faithfulness=0. |
| Q060 | answered | Judge discrepancy | Current liabilities 125,286 match the reference despite faithfulness=0. |
| Q061 | answered | Core covered | The core answer matches the supplied reference in this manual review; not an independent accuracy certification. |
| Q062 | insufficient_evidence | Unrequested prerequisite | The requested margin, two-point decline and reasons are quoted, but an extra prior-year fact blocks completion. |
| Q063 | answered | Core covered | The core answer matches the supplied reference in this manual review; not an independent accuracy certification. |
| Q064 | answered | Presentation | Core bookings explanation is present with some overlapping additional quotations. |
| Q065 | answered | Coverage detail | Consistent Azure AI stack is correctly quoted; the broader internal/customer/ISV explanation is omitted. |
| Q066 | answered | Coverage detail | 368 billion and 37% growth are present; the 35% constant-currency growth in the reference is omitted. |
| Q067 | insufficient_evidence | Wrong route/nontermination | RPO recognition is classified as a statement; 13 incompatible get_financial_statement calls end in GraphRecursionError, surfaced as insufficient evidence. |
| Q068 | answered | Core covered | The core answer matches the supplied reference in this manual review; not an independent accuracy certification. |
| Q069 | answered | Relevance | Capex level/drivers are covered; an extra revenue-growth variability quote is placed under capex variability. |
| Q070 | answered | Period clarity | Correct 67% guidance is followed by a historical 68% result under the same reasons section. |
| Q071 | answered | Core covered | The core answer matches the supplied reference in this manual review; not an independent accuracy certification. |
| Q072 | answered | Core covered | The core answer matches the supplied reference in this manual review; not an independent accuracy certification. |
| Q073 | answered | Relevance/attribution | Core storage commentary is present, but extra Powerwall, tariff and detached output-constraint quotes dilute the requested storage explanation. |
| Q074 | answered | Judge discrepancy | The tariff explanation matches the reference despite response_relevancy=0. |
| Q075 | insufficient_evidence | Boolean treated as numeric fact | The source says Optimus is not design-locked, but the plan also demands an unavailable fact for Optimus. |
| Q076 | answered | Substantive answer error | Reported total-margin headwinds are replaced with a future energy-storage margin-compression quote. Lower fixed-cost absorption is retrieved but omitted. |
| Q077 | answered | Substantive answer gap | Repeats that robotaxi metrics are not meaningful without giving the requested early-deployment/testing reason, available in the same source. |
| Q078 | answered | Judge discrepancy | Country list and year-end backlog match the reference despite factual_correctness=0.5. |
| Q079 | error | Provider request rejected | An OpenAIInvalidRequestError occurs during verification. The saved report lacks the raw provider rejection reason; do not infer quota/authentication/schema cause. |
| Q080 | answered | Substantive coverage gap | Quotes that AI5 is critical without explaining the Optimus/AI-chip dependency described in the reference; retrieved excerpts need further coverage review. |
| Q081 | answered | Judge discrepancy | Inputs and 11.5% calculation match the reference despite faithfulness=0.4. |
| Q082 | insufficient_evidence | Calculation dependency | Both ratio inputs are verified, but an additional reported gross profit percentage requirement blocks calculation. |
| Q083 | answered | Substantive answer error | Correct calculation is accompanied by an extra falsely labelled change of 141,218 million; actual change is 15,932 million. |
| Q084 | validation_failed | Invalid plan | Evidence requirement references do not match actual requested objects; rejected before retrieval. |
| Q085 | validation_failed | Invalid plan | Evidence requirement references do not match actual requested objects; rejected before retrieval. |
| Q086 | answered | Core covered | Both RPO values and 99 billion increase match the reference. |
| Q087 | insufficient_evidence | Guidance fact binding | The 67% guidance is retrieved but unavailable in the final comparison; the earlier 69% is retained. |
| Q088 | insufficient_evidence | Retrieval plus audit coverage | Spending guidance and physical datacenter dependencies are retrieved, but audit demands financial details; the expected capacity-constraint wording was not found in saved contexts. |
| Q089 | answered | Substantive answer error | Interpretation assigns the 2024 annual margin of 18.4% to the 2025 comparison; the cited table gives 17.8% for 2025. Quarterly excluding-credit versus annual basis also needs clarity. |
| Q090 | insufficient_evidence | Subject/coverage loss | Energy-storage subject is lost in the 2025 requirements; automotive backlog/shared spending are retained while retrieved storage outlook is rejected for unrequested figures. |
