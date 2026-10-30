# Current production-profile comparison check

Both comparisons were attempted once using the existing GPT-4o mini planning
and evidence settings. **Neither produced an accepted final answer before its
six-request cap. Answer quality remains unverified under this profile.**

The user declined the proposed additional allowance and authorized only the
12 remaining requests. Those were divided equally between the two questions.
The shared ledger advanced from 388 to **400/400** and no allowance was added.

## Conditions and results

The existing index `phase1-v2-repair-20261024`, 120-second application deadline,
30-second provider timeout and GPT-4o mini semantic parser were preserved.
Answer caching, OpenAI SDK retries, tracing and RAGAS judging were disabled in
the test processes. The production `.env` was not edited.

| Case | Actual provider requests | Observed time until stop | Estimated token cost, USD | Outcome |
| --- | --- | --- | --- | --- |
| Microsoft/Tesla 2024 business models | 6: 4 chat + 2 embedding | 39.27 s | 0.00549604 | Inconclusive: budget stopped verification/recovery |
| Tesla 2025 filing versus Q4 call | 6: 4 chat + 2 embedding | 21.28 s | 0.00287165 | Inconclusive: budget stopped verification |
| Total | 12: 8 chat + 4 embedding | Separate requests, not a latency benchmark | **0.00836769** | No final-answer acceptance |

Business comparison requests 389-394 completed planning, two scoped retrievals,
evidence selection, interpretation synthesis and one claim audit. The filing/call
comparison requests 395-400 completed semantic parsing, planning, two retrievals,
evidence selection and interpretation synthesis. Synthesis is recorded under
the existing `finrag.answer_evidence_verification` stage.

All 12 provider responses were HTTP 200 and included token usage. The ledger
contains 47,885 chat tokens and 177 embedding tokens for these attempts. The
cost estimate uses those actual responses and the existing configured pricing
catalog, including cached-input usage. It is not an invoice or an estimate of
the additional work required to complete either answer.

The observed times include the local HTTP path and end when the capped attempt
returns. They are not successful completion times and cannot establish a speed
improvement over the previously accepted stronger-model runs.

## Budget-stop diagnosis

Both application responses misleadingly report `provider_unavailable`, even
though every transmitted provider request succeeded. Subsequent outbound calls
were blocked locally by the acceptance runner's request guard. The OpenAI SDK
wraps that guard's `RequestAborted` exception in `APIConnectionError`, which the
existing classifier maps to `provider_unavailable`.

This wrapping was reproduced offline with an in-memory HTTP transport and
socket connections blocked. No additional provider call was made. The original
application errors remain preserved in the ledger; the review explicitly labels
these attempts `inconclusive_budget_exhausted` rather than changing their raw
results or claiming a provider outage.

Application telemetry also counts locally blocked model invocations and marks
their usage unavailable. Therefore the reviewed spending/request totals come
from the transmitted-request ledger, not from treating attempted model calls
as paid requests. The captured usage for all 12 transmitted requests is known.

Hashes of production Python files and `.env` match the preflight record.
No production logic, model settings, retrieval, generation or error handling
was changed. The earlier stronger-profile acceptance reviews remain historical
results; they do not constitute acceptance of these incomplete attempts.

## Implication

This check does not establish that GPT-4o mini is suitable or unsuitable for the
two comparisons. An evenly split 12-request budget was insufficient to complete
them through the current pipeline. Do not treat it as approval to run the full
90-question paid suite. The immediate zero-cost follow-up is to make budget
stops visible accurately and review the saved traces before considering further
paid validation; no such production change was made during this task.

Artifacts:

- [Session preflight](results/current_profile_20261027.preflight.json).
- [Reviewed results and offline exception reproduction](results/current_profile_20261027.review.json).
- [Shared request ledger](results/comparison_acceptance_20261026.json), requests 389-400, final two result rows.
- [Business run log](results/current_profile_business_20261027.log) and [filing/call run log](results/current_profile_transcript_20261027.log).
- [Offline review script](results/review_current_profile_20261027.py).
