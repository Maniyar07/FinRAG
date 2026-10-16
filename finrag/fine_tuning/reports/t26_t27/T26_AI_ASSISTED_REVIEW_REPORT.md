# T26 AI-Assisted Content Review

## Scope

- Reviewed examples: 40 held-out test cases
- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Adapter: `finrag-lora`
- Dataset SHA-256: `ce5ee27758605e8d7cf37583011b42eccaa9af3a6c365c56e4b6b7866cb43f07`
- Responses were randomized as A/B before content scoring.
- JSON formatting was excluded from content scoring.

## Results

| Metric | Base | LoRA adapter |
|---|---:|---:|
| Factual correctness | 35/40 (87.5%) | 37/40 (92.5%) |
| Completeness | 31/40 (77.5%) | 33/40 (82.5%) |
| Numerical correctness | 29/31 (93.5%) | 28/30 (93.3%) |
| Evidence support | 36/40 (90.0%) | 38/40 (95.0%) |
| Hallucination rate | 4/40 (10.0%) | 2/40 (5.0%) |
| Overall acceptable | 30/40 (75.0%) | 33/40 (82.5%) |

## Pairwise preference

- Base preferred: 7
- Adapter preferred: 10
- Tie: 23

## Material findings

- The adapter consistently obeyed the output contract, but content review still found targeted omissions and row-selection errors.
- Test 3 contains a flawed reference target: the source reports a dividend of USD 0.83 per share, not USD 0.83 million. The base answer is correct on this point and the adapter reproduces the bad target unit.
- Test 13 is ambiguous in wording; the validated target expects the operating-lease discount rate, while both responses selected other lease rows.
- The training/test data should be corrected before any future retraining or final production-quality claim.

## Review status

This AI-assisted review completes the content-analysis artifact for T26. It is not an independent human review. A human owner should spot-check the flagged cases and record sign-off before production acceptance.
