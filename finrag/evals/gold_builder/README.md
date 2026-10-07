# Grounded gold-dataset builder

This workflow creates reference answers from the immutable indexed corpus without
using `ChatService`. That separation prevents the RAG system under evaluation from
choosing its own gold evidence.

The checked-in `questions_30.json` contains 30 question specifications. A
specification identifies every required source and provides search hints, but it
does not contain a reference answer. Reference answers are created from retrieved
parent evidence, deterministically validated, and blocked from export until a human
reviewer approves them.

Run commands from the `finrag/` directory.

## 1. Check evidence retrieval without API calls

Start with a few rows:

```powershell
.\.venv\Scripts\python.exe -m evals.gold_builder.builder build `
  --retrieve-only `
  --ids 61 62 63
```

Then check all 30:

```powershell
.\.venv\Scripts\python.exe -m evals.gold_builder.builder build --retrieve-only
```

This writes ignored working files under `evals/gold_builder/work/`.

## 2. Extract grounded draft answers

An OpenAI API key is required. Each question is checkpointed immediately, so an
interrupted run can resume without repeating completed calls.

```powershell
.\.venv\Scripts\python.exe -m evals.gold_builder.builder build `
  --model gpt-4o-mini `
  --resume
```

The extractor gets a maximum of two attempts per question. Evidence is ranked using
both lexical and exact-number coverage, then reduced to compact source-faithful
windows. The model receives a per-requirement evidence diagnostic and a second
attempt receives the deterministic validation errors from the first attempt.

Outputs:

- `work/drafts.json`: complete questions, focused evidence, every extraction attempt,
  the final structured draft, and deterministic checks.
- `work/review.csv`: compact human-review sheet.

## 3. Review

Open `work/review.csv`. For every row:

1. Compare the reference with the displayed evidence quotes.
2. Check company, metric, period, unit, qualifier, and reported-versus-guidance status.
3. Check every arithmetic input and result.
4. Set `status` to `approved`, `rejected`, or `needs_edit`.
5. Add reviewer notes when useful.

The CSV is an approval surface. Edit a faulty reference in `drafts.json`, rerun
validation, regenerate the review CSV, and then approve it. Never approve a row that
has `validation_passed=False`.

## 4. Revalidate after edits

```powershell
.\.venv\Scripts\python.exe -m evals.gold_builder.builder validate
```

Validation checks:

- all required evidence targets are covered;
- every cited evidence ID exists and belongs to the claimed requirement;
- every evidence quote occurs in the retrieved source excerpt;
- reported numeric facts occur in their cited evidence;
- reference numbers occur in evidence, the question, or a verified calculation;
- calculation operations and operand order match the question specification;
- arithmetic is recomputed with `Decimal` and declared rounding;
- every required source is cited;
- references contain no builder citation markup.

## 5. Export approved records

Export is blocked unless all records are approved and pass deterministic validation:

```powershell
.\.venv\Scripts\python.exe -m evals.gold_builder.builder export
```

For an intentionally partial development file:

```powershell
.\.venv\Scripts\python.exe -m evals.gold_builder.builder export --allow-partial
```

The default final file is `evals/ragas_gold_new_30.json`, compatible with the existing
RAGAS evaluator.

## 6. Validate and evaluate the exported set

```powershell
.\.venv\Scripts\python.exe evals\evaluate_ragas.py `
  --dataset evals\ragas_gold_new_30.json `
  --validate-only

.\.venv\Scripts\python.exe evals\evaluate_ragas.py `
  --dataset evals\ragas_gold_new_30.json `
  --index-version phase1-v2 `
  --generate-only `
  --output evals\results\new_30_answers.json

.\.venv\Scripts\python.exe evals\evaluate_ragas.py `
  --dataset evals\ragas_gold_new_30.json `
  --index-version phase1-v2 `
  --score-input evals\results\new_30_answers.json `
  --output evals\results\new_30_ragas.json
```

## Scientific limitations

Passing the local validators proves provenance, numeric support, source coverage, and
arithmetic consistency. It cannot prove that a narrative interpretation is the only
reasonable interpretation. Human review is therefore mandatory before export.

