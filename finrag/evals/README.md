# FinRAG evaluation

This directory contains the canonical evaluation dataset, runners, audit notes,
and regression guidance for FinRAG. Generated run outputs belong under
`evals/results/`, which is intentionally ignored by Git.

## Canonical files

- `ragas_gold_90.json`: the 90-question gold set used for current LangGraph regression runs.
- `ragas_gold.json`: the smaller default source-grounded evaluation set.
- `evaluate_ragas.py`: end-to-end retrieval and generation evaluation.
- `MANUAL_REGRESSION_QUESTIONS.md`: 30-question manual end-to-end checklist.
- `gold_audit.md`: provenance and corrections for the gold set.
- `REGRESSION_CHECKLIST.md`: checks required before accepting pipeline changes.
- `../requirements-eval.txt`: evaluation-only dependencies.

The legacy 30-question JSONL dataset is retained as `ragas_gold_old.jsonl` only
for historical reproducibility; it is not the default evaluation input.

## Validate without model calls

From the `finrag/` directory:

```powershell
.\.venv\Scripts\python.exe evals\evaluate_ragas.py --validate-only
```

## Run the current evaluation

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-eval.txt
.\.venv\Scripts\python.exe evals\evaluate_ragas.py --index-version phase1-v2
```

The default detailed output is `evals/results/ragas_results.json`. Treat scores
as regression signals rather than universal accuracy: review low-scoring rows,
a sample of high-scoring rows, citation validity, and comparison-group coverage.

## Run the 90-question LangGraph evaluation

The checked-in 90-question set is `ragas_gold_90.json`; pass it explicitly
because the evaluator's default dataset is `ragas_gold.json`.

Generate answers and save one checkpoint after each question, without running
the separate RAGAS judge calls:

```powershell
.\.venv\Scripts\python.exe evals\evaluate_ragas.py `
  --dataset evals\ragas_gold_90.json `
  --generate-only `
  --output evals\results\langgraph_90_answers.json
```

Each result records the gold outcome and expected decision, actual decision and
answer, per-question generation latency, retrieved contexts and source
metadata, extracted citations with source-ID validation, compact LangGraph
tool calls and results, calculator inputs/results, runtime and tool errors, and
scope matching. The report summary counts expected decision matches and
expected clarifications separately; an actual `clarify` decision passes when
the row declares `"expected_decision": "clarify"`.

Existing answerable gold rows without `expected_decision` are treated as
`answered` because they provide a reference answer. For a clarification case,
add an explicit decision and a concise description of the expected behavior:

```json
{
  "expected_decision": "clarify",
  "expected_outcome": "Ask the user to choose the fiscal year before searching."
}
```

`reference` may be omitted for a non-answer expectation. Its `scope` can leave
unknown fields as empty lists. Such rows are not sent to RAGAS answer-quality
scoring. To score a saved answer report later, use a different output path:

```powershell
.\.venv\Scripts\python.exe evals\evaluate_ragas.py `
  --dataset evals\ragas_gold_90.json `
  --score-input evals\results\langgraph_90_answers.json `
  --output evals\results\langgraph_90_scored.json
```

Generated reports remain under `evals/results/` and should not be committed.

## Run targeted gold questions

Use the canonical gold dataset to rerun selected LangGraph questions after a fix:

```powershell
.\.venv\Scripts\python.exe evals\evaluate_ragas.py `
  --dataset evals\ragas_gold_90.json `
  --ids 81 `
  --generate-only `
  --output evals\results\q081_after_fix.json
```

Use a new output filename for each comparison. The evaluator records expected and
actual decisions, citations, tool calls/results, calculator inputs/results, errors,
and per-question latency. Rows and summaries also include per-stage model usage
and separate embedding usage; RAGAS scoring records its judge usage separately.
Recovered tool errors remain in the trace without failing a fully validated
answer. Unknown provider prices remain unknown, and LLM cost excludes embedding
and reranking charges. No backend-selection or multi-hop flag is used; all
in-scope requests run through LangGraph. Keep generated reports under `evals/results`
and out of commits.
