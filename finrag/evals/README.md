# FinRAG evaluation

This directory contains the canonical evaluation dataset, runners, audit notes,
and regression guidance for FinRAG. Generated run outputs belong under
`evals/results/`, which is intentionally ignored by Git.

## Canonical files

- `ragas_gold.json`: the current 60-question, source-grounded evaluation set.
- `evaluate_ragas.py`: end-to-end retrieval and generation evaluation.
- `smoke_multihop.py`: focused multi-hop smoke evaluation.
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
