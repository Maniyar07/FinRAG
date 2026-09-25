# FinRAG evaluation

This directory contains the canonical evaluation dataset, runners, audit notes,
and regression guidance for FinRAG. Generated run outputs belong under
`evals/results/`, which is intentionally ignored by Git.

## Canonical files

- `ragas_gold.json`: the current 60-question, source-grounded evaluation set.
- `evaluate_ragas.py`: end-to-end retrieval and generation evaluation.
- `smoke_multihop.py`: focused multi-hop smoke evaluation.
- `regression_cases.json`: machine-checkable ordinary and multi-hop regression cases.
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

## Run the deterministic regression baseline

Validate the case file without loading the index or calling a model:

```powershell
.\.venv\Scripts\python.exe evals\smoke_multihop.py --validate-only
```

Run a small subset before a full live evaluation:

```powershell
.\.venv\Scripts\python.exe evals\smoke_multihop.py --start 1 --limit 3
```

Run the full baseline and save the detailed report:

```powershell
.\.venv\Scripts\python.exe evals\smoke_multihop.py `
  --cases evals\regression_cases.json `
  --output evals\results\baseline_before_refactor.json `
  --index-version phase1-v2
```

The command exits with status `1` if any declared expectation fails. Reports
include answers, latency, scope, sources, validated facts, calculations,
multi-hop selection, issues, generation mode, and reranker status. Generated
reports remain under `evals/results/` and should not be committed. When a Cohere
trial quota is exhausted, disable reranking consistently for both the baseline
and comparison run rather than comparing two different configurations.
