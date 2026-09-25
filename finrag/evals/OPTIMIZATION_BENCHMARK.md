# Optimization benchmark

Use `optimization_cases.json` to measure retrieval changes without changing the
production pipeline. Keep the index, environment variables, models, reranker,
and multi-hop configuration identical between runs.

## 1. Validate the questions

```powershell
.\.venv\Scripts\python.exe evals\smoke_multihop.py `
  --cases evals\optimization_cases.json `
  --validate-only
```

## 2. Save the baseline before changing code

```powershell
.\.venv\Scripts\python.exe evals\smoke_multihop.py `
  --cases evals\optimization_cases.json `
  --output evals\results\optimization_before.json `
  --index-version phase1-v2
```

## 3. Save the optimized run

Restart the process after applying the optimization, then run:

```powershell
.\.venv\Scripts\python.exe evals\smoke_multihop.py `
  --cases evals\optimization_cases.json `
  --output evals\results\optimization_after.json `
  --index-version phase1-v2
```

## 4. Generate the comparison table

```powershell
.\.venv\Scripts\python.exe evals\compare_optimization_runs.py `
  --before evals\results\optimization_before.json `
  --after evals\results\optimization_after.json `
  --output evals\results\optimization_comparison.md
```

Each run stores candidate count, elapsed time, final sources, retrieval modes,
reranker status, answer checks, citation IDs, and invalid citation IDs. The
comparison calculates candidate and latency reductions automatically.

`cohere_scored_sources` counts final sources carrying Cohere scores. It is not
the exact paid input size sent to Cohere because the current production trace
does not expose that internal pool size.

The automatic citation metric is source utilization, not semantic claim
coverage. Manually open every cited source and calculate:

```text
supported factual claims / total factual claims * 100
```

Do not invent a historical latency value. If no compatible pre-change report
exists, save the current run as the baseline for the next optimization. Old
query logs may provide candidate counts, but they do not contain total latency.
