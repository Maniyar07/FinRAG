# FinRAG RAGAS evaluation

This evaluation runs the real FinRAG scope resolver, hybrid retriever, parent-context builder and answer generator. It then scores both retrieval and generation. It does not evaluate a manually pasted answer.

## Files

- `ragas_gold.jsonl`: 30 corrected, uniquely identified, source-grounded questions (10 Tesla, 10 Microsoft and 10 JPMorgan Chase).
- `evaluate_ragas.py`: preflight validation and full RAGAS runner.
- `requirements-eval.txt`: evaluation-only dependency pin at the project root.
- `gold_audit.md`: changes made to the submitted questions and remaining source caveat.

Each JSONL record has one reference answer, an explicit retrieval scope, and one or more independently attributed reference contexts. Cross-document and cross-company records therefore retain the source, section, physical PDF page and speaker for each piece of evidence.

## Before evaluation

Use the six 2024 source files with these exact canonical names in `Data/`:

```text
JPM_2024_10K.pdf
JPM_2024_TRANSCRIPT.pdf
MSFT_2024_10K.pdf
MSFT_2024_TRANSCRIPT.html
TSLA_2024_10K.pdf
TSLA_2024_TRANSCRIPT.html
```

If the `v1` index was built before all six files were present, build a new version instead of modifying the old index:

```powershell
python ingest_documents.py --index-version ragas-v1
```

Install evaluation dependencies:

```powershell
python -m pip install -r requirements-eval.txt
```

## Run safely

First validate the dataset without RAGAS calls or judge cost:

```powershell
python evals/evaluate_ragas.py --validate-only
```

Then run a small end-to-end smoke evaluation:

```powershell
python evals/evaluate_ragas.py --index-version ragas-v1 --limit 3
```

Finally run all 30 records:

```powershell
python evals/evaluate_ragas.py --index-version ragas-v1
```

Detailed results are written to `evals/results/ragas_results.json`. The output includes source coverage plus RAGAS faithfulness, answer relevancy, factual correctness, context precision and context recall. A score is diagnostic, not proof: review low-scoring rows and a sample of high-scoring rows manually.

`--allow-missing-sources` is only for checking JSONL structure before every source is available. Do not use it to present a full evaluation result.
