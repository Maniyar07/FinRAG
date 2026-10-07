# Dataset factory

This package owns the complete example lifecycle:

1. mine or generate evidence-grounded candidates;
2. validate schemas, numbers, quotations, hashes and duplicates;
3. guard the locked benchmark;
4. record independent review decisions;
5. create group-isolated splits; and
6. export conversational SFT JSONL.

For the approved expanded master, `prepare-final` also adds deterministic,
source-grounded calculation and evidence-safety coverage and caps the final export
without discarding rare capabilities:

```powershell
.\.venv\Scripts\python.exe -m fine_tuning prepare-final
```

It does not load a trainable base model or produce adapter weights.

Run through the stable root command:

```powershell
.\.venv\Scripts\python.exe -m fine_tuning --help
```

The explicit package entry point is equivalent:

```powershell
.\.venv\Scripts\python.exe -m fine_tuning.dataset_factory --help
```

Generated files belong in `fine_tuning/work/`; source documents and existing RAG
indexes are read-only inputs.
