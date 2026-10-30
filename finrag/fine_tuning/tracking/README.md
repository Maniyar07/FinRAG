# FinRAG MLflow experiment tracking (T28)

This package records the completed Week-8 QLoRA, quality-evaluation, and serving
results in one organized MLflow experiment. It is independent of the FastAPI
runtime and does not retrain or invoke either model.

## What is tracked

The importer creates seven runs with a shared import ID and dataset hash:

1. QLoRA training configuration, losses, runtime, hardware, and adapter hash.
2. Base-model content-quality evaluation.
3. LoRA-adapter content-quality evaluation.
4. Base serving benchmark at concurrency 1.
5. Adapter serving benchmark at concurrency 1.
6. Base serving benchmark at concurrency 2.
7. Adapter serving benchmark at concurrency 2.

Only reviewed JSON/Markdown summaries are logged as artifacts. Raw JSONL, review
CSVs, prompts, source evidence, model weights, and secrets are explicitly blocked
by the tracker.

The results are a historical import. The original Colab run did not capture a Git
commit, so every run is tagged `finrag.lineage_quality=partial_historical`; both
that limitation and the import-time Git state remain visible.

## Run locally from `finrag/`

Use an isolated environment so MLflow does not change application dependencies:

```powershell
python -m venv fine_tuning\work\mlflow\venv
.\fine_tuning\work\mlflow\venv\Scripts\python.exe -m pip install `
  -r fine_tuning\tracking\requirements.txt
```

Validate all inputs and see the planned runs without writing anything:

```powershell
.\fine_tuning\work\mlflow\venv\Scripts\python.exe -m fine_tuning.tracking --dry-run
```

Create the experiment:

```powershell
.\fine_tuning\work\mlflow\venv\Scripts\python.exe -m fine_tuning.tracking
```

The command is idempotent. A second run skips matching records instead of creating
duplicates. Use `--force` only when a deliberate duplicate snapshot is required.

## Open the comparison UI

```powershell
.\fine_tuning\work\mlflow\venv\Scripts\python.exe -m mlflow server `
  --backend-store-uri "sqlite:///fine_tuning/work/mlflow/mlflow.db" `
  --host 127.0.0.1 `
  --port 5000
```

Open <http://127.0.0.1:5000>, select `finrag-week8-lora`, filter by
`finrag.run_kind`, and compare the base/adapter runs. The local database,
artifacts, environment, and import receipt are under `fine_tuning/work/mlflow/`
and are intentionally ignored by Git.

For a shared team server, set `MLFLOW_TRACKING_URI` and
`MLFLOW_EXPERIMENT_NAME`; the importer uses those values without code changes.
