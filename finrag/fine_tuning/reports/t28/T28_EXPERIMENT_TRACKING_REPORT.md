# T28 — MLflow Experiment Tracking Report

## Outcome

The completed QLoRA training, base-versus-adapter quality evaluation, and vLLM
serving benchmarks are recorded in the local MLflow experiment
`finrag-week8-lora`. The experiment contains seven finished runs and uses one
dataset lineage hash across training and evaluation.

- MLflow version: `3.17.1`
- Backend: local SQLite
- Experiment ID: `1`
- Import ID: `3f776896b1c14383db5667c77e4c48609ede1aed509e3e5da3ff498b00b3cf97`
- Dataset SHA-256: `ce5ee27758605e8d7cf37583011b42eccaa9af3a6c365c56e4b6b7866cb43f07`
- Run status: `7/7 FINISHED`
- Idempotency check: second import created `0` and skipped all `7` existing runs

The local database and artifact store remain under `fine_tuning/work/mlflow/`
and are intentionally excluded from Git. Stable verification metadata is in
[`mlflow_import_manifest.json`](mlflow_import_manifest.json).

## Organized run layout

| Order | Run | Kind | Model role |
|---:|---|---|---|
| 1 | `01-training-qlora` | Training | Adapter training |
| 2 | `02-quality-base` | Content evaluation | Base |
| 3 | `03-quality-adapter` | Content evaluation | Adapter |
| 4 | `04-serving-c1-base` | Serving benchmark | Base |
| 5 | `05-serving-c1-adapter` | Serving benchmark | Adapter |
| 6 | `06-serving-c2-base` | Serving benchmark | Base |
| 7 | `07-serving-c2-adapter` | Serving benchmark | Adapter |

Every run is tagged by task, run group, run kind, model role, import ID, dataset
hash, and lineage quality. This supports direct filtering and comparison in the
MLflow UI.

## Training run

The training run records the base model, 800-example dataset split counts,
QLoRA/LoRA parameters, L4 hardware, optimizer, scheduler, adapter hash, runtime,
final training loss, and validation loss by epoch.

| Metric | Value |
|---|---:|
| Runtime | 7,145.04 s |
| Final training loss | 0.053305 |
| Validation loss, epoch 1 | 0.038936 |
| Validation loss, epoch 2 | 0.027865 |
| Validation loss, epoch 3 | 0.027835 |

## Base-versus-adapter quality

| Metric | Base | LoRA adapter |
|---|---:|---:|
| Factual correctness | 87.5% | 92.5% |
| Completeness | 77.5% | 82.5% |
| Numerical correctness | 93.55% | 93.33% |
| Evidence support | 90.0% | 95.0% |
| Hallucination rate | 10.0% | 5.0% |
| Overall acceptable | 75.0% | 82.5% |

These content scores are AI-assisted expert adjudication, not independent human
review. That limitation is retained in the tracked summary artifact.

## Serving comparison

| Concurrency | Model | P50 latency | P95 latency | Requests/s | Valid JSON |
|---:|---|---:|---:|---:|---:|
| 1 | Base | 5,175.21 ms | 19,551.29 ms | 0.132 | 0/40 |
| 1 | Adapter | 5,942.20 ms | 8,593.06 ms | 0.172 | 40/40 |
| 2 | Base | 5,641.23 ms | 27,937.49 ms | 0.215 | 0/40 |
| 2 | Adapter | 7,048.26 ms | 9,569.24 ms | 0.297 | 40/40 |

## Privacy and safety controls

- Only bounded `.json`, `.md`, and `.txt` summaries can be logged.
- `.csv`, `.jsonl`, and model/data binary artifacts are blocked.
- Raw questions, retrieved evidence, review rows, training rows, secrets, and
  adapter weights are not copied into MLflow.
- Mismatched dataset hashes fail before any run is created.
- Failed tracking operations terminate their run with `FAILED` status and a
  failure-type tag.

## Lineage limitation

The original Colab job was completed before MLflow was integrated and did not
record its source Git commit. The import therefore records the available immutable
dataset and adapter hashes, logs import-time Git state separately, and marks every
run `finrag.lineage_quality=partial_historical`. Future training runs should start
tracking before model loading so their source revision and failures are captured
at execution time.
