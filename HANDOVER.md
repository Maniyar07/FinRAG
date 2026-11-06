**FinRAG project handover — 30 October 2026**

This release is a snapshot from **30 October 2026**. For the subsequently reviewed
implementation and documentation status, see [CURRENT_STATE.md](CURRENT_STATE.md).

The application, tests, evaluation tools, documentation, and experiment
tracking code are on the `langgraph-experiment` branch. The source documents are
tracked in Git. Download the additional local data from the
[handover release](https://github.com/Maniyar07/FinRAG/releases/tag/handover-2026-10-30).

| Release attachment | Restores |
| --- | --- |
| `finetuning-artifacts.zip` | `finrag/fine_tuning/artifacts/`: LoRA adapter, tokenizer, and configuration |
| `experiment-work.zip` | `finrag/fine_tuning/work/`: reviewed datasets, generated outputs, MLflow database, and logged artifacts |
| `gold-builder-work.zip` | `finrag/evals/gold_builder/work/`: draft questions and retrieval reviews |
| `runtime-indexes.zip` | `finrag/Data/indexes/`: the saved retrieval indexes |
| `evaluation-results.zip` | `finrag/evals/results/`: captured evaluations, reviews, and regression evidence |
| `backup-manifest.json` | File inventory, SHA-256 checksums, database snapshot method, and exclusions |
| `SHA256SUMS.txt` | SHA-256 checksums of the five ZIP files |
| `RESTORE.md` | A copy of these instructions |
| `offline-tests.xml` | JUnit results from the handover validation |

The ZIP attachments preserve project-relative paths. They are separate downloads
from GitHub's automatically generated source archives and are not included by
`git clone`.

**Restore the project**

Clone the branch on the other computer:

```powershell
git clone --branch langgraph-experiment https://github.com/Maniyar07/FinRAG.git
cd FinRAG
```

Download the five ZIP files and `SHA256SUMS.txt` from the release into this new
repository root. Compare each result below with its entry in `SHA256SUMS.txt`:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath finetuning-artifacts.zip, experiment-work.zip, gold-builder-work.zip, runtime-indexes.zip, evaluation-results.zip
```

Extract the downloads into the repository root. Use a fresh clone so that the
saved data does not overwrite another machine's newer experiments or indexes.

```powershell
Expand-Archive -LiteralPath finetuning-artifacts.zip -DestinationPath . -Force
Expand-Archive -LiteralPath experiment-work.zip -DestinationPath . -Force
Expand-Archive -LiteralPath gold-builder-work.zip -DestinationPath . -Force
Expand-Archive -LiteralPath runtime-indexes.zip -DestinationPath . -Force
Expand-Archive -LiteralPath evaluation-results.zip -DestinationPath . -Force
```

Create a new Python environment and local configuration:

```powershell
cd finrag
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Set your authorized API credentials in the new `.env` file and select
`FINRAG_INDEX_VERSION=phase1-v2`. Start the application:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. See [the application README](finrag/README.md)
for configuration, index coverage, and optional dependencies.

**Experiment tracking on a different computer**

The MLflow SQLite database was preserved using SQLite's consistent backup API.
Its historical artifact URIs may still contain absolute paths from the original
laptop. The original database and logged files remain in `experiment-work.zip`.
To view the reviewed Week-8 summaries using paths on the new computer, import
them into a separate local database without replacing the historical snapshot.
Run these commands from `finrag/`:

```powershell
py -3.12 -m venv fine_tuning/work/mlflow/venv
.\fine_tuning\work\mlflow\venv\Scripts\python.exe -m pip install -r fine_tuning/tracking/requirements.txt
.\fine_tuning\work\mlflow\venv\Scripts\python.exe -m fine_tuning.tracking --tracking-uri sqlite:///fine_tuning/work/mlflow/restored.db --artifact-root fine_tuning/work/mlflow/restored-artifacts --receipt fine_tuning/work/mlflow/restored-receipt.json
.\fine_tuning\work\mlflow\venv\Scripts\python.exe -m mlflow server --backend-store-uri sqlite:///fine_tuning/work/mlflow/restored.db --host 127.0.0.1 --port 5000
```

This imports existing reviewed summaries; it does not retrain the model or make
model-provider requests. See [the tracking README](finrag/fine_tuning/tracking/README.md)
for the historical experiment's provenance and limitations.

**Update a clone you already restored**

Run the following from the outer repository folder (the one containing this file
and the inner `finrag/` folder). There is no need to clone again:

```powershell
git status --short
git switch langgraph-experiment
git pull --ff-only origin langgraph-experiment
git log -1 --oneline
```

If the terminal is in the inner application folder, run `cd ..` first. Review
`git status` for your own edits. If Git reports that local changes would be
overwritten, or that the branches have diverged, preserve those changes and
resolve the reported condition before retrying. Do not discard work to force an
update.

This updates tracked source and documentation, including `AGENTS.md` and
`CURRENT_STATE.md`. Your ignored `.env`, virtual environment, restored indexes,
model files, and experiment work are not replaced by these Git commands. Release
archives are separate snapshots and are not downloaded by `git pull`.

For a documentation-only update, no environment recreation, index extraction, or
training is needed. For later application changes, review the new commit's
instructions and requirements; restart the server after updating running code.

**Backup scope and validation (30 October 2026 snapshot)**

Credentials and `.env` files, Python environments, bytecode, caches, temporary
test and audit directories, application query logs under `Data/logs/`, and
redundant top-level generated ZIP bundles were not uploaded. The manifest lists
the exclusions within each packaged folder. Recreate environments from the
tracked requirements files and configure credentials on the new computer.

All five ZIP files passed CRC integrity checks. A scan of the selected source
and backup files found no matches for locally configured credentials or the
checked token/private-key patterns; this is not a guarantee that every possible
type of sensitive information was detected. Python syntax checks passed for
55 changed or new Python files. The full application test suite passed offline:
741 tests and 70 subtests, with 49 dependency warnings. External connections and
proxy use were disabled during this validation; local test HTTP traffic was
allowed. No fresh provider-backed quality evaluation was run for this handover.
