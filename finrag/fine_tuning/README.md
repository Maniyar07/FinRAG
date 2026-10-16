# FinRAG fine-tuning dataset factory

This directory is an additive, audit-first workflow for preparing a LoRA/QLoRA
dataset for FinRAG's **answer-generation model only**. It does not modify the
application, immutable indexes, source documents, canonical gold set, or current
OpenAI runtime.

## Focused architecture

```text
fine_tuning/
|-- dataset_factory/             # everything that creates or gates examples
|   |-- configs/
|   |   `-- dataset_policy.json
|   |-- tests/
|   |-- table_candidates.py      # deterministic table/number examples
|   |-- augmentation_candidates.py # verified calculations and evidence-safety examples
|   |-- teacher_candidates.py    # narrative and transcript examples
|   |-- teacher_synthesis.py     # multi-source examples
|   |-- ai_review.py             # independent semantic review
|   |-- validators.py            # schema, grounding and duplicate gates
|   `-- split_export.py          # leakage-safe splits and SFT export
|-- training/                    # only model-training concerns
|   |-- configs/
|   |   `-- qlora.example.json
|   |-- requirements.txt
|   `-- train_qlora.py
|-- inference/                   # opt-in adapter serving and base/LoRA benchmark
|-- work/                        # generated datasets and review audits
|-- artifacts/                   # adapters/checkpoints; created during training
|-- __main__.py                  # dataset-factory compatibility command
`-- README.md
```

Dataset construction and model training intentionally have separate dependencies
and entry points. `python -m fine_tuning ...` remains the stable dataset command;
`python -m fine_tuning.dataset_factory ...` is its explicit equivalent.

After training, use [`inference/README.md`](inference/README.md) to serve the
saved adapter and compare base versus LoRA latency, throughput, and output
format. The FastAPI model endpoint switch is optional and defaults to the
existing OpenAI configuration.

## Safety boundary

The workflow distinguishes three states:

1. **Candidate**: created from source evidence, but not trusted for training.
2. **Automatically validated**: schema, citations, numbers, arithmetic, hashes,
   duplicates, and split leakage pass deterministic checks.
3. **Training-ready**: an automatically validated candidate has also been approved
   by a human reviewer and does not overlap the locked benchmark.

Export refuses to produce a training set with fewer than 500 approved training rows
by default. Evaluation-reserved gold examples are never exported as training data.

## What is generated now

The initial build creates:

- 30 project-native, human-approved gold-builder examples as
  `evaluation_reserved` schema prototypes;
- balanced direct-extraction candidates from verified 10-K table cells;
- year-over-year calculation candidates with `Decimal`-verified arithmetic;
- targeted-table and complete-table candidates with structurally validated Markdown;
- conservative cross-company comparison candidates for unambiguous common metrics;
- a review CSV and a machine-readable validation report.

The 30 approved prototypes occur in `ragas_gold_90.json`. They demonstrate the
correct schema for transcript, guidance, ratio, comparison, and cross-document
answers, but the benchmark guard prevents them from leaking into training.

## Run the initial build

Run from the `finrag/` directory with its existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -m fine_tuning build
```

Generated files are written only under `fine_tuning/work/`, which is ignored by the
fine-tuning package's own `.gitignore`:

```text
fine_tuning/work/
|-- master_candidates.jsonl
|-- validation_report.json
`-- review.csv
```

The command is deterministic for an unchanged index and source configuration.

## Add narrative and multi-document candidates

Table candidates are deterministic. Narrative, guidance, transcript, and synthesis
answers require semantic judgment, so they use a separate teacher-generation stage
and remain `pending` until reviewed by a human. Pool inspection is local and free:

```powershell
.\.venv\Scripts\python.exe -m fine_tuning narrative-pool
.\.venv\Scripts\python.exe -m fine_tuning synthesis-pool
```

Teacher generation calls the configured model API and therefore may incur cost:

```powershell
.\.venv\Scripts\python.exe -m fine_tuning generate-narrative --target-count 250
.\.venv\Scripts\python.exe -m fine_tuning generate-synthesis --target-count 60
```

To expand an existing master without regenerating its questions, use the exclusion
gate. Transcript-only expansion is supported explicitly:

```powershell
.\.venv\Scripts\python.exe -m fine_tuning generate-narrative `
  --doc-type TRANSCRIPT `
  --exclude-dataset fine_tuning/work/combined_master.jsonl `
  --target-count 30

.\.venv\Scripts\python.exe -m fine_tuning generate-synthesis `
  --exclude-dataset fine_tuning/work/combined_master.jsonl `
  --target-count 35
```

The exclusion gate checks example IDs, canonical groups, and normalized question
fingerprints. Transcript generation also rejects safe-harbor boilerplate as
guidance; synthesis generation requires a separately answerable 10-K part and
transcript part rather than an inferred relationship.

Both generators checkpoint their work, require exact evidence quotations, validate
the structured response, and reject benchmark overlap. Merge them with the
deterministic master file before review:

```powershell
.\.venv\Scripts\python.exe -m fine_tuning merge `
  fine_tuning/work/master_candidates.jsonl `
  fine_tuning/work/narrative_candidates.jsonl `
  fine_tuning/work/synthesis_candidates.jsonl `
  --output fine_tuning/work/combined_master.jsonl
```

Do not use a generated answer for training merely because it passes automated
checks. Exact quotation proves evidence presence, not that an interpretation is
financially correct.

## Optional independent AI-assisted review

An independent semantic-review command can triage the combined dataset before a
named human reviewer signs it:

```powershell
.\.venv\Scripts\python.exe -m fine_tuning ai-review `
  --model gpt-4o `
  --batch-size 3 `
  --minimum-confidence 0.90 `
  --resume
```

This produces checkpointed decision records and `combined_review_ai.csv`. It
conservatively rejects unsupported implications, incorrect query types, weak
cross-document connections, and incomplete answers. The output explicitly says
`Human sign-off not recorded`; it must never be represented as literal human review.

## Review protocol

Open `fine_tuning/work/review.csv`. For every training candidate:

1. Confirm the question names the correct company, source filing, metric and period.
2. Confirm the target value is in the displayed source table.
3. Confirm units, scale, sign, and qualifiers.
4. For calculations, recompute old/new ordering and the percentage denominator.
5. Confirm that the question is useful and not merely a duplicate wording.
6. Set `status` to `approved`, `needs_edit`, or `rejected`.
7. For every approved training row, fill `reviewed_by` and an ISO-8601
   `reviewed_at` timestamp including its timezone, such as
   `2026-09-30T17:30:00+05:30`.
8. Add a reviewer note for any non-obvious decision.

Rows with `automated_status != passed` or `benchmark_overlap = True` cannot be
approved for training. The CSV includes the complete evidence blocks needed to
check the answer without searching through the index manually.

Apply review decisions:

```powershell
.\.venv\Scripts\python.exe -m fine_tuning apply-review
```

If you reviewed `combined_master.jsonl`, pass that dataset and its review CSV to the
corresponding command-line options instead of using the defaults.

## Split without leakage

After the combined dataset has enough high-quality approvals to satisfy the policy,
assign deterministic group-level splits. Aim above the bare 500-row requirement
because only 84% is assigned to training and some candidates will be rejected:

```powershell
.\.venv\Scripts\python.exe -m fine_tuning split
```

All variants of one canonical fact share a `group_id`; the validator blocks a group
from appearing in more than one split.

The configured dataset policy also enforces minimum coverage across numeric/table,
quantitative, narrative, transcript, and multi-source families and prevents one
query type from dominating the training set.

## Export for TRL

```powershell
.\.venv\Scripts\python.exe -m fine_tuning export
```

If the reviewed master passes every coverage minimum but narrowly exceeds the
single-query-type cap, export can deterministically downselect only the excess
dominant rows:

```powershell
.\.venv\Scripts\python.exe -m fine_tuning export --balance-query-types
```

This does not alter the master dataset or any review status. Every omitted example
ID, the selection-rule version, and the post-balance policy result are written to
`manifest.json`.

The output is conversational prompt-completion JSONL:

```json
{
  "prompt": [
    {"role": "system", "content": "...scope and retrieved evidence..."},
    {"role": "user", "content": "Question: ..."}
  ],
  "completion": [
    {"role": "assistant", "content": "{\"answer\":...,\"claims\":...,\"source_ids\":...}"}
  ]
}
```

Only assistant completion tokens should contribute to the SFT loss.

## Prepare the final 800-example dataset

After the expanded reviewed master has been split, the deterministic finalization
command adds 25 verified calculation examples and 25 canonical evidence-safety
examples. It then revalidates the full master, guards the locked benchmark,
preserves source-fact split groups, and removes only excess direct-extraction rows
to cap the exported dataset at exactly 800 examples:

```powershell
.\.venv\Scripts\python.exe -m fine_tuning prepare-final
```

The generated training files are under `fine_tuning/work/sft_final_800/`. The
manifest records file hashes, split counts, query-type coverage, policy results,
and every deterministically omitted example ID. Safety examples use the runtime's
exact insufficient-evidence response and never invent or cite an unavailable
source.

## Training

Create a separate training environment and install `training/requirements.txt`.
Copy `training/configs/qlora.example.json`, select a licensed instruction model
compatible with your hardware, and use new output directories for every run.

```powershell
python -m pip install -r fine_tuning\training\requirements.txt
python -m fine_tuning.training --config fine_tuning\training\configs\qlora.local.json
```

The older `python -m fine_tuning.train_qlora` entry point remains as a thin
compatibility wrapper; all training implementation is under `training/`.

The trainer refuses to overwrite a non-empty run directory. Adapter weights and
training artifacts belong under `fine_tuning/artifacts/` and are ignored by Git.
The selected checkpoint must be a licensed chat/instruction model whose tokenizer
defines a chat template; training fails early otherwise.

## Non-negotiable scientific rules

- Do not train on `ragas_gold_90.json` or its paraphrases.
- Do not approve synthetic narrative interpretations without source review.
- Do not split after paraphrasing; assign all variants through `group_id`.
- Do not compare an OpenAI model directly with a different local base model and call
  the difference a LoRA effect. Compare the same base model with and without LoRA.
- Keep retrieval fixed for generator-only A/B evaluation.
- Keep planner, ReAct tool-use, and reranker datasets separate from answer SFT.
- Never bypass the existing answer/citation/financial guardrails after deployment.
