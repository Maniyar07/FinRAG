# Serving and evaluating the FinRAG LoRA adapter

This is an opt-in staging workflow. The application's default model remains
OpenAI. The saved `adapter_final` directory contains LoRA weights, not the
Qwen base model. The exact training base is `Qwen/Qwen2.5-7B-Instruct`.

The current FastAPI route uses the LangGraph agent in `src/agent/graph.py`.
That route asks the model to call tools and complete several strict JSON
schemas; the LoRA dataset taught the simpler `answer`/`claims`/`source_ids`
schema. Passing the direct generation benchmark does not establish that the
adapter can operate this full agent. Keep the model switch in staging until
the end-to-end tests and human evidence review pass.

## 1. Verify and stage the adapter

The trained artifact is on Drive at:

`/content/drive/MyDrive/FinRAG_Training/runs/finrag_qwen25_7b_l4_v1/full/adapter_final`

Verify `adapter_config.json` and `adapter_model.safetensors` exist. Check that
`base_model_name_or_path` in `adapter_config.json` identifies the same Qwen
base model. Mount or copy the adapter to the GPU host. Under this repository,
`fine_tuning/artifacts/` is ignored by Git and may be used for a local copy.
Do not put the adapter into `train.jsonl` or replace the current model files.

## 2. Run one GPU model server

Use a fresh Linux GPU environment with a compatible vLLM installation. The
server command below registers both the unchanged base and the LoRA under
different names. This is a Linux shell command, not a Windows PowerShell
command. Replace `<adapter-path>` with the directory from step 1. The server
capacity is deliberately set to two so the later concurrency-two benchmark
measures true parallel capacity rather than a one-slot queue.

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --enable-lora \
  --lora-modules finrag-lora=<adapter-path> \
  --max-lora-rank 16 \
  --max-model-len 8192 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.85 \
  --no-enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --host 127.0.0.1 --port 8000
```

If vLLM reports insufficient GPU memory, inspect the actual request length
and available memory before changing the context limit. First reduce
`--max-num-seqs` to 1 for a smoke run; do not present that queued result as a
parallel concurrency-two result. The held-out split contains prompts longer
than 4,096 tokens. Prefix caching is disabled here so repeated base/adapter
prompts do not give the second model an unfair cache advantage. Keep this endpoint private; in
Colab, `127.0.0.1` is reachable only from processes in that Colab runtime.
To use a separate FastAPI host, provide a private authenticated connection.

Check the served names from the same environment:

```bash
curl http://127.0.0.1:8000/v1/models
```

Both `Qwen/Qwen2.5-7B-Instruct` and `finrag-lora` must appear.

On Colab, use a fresh L4 runtime, mount Drive, verify the two adapter files,
and copy `adapter_final` to `/content/adapter_final` before starting vLLM.
Copying avoids model-server reads from the slower Drive mount. Install vLLM
in that fresh runtime, then run the server as a background process and inspect
its log until `/v1/models` responds. The updated project and test JSONL must
also be present in that Colab runtime for the benchmark below. A notebook
runtime restart removes `/content` and stops the server, but leaves Drive
unchanged.

## 3. Compare base and adapter directly

Run the same held-out 40 examples against both served names. Use the project
environment, which already includes the `openai` Python package:

```bash
python fine_tuning/inference/benchmark_endpoint.py \
  --base-url http://127.0.0.1:8000/v1 \
  --dataset fine_tuning/work/sft_final_800/test.jsonl \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --adapter-model finrag-lora \
  --concurrency 1 --server-max-num-seqs 2 \
  --warmup 1 --max-tokens 2048 \
  --output fine_tuning/work/benchmarks/base_vs_adapter_c1.jsonl
```

This writes predictions and a summary under the ignored
`fine_tuning/work/benchmarks/` directory. It measures response latency
(median and p95), completed requests per second, output tokens per second
when the server reports usage, JSON/schema compliance, and source-ID
formatting. It uses the same prompt and generation parameters for both
models. It does **not** measure whether a financial claim is true or a
citation supports it; review those against the original evidence.
The summary reports how many source-ID checks are applicable: a valid
unsupported-answer example may intentionally cite no source.

To measure throughput under simultaneous load, keep the same running server
and run a second benchmark:

```bash
python fine_tuning/inference/benchmark_endpoint.py \
  --base-url http://127.0.0.1:8000/v1 \
  --dataset fine_tuning/work/sft_final_800/test.jsonl \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --adapter-model finrag-lora \
  --concurrency 2 --server-max-num-seqs 2 \
  --warmup 1 --max-tokens 2048 \
  --output fine_tuning/work/benchmarks/base_vs_adapter_c2.jsonl
```

`--server-max-num-seqs` is benchmark metadata and must equal the actual vLLM
setting; it does not reconfigure vLLM. The script rejects a client concurrency
above that capacity. For an intentionally queued stress test, explicitly add
`--allow-queued-load`; its summary is labelled `queued`, not parallel. Increase
concurrency only after the two-request run is stable. Throughput and latency
depend on prompt length, output length, server configuration, GPU, and
concurrent load. Do not compare a local vLLM run with a hosted OpenAI run as
if their hardware were the same.

## 4. Stage the full FastAPI pipeline

The optional environment settings are `FINRAG_LLM_BASE_URL` and
`FINRAG_LLM_API_KEY`. When `FINRAG_LLM_BASE_URL` is unset, the project uses
its existing OpenAI chat model. When it is set, all calls to
`get_llm_engine()` use that OpenAI-compatible endpoint. Keep
`OPENAI_API_KEY` for the project's existing OpenAI embeddings.

Run the full project and model server on the same host, or set the URL to an
authenticated private endpoint. For a base-model run, set
`FINRAG_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct`; for the adapter run, set
`FINRAG_LLM_MODEL=finrag-lora`. Start a new FastAPI process after each
configuration change; environment settings are read at process startup.
If FastAPI and vLLM share a host, use different ports (for example, vLLM on
8000 and FastAPI on 8001). Colab's `127.0.0.1` is **not** your Windows PC's
`127.0.0.1`; a Windows FastAPI process cannot reach that address. A full
pipeline run also needs access to the existing document index and embeddings.

For a Windows PowerShell session **only when the vLLM endpoint is actually
reachable from that PC**, the staging launch is:

```powershell
cd finrag
$env:FINRAG_LLM_BASE_URL = "http://127.0.0.1:8000/v1"
$env:FINRAG_LLM_API_KEY = "local"
$env:FINRAG_LLM_MODEL = "finrag-lora"
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8001
```

Use the real private server URL and authentication token if the model runs
on another host. Do not expose an unauthenticated vLLM endpoint publicly.

Before a full 90-question run, use one ordinary fact question, one table,
one calculation, one multi-source question, and one unsupported question.
Inspect tool calls, strict JSON schema failures, validated facts, citations,
and final answers. The present adapter may fail the agent's tool or schema
requirements even when its standalone answer JSON is valid.

The existing end-to-end evaluator records generation latency and verdicts.
From the `finrag` directory, run it twice with separate output paths and a
fresh process for each model:

```bash
python evals/evaluate_ragas.py --dataset evals/ragas_gold_90.json \
  --generate-only --output evals/results/base_vllm_answers.json

python evals/evaluate_ragas.py --dataset evals/ragas_gold_90.json \
  --generate-only --output evals/results/adapter_vllm_answers.json
```

Set `FINRAG_LLM_MODEL` to the appropriate served name before each command.
These end-to-end timings include retrieval, tools, and model calls; they
answer a different question from the direct model-serving benchmark.
The current 40-example test has already been used for diagnosis. Use a new,
human-audited held-out suite for final quality acceptance.

## 5. Finalize a blind content review

The automated benchmark checks structure and serving performance; it does not
prove that claims are factually correct or supported by the supplied evidence.
After completing an A/B review CSV, validate, unblind, and summarize it with:

```powershell
.\.venv\Scripts\python.exe fine_tuning\inference\finalize_human_review.py `
  --review-csv <blind-review.csv> `
  --review-key <blind-review-key.json> `
  --decisions <review-decisions.json> `
  --output-dir fine_tuning\work\t26_review_results
```

The command rejects missing test indices, invalid rubric values, and mismatched
A/B assignments. It writes a completed review CSV, an unblinded JSON summary,
and a Markdown report. Keep AI-assisted adjudication explicitly labelled as
such; it is not a substitute for independent human sign-off in production.

## Ollama option

Ollama can run a model from a compatible Safetensors or GGUF model directory.
The current Ollama import documentation describes model import, not a direct
load of this PEFT `adapter_final` directory. To use Ollama, first merge the
LoRA with the exact Qwen base model and export a supported model format,
then create an Ollama model from that artifact. Quantization or conversion
changes the serving setup, so compare its answers and throughput separately
from the direct LoRA run. vLLM is the shorter route for this adapter.
