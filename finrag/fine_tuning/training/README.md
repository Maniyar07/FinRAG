# LoRA/QLoRA training

This package owns only model-training concerns. It consumes the approved JSONL
splits produced by `fine_tuning.dataset_factory`; it does not generate, review, or
repair examples.

1. Create a separate training environment.
2. Install `fine_tuning/training/requirements.txt`.
3. Copy `configs/qlora.example.json` to an untracked local config.
4. Confirm the licensed chat/instruction checkpoint and use a new output directory.
5. Run:

```powershell
python -m fine_tuning.training --config fine_tuning/training/configs/qlora.local.json
```

The trainer checks CUDA availability, input files, tokenizer chat-template support,
and refuses to overwrite a non-empty output directory.

The checked-in example selects `Qwen/Qwen2.5-7B-Instruct` and the finalized
`sft_final_800` train/validation splits. The project's longest examples exceeded a
free T4 during the worst-case smoke test, so use an L4-class 24 GB GPU or larger
for the unchanged 8,192-token configuration. Do not reduce the limit without a
fresh tokenizer-length and evidence-truncation audit.
