from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _load_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Training config not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid training config: {error}") from error
    required = {
        "base_model", "output_dir", "train_file", "validation_file", "max_length",
        "learning_rate", "epochs", "per_device_batch_size",
        "gradient_accumulation_steps", "lora_rank", "lora_alpha", "lora_dropout", "seed",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Training config is missing: {sorted(missing)}")
    if payload["base_model"] == "replace-with-an-instruction-model":
        raise ValueError("Select a licensed instruction model in base_model before training")
    return payload


def train(config_path: Path) -> None:
    config = _load_config(config_path)
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as error:
        raise RuntimeError(
            "Training dependencies are missing. Create a separate environment and install "
            "fine_tuning/training/requirements.txt."
        ) from error

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a supported accelerator; CUDA was not detected")
    train_path = Path(config["train_file"]).resolve()
    validation_path = Path(config["validation_file"]).resolve()
    for path in (train_path, validation_path):
        if not path.is_file():
            raise ValueError(f"Dataset split not found: {path}")
    output_dir = Path(config["output_dir"]).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite non-empty training output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        raise ValueError(
            "The selected instruction model tokenizer has no chat template; choose a "
            "chat/instruct checkpoint or explicitly configure its verified template."
        )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=quantization,
        dtype=compute_dtype,
        device_map={"": torch.cuda.current_device()},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    dataset = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(validation_path)},
    )
    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(config["lora_rank"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        bias="none",
        target_modules="all-linear",
    )
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(config["epochs"]),
        per_device_train_batch_size=int(config["per_device_batch_size"]),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["learning_rate"]),
        warmup_ratio=0.03,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        max_length=int(config["max_length"]),
        completion_only_loss=True,
        packing=False,
        gradient_checkpointing=True,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=lora,
    )
    result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    manifest = {
        "base_model": config["base_model"],
        "adapter_dir": str(output_dir),
        "train_file": str(train_path),
        "validation_file": str(validation_path),
        "seed": int(config["seed"]),
        "metrics": result.metrics,
        "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }
    (output_dir / "finrag_training_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the FinRAG answer-generation QLoRA.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "configs" / "qlora.example.json"),
    )
    args = parser.parse_args()
    train(Path(args.config).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
