from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_prompt(system_prompt: str, instruction: str, user_input: str, output: str) -> str:
    user_content = f"{instruction}\n{user_input.strip()}" if user_input.strip() else instruction
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QLoRA model with TRL SFTTrainer.")
    parser.add_argument("--config", default="configs/qlora_config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))

    default_training: dict[str, Any] = {
        "output_dir": "outputs/jinyong-qlora",
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "num_train_epochs": 2,
        "max_seq_length": 1024,
        "save_steps": 100,
        "save_total_limit": 3,
        "logging_steps": 10,
        "report_to": "none",
        "fp16": False,
        "bf16": True,
        "packing": False,
        "eval_split_ratio": 0.05,
        "seed": 42,
        "eval_steps": 100,
        "gradient_checkpointing": True,
    }
    config.setdefault("training", {})
    for k, v in default_training.items():
        config["training"].setdefault(k, v)

    model_cfg = config["model"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]
    data_cfg = config["data"]

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map[model_cfg["bnb_4bit_compute_dtype"]]

    bnb_kwargs: dict[str, Any] = {
        "load_in_4bit": model_cfg["load_in_4bit"],
        "bnb_4bit_quant_type": model_cfg["bnb_4bit_quant_type"],
        "bnb_4bit_compute_dtype": compute_dtype,
        "bnb_4bit_use_double_quant": model_cfg["bnb_4bit_use_double_quant"],
    }
    # Optional: e.g. uint8 storage when set in YAML (BitsAndBytesConfig naming).
    if model_cfg.get("bnb_4bit_quant_storage") == "uint8":
        bnb_kwargs["bnb_4bit_quant_storage"] = torch.uint8
    bnb_config = BitsAndBytesConfig(**bnb_kwargs)

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_id"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=model_cfg["trust_remote_code"],
    )
    model.enable_input_require_grads()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["model_id"],
        trust_remote_code=model_cfg["trust_remote_code"],
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    _gc_kw: dict[str, Any] = {}
    if train_cfg["gradient_checkpointing"]:
        _gc_kw["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=train_cfg["gradient_checkpointing"],
        **_gc_kw,
    )
    lora_kwargs: dict[str, Any] = {
        "r": lora_cfg["r"],
        "lora_alpha": lora_cfg["lora_alpha"],
        "target_modules": lora_cfg["target_modules"],
        "lora_dropout": lora_cfg["lora_dropout"],
        "bias": lora_cfg["bias"],
        "task_type": lora_cfg["task_type"],
    }
    if lora_cfg.get("modules_to_save"):
        lora_kwargs["modules_to_save"] = lora_cfg["modules_to_save"]
    lora_config = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Required for gradient checkpointing + frozen base + LoRA: activations into
    # checkpointed blocks must require grad or backward skips LoRA (loss has no grad_fn).
    if train_cfg["gradient_checkpointing"] and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    dataset = load_dataset(
        "json",
        data_files=data_cfg["instruction_jsonl"],
        split="train",
    )
    dataset = dataset.train_test_split(
        test_size=train_cfg["eval_split_ratio"],
        seed=train_cfg["seed"],
    )

    system_prompt = data_cfg.get("system_prompt", "你是一位精通金庸武侠风格的写作助手。")

    def format_prompt(example: dict[str, str]) -> dict[str, str]:
        return {
            "text": build_prompt(
                system_prompt,
                example["instruction"],
                example.get("input", ""),
                example["output"],
            )
        }

    dataset = dataset.map(format_prompt, desc="Formatting prompts")

    sample_texts = [dataset["train"][i]["text"] for i in range(min(5, len(dataset["train"])))]
    if sample_texts:
        lengths = [len(tokenizer.encode(t)) for t in sample_texts]
        print(f"\nSample token lengths (first 5): {lengths}")
        print(f"max_seq_length = {train_cfg['max_seq_length']}")
        if max(lengths) > train_cfg["max_seq_length"]:
            print("[warn] Some samples exceed max_seq_length and will be truncated.")
        else:
            print("[ok] All checked samples fit within max_seq_length.")

    _sft_gc: dict[str, Any] = {}
    if train_cfg["gradient_checkpointing"]:
        # Trainer re-applies GC; without this, torch warns and may use reentrant checkpointing.
        _sft_gc["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    training_args = SFTConfig(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        num_train_epochs=train_cfg["num_train_epochs"],
        max_seq_length=train_cfg["max_seq_length"],
        save_strategy="steps",
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        report_to=train_cfg["report_to"],
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        packing=train_cfg["packing"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        seed=train_cfg["seed"],
        **_sft_gc,
    )

    def _format_example(example: dict[str, str]) -> str:
        return example["text"]

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        formatting_func=_format_example,
    )

    print(
        f"\nTraining on {len(dataset['train']):,} samples, "
        f"evaluating on {len(dataset['test']):,} samples"
    )
    print(f"packing={train_cfg['packing']}, max_seq_length={train_cfg['max_seq_length']}\n")
    trainer.train()

    adapter_dir = Path(train_cfg["output_dir"]) / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"\nSaved adapter to: {adapter_dir}")


if __name__ == "__main__":
    main()

