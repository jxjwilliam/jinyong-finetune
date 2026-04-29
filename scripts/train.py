from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_prompt(system_prompt: str, instruction: str, user_input: str, output: str) -> str:
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QLoRA model with TRL SFTTrainer.")
    parser.add_argument("--config", default="configs/qlora_config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))

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

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_cfg["load_in_4bit"],
        bnb_4bit_quant_type=model_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=model_cfg["bnb_4bit_use_double_quant"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_id"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=model_cfg["trust_remote_code"],
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["model_id"],
        trust_remote_code=model_cfg["trust_remote_code"],
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=data_cfg["instruction_jsonl"], split="train")
    dataset = dataset.train_test_split(
        test_size=train_cfg["eval_split_ratio"],
        seed=train_cfg["seed"],
    )

    def format_prompt(example: dict[str, str]) -> dict[str, str]:
        return {
            "text": build_prompt(
                data_cfg["system_prompt"],
                example["instruction"],
                example["input"],
                example["output"],
            )
        }

    dataset = dataset.map(format_prompt)

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
        report_to=train_cfg["report_to"],
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        packing=train_cfg["packing"],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )
    trainer.train()

    adapter_dir = Path(train_cfg["output_dir"]) / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"saved adapter to: {adapter_dir}")


if __name__ == "__main__":
    main()

