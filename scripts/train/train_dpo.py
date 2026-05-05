from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DPO adapter from preference pairs.")
    parser.add_argument("--config", default="configs/qlora_config.yaml")
    parser.add_argument("--preferences", default=None, help="Path to prompt/chosen/rejected JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    config = load_config(Path(args.config))
    model_cfg = config["model"]
    lora_cfg = config["lora"]
    train_cfg = config.get("training", {})
    dpo_cfg = config.get("dpo", {})

    preferences_path = args.preferences or dpo_cfg.get("preference_jsonl", "outputs/dpo/preferences.jsonl")
    preferences_file = Path(preferences_path)
    if not preferences_file.is_file():
        raise FileNotFoundError(f"DPO preferences not found: {preferences_file}")

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    compute_dtype = dtype_map[model_cfg["bnb_4bit_compute_dtype"]]
    bnb_kwargs: dict[str, Any] = {
        "load_in_4bit": model_cfg["load_in_4bit"],
        "bnb_4bit_quant_type": model_cfg["bnb_4bit_quant_type"],
        "bnb_4bit_compute_dtype": compute_dtype,
        "bnb_4bit_use_double_quant": model_cfg["bnb_4bit_use_double_quant"],
    }
    if model_cfg.get("bnb_4bit_quant_storage") == "uint8":
        bnb_kwargs["bnb_4bit_quant_storage"] = torch.uint8

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_id"],
        quantization_config=BitsAndBytesConfig(**bnb_kwargs),
        device_map="auto",
        trust_remote_code=model_cfg["trust_remote_code"],
    )
    model = prepare_model_for_kbit_training(model)
    peft_cfg = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )
    model = get_peft_model(model, peft_cfg)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_id"], trust_remote_code=model_cfg["trust_remote_code"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = load_dataset("json", data_files=str(preferences_file), split="train")
    output_dir = Path(dpo_cfg.get("output_dir", "outputs/jinyong-dpo"))
    train_args = DPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(dpo_cfg.get("per_device_train_batch_size", train_cfg.get("per_device_train_batch_size", 2))),
        gradient_accumulation_steps=int(dpo_cfg.get("gradient_accumulation_steps", train_cfg.get("gradient_accumulation_steps", 4))),
        learning_rate=float(dpo_cfg.get("learning_rate", train_cfg.get("learning_rate", 5e-5))),
        num_train_epochs=float(dpo_cfg.get("num_train_epochs", 1)),
        logging_steps=int(dpo_cfg.get("logging_steps", 10)),
        save_steps=int(dpo_cfg.get("save_steps", 100)),
        save_total_limit=int(dpo_cfg.get("save_total_limit", 2)),
        fp16=bool(dpo_cfg.get("fp16", train_cfg.get("fp16", False))),
        bf16=bool(dpo_cfg.get("bf16", train_cfg.get("bf16", True))),
        report_to=str(dpo_cfg.get("report_to", "none")),
        beta=float(dpo_cfg.get("beta", 0.1)),
        max_length=int(dpo_cfg.get("max_length", train_cfg.get("max_seq_length", 1024))),
        max_prompt_length=int(dpo_cfg.get("max_prompt_length", 512)),
    )
    trainer = DPOTrainer(
        model=model,
        args=train_args,
        processing_class=tokenizer,
        train_dataset=dataset,
    )
    trainer.train()

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Saved DPO adapter to: {adapter_dir}")


if __name__ == "__main__":
    main()

