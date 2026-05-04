#!/usr/bin/env python3
"""Merge a trained PEFT LoRA adapter into full Hugging Face weights (no quantization).

Use on AutoDL after training writes ``outputs/jinyong-qlora/adapter/`` (paths from
``configs/qlora_config.yaml``). Merged weights are suitable for ``convert_hf_to_gguf.py``
(``docs/LORA_TO_GGUF_GUIDE.md``) or transformers inference without PEFT.

From repo root in Jupyter::

    !python scripts/merge_lora.py --config configs/qlora_config.yaml

Or: ``%run scripts/merge_lora.py --config configs/qlora_config.yaml``

VRAM: Qwen2.5-7B in bf16/fp16 needs about one 24GB GPU for load+merge+save; if OOM,
pass ``--dtype float16``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML: {path}")
    return data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model (full-precision HF checkpoint)."
    )
    p.add_argument(
        "--config",
        default="configs/qlora_config.yaml",
        help="Training YAML (model_id, trust_remote_code, training.output_dir).",
    )
    p.add_argument(
        "--adapter",
        default=None,
        help="Adapter directory (default: {training.output_dir}/adapter from config).",
    )
    p.add_argument(
        "--merged-dir",
        default=None,
        help="Output directory for merged model (default: sibling of output_dir, "
        "e.g. outputs/jinyong-qlora -> outputs/jinyong-merged).",
    )
    p.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
        help="Load/merge dtype. auto = bf16 on CUDA if supported, else float16.",
    )
    p.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Max shard size for save_pretrained (e.g. 5GB).",
    )
    return p.parse_args()


def resolve_merged_dir(output_dir: str, merged_arg: str | None) -> Path:
    if merged_arg:
        return Path(merged_arg)
    out = Path(output_dir)
    name = out.name.replace("-qlora", "-merged")
    if name == out.name:
        name = f"{out.name}-merged"
    return out.parent / name


def resolve_torch_dtype(name: str) -> torch.dtype:
    if name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return getattr(torch, name)


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    base_id = model_cfg["model_id"]
    trust = bool(model_cfg.get("trust_remote_code", True))
    adapter_path = Path(args.adapter or (Path(train_cfg["output_dir"]) / "adapter"))
    merged_dir = resolve_merged_dir(train_cfg["output_dir"], args.merged_dir)
    torch_dtype = resolve_torch_dtype(args.dtype)

    if not adapter_path.is_dir():
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_path}\n"
            "Train first (scripts/train.py) or pass --adapter /path/to/adapter"
        )

    merged_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base model     : {base_id}")
    print(f"Adapter        : {adapter_path.resolve()}")
    print(f"Merged output  : {merged_dir.resolve()}")
    print(f"torch_dtype    : {torch_dtype}")
    print("Loading base model (full precision, no 4-bit)...")

    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=trust)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=trust,
        low_cpu_mem_usage=True,
    )

    print("Loading adapter...")
    model = PeftModel.from_pretrained(model, str(adapter_path), torch_dtype=torch_dtype)

    print("Merging LoRA into base weights (merge_and_unload)...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {merged_dir} ...")
    model.save_pretrained(
        str(merged_dir),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(str(merged_dir))

    print("Done.")
    print("Next: zip for download, e.g.  cd outputs && zip -r jinyong-merged.zip jinyong-merged/")
    print("      (adjust folder name if you used --merged-dir)")


if __name__ == "__main__":
    main()
