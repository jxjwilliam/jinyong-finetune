#!/usr/bin/env python3
"""Merge a trained PEFT LoRA adapter into full Hugging Face weights (no quantization).

Use on AutoDL after training writes ``outputs/jinyong-qlora/adapter/`` (paths from
``configs/qlora_config.yaml``). Merged weights are suitable for ``convert_hf_to_gguf.py``
(``docs/LORA_TO_GGUF_GUIDE.md``) or transformers inference without PEFT.

From repo root in Jupyter::

    !python scripts/train/merge_lora.py --config configs/qlora_config.yaml

Or: ``%run scripts/train/merge_lora.py --config configs/qlora_config.yaml``

VRAM: Qwen2.5-7B in bf16/fp16 needs about one 24GB GPU for load+merge+save; if OOM,
pass ``--dtype float16``.

If Hugging Face is unreachable (**Errno 101** / timeouts on AutoDL mainland), either::

    export HF_ENDPOINT=https://hf-mirror.com

(or ``--hf-endpoint https://hf-mirror.com``), then rerun, or merge fully offline::

    python scripts/train/merge_lora.py --config configs/qlora_config.yaml \\
        --local-files-only \\
        --base-model-path ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/<hash>

(Use the snapshot directory under ``hub/`` after a successful ``scripts/train/train.py`` download.)
"""
from __future__ import annotations

import argparse
import os
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
    p.add_argument(
        "--hf-endpoint",
        default=None,
        help="Sets HF_ENDPOINT before Hub access (e.g. https://hf-mirror.com for mainland).",
    )
    p.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not hit the Hub; use only files already cached or under --base-model-path.",
    )
    p.add_argument(
        "--base-model-path",
        default=None,
        help="Local directory of the base model (snapshot with config.json). "
        "If set, this path is loaded instead of the Hub model_id.",
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

    # Apply before first Hub/transformers disk access (mirrors HF CLI behaviour).
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint.rstrip("/")
        print(f"HF_ENDPOINT    : {os.environ['HF_ENDPOINT']}")

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    base_id_or_path = model_cfg["model_id"]
    trust = bool(model_cfg.get("trust_remote_code", True))
    if args.base_model_path:
        base_path = Path(args.base_model_path).expanduser().resolve()
        if not base_path.is_dir():
            raise FileNotFoundError(f"--base-model-path is not a directory: {base_path}")
        cfg_json = base_path / "config.json"
        if not cfg_json.is_file():
            raise FileNotFoundError(
                f"No config.json under {base_path} — pass the Hugging Face snapshot folder, "
                "not the hub repo root."
            )
        base_id_or_path = str(base_path)

    lf_only = args.local_files_only
    adapter_path = Path(args.adapter or (Path(train_cfg["output_dir"]) / "adapter"))
    merged_dir = resolve_merged_dir(train_cfg["output_dir"], args.merged_dir)
    torch_dtype = resolve_torch_dtype(args.dtype)

    if not adapter_path.is_dir():
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_path}\n"
            "Train first (scripts/train/train.py) or pass --adapter /path/to/adapter"
        )

    merged_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base model     : {base_id_or_path}")
    print(f"Adapter        : {adapter_path.resolve()}")
    print(f"Merged output  : {merged_dir.resolve()}")
    print(f"torch_dtype    : {torch_dtype}")
    if lf_only:
        print("Hub access     : local_files_only=True")

    uses_hub_name = (
        args.base_model_path is None
        and str(base_id_or_path) == str(model_cfg["model_id"])
    )
    if (
        uses_hub_name
        and not lf_only
        and os.environ.get("HF_ENDPOINT") is None
        and args.hf_endpoint is None
    ):
        print(
            "Tip: if huggingface.co is unreachable run with "
            "--hf-endpoint https://hf-mirror.com  or export HF_ENDPOINT=https://hf-mirror.com"
        )

    print("Loading base model (full precision, no 4-bit)...")

    tokenizer = AutoTokenizer.from_pretrained(
        base_id_or_path,
        trust_remote_code=trust,
        local_files_only=lf_only,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_id_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=trust,
        low_cpu_mem_usage=True,
        local_files_only=lf_only,
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
