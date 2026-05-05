#!/usr/bin/env python3
"""Upload the saved PEFT adapter directory to a Hugging Face model repo.

Requires: pip install huggingface_hub pyyaml
Auth: huggingface-cli login   or   export HF_TOKEN=...

Run from repo root, for example:
  python scripts/hub/upload_adapter_hf.py --repo-id jxjwilliam/jinyong-qwen2.5-7b-qlora
  python scripts/hub/upload_adapter_hf.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_card_markdown(*, repo_id: str) -> str:
    """HF model card with YAML front matter for the Hub UI."""
    return f"""---
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: peft
license: apache-2.0
language:
- zh
tags:
- lora
- qlora
- creative-writing
- wuxia
pipeline_tag: text-generation
---

# Jin Yong style QLoRA adapter (Qwen2.5-7B-Instruct)

PEFT LoRA adapter for Chinese wuxia-style creative writing in the manner of Jin Yong (金庸).

## Base model

Use with [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct). This repo contains **adapter weights and tokenizer files only**, not the full base checkpoint.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = "Qwen/Qwen2.5-7B-Instruct"
adapter = "{repo_id}"

tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, adapter)
```

For inference with the same memory profile as QLoRA training, load the base in 4-bit and match `bnb` settings to your training config.

## Limitations

Generative models may reflect training data biases; outputs are fictional and not factual. Users are responsible for appropriate use and for complying with applicable laws and policies.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload PEFT adapter folder to Hugging Face Hub.")
    p.add_argument(
        "--config",
        default="configs/qlora_config.yaml",
        help="YAML config (uses training.output_dir / adapter subfolder).",
    )
    p.add_argument(
        "--repo-id",
        default="jxjwilliam/jinyong-qwen2.5-7b-qlora",
        help="Target HF model repo id (namespace/name).",
    )
    p.add_argument(
        "--adapter-subdir",
        default="adapter",
        help="Directory name under output_dir where save_pretrained wrote the adapter.",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private (only applies when the repo is created).",
    )
    p.add_argument(
        "--no-write-model-card",
        action="store_true",
        help="Do not overwrite README.md in the adapter folder before upload.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print paths and exit without calling the Hub API.",
    )
    p.add_argument(
        "--commit-message",
        default="Upload Jin Yong QLoRA adapter (Qwen2.5-7B-Instruct)",
        help="Git commit message for the Hub upload.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Missing dependency: pip install huggingface_hub", file=sys.stderr)
        return 1

    root = Path(__file__).resolve().parents[2]
    config_path = (root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(config_path)
    train_cfg = cfg.get("training") or {}
    output_dir = train_cfg.get("output_dir", "outputs/jinyong-qlora")
    adapter_dir = (root / output_dir / args.adapter_subdir).resolve()

    if not adapter_dir.is_dir():
        print(f"Adapter directory not found: {adapter_dir}", file=sys.stderr)
        return 1
    required = ("adapter_config.json", "adapter_model.safetensors")
    missing = [name for name in required if not (adapter_dir / name).is_file()]
    if missing:
        print(f"Missing files in {adapter_dir}: {missing}", file=sys.stderr)
        return 1

    print(f"Adapter folder: {adapter_dir}")
    print(f"Repo: {args.repo_id}")

    if args.dry_run:
        if not args.no_write_model_card:
            print("(Would write README.md model card before upload; skipped in dry-run.)")
        print("Dry run: no upload.")
        return 0

    if not args.no_write_model_card:
        readme = adapter_dir / "README.md"
        readme.write_text(model_card_markdown(repo_id=args.repo_id), encoding="utf-8")
        print(f"Wrote model card: {readme}")

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
    )
    print(f"Done: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
