from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def judge_pairwise(
    *,
    judge_model: str,
    api_base: str,
    instruction: str,
    user_input: str,
    completion_a: str,
    completion_b: str,
) -> str:
    import requests

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is required for pairwise DPO judging.")

    prompt = (
        "你是严格的中文武侠文本评审。请在 A/B 中选更符合金庸风格且更遵循指令的一项。"
        "只返回一个字符：A 或 B。\n\n"
        f"instruction:\n{instruction}\n\n"
        f"input:\n{user_input}\n\n"
        f"A:\n{completion_a}\n\n"
        f"B:\n{completion_b}\n"
    )
    payload = {
        "model": judge_model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": "只输出 A 或 B。"},
            {"role": "user", "content": prompt},
        ],
    }
    resp = requests.post(
        f"{api_base.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=60,
    )
    resp.raise_for_status()
    out = str(resp.json()["choices"][0]["message"]["content"]).strip().upper()
    if out.startswith("A"):
        return "A"
    if out.startswith("B"):
        return "B"
    return "A"


def build_user_content(instruction: str, user_input: str) -> str:
    return f"{instruction}\n{user_input.strip()}" if user_input.strip() else instruction


def generate_completion(
    *,
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    instruction: str,
    user_input: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    import torch

    user_content = build_user_content(instruction, user_input)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DPO preference pairs via pairwise GPT judging.")
    parser.add_argument("--config", default="configs/qlora_config.yaml")
    parser.add_argument("--max-prompts", type=int, default=20)
    parser.add_argument("--output", default="outputs/dpo/preferences.jsonl")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--judge-api-base", default="https://api.openai.com/v1")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = load_yaml(Path(args.config))
    dpo_cfg = config.get("dpo", {})
    eval_cfg = config.get("eval", {})

    prompts_path = Path(dpo_cfg.get("prompt_set", eval_cfg.get("prompts_jsonl", "scripts/eval/prompts_v2_typed20.jsonl")))
    prompts = load_jsonl(prompts_path)[: args.max_prompts]
    if not prompts:
        raise ValueError(f"No prompts loaded from {prompts_path}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_out = Path(config["training"]["output_dir"])
    model_dir = Path(args.model_dir) if args.model_dir else (train_out / "merged")
    if not model_dir.exists():
        model_dir = train_out / "adapter"
    if not model_dir.exists():
        model_dir = Path(config["model"]["model_id"])

    trust_remote_code = bool(config["model"].get("trust_remote_code", True))
    on_cuda = torch.cuda.is_available()
    on_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if on_cuda:
        compute_dtype = config["model"].get("bnb_4bit_compute_dtype", "bfloat16")
        torch_dtype: torch.dtype = torch.bfloat16 if compute_dtype == "bfloat16" else torch.float16
    elif on_mps:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=trust_remote_code, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter_config = model_dir / "adapter_config.json"
    if adapter_config.is_file():
        adapter_meta = json.loads(adapter_config.read_text(encoding="utf-8"))
        base_model_id = str(adapter_meta.get("base_model_name_or_path") or config["model"]["model_id"])
        if on_cuda:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(base_model, str(model_dir))
        else:
            # On Mac/CPU we avoid accelerate auto offload path for adapter directories.
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
            )
            model = PeftModel.from_pretrained(base_model, str(model_dir))
            target_device = "mps" if on_mps else "cpu"
            model = model.to(target_device)
    else:
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": torch_dtype,
        }
        if on_cuda:
            load_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), **load_kwargs)
        if not on_cuda:
            model = model.to("mps" if on_mps else "cpu")

    model.eval()

    system_prompt = config["data"].get("system_prompt", "你是一位精通金庸武侠风格的写作助手。")
    judge_model = dpo_cfg.get("judge_model", "gpt-4o")

    rows: list[dict[str, str]] = []
    for item in prompts:
        instruction = str(item.get("instruction", "")).strip()
        user_input = str(item.get("input", "")).strip()
        with contextlib.suppress(RuntimeError):
            if on_cuda:
                torch.cuda.empty_cache()

        completion_a = generate_completion(
            model=model,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            instruction=instruction,
            user_input=user_input,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        completion_b = generate_completion(
            model=model,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            instruction=instruction,
            user_input=user_input,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        winner = judge_pairwise(
            judge_model=judge_model,
            api_base=args.judge_api_base,
            instruction=instruction,
            user_input=user_input,
            completion_a=completion_a,
            completion_b=completion_b,
        )
        chosen, rejected = (completion_a, completion_b) if winner == "A" else (completion_b, completion_a)
        rows.append(
            {
                "prompt": build_user_content(instruction, user_input),
                "chosen": chosen,
                "rejected": rejected,
            }
        )
        print(f"[ok] prompt={item.get('id', '?')} winner={winner}")

    write_jsonl(Path(args.output), rows)
    print(f"Saved DPO preference dataset: {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

