from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from scripts.eval.judge_gpt4o import DIMENSIONS, JudgeConfig, judge_one

if TYPE_CHECKING:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 5-dim rubric eval on fixed typed prompts.")
    parser.add_argument("--config", default="configs/qlora_config.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prompts", default=None, help="Override eval prompts JSONL.")
    parser.add_argument("--output-dir", default=None, help="Override eval output dir.")
    parser.add_argument("--model-dir", default=None, help="Optional explicit model dir.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-api-base", default="https://api.openai.com/v1")
    parser.add_argument("--gate-min-avg", type=float, default=-1.0)
    parser.add_argument("--gate-max-drop", type=float, default=-1.0)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def resolve_model_dir(config: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override)
    out_dir = Path(config["training"]["output_dir"])
    merged = out_dir / "merged"
    if merged.exists():
        return merged
    return out_dir / "adapter"


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_user_content(instruction: str, user_input: str) -> str:
    return f"{instruction}\n{user_input.strip()}" if user_input.strip() else instruction


def generate_text(
    model: "AutoModelForCausalLM",
    tokenizer: "AutoTokenizer",
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


def load_previous_summary(summary_path: Path, run_id: str) -> dict[str, float] | None:
    if not summary_path.exists():
        return None
    latest: dict[str, Any] | None = None
    with summary_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("run_id") == run_id:
                continue
            latest = obj
    if latest is None:
        return None
    return latest.get("dimension_avg", None)


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = load_config(Path(args.config))
    eval_cfg = config.get("eval", {})

    prompts_path = Path(args.prompts if args.prompts else eval_cfg.get("prompts_jsonl", "scripts/eval/prompts_v2_typed20.jsonl"))
    output_dir = Path(args.output_dir if args.output_dir else eval_cfg.get("output_dir", "outputs/eval"))
    judge_model = args.judge_model if args.judge_model else eval_cfg.get("judge_model", "gpt-4o")
    gate_min_avg = args.gate_min_avg if args.gate_min_avg > 0 else float(eval_cfg.get("gate_min_avg", 0.0))
    gate_max_drop = args.gate_max_drop if args.gate_max_drop > 0 else float(eval_cfg.get("gate_max_drop", 0.0))

    prompts = load_jsonl(prompts_path)
    if not prompts:
        raise ValueError("No prompts loaded for evaluation.")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_dir = resolve_model_dir(config, args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    trust_remote_code = bool(config["model"].get("trust_remote_code", True))
    compute_dtype = config["model"].get("bnb_4bit_compute_dtype", "bfloat16")
    torch_dtype: torch.dtype = torch.bfloat16 if compute_dtype == "bfloat16" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=trust_remote_code, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model.eval()

    system_prompt = config["data"].get("system_prompt", "你是一位精通金庸武侠风格的写作助手。")
    judge_cfg = JudgeConfig(model=judge_model, api_base=args.judge_api_base)

    base_output_dir = output_dir
    run_dir = base_output_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    generations_path = run_dir / "generations.jsonl"
    summary_json_path = run_dir / "summary.json"
    trend_path = base_output_dir / "eval_results.jsonl"
    summary_trend_path = base_output_dir / "summary_history.jsonl"

    for prompt in prompts:
        generation = generate_text(
            model=model,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            instruction=prompt["instruction"],
            user_input=prompt.get("input", ""),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        row = {
            "run_id": args.run_id,
            "prompt_id": prompt["id"],
            "category": prompt.get("category", ""),
            "instruction": prompt["instruction"],
            "input": prompt.get("input", ""),
            "output": generation,
        }
        append_jsonl(generations_path, row)

    generation_rows = load_jsonl(generations_path)
    judged_rows: list[dict[str, Any]] = []
    dimension_sums = {dim: 0.0 for dim in DIMENSIONS}
    for row in generation_rows:
        judged = judge_one(
            {
                "instruction": row["instruction"],
                "input": row.get("input", ""),
                "output": row["output"],
            },
            cfg=judge_cfg,
        )
        scored = {
            "timestamp": datetime.now(UTC).isoformat(),
            "run_id": args.run_id,
            "prompt_id": row["prompt_id"],
            "category": row.get("category", ""),
            "judge_model": judge_model,
            "scores": judged["scores"],
            "avg": judged["avg"],
            "brief": judged["brief"],
        }
        judged_rows.append(scored)
        append_jsonl(trend_path, scored)
        for dim in DIMENSIONS:
            dimension_sums[dim] += float(scored["scores"][dim])

    count = max(1, len(judged_rows))
    dimension_avg = {dim: round(dimension_sums[dim] / count, 4) for dim in DIMENSIONS}
    overall_avg = round(sum(dimension_avg.values()) / len(DIMENSIONS), 4)
    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "run_id": args.run_id,
        "judge_model": judge_model,
        "count": len(judged_rows),
        "overall_avg": overall_avg,
        "dimension_avg": dimension_avg,
    }

    previous_dimension_avg = load_previous_summary(summary_trend_path, args.run_id)
    gate_failures: list[str] = []
    if gate_min_avg > 0 and overall_avg < gate_min_avg:
        gate_failures.append(
            f"overall_avg {overall_avg:.4f} below gate_min_avg {gate_min_avg:.4f}"
        )
    if gate_max_drop > 0 and previous_dimension_avg:
        for dim in DIMENSIONS:
            prev = float(previous_dimension_avg.get(dim, dimension_avg[dim]))
            drop = prev - float(dimension_avg[dim])
            if drop > gate_max_drop:
                gate_failures.append(
                    f"{dim} dropped by {drop:.4f}, max allowed {gate_max_drop:.4f}"
                )
    summary["gate_passed"] = len(gate_failures) == 0
    summary["gate_failures"] = gate_failures

    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    append_jsonl(summary_trend_path, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if gate_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

