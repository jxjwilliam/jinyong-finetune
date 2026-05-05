from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for Jin Yong style QLoRA model.")
    parser.add_argument("--config", default="configs/qlora_config.yaml", help="Path to config file.")
    parser.add_argument("--model-dir", default=None, help="Path to merged model directory (overrides config output_dir).")
    parser.add_argument("--prompt", default=None, help="Input prompt (if not provided, enter interactive mode).")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))

    model_dir = Path(args.model_dir) if args.model_dir else Path(config["training"]["output_dir"]) / "merged"
    if not model_dir.exists():
        model_dir = Path(config["training"]["output_dir"])

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=config["model"].get("trust_remote_code", True),
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=config["model"].get("trust_remote_code", True),
        torch_dtype=torch.bfloat16 if config["model"].get("bnb_4bit_compute_dtype") == "bfloat16" else torch.float16,
        device_map="auto",
    )
    model.eval()

    system_prompt = config["data"].get(
        "system_prompt", "你是一位精通金庸武侠风格的写作助手。请根据用户的要求，创作符合金庸武侠小说风格的原创内容。"
    )

    def generate_response(instruction: str, user_input: str = "") -> str:
        user_content = f"{instruction}\n{user_input.strip()}" if user_input.strip() else instruction
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    if args.prompt:
        print(generate_response(args.prompt))
    else:
        print("Interactive mode (type 'exit' to quit)")
        while True:
            instruction = input("Instruction: ")
            if instruction.lower() == "exit":
                break
            user_input = input("User input (optional): ")
            response = generate_response(instruction, user_input)
            print(f"Response: {response}\n")


if __name__ == "__main__":
    main()
