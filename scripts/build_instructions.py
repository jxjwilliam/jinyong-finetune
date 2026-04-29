from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from clean_text import clean_novel

DEFAULT_INSTRUCTION = "以金庸武侠小说的风格，续写以下段落："


@dataclass
class Pair:
    instruction: str
    input: str
    output: str


def sliding_window_pairs(text: str, chunk_size: int, overlap: int) -> list[Pair]:
    chars = list(text)
    pairs: list[Pair] = []
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than chunk_size")

    i = 0
    while i + 2 * chunk_size <= len(chars):
        prompt = "".join(chars[i : i + chunk_size]).strip()
        continuation = "".join(chars[i + chunk_size : i + 2 * chunk_size]).strip()
        if prompt and continuation:
            pairs.append(Pair(DEFAULT_INSTRUCTION, prompt, continuation))
        i += step
    return pairs


def validate_pairs(pairs: Iterable[Pair], min_output_chars: int = 30) -> list[Pair]:
    validated: list[Pair] = []
    for pair in pairs:
        if not pair.instruction.strip():
            continue
        if not pair.input.strip():
            continue
        if len(pair.output.strip()) < min_output_chars:
            continue
        validated.append(pair)
    return validated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build instruction JSONL from raw novels.")
    parser.add_argument("--input-dir", default="data/raw")
    parser.add_argument("--output", default="data/instructions/jinyong_sft.jsonl")
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--min-output-chars", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")

    all_pairs: list[Pair] = []
    for txt_file in txt_files:
        text = txt_file.read_text(encoding="utf-8")
        cleaned = clean_novel(text)
        file_pairs = sliding_window_pairs(cleaned, args.chunk_size, args.overlap)
        all_pairs.extend(file_pairs)
        print(f"{txt_file.name}: {len(file_pairs)} raw pairs")

    validated = validate_pairs(all_pairs, min_output_chars=args.min_output_chars)
    print(f"validated pairs: {len(validated)} / {len(all_pairs)}")

    if args.dry_run:
        return

    with output_path.open("w", encoding="utf-8") as f:
        for pair in validated:
            row = {
                "instruction": pair.instruction,
                "input": pair.input,
                "output": pair.output,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()

