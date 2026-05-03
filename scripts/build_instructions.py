from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

try:
    from clean_text import clean_novel
except ImportError:
    def clean_novel(text: str) -> str:
        return text

DEFAULT_INSTRUCTION = "以金庸武侠小说的风格，续写以下段落："

TYPED_TEMPLATES: tuple[str, ...] = (
    "以金庸武侠风格，描写一场高手之间的内力比拼，约200字",
    "以金庸风格写一段江湖儿女的离别场景，情感含蓄，约200字",
    "描写一位武功高强但性格孤傲的侠客初入客栈的场景，约200字",
    "用金庸笔法写出两个门派之间因误会而起的冲突，约200字",
    "以金庸笔法描写一位高手施展轻功的场景，约200字",
    "写一段金庸风格的武学秘籍传授场景，师父语气庄重，约200字",
    "描写一场以少胜多的江湖打斗，主角以智取胜，约200字",
    "以金庸风格写一段两位旧识重逢却各怀心事的对话，约200字",
    "描写一个初出茅庐的少年第一次见识真正高手的震撼，约200字",
    "以金庸笔法写出一位反派的出场，气势逼人却不失深度，约200字",
    "用金庸风格描写江湖门派的拜师仪式，约200字",
    "写一段武功秘籍的文字描述，风格古朴，暗含哲理，约200字",
    "以金庸风格描写两位武林高手以棋局论道的场景，约200字",
    "写一段江湖恩怨中的临终托付场景，情真意切，约200字",
    "以金庸风格描写一场追逐战，穿越山林水泽，约200字",
    "描写一位隐居高人被迫出山的内心挣扎，约200字",
    "以金庸笔法写出一段武功心法的顿悟场景，约200字",
    "描写江湖中一次重大武林大会的开场，约200字",
    "写一段金庸风格的毒功与解毒的对决，约200字",
    "以金庸风格描写一位侠客独自面对绝境的内心独白，约200字",
)


@dataclass
class Pair:
    instruction: str
    input: str
    output: str


def load_yaml(path: Path) -> dict[str, Any] | None:
    try:
        import yaml
    except ModuleNotFoundError:
        return None
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def sliding_segments(text: str, chunk_size: int, overlap: int) -> list[tuple[str, str]]:
    chars = list(text)
    segments: list[tuple[str, str]] = []
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than chunk_size")

    i = 0
    while i + 2 * chunk_size <= len(chars):
        prompt = "".join(chars[i : i + chunk_size]).strip()
        continuation = "".join(chars[i + chunk_size : i + 2 * chunk_size]).strip()
        if prompt and continuation:
            segments.append((prompt, continuation))
        i += step
    return segments


def continuation_pairs(segments: list[tuple[str, str]]) -> list[Pair]:
    return [Pair(DEFAULT_INSTRUCTION, inp, out) for inp, out in segments]


def load_typed_pairs_from_jsonl(path: Path) -> list[Pair]:
    pairs: list[Pair] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pairs.append(
                Pair(
                    instruction=obj["instruction"],
                    input=obj.get("input", ""),
                    output=obj["output"],
                )
            )
    return pairs


def validate_pairs(pairs: Iterable[Pair], min_output_chars: int = 50) -> list[Pair]:
    validated: list[Pair] = []
    for pair in pairs:
        if not pair.instruction.strip():
            continue
        if len(pair.output.strip()) < min_output_chars:
            continue
        validated.append(pair)
    return validated


def train_val_counts(n_total: int, eval_ratio: float) -> tuple[int, int]:
    if n_total <= 0:
        return 0, 0
    if eval_ratio <= 0:
        return n_total, 0
    n_test = max(1, int(round(n_total * eval_ratio)))
    if n_total == 1:
        return 1, 0
    n_test = min(n_test, n_total - 1)
    return n_total - n_test, n_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build instruction JSONL from cleaned novels."
    )
    parser.add_argument("--config", default="configs/qlora_config.yaml")
    parser.add_argument("--input-dir", default=None, help="Directory of cleaned .txt novels")
    parser.add_argument("--output", default=None, help="Output JSONL path")
    parser.add_argument(
        "--typed-jsonl",
        default=None,
        help="Path to pre-generated typed pairs JSONL (from generate_typed_pairs.py). If omitted, typed pairs are skipped.",
    )
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--min-output-chars", type=int, default=50)
    parser.add_argument("--apply-clean", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Shuffle seed before capping with --max-pairs",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stats", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    default_in = "data/processed"
    default_out = "data/instructions/jinyong_sft.jsonl"
    eval_ratio = 0.05
    seed = args.seed if args.seed is not None else 42

    if config_path.is_file():
        cfg = load_yaml(config_path)
        if cfg:
            data_cfg = cfg.get("data") or {}
            train_cfg = cfg.get("training") or {}
            default_in = data_cfg.get("processed_txt_dir") or default_in
            default_out = data_cfg.get("instruction_jsonl") or default_out
            eval_ratio = float(train_cfg.get("eval_split_ratio", eval_ratio))
            if args.seed is None:
                seed = int(train_cfg.get("seed", seed))

    input_dir = Path(args.input_dir or default_in)
    output_path = Path(args.output or default_out)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")

    all_segments: list[tuple[str, str]] = []
    for txt_file in txt_files:
        text = txt_file.read_text(encoding="utf-8")
        if args.apply_clean:
            text = clean_novel(text)
        segs = sliding_segments(text, args.chunk_size, args.overlap)
        all_segments.extend(segs)
        print(f"  {txt_file.name}: {len(segs):,} windows")

    cont_pairs = continuation_pairs(all_segments)
    cont_valid = validate_pairs(cont_pairs, args.min_output_chars)

    typed_valid: list[Pair] = []
    if args.typed_jsonl:
        typed_path = Path(args.typed_jsonl)
        if typed_path.is_file():
            raw_typed = load_typed_pairs_from_jsonl(typed_path)
            typed_valid = validate_pairs(raw_typed, args.min_output_chars)
            print(f"\nLoaded {len(raw_typed):,} typed pairs -> {len(typed_valid):,} valid")
        else:
            print(f"[warn] --typed-jsonl not found: {typed_path}, skipping")
    else:
        print("\n[info] No --typed-jsonl provided. Only continuation pairs will be used.")
        print("       Run generate_typed_pairs.py first for better instruction-following.")

    combined = cont_valid + typed_valid

    if args.seed is not None or args.max_pairs is not None:
        rng = random.Random(seed)
        rng.shuffle(combined)
    if args.max_pairs is not None and len(combined) > args.max_pairs:
        combined = combined[: args.max_pairs]

    if args.stats or args.dry_run:
        n_train, n_val = train_val_counts(len(combined), eval_ratio)
        print(f"\nContinuation pairs : {len(cont_valid):,}")
        print(f"Typed scene pairs  : {len(typed_valid):,}")
        print(f"Total valid pairs  : {len(combined):,}")
        print(f"Train / Val split  : {n_train:,} / {n_val:,} "
              f"(eval_ratio={eval_ratio}, seed={seed})")

    if args.dry_run:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for pair in combined:
            row = {
                "instruction": pair.instruction,
                "input": pair.input,
                "output": pair.output,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved -> {output_path}  ({len(combined):,} rows)")


if __name__ == "__main__":
    main()
