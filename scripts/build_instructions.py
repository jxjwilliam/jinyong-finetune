from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from clean_text import clean_novel

DEFAULT_INSTRUCTION = "以金庸武侠小说的风格，续写以下段落："

TYPED_TEMPLATES: tuple[str, ...] = (
    "以金庸武侠风格，描写一场高手之间的内力比拼",
    "以金庸风格写一段江湖儿女的离别场景，情感含蓄",
    "描写一位武功高强但性格孤傲的侠客初入客栈的场景",
    "用金庸笔法写出两个门派之间因误会而起的冲突",
    "以金庸笔法描写一位高手施展轻功的场景",
    "写一段金庸风格的武学秘籍传授场景，师父语气庄重",
    "描写一场以少胜多的江湖打斗，主角以智取胜",
    "以金庸风格写一段两位旧识重逢却各怀心事的对话",
    "描写一个初出茅庐的少年第一次见识真正高手的震撼",
    "以金庸笔法写出一位反派的出场，气势逼人却不失深度",
    "用金庸风格描写江湖门派的拜师仪式",
    "写一段武功秘籍的文字描述，风格古朴，暗含哲理",
    "以金庸风格描写两位武林高手以棋局论道的场景",
    "写一段江湖恩怨中的临终托付场景，情真意切",
    "以金庸风格描写一场追逐战，穿越山林水泽",
    "描写一位隐居高人被迫出山的内心挣扎",
    "以金庸笔法写出一段武功心法的顿悟场景",
    "描写江湖中一次重大武林大会的开场",
    "写一段金庸风格的毒功与解毒的对决",
    "以金庸风格描写一位侠客独自面对绝境的内心独白",
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


def typed_scene_pairs(segments: list[tuple[str, str]], stride: int) -> list[Pair]:
    if stride < 1:
        raise ValueError("typed_stride must be >= 1")
    pairs: list[Pair] = []
    typed_idx = 0
    for i, (inp, out) in enumerate(segments):
        if i % stride != 0:
            continue
        tpl = TYPED_TEMPLATES[typed_idx % len(TYPED_TEMPLATES)]
        typed_idx += 1
        pairs.append(Pair(tpl, inp, out))
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
        description="Build instruction JSONL from cleaned novels (continuation + typed scene pairs).",
    )
    parser.add_argument(
        "--config",
        default="configs/qlora_config.yaml",
        help="YAML for defaults: processed dir, JSONL path, eval split ratio, seed",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory of .txt novels (default: data.processed_txt_dir from config)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: data.instruction_jsonl from config)",
    )
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--typed-stride", type=int, default=8, help="Emit one typed pair every N sliding windows")
    parser.add_argument("--min-output-chars", type=int, default=30)
    parser.add_argument(
        "--apply-clean",
        action="store_true",
        help="Run clean_novel() on each file (use if --input-dir points at raw exports)",
    )
    parser.add_argument("--max-pairs", type=int, default=None, help="Cap total rows after validation")
    parser.add_argument("--dry-run", action="store_true", help="Do not write JSONL")
    parser.add_argument("--stats", action="store_true", help="Print pair counts and train/val split estimate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    input_dir: Path
    output_path: Path
    eval_ratio = 0.05
    seed = 42

    default_in = "data/processed"
    default_out = "data/instructions/jinyong_sft.jsonl"
    eval_ratio = 0.05
    seed = 42
    if config_path.is_file():
        cfg = load_yaml(config_path)
        if cfg is not None:
            data_cfg = cfg.get("data") or {}
            train_cfg = cfg.get("training") or {}
            default_in = data_cfg.get("processed_txt_dir") or default_in
            default_out = data_cfg.get("instruction_jsonl") or default_out
            eval_ratio = float(train_cfg.get("eval_split_ratio", eval_ratio))
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
        print(f"{txt_file.name}: {len(segs)} sliding windows")

    cont_pairs = continuation_pairs(all_segments)
    typed_pairs = typed_scene_pairs(all_segments, args.typed_stride)
    combined = cont_pairs + typed_pairs

    validated = validate_pairs(combined, min_output_chars=args.min_output_chars)
    if args.max_pairs is not None and len(validated) > args.max_pairs:
        validated = sorted(
            validated,
            key=lambda p: (p.instruction, p.input, p.output),
        )[: args.max_pairs]

    n_cont_v = len(validate_pairs(cont_pairs, min_output_chars=args.min_output_chars))
    n_typ_v = len(validate_pairs(typed_pairs, min_output_chars=args.min_output_chars))

    if args.stats or args.dry_run:
        n_train, n_val = train_val_counts(len(validated), eval_ratio)
        print(f"Continuation pairs:  {n_cont_v:,}")
        print(f"Typed scene pairs:   {n_typ_v:,}")
        print(f"Total pairs:         {len(validated):,}")
        print(
            f"Train / Val split:   {n_train:,} / {n_val:,}  "
            f"(eval_ratio={eval_ratio}, seed={seed})"
        )

    if args.dry_run:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for pair in validated:
            row = {
                "instruction": pair.instruction,
                "input": pair.input,
                "output": pair.output,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.stats:
        print(f"Saved → {output_path}")
    else:
        print(f"saved: {output_path} ({len(validated):,} rows)")


if __name__ == "__main__":
    main()
