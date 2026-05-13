from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable

_repo_root = Path(__file__).resolve().parents[2]
_lib = _repo_root / "scripts" / "lib"
if str(_lib) not in sys.path:
    sys.path.insert(0, str(_lib))

try:
    from clean_text import clean_novel
except ImportError:
    def clean_novel(text: str) -> str:
        return text

from instruction_jsonl import Pair, load_pairs_jsonl, pair_to_json_obj
from dedup_pairs import dedup_continuation_pairs

DEFAULT_INSTRUCTION = "以金庸武侠小说的风格，续写以下段落："


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


def expand_typed_jsonl_paths(entries: list[str] | None) -> list[Path]:
    """Flatten repeatable ``--typed-jsonl`` args and comma-separated paths."""
    if not entries:
        return []
    out: list[Path] = []
    for entry in entries:
        for part in entry.split(","):
            p = part.strip()
            if p:
                out.append(Path(p))
    return out


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
        action="append",
        default=None,
        metavar="PATH",
        help=(
            "Typed pairs JSONL (repeat flag or comma-separated in one arg). "
            "Skipped if path missing. See docs/v1/TYPED_PAIRS_PIPELINE.md."
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--min-output-chars", type=int, default=50)
    parser.add_argument("--apply-clean", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument(
        "--dedup-continuation",
        action="store_true",
        help="Apply MinHash LSH deduplication to continuation pairs before merging typed pairs.",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.85,
        help="MinHash LSH similarity threshold for continuation deduplication.",
    )
    parser.add_argument(
        "--min-typed-ratio",
        type=float,
        default=None,
        help="Minimum typed pairs ratio (typed/total). Overrides config data.typed_pairs.min_ratio_vs_continuation.",
    )
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
    min_typed_ratio = args.min_typed_ratio if args.min_typed_ratio is not None else 0.0

    if config_path.is_file():
        cfg = load_yaml(config_path)
        if cfg:
            data_cfg = cfg.get("data") or {}
            train_cfg = cfg.get("training") or {}
            typed_cfg = data_cfg.get("typed_pairs") or {}
            default_in = data_cfg.get("processed_txt_dir") or default_in
            default_out = data_cfg.get("instruction_jsonl") or default_out
            eval_ratio = float(train_cfg.get("eval_split_ratio", eval_ratio))
            if args.min_typed_ratio is None:
                min_typed_ratio = float(typed_cfg.get("min_ratio_vs_continuation", min_typed_ratio))
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
    dedup_report: dict[str, Any] | None = None
    if args.dedup_continuation:
        deduped, dedup_stats = dedup_continuation_pairs(
            cont_valid,
            threshold=args.dedup_threshold,
            num_perm=128,
        )
        dedup_report = {
            "before": dedup_stats.before,
            "after": dedup_stats.after,
            "removed": dedup_stats.removed,
            "removed_ratio": round(dedup_stats.removed_ratio, 6),
            "threshold": args.dedup_threshold,
        }
        cont_valid = deduped

    typed_valid: list[Pair] = []
    typed_paths = expand_typed_jsonl_paths(args.typed_jsonl)
    if typed_paths:
        raw_total = 0
        for typed_path in typed_paths:
            if not typed_path.is_file():
                print(f"[warn] --typed-jsonl not found: {typed_path}, skipping")
                continue
            raw_typed = load_pairs_jsonl(typed_path)
            raw_total += len(raw_typed)
            batch = validate_pairs(raw_typed, args.min_output_chars)
            typed_valid.extend(batch)
            print(f"\nLoaded {typed_path}: {len(raw_typed):,} rows -> {len(batch):,} valid")
        print(f"Typed pairs total (valid): {len(typed_valid):,} (from {raw_total:,} raw rows)")
    else:
        print("\n[info] No --typed-jsonl provided. Only continuation pairs will be used.")
        print("       Run scripts/gen/generate_typed_pairs.py claude|openai … (see docs/v1/TYPED_PAIRS_PIPELINE.md).")

    combined = cont_valid + typed_valid

    if args.seed is not None or args.max_pairs is not None:
        rng = random.Random(seed)
        rng.shuffle(combined)
    if args.max_pairs is not None and len(combined) > args.max_pairs:
        combined = combined[: args.max_pairs]

    if args.stats or args.dry_run:
        n_train, n_val = train_val_counts(len(combined), eval_ratio)
        typed_ratio = (len(typed_valid) / len(combined)) if combined else 0.0
        print(f"\nContinuation pairs : {len(cont_valid):,}")
        print(f"Typed scene pairs  : {len(typed_valid):,}")
        print(f"Total valid pairs  : {len(combined):,}")
        print(f"Typed ratio        : {typed_ratio:.3f}")
        print(f"Train / Val split  : {n_train:,} / {n_val:,} "
              f"(eval_ratio={eval_ratio}, seed={seed})")
        if dedup_report is not None:
            print(
                f"Dedup report       : before={dedup_report['before']:,}, "
                f"after={dedup_report['after']:,}, removed={dedup_report['removed']:,} "
                f"({dedup_report['removed_ratio']:.2%}, threshold={dedup_report['threshold']})"
            )

    typed_ratio = (len(typed_valid) / len(combined)) if combined else 0.0
    if min_typed_ratio > 0 and typed_ratio < min_typed_ratio:
        raise ValueError(
            f"typed ratio too low: {typed_ratio:.3f} < {min_typed_ratio:.3f}. "
            "Increase --per-template or add more --typed-jsonl inputs."
        )

    if args.dry_run:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for pair in combined:
            fh.write(json.dumps(pair_to_json_obj(pair), ensure_ascii=False) + "\n")

    print(f"\nSaved -> {output_path}  ({len(combined):,} rows)")
    if dedup_report is not None:
        dedup_report_path = Path("outputs/data/dedup_report.json")
        dedup_report_path.parent.mkdir(parents=True, exist_ok=True)
        dedup_report_path.write_text(
            json.dumps(dedup_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Dedup report -> {dedup_report_path}")


if __name__ == "__main__":
    main()
