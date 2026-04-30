"""
Implement scripts/clean_text.py using @file:configs/qlora_config.yaml and @notepad:jinyong-project. The script should: accept a directory of .txt files, detect and normalize encoding (GB2312/UTF-8), strip chapter headers matching Chinese patterns, remove HTML artifacts, normalize punctuation, collapse whitespace, and write cleaned output to data/processed/. Add a --dry-run flag that prints stats without writing.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

CHAPTER_PATTERNS: tuple[str, ...] = (
    r"第[零一二三四五六七八九十百千]+[回章节卷].*?\n",
    r"^\s*[（(]\s*[一二三四五六七八九十]+\s*[)）].*?\n",
)


def detect_encoding(raw: bytes) -> str:
    """Pick the first encoding that decodes without error (UTF-8 / GB family / Big5)."""
    for enc in ("utf-8", "gb2312", "gbk", "big5"):
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "utf-8"


def clean_novel(text: str) -> str:
    for pattern in CHAPTER_PATTERNS:
        text = re.sub(pattern, "\n", text, flags=re.MULTILINE)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u3000", "")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_data_dirs_from_config(config_path: Path) -> tuple[str, str] | None:
    if not config_path.is_file():
        return None
    try:
        import yaml
    except ModuleNotFoundError:
        return None
    with open(config_path, encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}
    data = cfg.get("data") or {}
    raw_dir = data.get("raw_txt_dir")
    out_dir = data.get("processed_txt_dir")
    if isinstance(raw_dir, str) and isinstance(out_dir, str):
        return raw_dir, out_dir
    return None


def process_file(src: Path, dst_dir: Path, *, dry_run: bool) -> dict[str, Any]:
    raw_bytes = src.read_bytes()
    enc = detect_encoding(raw_bytes)
    raw = raw_bytes.decode(enc, errors="replace")
    cleaned = clean_novel(raw)
    raw_len = len(raw)
    clean_len = len(cleaned)
    reduction = round((1 - clean_len / raw_len) * 100, 1) if raw_len else 0.0
    stats: dict[str, Any] = {
        "file": src.name,
        "encoding": enc,
        "raw_chars": raw_len,
        "clean_chars": clean_len,
        "reduction_pct": reduction,
    }
    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        dst.write_text(cleaned, encoding="utf-8")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean raw Jin Yong novel text files.")
    parser.add_argument(
        "--config",
        default="configs/qlora_config.yaml",
        help="YAML config for default raw/processed dirs (data.raw_txt_dir, data.processed_txt_dir)",
    )
    parser.add_argument("--input-dir", default=None, help="Directory of raw .txt files (overrides config)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for cleaned .txt files (overrides config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print per-file stats without writing cleaned files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    from_config = load_data_dirs_from_config(config_path)
    default_raw, default_processed = (
        from_config if from_config is not None else ("data/raw", "data/processed")
    )
    input_dir = Path(args.input_dir or default_raw)
    output_dir = Path(args.output_dir or default_processed)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")

    total_raw = 0
    total_clean = 0
    for src in txt_files:
        stats = process_file(src, output_dir, dry_run=args.dry_run)
        total_raw += stats["raw_chars"]
        total_clean += stats["clean_chars"]
        tag = "[DRY RUN] " if args.dry_run else ""
        print(
            f"{tag}{stats['file']:30s} {stats['encoding']:8s} "
            f"{stats['raw_chars']:>10,} → {stats['clean_chars']:>10,} "
            f"chars (-{stats['reduction_pct']}%)"
        )

    print(f"\nTotal: {total_raw:,} → {total_clean:,} chars")


if __name__ == "__main__":
    main()
