from __future__ import annotations

import argparse
import re
from pathlib import Path


def clean_novel(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u3000", "")
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"第[零一二三四五六七八九十百千]+[回章节].*?\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def process_file(src: Path, dst: Path) -> None:
    raw = src.read_text(encoding="utf-8")
    cleaned = clean_novel(raw)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(cleaned, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean raw Jin Yong novel text files.")
    parser.add_argument("--input-dir", default="data/raw", help="Directory of raw .txt files")
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for cleaned .txt files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")

    for src in txt_files:
        dst = output_dir / src.name
        process_file(src, dst)
        print(f"cleaned: {src} -> {dst}")


if __name__ == "__main__":
    main()

