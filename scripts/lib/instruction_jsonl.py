"""Shared SFT JSONL row shape for instruction tuning.

Used by ``build_instructions.py`` (merge + validation) and typed generators so
the ``{instruction, input, output}`` schema stays in one place.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO


@dataclass
class Pair:
    instruction: str
    input: str
    output: str


def pair_from_json_obj(obj: dict) -> Pair:
    return Pair(
        instruction=obj["instruction"],
        input=obj.get("input", ""),
        output=obj["output"],
    )


def pair_to_json_obj(pair: Pair) -> dict[str, str]:
    return {
        "instruction": pair.instruction,
        "input": pair.input,
        "output": pair.output,
    }


def typed_pair_dict(instruction: str, output: str) -> dict[str, str]:
    """One JSONL object for typed-scene rows (empty context)."""
    return pair_to_json_obj(Pair(instruction=instruction.strip(), input="", output=output.strip()))


def load_pairs_jsonl(path: Path) -> list[Pair]:
    pairs: list[Pair] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pairs.append(pair_from_json_obj(obj))
    return pairs


def write_jsonl_line(fh: TextIO, row: dict[str, str]) -> None:
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl_rows(path: Path, rows: Iterable[dict[str, str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            write_jsonl_line(fh, row)
            n += 1
    return n


def count_nonempty_jsonl_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())
