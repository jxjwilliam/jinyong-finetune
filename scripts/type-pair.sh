#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/data/instructions"
cd "${ROOT}"

python scripts/generate_more_types_pairs.py --providers deepseek  --per-template 5 \
  --output "${OUT}/more_types_deepseek.jsonl"

python scripts/generate_more_types_pairs.py --providers kimi --per-template 5 \
  --output "${OUT}/more_types_kimi.jsonl"

python scripts/generate_more_types_pairs.py --providers glm --per-template 5 \
  --output "${OUT}/more_types_glm.jsonl"

python scripts/generate_more_types_pairs.py --providers minimax --per-template 5 \
  --output "${OUT}/more_types_minimax.jsonl"

cat "${OUT}/typed_pairs.jsonl" \
  "${OUT}/more_types_deepseek.jsonl" \
  "${OUT}/more_types_kimi.jsonl" \
  "${OUT}/more_types_minimax.jsonl" \
  "${OUT}/more_types_glm.jsonl" \
  > "${OUT}/all_typed.jsonl"

python scripts/build_instructions.py --typed-jsonl "${OUT}/all_typed.jsonl" --stats
