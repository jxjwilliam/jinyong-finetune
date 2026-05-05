#!/usr/bin/env bash
set -euo pipefail
# Requires ANTHROPIC_API_KEY for `claude`; DEEPSEEK/KIMI/MINIMAX/GLM keys (see .env.example) for `openai`.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${ROOT}/data/instructions"
cd "${ROOT}"

python scripts/gen/generate_typed_pairs.py claude --per-template 5 \
  --output "${OUT}/typed_pairs.jsonl"

python scripts/gen/generate_typed_pairs.py openai --providers deepseek --per-template 5 \
  --output "${OUT}/more_types_deepseek.jsonl"

python scripts/gen/generate_typed_pairs.py openai --providers kimi --per-template 5 \
  --output "${OUT}/more_types_kimi.jsonl"

python scripts/gen/generate_typed_pairs.py openai --providers glm --per-template 5 \
  --output "${OUT}/more_types_glm.jsonl"

python scripts/gen/generate_typed_pairs.py openai --providers minimax --per-template 5 \
  --output "${OUT}/more_types_minimax.jsonl"

# generate jinyong_sft.jsonl
python scripts/data/build_instructions.py \
  --typed-jsonl "${OUT}/typed_pairs.jsonl" \
  --typed-jsonl "${OUT}/more_types_deepseek.jsonl" \
  --typed-jsonl "${OUT}/more_types_kimi.jsonl" \
  --typed-jsonl "${OUT}/more_types_minimax.jsonl" \
  --typed-jsonl "${OUT}/more_types_glm.jsonl" \
  --stats
