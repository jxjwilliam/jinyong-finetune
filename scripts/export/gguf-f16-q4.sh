#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LLAMA_DIR="${LLAMA_DIR:-${HOME}/my-tools/llama.cpp}"
MODELS_DIR="${MODELS_DIR:-${REPO_ROOT}/models}"
MERGED_DIR="${1:-${REPO_ROOT}/outputs/jinyong-merged}"
F16_OUT="${MODELS_DIR}/jinyong-f16.gguf"
Q4_OUT="${MODELS_DIR}/jinyong-q4_k_m.gguf"
Q5_OUT="${MODELS_DIR}/jinyong-q5_k_m.gguf"
BUILD_Q5="${BUILD_Q5:-0}"
KEEP_F16="${KEEP_F16:-0}"

CONVERT_SCRIPT="${LLAMA_DIR}/convert_hf_to_gguf.py"
QUANT_BIN="${LLAMA_DIR}/llama-quantize"

usage() {
  cat <<'EOF'
Usage:
  scripts/export/gguf.sh [MERGED_DIR]

Defaults:
  MERGED_DIR = ./outputs/jinyong-merged
  LLAMA_DIR  = ~/my-tools/llama.cpp
  MODELS_DIR = ./models

Optional env:
  BUILD_Q5=1   Also build ./models/jinyong-q5_k_m.gguf
  KEEP_F16=1   Keep ./models/jinyong-f16.gguf after quantization
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -d "${MERGED_DIR}" ]]; then
  echo "Merged model directory not found: ${MERGED_DIR}"
  echo "Expected merged model from LoRA merge step."
  echo "Example: ~/models/jinyong-merged"
  exit 1
fi

if [[ ! -f "${CONVERT_SCRIPT}" ]]; then
  echo "convert_hf_to_gguf.py not found: ${CONVERT_SCRIPT}"
  echo "Set LLAMA_DIR or install llama.cpp at ~/my-tools/llama.cpp"
  exit 1
fi

if [[ ! -x "${QUANT_BIN}" ]]; then
  echo "llama-quantize not executable: ${QUANT_BIN}"
  echo "Build llama.cpp first (e.g. make -j\$(sysctl -n hw.logicalcpu))"
  exit 1
fi

mkdir -p "${MODELS_DIR}"

echo "[1/3] Convert merged HF model to f16 GGUF..."
python "${CONVERT_SCRIPT}" "${MERGED_DIR}" \
  --outfile "${F16_OUT}" \
  --outtype f16

echo "[2/3] Quantize to q4_k_m (recommended)..."
"${QUANT_BIN}" "${F16_OUT}" "${Q4_OUT}" q4_k_m

if [[ "${BUILD_Q5}" == "1" ]]; then
  echo "[3/3] Quantize to q5_k_m (optional)..."
  "${QUANT_BIN}" "${F16_OUT}" "${Q5_OUT}" q5_k_m
else
  echo "[3/3] Skip optional q5_k_m (set BUILD_Q5=1 to enable)."
fi

echo "Generated files:"
ls -lh "${MODELS_DIR}"/jinyong-*.gguf

if [[ "${KEEP_F16}" == "1" ]]; then
  echo "Keeping f16 intermediate: ${F16_OUT}"
else
  rm -f "${F16_OUT}"
  echo "Removed f16 intermediate: ${F16_OUT}"
fi

echo "Done: ${Q4_OUT}"
