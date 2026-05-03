#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MERGED_DIR="${1:-${REPO_ROOT}/outputs/jinyong-merged}"
F16_OUT="${REPO_ROOT}/models/jinyong-f16.gguf"
Q4_OUT="${REPO_ROOT}/models/jinyong-q4.gguf"

if [[ ! -d "${MERGED_DIR}" ]]; then
  echo "Merged model directory not found: ${MERGED_DIR}"
  echo "Expected merged model from latest adapter at outputs/jinyong-merged"
  exit 1
fi

mkdir -p "${REPO_ROOT}/models"

echo "[1/2] Convert merged HF model to f16 GGUF..."
python "${HOME}/my-tools/llama.cpp/convert_hf_to_gguf.py" "${MERGED_DIR}" \
  --outfile "${F16_OUT}" \
  --outtype f16

echo "[2/2] Quantize GGUF to q4_k_m..."
"${HOME}/my-tools/llama.cpp/llama-quantize" "${F16_OUT}" "${Q4_OUT}" q4_k_m

echo "Done: ${Q4_OUT}"
rm -f "${F16_OUT}"
