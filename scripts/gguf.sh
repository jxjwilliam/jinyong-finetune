#!/bin/bash

# Step 1: Convert to f16 GGUF
python ~/my-tools/llama.cpp/convert_hf_to_gguf.py ./outputs/jinyong-merged \
  --outfile ./jinyong-f16.gguf \
  --outtype f16

# Step 2: Quantize to q4_k_m
~/my-tools/llama.cpp/llama-quantize ./jinyong-f16.gguf ./jinyong-q4.gguf q4_k_m

# Optional: remove f16 intermediate (it's ~14GB)
rm ./jinyong-f16.gguf