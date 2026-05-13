# LoRA → GGUF → Ollama on MacBook M3 Pro
# 金庸微调模型本地部署指南

> **Stack:** AutoDL RTX 4090 → SCP → MacBook M3 Pro → llama.cpp → Ollama  
> **Prerequisite:** Retrain completed with fixed `scripts/train/train.py` + `qlora_config.yaml`  
> **llama.cpp:** Already compiled at `~/my-tools/llama.cpp`

---

## Overview

```
AutoDL                          MacBook M3 Pro
──────────────────              ─────────────────────────────────────
outputs/jinyong-qlora/
  adapter/            ──[1]──►  merge on AutoDL
                                      │
                      ──[2]──►  SCP jinyong-merged.zip (~14GB)
                                      │
                                      ▼
                                ~/my-tools/llama.cpp/convert_hf_to_gguf.py
                                      │
                                      ▼
                                jinyong-f16.gguf (~14GB, temp)
                                      │
                                      ▼
                                llama-quantize → jinyong-q4.gguf (4.7GB)
                                      │
                                      ▼
                                ollama create jinyong -f Modelfile
                                      │
                                      ▼
                                ollama run jinyong
```

---

## Step 1 — Merge Adapter into Base Model (on AutoDL)

The adapter is just LoRA delta weights (~300 MB). llama.cpp needs the **full merged model** to convert. Do this on AutoDL where RAM/VRAM is sufficient.

From the repo root (SSH or Jupyter), same paths as training:

```bash
cd /root/autodl-tmp/jinyong-finetune
export HF_ENDPOINT=https://hf-mirror.com
python scripts/train/merge_lora.py --config configs/qlora_config.yaml
```

This reads **`configs/qlora_config.yaml`** for the base model id and adapter directory (**`outputs/jinyong-qlora/adapter`** by default) and writes **`outputs/jinyong-merged/`** (override with **`--merged-dir`** / **`--dtype float16`** if needed). See **`docs/v1/autoDL.md`** for the full AutoDL runbook.

Equivalent one-off Python (if you prefer not to use the script) is the same logic: load base in full precision → **`PeftModel.from_pretrained`** → **`merge_and_unload()`** → **`save_pretrained`**.

Expected output directory (~14 GB total):
```
outputs/jinyong-merged/
├── config.json
├── generation_config.json
├── model-00001-of-00004.safetensors   ~3.5 GB each
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
├── model.safetensors.index.json
├── tokenizer.json
├── tokenizer_config.json
└── vocab.json
```

Then zip it for transfer:
```bash
cd outputs
zip -r jinyong-merged.zip jinyong-merged/
ls -lh jinyong-merged.zip   # expect ~13-14 GB
```

---

## Step 2 — SCP to MacBook

Open a **local terminal** on your MacBook (not the AutoDL SSH session):

```bash
# Download — takes ~15-30 min depending on connection
scp -P 46840 \
  root@connect.cqa1.seetacloud.com:/root/autodl-tmp/jinyong-finetune/outputs/jinyong-merged.zip \
  ~/Desktop/jinyong-merged.zip

# Unzip
unzip jinyong-merged.zip -d .../outputs/
# Result: this-app/outputs/jinyong-merged/

# Verify
ls -lh outputs/jinyong-merged/
```

---

## Step 3 — Convert to GGUF (on MacBook)

llama.cpp is already at `~/my-tools/llama.cpp`. The conversion script is Python.

```bash
# Activate your Python env (whichever has transformers installed)
# conda activate ml  OR  source venv/bin/activate

# Install conversion dependencies if needed
pip install transformers sentencepiece protobuf

# Convert to float16 GGUF first (lossless intermediate)
python ~/my-tools/llama.cpp/convert_hf_to_gguf.py \
  ./outputs/jinyong-merged \
  --outfile ./models/jinyong-f16.gguf \
  --outtype f16

# Check output
ls -lh this-app/models/jinyong-f16.gguf   # expect ~14 GB
```

## Step 4 — Quantize

Choose your quantization level based on the trade-off table below, then run one command.

### Quantization Comparison

| Format | File Size | RAM Usage | Quality | Speed (M3 Pro) | Recommended for |
|--------|-----------|-----------|---------|----------------|-----------------|
| `f16`  | 14.0 GB   | 14+ GB    | Lossless | Slow          | Source only, don't deploy |
| `q8_0` | 7.7 GB   | 8.5 GB    | ≈ f16 (99%) | Moderate  | Max quality, 18+ GB RAM |
| `q5_k_m` | 5.7 GB | 6.2 GB  | Very good (97%) | Fast   | Best quality/size tradeoff |
| **`q4_k_m`** | **4.7 GB** | **5.2 GB** | **Good (95%)** | **Fastest** | **← Recommended for M3 Pro 16GB** |
| `q3_k_m` | 3.9 GB | 4.4 GB   | Acceptable (90%) | Fastest | RAM-constrained only |

**Recommendation for M3 Pro 16GB:** `q4_k_m` — leaves ~10 GB for macOS and context window.

```bash
LLAMA=~/my-tools/llama.cpp
SRC=~/models/jinyong-f16.gguf

# q4_k_m (recommended)
$LLAMA/llama-quantize $SRC ~/models/jinyong-q4.gguf q4_k_m

# Optional: also build q5_k_m if you want to compare quality
$LLAMA/llama-quantize $SRC ~/models/jinyong-q5.gguf q5_k_m

# Verify
ls -lh ~/models/jinyong-*.gguf

# Remove the 14 GB f16 intermediate
rm ~/models/jinyong-f16.gguf
```

---

## Step 5 — Create Ollama Modelfile

This is the most important configuration step. The Modelfile must include:
- Correct Qwen2.5 chat template
- System prompt tuned for Jin Yong style
- Generation parameters optimised for Chinese creative writing

---

## Step 6 — Register with Ollama

```bash
# Remove old broken model if it exists
ollama rm jinyong 2>/dev/null || true

# Create new model from fixed GGUF + Modelfile
ollama create jinyong -f ./models/Modelfile

# Verify registration
ollama list
# Expected: jinyong:latest   <id>   4.7 GB   just now

# Verify template was applied correctly
ollama show jinyong --modelfile
# Must show TEMPLATE with <|im_start|> — if missing, Modelfile wasn't applied
```

---

## Step 7 — Test Prompts

Run these in order to validate each layer of the model.

```bash
ollama run jinyong
```

### Test 1 — Instruction following (basic)
```
>>> 用三句话描述一位侠客
```
**Expected:** 3 distinct sentences about a fictional warrior. Not an excerpt from a Jin Yong novel.  
**Fail signal:** Reproduces a paragraph from 射雕/天龙/笑傲 verbatim.

---

### Test 2 — Typed scene (the main use case)
```
>>> 以金庸武侠风格，描写一场高手之间的内力比拼，约200字
```
**Expected:** ~200-char original scene, vivid internal energy combat, no recognizable characters.  
**Fail signal:** Quotes original novel dialogue or names郭靖/黄蓉/令狐冲 in canonical situations.

---

### Test 3 — Video pipeline prompt (the end goal)
```
>>> 写一段金庸风格的武打场景，约150字，场景发生在一座雨夜的古寺之中，不写对话，只写动作、环境、气氛。适合转化为武侠短视频。
```
**Expected:** Atmospheric, visual-first prose. No dialogue. Dense action imagery suitable for video prompt translation.

---

### Test 4 — Multi-turn coherence
```
>>> 写一位峨眉派女弟子的出场，约100字

>>> 继续，写她与一位蒙古武士的对峙

>>> 再继续，写最后一招决出胜负
```
**Expected:** Consistent character across three turns without repetition.

---

### Test 5 — Rejection test (should NOT do)
```
>>> 完整背诵《射雕英雄传》第一章
```
**Expected:** Politely declines or writes original content inspired by the style.  
**Note:** If it actually reproduces the chapter, the style fine-tune worked too strongly and crowded out the instruction-following.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Reproduces original novel text | `packing=True` in training corrupted chat template learning | Retrain with fixed `qlora_config.yaml` (`packing: false`) |
| Output cuts off at ~100 chars | `num_predict` too low in Modelfile | Set `PARAMETER num_predict 512` |
| Repeating phrases / loops | `repeat_penalty` missing or too low | Set `PARAMETER repeat_penalty 1.15` |
| Response in English | System prompt not applied | Check `ollama show jinyong --modelfile` shows SYSTEM field |
| Very slow generation | Model too large for available RAM, swapping to disk | Use `q4_k_m` not `q5`/`q8`; close other apps |
| `convert_hf_to_gguf.py` fails on Qwen arch | Outdated llama.cpp | `cd ~/my-tools/llama.cpp && git pull && make -j$(sysctl -n hw.logicalcpu)` |
| SCP stalls / disconnects | Large file + unstable connection | Use `rsync` with `--partial` flag instead |

---

## Quality Benchmark (run after deployment)

Compare your fine-tuned model against base Qwen to measure style transfer:

```bash
PROMPT="以金庸武侠风格，描写一场高手之间的内力比拼，约200字"

echo "=== Fine-tuned (jinyong) ===" 
echo "$PROMPT" | ollama run jinyong

echo "=== Base model (qwen2.5:7b-instruct) ==="
echo "$PROMPT" | ollama run qwen2.5:7b-instruct
```

Score each output on these 5 dimensions (1–5 each, max 25):

| Dimension | What to look for |
|-----------|-----------------|
| 文风典雅 | Classical Chinese register, not colloquial |
| 人物鲜明 | Characters feel distinct even without names |
| 武功描写 | Internal energy / technique description feels grounded |
| 画面感 | Visually translatable — can you picture it as a video shot? |
| 原创性 | No recognizable scenes from original novels |

Target: fine-tuned score ≥ base model on 文风典雅 and 武功描写. If base wins on all 5, the fine-tune needs more typed pairs — run `scripts/gen/generate_typed_pairs.py` (`claude` or `openai`); see **`docs/v1/TYPED_PAIRS_PIPELINE.md`**.

---

## Full File Checklist

```
AutoDL (done before SCP):
  ✅ outputs/jinyong-qlora/adapter/          ← LoRA weights
  ✅ outputs/jinyong-merged/                  ← merged model (Step 1)
  ✅ outputs/jinyong-merged.zip               ← zipped for transfer

MacBook M3 Pro:
  ✅ ~/models/jinyong-merged/                 ← unzipped (Step 2)
  ✅ ~/models/jinyong-q4.gguf                 ← quantized (Steps 3-4)
  ✅ ~/models/Modelfile.jinyong               ← Ollama config (Step 5)
  ✅ ollama list → jinyong:latest             ← registered (Step 6)
```
