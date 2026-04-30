# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Fine-tune `Qwen/Qwen2.5-7B-Instruct` with QLoRA (4-bit NF4 quantization) to generate Chinese wuxia fiction in the style of Jin Yong (金庸). Designed to run on Kaggle free-tier T4 GPU (16GB VRAM).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Key Commands

```bash
# Clean raw novel text (defaults from configs/qlora_config.yaml data.raw_txt_dir / processed_txt_dir)
python scripts/clean_text.py
python scripts/clean_text.py --dry-run

# Build instruction pairs: continuation windows + typed scene templates (defaults: data/processed → JSONL path in YAML)
python scripts/build_instructions.py --stats
python scripts/build_instructions.py --dry-run --stats
python scripts/build_instructions.py --chunk-size 300 --overlap 100

# Train (all hyperparameters come from the YAML config)
python scripts/train.py --config configs/qlora_config.yaml
```

`build_instructions.py` optionally imports `clean_novel` when `--apply-clean` is set. Run `python scripts/…` from the repo root so `scripts/` is on `sys.path`, or use the notebooks (they `chdir` to the repo root).

## Architecture & Data Flow

```
data/raw/*.txt
  → scripts/clean_text.py        strips headers, HTML, fullwidth spaces, normalizes whitespace
  → data/processed/*.txt         (optional intermediate)
  → scripts/build_instructions.py  sliding-window continuations (chunk=300, overlap=100) + typed scene pairs (20 templates)
  → data/instructions/jinyong_sft.jsonl   {instruction, input, output} rows
  → scripts/train.py             loads YAML config, 4-bit Qwen2.5-7B, QLoRA r=64
  → outputs/jinyong-qlora/adapter/        saved LoRA adapter (not full weights)
```

### JSONL schema
```json
{"instruction": "以金庸武侠小说的风格，续写以下段落：", "input": "（上文300字）", "output": "（续写300字）"}
```

### Training config (`configs/qlora_config.yaml`)
Single source of truth for all hyperparameters — no inline constants in `train.py`. Key values: `r=64`, `lora_alpha=128`, `batch_size=2`, `gradient_accumulation_steps=8` (effective batch=16), `learning_rate=2e-4`, `max_seq_length=1024`, `fp16=true`, `eval_split_ratio=0.05`.

### ChatML prompt format
```
<|im_start|>system
你是一位精通金庸武侠风格的写作助手。<|im_end|>
<|im_start|>user
{instruction}\n{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>
```

## Constraints & Conventions

- **Platform target:** Kaggle T4 (16GB VRAM) — use `fp16=True, bf16=False`; T4 does not support bf16.
- **Model order:** always call `prepare_model_for_kbit_training()` before `get_peft_model()`.
- **Tokenizer:** set `pad_token = eos_token` and `padding_side = "right"` for causal LM fine-tuning.
- **LoRA targets:** all 7 projection layers — `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- **Dtype:** always use explicit `torch.dtype` values; never rely on defaults.
- **Inference:** wrap `model.generate()` in `torch.no_grad()`.
- **Paths:** no hardcoded absolute paths; use relative paths or env vars (`MODEL_ID`, `SEED` via `.env`).
- **Text:** all files UTF-8; preserve Chinese punctuation; avoid modern slang in training data.
- **Git:** `outputs/`, `data/raw/`, `data/processed/`, and `data/instructions/*.jsonl` are git-ignored (only `.gitkeep` placeholders are tracked).

## Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/01_data_prep.ipynb` | Runs `build_instructions.py --dry-run`, inspects pair counts |
| `notebooks/02_train.ipynb` | Runs `train.py` with the YAML config |
| `notebooks/03_inference.ipynb` | Loads base tokenizer for generation testing |
