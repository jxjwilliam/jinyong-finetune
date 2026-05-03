# Jin Yong Fine-Tune (AutoDL + QLoRA)

Fine-tune **`Qwen/Qwen2.5-7B-Instruct`** with **QLoRA** (4-bit NF4) for Jin Yong–style Chinese wuxia generation. The default training profile targets **[AutoDL](https://www.autodl.com)** with **NVIDIA RTX 4090 (24 GB VRAM)**; **Kaggle / Colab** remain optional with a smaller-GPU config (see `.cursor/rules/autodl.mdc`).

## Project Layout

```
jinyong-finetune/
├── configs/
│   └── qlora_config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── instructions/
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── 02_train.ipynb
│   └── 03_inference.ipynb
├── outputs/
├── scripts/
│   ├── clean_text.py
│   ├── build_instructions.py
│   ├── generate_typed_pairs.py
│   ├── generate_more_types_pairs.py
│   └── train.py
├── requirements.txt
└── .cursor/rules/
```

## Quick Start (Local)

1. Create environment:

   `python3 -m venv .venv && source .venv/bin/activate`

2. Install dependencies:

   `pip install -r requirements.txt`

   For training you also need a PyTorch + CUDA stack and packages such as `transformers`, `peft`, `trl`, `accelerate`, `bitsandbytes`, `datasets` (see `notebooks/02_train.ipynb` pip cell for pinned examples).

**AutoDL env:**
   `pip install -U bitsandbytes accelerate peft transformers datasets trl torch==2.1.0  # 适配4090的torch版本`

3. Put novel text files under `data/raw/`.

4. Clean encodings and noise, then build the instruction JSONL (from cleaned `data/processed/`):

   `python scripts/clean_text.py`

   `python scripts/build_instructions.py --stats`

   To preview counts without writing JSONL: `python scripts/build_instructions.py --dry-run --stats`.

   If you skip `clean_text.py` and point `--input-dir` at raw exports, add `--apply-clean`.

   For stronger instruction-following, generate typed pairs then merge:

   `python scripts/generate_typed_pairs.py --output data/instructions/typed_pairs.jsonl --per-template 20`

   Or use alternative Chinese-model APIs via `.env` keys (DeepSeek/Kimi/MiniMax/GLM):  
   `pip install openai python-dotenv && python scripts/generate_more_types_pairs.py --dry-run`

   `python scripts/build_instructions.py --typed-jsonl data/instructions/typed_pairs.jsonl --stats`

## Quick Start (AutoDL)

1. Clone this repo on the instance (e.g. under `/root/autodl-tmp/jinyong-finetune`).
2. Open **`notebooks/02_train.ipynb`** or run the same commands from the repo root (GPU check + install cell, then `clean_text` → `build_instructions` → `train.py`).
3. Populate **`data/raw/`** (upload, `scp`, or `kaggle datasets download …` if you use the Kaggle API on the box).
4. Training reads **`configs/qlora_config.yaml`** (`bf16`, `packing: false`, effective batch 16 on 4090).
5. Artifacts: **`outputs/jinyong-qlora/adapter/`**. Zip and download before the instance recycles. Merge / GGUF / Ollama: **`docs/LORA_TO_GGUF_GUIDE.md`**.

## Quick Start (Kaggle / Colab, optional)

1. Upload this repo to GitHub.
2. Clone in a GPU notebook; attach or download the **Jinyong Wuxia** dataset: `kaggle datasets download -d evilpsycho42/jinyong-wuxia -p data/raw --unzip`
3. Run `notebooks/01_data_prep.ipynb` then `notebooks/02_train.ipynb`.
4. On **T4 (16 GB)** you may need a copied YAML with **`fp16: true`**, **`bf16: false`**, **`bnb_4bit_compute_dtype: float16`**, and smaller `per_device_train_batch_size` — see `.cursor/rules/autodl.mdc`.

## Dataset Schema

Each JSONL row:

```json
{
  "instruction": "以金庸武侠小说的风格，续写以下段落：",
  "input": "（上文）",
  "output": "（续写）"
}
```

## Notes

- Default **`configs/qlora_config.yaml`** is tuned for **RTX 4090** (bf16 + bnb float16 compute as bfloat16 where set in YAML).
- `outputs/` and raw/processed/instruction datasets are ignored by git.
- Keep text UTF-8 encoded.
