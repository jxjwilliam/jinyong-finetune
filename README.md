# Jin Yong Fine-Tune (AutoDL + QLoRA)

Fine-tune **`Qwen/Qwen2.5-7B-Instruct`** with **QLoRA** (4-bit NF4) for Jin Yong–style Chinese wuxia generation. The default training profile targets **[AutoDL](https://www.autodl.com)** with **NVIDIA RTX 4090 (24 GB VRAM)**; **Kaggle / Colab** remain optional with a smaller-GPU config (see **`.cursor/rules/autodl.mdc`**).

**AutoDL end-to-end steps:** **`docs/autoDL.md`**

## Project Layout

```
jinyong-finetune/
├── configs/
│   └── qlora_config.yaml
├── docs/
│   ├── autoDL.md              # AutoDL Jupyter / SSH runbook
│   ├── LORA_TO_GGUF_GUIDE.md  # merge → GGUF → Ollama
│   └── INFERENCE_GUIDE.md
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
│   ├── train.py
│   └── merge_lora.py          # full HF merge after training (for GGUF / non-PEFT)
├── requirements.txt
└── .cursor/rules/
```

## Quick Start (Local)

1. Create environment:

   `python3 -m venv .venv && source .venv/bin/activate`

2. Install dependencies:

   `pip install -r requirements.txt`

   For training you also need a **PyTorch + CUDA** stack plus **`transformers`**, **`peft`**, **`trl`**, **`accelerate`**, **`bitsandbytes`**, **`datasets`**. The **`notebooks/02_train.ipynb`** setup cell installs an unpinned set suitable for current **`scripts/train.py`**.

3. **AutoDL-style stack (if you mirror the cloud env):**

   `pip install -U bitsandbytes accelerate peft transformers datasets trl pyyaml`

   Install **torch** for your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/) if it is missing.

4. Put novel text files under **`data/raw/`**.

5. Clean encodings and noise, then build the instruction JSONL (from cleaned **`data/processed/`**):

   `python scripts/clean_text.py`

   Preview counts **without** writing JSONL:

   `python scripts/build_instructions.py --dry-run --stats`

   **Write** the JSONL (default path from YAML, e.g. **`data/instructions/jinyong_sft.jsonl`**):

   `python scripts/build_instructions.py --stats`

   If you skip **`clean_text.py`** and point **`--input-dir`** at raw exports, add **`--apply-clean`**.

   For stronger instruction-following, generate typed pairs then merge:

   `python scripts/generate_typed_pairs.py --output data/instructions/typed_pairs.jsonl --per-template 20`

   Or use alternative Chinese-model APIs via **`.env`** keys (DeepSeek/Kimi/MiniMax/GLM):

   `pip install openai python-dotenv && python scripts/generate_more_types_pairs.py --dry-run`

   `python scripts/build_instructions.py --typed-jsonl data/instructions/typed_pairs.jsonl --stats`

6. Train:

   `python scripts/train.py --config configs/qlora_config.yaml`

7. (Optional) Merge LoRA into full HF weights for GGUF or non-PEFT inference:

   `python scripts/merge_lora.py --config configs/qlora_config.yaml`

## Quick Start (AutoDL)

1. Clone the repo on the instance (e.g. **`/root/autodl-tmp/jinyong-finetune`**).
2. Follow **`docs/autoDL.md`** (environment, **`data/raw/`**, **`clean_text`** → **`build_instructions`** → **`train.py`** → optional **`merge_lora.py`**, zip/download).
3. Use **`notebooks/01_data_prep.ipynb`** / **`02_train.ipynb`** / **`03_inference.ipynb`** from repo root if you prefer Jupyter; they match the same YAML and scripts as the CLI.
4. Training reads **`configs/qlora_config.yaml`** (**bf16**, **`packing: false`**, effective batch 16 on 4090 by default).
5. **LoRA artifacts:** **`outputs/jinyong-qlora/adapter/`** — zip and download before the instance recycles.
6. **Merged HF checkpoint (optional, on GPU):** **`scripts/merge_lora.py`** → default **`outputs/jinyong-merged/`**, then zip. **GGUF / Ollama:** **`docs/LORA_TO_GGUF_GUIDE.md`**.

## Quick Start (Kaggle / Colab, optional)

1. Upload this repo to GitHub.
2. Clone in a GPU notebook; attach or download the **Jinyong Wuxia** dataset: `kaggle datasets download -d evilpsycho42/jinyong-wuxia -p data/raw --unzip`
3. Run **`notebooks/01_data_prep.ipynb`** then **`notebooks/02_train.ipynb`** (or the same shell commands as **`docs/autoDL.md`**).
4. On **T4 (16 GB)** you may need a copied YAML with **`fp16: true`**, **`bf16: false`**, **`bnb_4bit_compute_dtype: float16`**, and smaller **`per_device_train_batch_size`** — see **`.cursor/rules/autodl.mdc`**.

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

- Default **`configs/qlora_config.yaml`** is tuned for **RTX 4090** (**bf16**, **`bnb_4bit_compute_dtype: bfloat16`**).
- **`outputs/`** and raw/processed/instruction datasets are ignored by git.
- Keep text **UTF-8** encoded.
