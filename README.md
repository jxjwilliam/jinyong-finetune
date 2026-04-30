# Jin Yong Fine-Tune (Kaggle-First)

This repository scaffolds a Kaggle-first workflow to fine-tune `Qwen/Qwen2.5-7B-Instruct` with QLoRA for Jin Yong style wuxia generation.

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
│   └── train.py
├── requirements.txt
└── .cursor/rules/
```

## Quick Start (Local)

1. Create environment:

   `python3 -m venv .venv && source .venv/bin/activate`

2. Install dependencies:

   `pip install -r requirements.txt`

3. Put novel text files under `data/raw/`.

4. Clean encodings and noise, then build the instruction JSONL (from cleaned `data/processed/`):

   `python scripts/clean_text.py`

   `python scripts/build_instructions.py --stats`

   To preview counts without writing JSONL: `python scripts/build_instructions.py --dry-run --stats`.

   If you skip `clean_text.py` and point `--input-dir` at raw exports, add `--apply-clean`.

## Quick Start (Kaggle)

1. Upload this repo to GitHub.
2. Clone in Kaggle notebook.
3. Download raw novels (optional locally): `kaggle datasets download -d evilpsycho42/jinyong-wuxia -p data/raw --unzip`
4. Run `notebooks/01_data_prep.ipynb` to clean text and build JSONL, or attach a dataset that already contains `jinyong_sft.jsonl`.
5. Run `notebooks/02_train.ipynb` cells.

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

- Default settings are tuned for Kaggle T4 16GB.
- `outputs/` and raw/processed datasets are ignored by git.
- Keep text UTF-8 encoded.

