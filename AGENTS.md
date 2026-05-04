# Repository Guidelines

## Project Structure & Module Organization
This repository is organized around a QLoRA fine-tuning pipeline for Jin Yong–style Chinese text generation.

- `scripts/`: primary CLI entry points for data cleaning, instruction building, training, LoRA merge, and export helpers.
- `configs/`: YAML runtime configuration, especially `configs/qlora_config.yaml`.
- `data/`: working dataset area. Use `data/raw/` for source text, `data/processed/` for cleaned text, and `data/instructions/` for JSONL training data.
- `notebooks/`: Jupyter workflows mirroring the CLI pipeline.
- `docs/`: runbooks for AutoDL, inference, and GGUF/Ollama export.
- `models/` and `outputs/`: model templates and generated artifacts.

## Build, Test, and Development Commands
- `python3 -m venv .venv && source .venv/bin/activate`: create and activate the local environment.
- `pip install -r requirements.txt`: install pinned base dependencies.
- `python scripts/clean_text.py`: normalize raw novel text into `data/processed/`.
- `python scripts/build_instructions.py --stats`: build the JSONL dataset and print sample counts.
- `python scripts/train.py --config configs/qlora_config.yaml`: run QLoRA fine-tuning.
- `python scripts/merge_lora.py --config configs/qlora_config.yaml`: merge adapter weights for full-model export.

## Coding Style & Naming Conventions
Use Python 3.12+ style with 4-space indentation, type hints where practical, and UTF-8 file encoding. Follow the existing script style: small focused functions, `snake_case` for functions/files/variables, and explicit CLI arguments via `argparse`. Keep config keys lowercase with underscores. Shell helpers in `scripts/*.sh` should remain idempotent and portable.

## Testing Guidelines
There is no dedicated automated test suite yet. Validate changes with targeted dry runs:

- `python scripts/build_instructions.py --dry-run --stats`
- `python scripts/generate_typed_pairs.py openai --dry-run`

For training-related changes, confirm the pipeline still reads `configs/qlora_config.yaml` correctly and note any GPU or dependency assumptions in the PR.

## Commit & Pull Request Guidelines
Recent history uses short imperative subjects, often with prefixes like `feat:` and `fix:`. Keep commits focused and descriptive, for example: `fix: preserve prompt formatting in train.py`.

PRs should include:
- a concise summary of behavior changes
- affected paths or commands
- config or environment assumptions
- sample output or screenshots when notebooks, docs, or inference results change

## Security & Configuration Tips
Do not commit secrets from `.env`. Keep large datasets, checkpoints, and generated outputs out of Git unless explicitly required.
