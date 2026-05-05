# Scripts layout

Run CLIs from the **repository root** so paths like `configs/qlora_config.yaml` resolve.

| Folder | Role |
|--------|------|
| **`data/`** | Raw/processed text and instruction JSONL (`clean_text.py`, `build_instructions.py`). |
| **`gen/`** | LLM APIs → typed JSONL (`generate_typed_pairs.py`). |
| **`lib/`** | Shared modules (`instruction_jsonl.py`, `typed_prompts.py`) — import only, not standalone CLIs. |
| **`train/`** | QLoRA SFT and LoRA merge (`train.py`, `merge_lora.py`). |
| **`infer/`** | Local transformers inference (`inference.py`). |
| **`export/`** | GGUF / Ollama (`convert_to_gguf.py`, `gguf.sh`, `ollama.sh`). |
| **`hub/`** | Hugging Face Hub uploads (`upload_adapter_hf.py`, `upload_raw_corpus_hf.py`). |
| **`shell/`** | Multi-step bash orchestration (`type-pair.sh`). |
