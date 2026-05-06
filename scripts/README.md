# Scripts layout

Run CLIs from the **repository root** so paths like `configs/qlora_config.yaml` resolve.

| Folder | Role |
| --- | --- |
| **`data/`** | Raw/processed text and SFT JSONL build (`clean_text.py`, `build_instructions.py`, `dedup_pairs.py`). |
| **`gen/`** | LLM APIs -> typed JSONL (`generate_typed_pairs.py`). |
| **`eval/`** | Rubric evaluation + GPT judge + benchmark prompts (`eval_rubric.py`, `judge_gpt4o.py`, `prompts_v2_typed20.jsonl`). |
| **`dpo/`** | Preference dataset construction (`build_preference_pairs.py`). |
| **`train/`** | QLoRA SFT, DPO, and LoRA merge (`train.py`, `train_dpo.py`, `merge_lora.py`). |
| **`infer/`** | Local transformers inference + template library (`inference.py`, `prompt_library.py`). |
| **`server/`** | Local API service and streaming SSE wrapper for Ollama (`stream_api.py`). |
| **`lib/`** | Shared modules (`instruction_jsonl.py`, `typed_prompts.py`) — import only, not standalone CLIs. |
| **`export/`** | GGUF / Ollama (`convert_to_gguf.py`, `gguf.sh`, `ollama.sh`). |
| **`hub/`** | Hugging Face Hub uploads (`upload_adapter_hf.py`, `upload_raw_corpus_hf.py`). |
| **`shell/`** | Multi-step bash orchestration (`type-pair.sh`). |

Common local flow:

1. `python scripts/data/build_instructions.py --dedup-continuation --stats`
2. `python scripts/train/train.py --config configs/qlora_config.yaml`
3. `python scripts/eval/eval_rubric.py --config configs/qlora_config.yaml --run-id sft_latest`
4. (Optional) `python scripts/dpo/build_preference_pairs.py --config configs/qlora_config.yaml --max-prompts 20`
5. (Optional) `python scripts/train/train_dpo.py --config configs/qlora_config.yaml`
6. (Optional) `python scripts/server/stream_api.py`
