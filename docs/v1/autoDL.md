# AutoDL cloud runbook

This guide focuses on **[AutoDL](https://www.autodl.com)** (Jupyter + SSH) for **QLoRA** fine-tuning of **`Qwen/Qwen2.5-7B-Instruct`** in this repo. Hyperparameters and paths are defined in **`configs/qlora_config.yaml`**; training code is **`scripts/train/train.py`**.

## What you get on the instance

| Stage | Output |
|--------|--------|
| Data prep | `data/processed/*.txt`, then `data/instructions/*.jsonl` (path from YAML `data.instruction_jsonl`) |
| Training | **`outputs/jinyong-qlora/adapter/`** — LoRA weights + tokenizer (not a full merged model) |
| Merge (optional) | **`outputs/jinyong-merged/`** (default) — full-precision HF checkpoint for GGUF export or non-PEFT inference |

Zip adapters or merged folders **before** the container is released; AutoDL storage is ephemeral unless you use persistent volumes.

---

## 1. Create the machine and open Jupyter

1. Choose a template with **NVIDIA RTX 4090 (24 GB)** (or similar Ada GPU with bf16).
2. Start the instance and open **Jupyter** or **SSH**; work from a clone of this repo (typical path: `/root/autodl-tmp/jinyong-finetune`).

---

## 2. Clone the repository

```bash
cd /root/autodl-tmp
git clone <YOUR_REPO_URL> jinyong-finetune
cd jinyong-finetune
```

Use **HTTPS** or **SSH** depending on your Git host. All following commands assume **repository root** as the current directory.

---

## 3. Python environment (CUDA)

AutoDL images usually ship a **CUDA-enabled PyTorch**. Install the training stack (unpinned versions track current `scripts/train/train.py` / TRL APIs):

```bash
pip install -U bitsandbytes accelerate peft transformers datasets trl pyyaml
```

If imports fail, install a **torch** wheel that matches the image’s CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/), then rerun the line above.

**Hugging Face Hub slow or blocked:** before the first model download:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

(or set the same in Jupyter: `import os; os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")`).

---

## 4. Novel corpus under `data/raw/`

- **Upload** `.txt` files via Jupyter, or **`scp`** from your laptop, or **`kaggle datasets download`** on the box (requires `~/.kaggle/kaggle.json`).
- Default raw directory comes from YAML: **`data.raw_txt_dir`** (usually `data/raw`).

---

## 5. Clean text

Writes cleaned novels to **`data.processed_txt_dir`** (usually `data/processed`):

```bash
python scripts/data/clean_text.py --dry-run   # stats only
python scripts/data/clean_text.py             # write processed/*.txt
```

---

## 6. Instruction JSONL (continuations ± typed scenes)

Paths default from **`configs/qlora_config.yaml`** (`data.instruction_jsonl`, `data.processed_txt_dir`).

**Preview counts only (does not write the JSONL):**

```bash
python scripts/data/build_instructions.py --dry-run --stats
```

**Write the JSONL** (merge **`typed_pairs.jsonl`** if that file exists):

```bash
# Optional: API-generated typed scenes (see README)
# python scripts/gen/generate_typed_pairs.py claude --output data/instructions/typed_pairs.jsonl

python scripts/data/build_instructions.py --typed-jsonl data/instructions/typed_pairs.jsonl --stats
# Multiple typed files (merged and shuffled together):
# python scripts/data/build_instructions.py \
#   --typed-jsonl data/instructions/typed_pairs.jsonl \
#   --typed-jsonl data/instructions/more_types_deepseek.jsonl \
#   --stats
# If you have no typed_pairs file, omit --typed-jsonl:
# python scripts/data/build_instructions.py --stats
```

**Note:** `build_instructions.py --stats` **without** `--dry-run` **will write** the output file.

---

## 7. Train (QLoRA)

```bash
python scripts/train/train.py --config configs/qlora_config.yaml
```

- Uses **bf16** + **4-bit NF4** + **LoRA** as in the YAML; **`packing: false`** is required for correct ChatML formatting.
- Gradient checkpointing is enabled in YAML; **`scripts/train/train.py`** calls **`enable_input_require_grads()`** so LoRA gradients flow correctly with checkpointing.

**Artifacts:** `outputs/jinyong-qlora/adapter/` (adapter + tokenizer). Zip for download:

```bash
cd outputs && zip -r jinyong-adapter.zip jinyong-qlora/adapter/
```

---

## 8. Merge LoRA into a full HF model (on AutoDL, before GGUF)

`llama.cpp` conversion needs a **merged** full model, not the adapter alone. On the same GPU host (about one 24 GB card for 7B bf16/fp16):

```bash
python scripts/train/merge_lora.py --config configs/qlora_config.yaml
```

If **`huggingface.co`** is unreachable (**Errno 101**), use a mirror or stay offline (base must already be cached from training):

```bash
export HF_ENDPOINT=https://hf-mirror.com
python scripts/train/merge_lora.py --config configs/qlora_config.yaml
# or inline:
python scripts/train/merge_lora.py --config configs/qlora_config.yaml --hf-endpoint https://hf-mirror.com

# Fully offline — point at the hub snapshot directory (see scripts/train/merge_lora.py --help)
python scripts/train/merge_lora.py --config configs/qlora_config.yaml --local-files-only \
  --base-model-path ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/<hash>
```

Defaults: adapter = **`{training.output_dir}/adapter`**, merged output = **`outputs/jinyong-merged`** when `output_dir` is `outputs/jinyong-qlora`. Override with **`--adapter`** / **`--merged-dir`**. If VRAM is tight: **`--dtype float16`**.

```bash
cd outputs && zip -r jinyong-merged.zip jinyong-merged/
```

Further steps (SCP to Mac, **convert_hf_to_gguf.py**, Ollama): **`docs/v1/LORA_TO_GGUF_GUIDE.md`**.

---

## 9. Jupyter notebooks on AutoDL

| Notebook | Role |
|----------|------|
| **`notebooks/01_data_prep.ipynb`** | Resolve repo root, `clean_text`, preview (`--dry-run --stats`) vs write `build_instructions`, sample JSONL from YAML path |
| **`notebooks/02_train.ipynb`** | GPU check, pip installs, optional Kaggle download, data pipeline, **`scripts/train/train.py`** (streaming logs, no `capture_output` trap) |
| **`notebooks/03_inference.ipynb`** | Load 4-bit base + adapter for smoke tests (CUDA required) |

Use the same **`configs/qlora_config.yaml`** as the CLI. In a cell:

```python
!python scripts/train/merge_lora.py --config configs/qlora_config.yaml
```

---

## 10. Inference and docs elsewhere

- **PEFT / transformers** usage and prompts: **`docs/v1/INFERENCE_GUIDE.md`**
- **GGUF / Ollama** after merge: **`docs/v1/LORA_TO_GGUF_GUIDE.md`**

---

## Checklist (copy-friendly)

1. [ ] Repo cloned under `/root/autodl-tmp/...`, shell at repo root  
2. [ ] `pip install …` bitsandbytes, accelerate, peft, transformers, datasets, trl, pyyaml (+ torch if needed)  
3. [ ] `data/raw/*.txt` present  
4. [ ] `python scripts/data/clean_text.py`  
5. [ ] `python scripts/data/build_instructions.py --dry-run --stats` then write with `… --stats` (and `--typed-jsonl` if you have typed pairs)  
6. [ ] `python scripts/train/train.py --config configs/qlora_config.yaml`  
7. [ ] Zip **`outputs/jinyong-qlora/adapter/`**  
8. [ ] (Optional) `python scripts/train/merge_lora.py` → zip **`outputs/jinyong-merged/`**  
9. [ ] Download zips before stopping the instance  

---

## T4 / smaller GPUs (not the default YAML)

Copy **`configs/qlora_config.yaml`** to a new file and set **`fp16: true`**, **`bf16: false`**, **`bnb_4bit_compute_dtype: float16`**, and reduce **`per_device_train_batch_size`** if needed. See **`.cursor/rules/autodl.mdc`**.
