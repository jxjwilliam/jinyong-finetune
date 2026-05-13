# Jin Yong Fine-Tune (AutoDL + QLoRA)

Fine-tune **`Qwen/Qwen2.5-7B-Instruct`** with **QLoRA** (4-bit NF4) for Jin Yong–style Chinese wuxia generation. The default training profile targets **[AutoDL](https://www.autodl.com)** with **NVIDIA RTX 4090 (24 GB VRAM)**; **Kaggle / Colab** remain optional with a smaller-GPU config (see **`.cursor/rules/autodl.mdc`**).

**AutoDL end-to-end steps:** **`docs/v1/autoDL.md`**  
**Typed scene generation + merge:** **`docs/v1/TYPED_PAIRS_PIPELINE.md`** (architecture diagrams + CLI)

## Project Layout

```text
jinyong-finetune/
├── configs/
│   └── qlora_config.yaml
├── docs/
│   ├── autoDL.md                 # AutoDL Jupyter / SSH runbook
│   ├── TYPED_PAIRS_PIPELINE.md   # typed scenes: buckets, diagrams, CLI
│   ├── LORA_TO_GGUF_GUIDE.md     # merge → GGUF → Ollama
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
│   ├── README.md                 # script index by category
│   ├── data/                     # raw → processed → instruction JSONL
│   ├── gen/                      # LLM API → typed JSONL
│   ├── lib/                      # shared Pair / prompts (imported, not CLI)
│   ├── train/                    # QLoRA + LoRA merge
│   ├── infer/                    # local transformers inference
│   ├── export/                   # GGUF + Ollama helpers
│   ├── hub/                      # Hugging Face upload helpers
│   └── shell/                    # multi-step shell orchestration
├── requirements.txt
└── .cursor/rules/
```

## Quick Start (Local)

1. Create environment:

   `python3 -m venv .venv && source .venv/bin/activate`

2. Install dependencies:

   `pip install -r requirements.txt`

   For training you also need a **PyTorch + CUDA** stack plus **`transformers`**, **`peft`**, **`trl`**, **`accelerate`**, **`bitsandbytes`**, **`datasets`**. The **`notebooks/02_train.ipynb`** setup cell installs an unpinned set suitable for current **`scripts/train/train.py`**.

3. **AutoDL-style stack (if you mirror the cloud env):**

   `pip install -U bitsandbytes accelerate peft transformers datasets trl pyyaml`

   Install **torch** for your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/) if it is missing.

4. Put novel text files under **`data/raw/`**.

5. Clean encodings and noise, then build the instruction JSONL (from cleaned **`data/processed/`**):

   `python scripts/data/clean_text.py`

   **Input** scripts/shell/type-pair.sh: 5 jsonls = all_typed.jsonl
   **Write** the JSONL (default path from YAML, e.g. **`data/instructions/jinyong_sft.jsonl`**):

   `python scripts/data/build_instructions.py --typed-jsonl  data/instructions/all_typed.jsonl --stats --dry-run`

   `python scripts/data/build_instructions.py --typed-jsonl  data/instructions/all_typed.jsonl --stats`

   `ll data/instructions/jinyong_sft.jsonl`

6. Train:

   `python scripts/train/train.py --config configs/qlora_config.yaml`

7. Merge LoRA into full HF weights for GGUF or non-PEFT inference:

   `python scripts/train/merge_lora.py --config configs/qlora_config.yaml --hf-endpoint https://hf-mirror.com`

## Quick Start (AutoDL)

1. Clone the repo on the instance (e.g. **`/root/autodl-tmp/jinyong-finetune`**).
2. Follow **`docs/v1/autoDL.md`** (environment, **`data/raw/`**, **`scripts/data/clean_text.py`** → **`scripts/data/build_instructions.py`** → **`scripts/train/train.py`** → optional **`scripts/train/merge_lora.py`**, zip/download).
3. Use **`notebooks/01_data_prep.ipynb`** / **`02_train.ipynb`** / **`03_inference.ipynb`** from repo root if you prefer Jupyter; they match the same YAML and scripts as the CLI.
4. Training reads **`configs/qlora_config.yaml`** (**bf16**, **`packing: false`**, effective batch 16 on 4090 by default).
5. **LoRA artifacts:** **`outputs/jinyong-qlora/adapter/`** — zip and download before the instance recycles.
6. **Merged HF checkpoint (optional, on GPU):** **`scripts/train/merge_lora.py`** → default **`outputs/jinyong-merged/`**, then zip. **GGUF / Ollama:** **`docs/v1/LORA_TO_GGUF_GUIDE.md`**.

## Step-by-Step Workflow

Use this end-to-end sequence if you want a single checklist from raw text to deployable model:

1. Prepare environment: create/activate `.venv`, install `requirements.txt`, and ensure a CUDA-compatible PyTorch build is installed.
2. Place source novels under **`data/raw/`**.
3. Normalize text with **`python scripts/data/clean_text.py`**.
4. Build continuation-style SFT data with **`python scripts/data/build_instructions.py --stats`**.
   For better quality, enable near-duplicate removal on continuation pairs:
   **`python scripts/data/build_instructions.py --dedup-continuation --dedup-threshold 0.85 --stats`**.
5. (Optional) Generate typed scene pairs with **`scripts/gen/generate_typed_pairs.py`** and merge via one or more `--typed-jsonl` flags in **`scripts/data/build_instructions.py`**.
6. Train LoRA adapter with **`python scripts/train/train.py --config configs/qlora_config.yaml`**.
7. Validate generation using **`notebooks/03_inference.ipynb`** or your local inference script.
   You can also use built-in prompt templates:
   - List templates: **`python scripts/infer/inference.py --list-templates`**
   - Render one template: **`python scripts/infer/inference.py --template-id battle_duel_01 --template-slots-json '{"fighter_a":"郭靖","fighter_b":"欧阳锋","weapon_a":"降龙掌","weapon_b":"蛇杖","location":"华山绝顶"}'`**
8. Run automated quality evaluation (20 typed prompts + GPT judge):
   **`python scripts/eval/eval_rubric.py --config configs/qlora_config.yaml --run-id <run_id>`**
9. (Optional) Build DPO preference pairs and run DPO:
   - **`python scripts/dpo/build_preference_pairs.py --config configs/qlora_config.yaml --max-prompts 20`**
   - **`python scripts/train/train_dpo.py --config configs/qlora_config.yaml`**
   - **`python scripts/eval/eval_rubric.py --config configs/qlora_config.yaml --run-id dpo_<run_id>`**
10. (Optional) Merge LoRA into full HF weights using **`python scripts/train/merge_lora.py --config configs/qlora_config.yaml`**.
11. (Optional) Export to GGUF/Ollama using **`docs/v1/LORA_TO_GGUF_GUIDE.md`**.
12. Archive/download artifacts from **`outputs/`** before ephemeral cloud instances are recycled.

## SFT -> DPO Workflow

Recommended staged quality loop:

1. Run SFT and keep a baseline eval result:
   - `python scripts/train/train.py --config configs/qlora_config.yaml`
   - `python scripts/eval/eval_rubric.py --config configs/qlora_config.yaml --run-id sft_baseline`
2. Build DPO preferences from sampled pairs:
   - `python scripts/dpo/build_preference_pairs.py --config configs/qlora_config.yaml --max-prompts 20`
3. Train DPO adapter:
   - `python scripts/train/train_dpo.py --config configs/qlora_config.yaml`
4. Re-run eval and compare with SFT baseline:
   - `python scripts/eval/eval_rubric.py --config configs/qlora_config.yaml --run-id dpo_run_001`

The run should be treated as successful only when DPO scores do not regress on the 5-dim rubric.

## Streaming API Usage

Start server:

`python scripts/server/stream_api.py`

### Browser streaming fetch example

```javascript
const response = await fetch("http://127.0.0.1:8000/v1/generate/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    prompt: "以金庸风格写一段华山夜战",
    max_tokens: 256,
    temperature: 0.7
  })
});
const reader = response.body.getReader();
const decoder = new TextDecoder("utf-8");
while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  console.log(decoder.decode(value, { stream: true }));
}
```

### cURL stream test

```bash
curl -N -X POST http://127.0.0.1:8000/v1/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt":"以金庸风格写一段华山夜战","max_tokens":256,"temperature":0.7}'
```

## Quick Start (Kaggle / Colab, optional)

1. Upload this repo to GitHub.
2. Clone in a GPU notebook; attach or download the **Jinyong Wuxia** dataset: `kaggle datasets download -d evilpsycho42/jinyong-wuxia -p data/raw --unzip`
3. Run **`notebooks/01_data_prep.ipynb`** then **`notebooks/02_train.ipynb`** (or the same shell commands as **`docs/v1/autoDL.md`**).
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
- My two Hugging Face contributions are available at **`jxjwilliam/jinyong-wuxia`** (dataset) and **`jxjwilliam/jinyong-qwen2.5-7b-qlora`** (LoRA adapter): [Dataset](https://huggingface.co/datasets/jxjwilliam/jinyong-wuxia), [Model](https://huggingface.co/jxjwilliam/jinyong-qwen2.5-7b-qlora).
- Eval outputs are stored in **`outputs/eval/`** (`eval_results.jsonl` and per-run `summary.json`).
- Dedup reports are stored in **`outputs/data/dedup_report.json`** when `--dedup-continuation` is enabled.
- SSE API server is available at **`scripts/server/stream_api.py`**:
  - Start: **`python scripts/server/stream_api.py`**
  - Stream endpoint: **`POST /v1/generate/stream`**
