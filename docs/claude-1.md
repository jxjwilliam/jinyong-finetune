Q: For the Query: I want to fine-tune an LLM to generate novels in the style and tone of Jin Yong (金庸), the renowned Chinese wuxia author. I've found a relevant dataset at [Kaggle: Jinyong Wuxia](https://www.kaggle.com/datasets/evilpsycho42/jinyong-wuxia), but recommendations for additional datasets on Hugging Face or elsewhere would be greatly appreciated. The attached is the answer.

Based on above, write a implementation plan on how to do it? I have the following ide/tui for development options: cursor, vscode+copilot,  opencode.

write a detailed plan for the fine-tune implementation.  ask me questions if have then to start.

A: Great context. Here's the full picture before I write the plan:

- **Free T4 GPU = 16GB VRAM** → Qwen2.5-7B with QLoRA is your target model (14B won't fit)
- **Kaggle > Colab** for this project — 30hr/week GPU quota, persistent datasets, no session timeouts killing long runs
- **Skip Stage 1** (continued pretraining) — requires days of compute; free tier can't do it. Go straight to instruction fine-tuning (Stage 2), which still yields strong stylistic results
- **IDE pick: Cursor** — best for this workflow; you'll be iterating on Python scripts + notebooks and Cursor's inline AI is excellent for debugging HuggingFace/PEFT errors

---

# 🏯 Jin Yong LLM Fine-Tuning — Implementation Plan

## Phase 0 — Project Structure & Tooling

### IDE: Cursor (recommended over VSCode+Copilot and OpenCode for this)
Cursor wins here because:
- Inline `Cmd+K` edits are ideal for iterating on training configs without breaking indentation
- `@codebase` context lets it reason across your data prep + training scripts together
- Better at HuggingFace/PEFT debugging than Copilot (which skews toward web dev patterns)
- OpenCode is great for agentic tasks but overkill for a focused ML project

### Repo Layout
```
jinyong-finetune/
├── data/
│   ├── raw/               # Raw .txt novel files from Kaggle
│   ├── processed/         # Cleaned, chunked JSONL
│   └── instructions/      # Instruction pairs (stage 2 dataset)
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── 02_train.ipynb
│   └── 03_inference.ipynb
├── scripts/
│   ├── clean_text.py
│   ├── build_instructions.py
│   └── train.py
├── configs/
│   └── qlora_config.yaml
├── outputs/               # Saved LoRA adapters
└── README.md
```

Set this up locally in Cursor, push to GitHub, then clone into Kaggle notebooks via `!git clone`.

---

## Phase 1 — Data Pipeline

### Step 1.1 — Download & Audit Raw Data
Download the [Kaggle Jinyong Wuxia dataset](https://www.kaggle.com/datasets/evilpsycho42/jinyong-wuxia). You'll get ~15 `.txt` files, one per novel (射雕英雄传, 神雕侠侣, 笑傲江湖, etc.).

**First audit in a notebook:**
```python
import os, glob

novel_files = glob.glob("data/raw/*.txt")
for f in novel_files:
    size = os.path.getsize(f) / 1024
    with open(f, encoding='utf-8') as fh:
        chars = len(fh.read())
    print(f"{os.path.basename(f)}: {size:.0f}KB, {chars:,} chars")

# Expected: ~4M total characters across all 15 novels
```

Watch for: mixed encodings (GB2312 vs UTF-8), HTML artifacts, chapter header noise.

### Step 1.2 — Clean the Text (`clean_text.py`)
```python
import re

def clean_novel(text: str) -> str:
    # Remove chapter headers like 第一回、第二章
    text = re.sub(r'第[零一二三四五六七八九十百千]+[回章节].*?\n', '\n', text)
    # Remove HTML tags if any
    text = re.sub(r'<[^>]+>', '', text)
    # Normalize fullwidth punctuation
    text = text.replace('　', '').replace('\r\n', '\n')
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
```

### Step 1.3 — Build Instruction Pairs (`build_instructions.py`)

This is the most impactful step. You need ~2,000–5,000 high-quality instruction pairs. Generate them semi-automatically:

**Strategy A — Sliding window continuation pairs** (easiest, highest volume):
```python
def make_continuation_pairs(text: str, chunk_size=300, overlap=100):
    """Split text into (prompt, continuation) pairs."""
    pairs = []
    chars = list(text)
    i = 0
    while i + chunk_size * 2 < len(chars):
        prompt_text = ''.join(chars[i:i+chunk_size])
        continuation = ''.join(chars[i+chunk_size:i+chunk_size*2])
        pairs.append({
            "instruction": "以金庸武侠小说的风格，续写以下段落：",
            "input": prompt_text,
            "output": continuation
        })
        i += chunk_size - overlap
    return pairs
```

**Strategy B — Typed scene instructions** (higher quality, needs manual templates):

Create ~20 instruction templates, then fill them from novel excerpts:

```python
TEMPLATES = [
    {
        "instruction": "以金庸武侠风格，描写一场高手之间的内力比拼",
        "tags": ["fight", "neili"]
    },
    {
        "instruction": "以金庸风格写一段江湖儿女的离别场景，情感含蓄",
        "tags": ["romance", "farewell"]
    },
    {
        "instruction": "描写一位武功高强但性格孤傲的侠客初入客栈的场景",
        "tags": ["character", "setting"]
    },
    {
        "instruction": "用金庸笔法写出两个门派之间因误会而起的冲突",
        "tags": ["conflict", "jianghu"]
    },
    # Add 15-20 more covering: 轻功, 暗器, 毒功, 医术, 门派规矩, 武学秘籍...
]
```

Then use Claude (or GPT-4) to help you extract matching passages from the novels and pair them with templates. This is the one place where spending an hour with an AI assistant pays dividends.

**Target dataset size:**
```
Continuation pairs:   ~3,000   (auto-generated)
Typed scene pairs:    ~500     (semi-manual)
Total:                ~3,500 pairs → train/val 95/5 split
```

Save as JSONL:
```python
import json

with open("data/instructions/jinyong_sft.jsonl", "w", encoding="utf-8") as f:
    for pair in all_pairs:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")
```

---

## Phase 2 — Environment Setup on Kaggle

### Kaggle Notebook Setup
```python
# Cell 1 — Install dependencies (runs ~3 min)
!pip install -q \
    transformers==4.44.0 \
    peft==0.12.0 \
    trl==0.10.1 \
    accelerate==0.34.0 \
    bitsandbytes==0.43.3 \
    datasets==2.21.0 \
    wandb  # optional but useful for loss curves
```

```python
# Cell 2 — Clone your repo
!git clone https://github.com/YOUR_USERNAME/jinyong-finetune.git
%cd jinyong-finetune
```

Upload `jinyong_sft.jsonl` as a Kaggle Dataset (not inline — Kaggle Datasets persist between sessions, notebook code doesn't).

---

## Phase 3 — Model Selection & QLoRA Config

### Model: `Qwen/Qwen2.5-7B-Instruct`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,   # saves ~0.4GB extra
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```

**Expected VRAM after loading: ~5.5GB** — leaves ~10GB headroom for activations + optimizer states.

### LoRA Config
```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],  # include FFN layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected: ~1.7% of params trainable (~120M / 7B)
```

---

## Phase 4 — Training

### Dataset Formatting
```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="data/instructions/jinyong_sft.jsonl", split="train")
dataset = dataset.train_test_split(test_size=0.05, seed=42)

def format_prompt(example):
    """Qwen2.5 chat template format."""
    return {
        "text": f"<|im_start|>system\n你是一位精通金庸武侠风格的写作助手。<|im_end|>\n"
                f"<|im_start|>user\n{example['instruction']}\n{example['input']}<|im_end|>\n"
                f"<|im_start|>assistant\n{example['output']}<|im_end|>"
    }

dataset = dataset.map(format_prompt)
```

### SFTTrainer Config
```python
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="outputs/jinyong-qlora",
    
    # Batch & gradient — tuned for T4 16GB
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,   # effective batch = 16
    
    # Learning rate
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    
    # Duration — ~2 epochs fits in a Kaggle session
    num_train_epochs=2,
    max_steps=-1,
    
    # Sequence length — 1024 is safe on T4; push to 2048 if no OOM
    max_seq_length=1024,
    
    # Saving
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    
    # Logging
    logging_steps=10,
    report_to="none",   # swap to "wandb" if you set up W&B
    
    # Mixed precision
    bf16=False,   # T4 doesn't support bf16 — use fp16
    fp16=True,
    
    # Packing short sequences for efficiency
    packing=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
)

trainer.train()
```

**Expected training time on T4:**
- 3,500 pairs × 2 epochs ≈ ~2.5–3.5 hours
- Well within the Kaggle session limit

### Save the Adapter
```python
# Save LoRA adapter only (~300MB, not 14GB)
model.save_pretrained("outputs/jinyong-qlora-adapter")
tokenizer.save_pretrained("outputs/jinyong-qlora-adapter")

# Download from Kaggle → Kaggle Output tab → Add to dataset for persistence
```

---

## Phase 5 — Inference & Evaluation

### Load & Generate
```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "outputs/jinyong-qlora-adapter")

def generate(instruction, input_text="", max_new_tokens=300):
    prompt = (
        f"<|im_start|>system\n你是一位精通金庸武侠风格的写作助手。<|im_end|>\n"
        f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.85,
            top_p=0.92,
            repetition_penalty=1.15,   # critical for Chinese — prevents loops
            do_sample=True,
        )
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

# Test
print(generate(
    "以金庸武侠风格，写一段两位高手在客栈初次相遇的场景",
    "主角：一位云游四海的剑客，对手：神秘的黑衣人"
))
```

### Evaluation Checklist (human eval — the only metric that matters)
Score each output 1–5 on:

| Dimension | What to look for |
|---|---|
| **词汇风格** | Period-appropriate vocabulary, no modern slang |
| **武功描写** | Specific, vivid, not generic kung-fu clichés |
| **人物对话** | Terse, meaningful, carries subtext |
| **叙事节奏** | Sentence rhythm matches Jin Yong's measured pacing |
| **江湖气** | Feels like the jianghu world, not a generic fantasy |

Generate 20–30 samples across different instruction types and score them. Anything averaging below 3 means the instruction dataset needs more quality pairs, not more training steps.

---

## Phase 6 — Iteration Loop

```
Run training → Generate samples → Human eval → Identify weak spots
       ↑                                               |
       └───────── Add targeted instruction pairs ──────┘
```

The most common weak spots and fixes:

| Problem | Fix |
|---|---|
| Generic/repetitive fight scenes | Add more typed fight-scene pairs with specific techniques (降龙十八掌, 独孤九剑) |
| Modern vocabulary leaking in | Add a cleaning pass to flag/remove modern words from training data |
| Outputs too short | Increase `max_new_tokens`, check if `packing=True` is truncating outputs |
| Characters sound the same | Add character-voice pairs (令狐冲 vs 张无忌 vs 郭靖 have very different speech) |

---

## Timeline Estimate

| Phase | Time | Where |
|---|---|---|
| Phase 0–1: Setup + data prep | 4–6 hrs | Local in Cursor |
| Phase 2–3: Environment + model load | 1 hr | Kaggle |
| Phase 4: First training run | 3–4 hrs | Kaggle (leave running) |
| Phase 5: Inference + eval | 1–2 hrs | Kaggle |
| Phase 6: One iteration cycle | 2–3 hrs | Both |
| **Total to first good results** | **~12–16 hrs** | |

---

The biggest risk is Phase 1 — data quality. Spend more time there than feels necessary. A smaller set of well-crafted instruction pairs beats a large noisy one every time. Want me to help generate a starter set of the typed instruction templates (the 20 scene types for Strategy B)?