Good — let me inspect the actual file contents before writing the plan.Good. I can see the full picture now. Here's my analysis and what needs to happen next.

---

## Repo Audit

**What's in place ✅**
- Correct directory scaffold (`scripts/`, `notebooks/`, `data/`, `.cursor/rules/`, `configs/`)
- `requirements.txt` with correct pinned versions
- `.gitignore`, `.env.example`, `README.md`
- 82.7% Python / 17.3% Notebook split — structure is right

**What's missing / empty ❌**
- `.cursor/rules/*.mdc` — directories exist but files need content
- `scripts/clean_text.py` — stub or empty
- `scripts/build_instructions.py` — stub or empty
- `scripts/train.py` — stub or empty
- `configs/qlora_config.yaml` — empty
- `notebooks/01_data_prep.ipynb`, `02_train.ipynb`, `03_inference.ipynb` — empty
- `data/raw/` — no novel files yet (expected, they're gitignored)

**The gap:** Skeleton exists, nothing is runnable yet. All the real work is ahead.

---

# Next Steps Plan — Ordered by Dependency

## Step 1 — Fill `.cursor/rules/` (30 min, do this first)
This shapes every AI interaction in Cursor going forward. Do it before writing any code.

In Cursor, create these 3 files — paste the content from our previous conversation:

```
.cursor/rules/ml-python.mdc
.cursor/rules/huggingface.mdc
.cursor/rules/jinyong-project.mdc
```

Then create your 3 Notepads: `runs`, `eval`, `errors`.

Commit:
```bash
git add .cursor/rules/
git commit -m "feat: add cursor rules for ML/HF/project context"
git push
```

---

## Step 2 — `configs/qlora_config.yaml` (15 min)

This is the single source of truth for all training hyperparameters. Both `train.py` and `02_train.ipynb` will read from it — no magic numbers scattered in code.

```yaml
# configs/qlora_config.yaml

model:
  id: "Qwen/Qwen2.5-7B-Instruct"
  trust_remote_code: true

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "float16"   # T4 — no bfloat16
  bnb_4bit_use_double_quant: true

lora:
  r: 64
  lora_alpha: 128
  lora_dropout: 0.05
  bias: "none"
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

training:
  output_dir: "outputs/jinyong-qlora"
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8       # effective batch = 16
  learning_rate: 2.0e-4
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.05
  num_train_epochs: 2
  max_seq_length: 1024
  fp16: true
  bf16: false
  packing: true
  save_strategy: "steps"
  save_steps: 100
  save_total_limit: 3
  logging_steps: 10
  report_to: "none"

data:
  train_file: "data/instructions/jinyong_sft.jsonl"
  test_split: 0.05
  seed: 42

system_prompt: "你是一位精通金庸武侠风格的写作助手。"
```

Commit: `feat: add qlora training config`

---

## Step 3 — `scripts/clean_text.py` (1–2 hrs)

This is **local work in Cursor**. Use Agent mode (`Cmd+Shift+I`) and say:

> "Implement `scripts/clean_text.py` using `@file:configs/qlora_config.yaml` and `@notepad:jinyong-project`. The script should: accept a directory of .txt files, detect and normalize encoding (GB2312/UTF-8), strip chapter headers matching Chinese patterns, remove HTML artifacts, normalize punctuation, collapse whitespace, and write cleaned output to `data/processed/`. Add a `--dry-run` flag that prints stats without writing."

The full implementation should look like:

```python
# scripts/clean_text.py
import re, os, glob, argparse
from pathlib import Path

CHAPTER_PATTERNS = [
    r'第[零一二三四五六七八九十百千]+[回章节卷].*?\n',
    r'^\s*[（(]\s*[一二三四五六七八九十]+\s*[)）].*?\n',
]

def detect_encoding(filepath: str) -> str:
    """Try UTF-8 first, fall back to GB2312/GBK."""
    for enc in ['utf-8', 'gb2312', 'gbk', 'big5']:
        try:
            with open(filepath, encoding=enc) as f:
                f.read()
            return enc
        except UnicodeDecodeError:
            continue
    return 'utf-8'  # last resort

def clean_novel(text: str) -> str:
    for pattern in CHAPTER_PATTERNS:
        text = re.sub(pattern, '\n', text, flags=re.MULTILINE)
    text = re.sub(r'<[^>]+>', '', text)          # HTML tags
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = text.replace('　', '')                 # fullwidth space
    text = re.sub(r'\n{3,}', '\n\n', text)       # excess blank lines
    return text.strip()

def process_file(src: str, dst_dir: str, dry_run: bool = False) -> dict:
    enc = detect_encoding(src)
    with open(src, encoding=enc, errors='replace') as f:
        raw = f.read()
    cleaned = clean_novel(raw)
    stats = {
        'file': os.path.basename(src),
        'encoding': enc,
        'raw_chars': len(raw),
        'clean_chars': len(cleaned),
        'reduction_pct': round((1 - len(cleaned)/len(raw)) * 100, 1)
    }
    if not dry_run:
        Path(dst_dir).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(dst_dir, os.path.basename(src))
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)
    return stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='data/raw')
    parser.add_argument('--dst', default='data/processed')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    files = glob.glob(f"{args.src}/*.txt")
    if not files:
        print(f"No .txt files found in {args.src}")
        exit(1)

    total_raw = total_clean = 0
    for f in sorted(files):
        stats = process_file(f, args.dst, args.dry_run)
        total_raw += stats['raw_chars']
        total_clean += stats['clean_chars']
        tag = "[DRY RUN] " if args.dry_run else ""
        print(f"{tag}{stats['file']:30s} {stats['encoding']:8s} "
              f"{stats['raw_chars']:>10,} → {stats['clean_chars']:>10,} "
              f"chars (-{stats['reduction_pct']}%)")

    print(f"\nTotal: {total_raw:,} → {total_clean:,} chars")
```

Commit: `feat: implement clean_text.py with encoding detection`

---

## Step 4 — Download Raw Data (20 min, parallel with Step 3)

```bash
# Install Kaggle CLI locally
pip install kaggle

# Put your kaggle.json in ~/.kaggle/ (download from Kaggle → Account → API)
chmod 600 ~/.kaggle/kaggle.json

# Download the dataset
kaggle datasets download -d evilpsycho42/jinyong-wuxia -p data/raw --unzip
```

Run your first validation immediately:
```bash
python scripts/clean_text.py --dry-run
```

Expected output: 15 files, ~4M total chars, 5–15% reduction each.

---

## Step 5 — `scripts/build_instructions.py` (2–3 hrs, the most important step)

This builds your training dataset. Use Cursor Agent:

> "Implement `scripts/build_instructions.py`. Read cleaned .txt files from `data/processed/`. Generate two types of pairs: (A) sliding window continuation pairs with configurable chunk_size=300 and overlap=100, (B) typed scene pairs using a template list. Write to `data/instructions/jinyong_sft.jsonl` as UTF-8 JSONL. Add `--max-pairs`, `--dry-run`, and `--stats` flags."

The typed templates to include (Strategy B from Phase 1):

```python
TYPED_TEMPLATES = [
    "以金庸武侠风格，描写一场高手之间的内力比拼",
    "以金庸风格写一段江湖儿女的离别场景，情感含蓄",
    "描写一位武功高强但性格孤傲的侠客初入客栈的场景",
    "用金庸笔法写出两个门派之间因误会而起的冲突",
    "以金庸笔法描写一位高手施展轻功的场景",
    "写一段金庸风格的武学秘籍传授场景，师父语气庄重",
    "描写一场以少胜多的江湖打斗，主角以智取胜",
    "以金庸风格写一段两位旧识重逢却各怀心事的对话",
    "描写一个初出茅庐的少年第一次见识真正高手的震撼",
    "以金庸笔法写出一位反派的出场，气势逼人却不失深度",
    "用金庸风格描写江湖门派的拜师仪式",
    "写一段武功秘籍的文字描述，风格古朴，暗含哲理",
    "以金庸风格描写两位武林高手以棋局论道的场景",
    "写一段江湖恩怨中的临终托付场景，情真意切",
    "以金庸风格描写一场追逐战，穿越山林水泽",
    "描写一位隐居高人被迫出山的内心挣扎",
    "以金庸笔法写出一段武功心法的顿悟场景",
    "描写江湖中一次重大武林大会的开场",
    "写一段金庸风格的毒功与解毒的对决",
    "以金庸风格描写一位侠客独自面对绝境的内心独白",
]
```

After running:
```bash
python scripts/build_instructions.py --stats
```

Expected output:
```
Continuation pairs:  3,142
Typed scene pairs:     412
Total pairs:         3,554
Train / Val split:   3,376 / 178
Saved → data/instructions/jinyong_sft.jsonl
```

Commit: `feat: implement build_instructions.py with 20 typed templates`

---

## Step 6 — `notebooks/01_data_prep.ipynb` (30 min)

This is a **Kaggle-runnable** version of Steps 3–5 combined. Its job is to run the pipeline end-to-end on Kaggle where you can't run local scripts.

Structure:
```
Cell 1: !pip install (nothing extra needed — all stdlib)
Cell 2: !git clone https://github.com/jxjwilliam/jinyong-finetune.git
Cell 3: Mount/attach Kaggle dataset (raw novel files)
Cell 4: !python scripts/clean_text.py
Cell 5: !python scripts/build_instructions.py --stats
Cell 6: Inspect 5 random samples from the JSONL
Cell 7: Save jinyong_sft.jsonl as a Kaggle Dataset output
```

The last step is critical — save it as a persistent Kaggle Dataset so `02_train.ipynb` can attach it as input without regenerating.

---

## Step 7 — `notebooks/02_train.ipynb` (1 hr to write, 3 hrs to run)

This is the main Kaggle training notebook. Structure:

```
Cell 1:  !pip install -q transformers==4.44.0 peft==0.12.0 trl==0.10.1 ...
Cell 2:  !git clone ... && %cd jinyong-finetune
Cell 3:  import + GPU check (assert T4 or better)
Cell 4:  Load config from configs/qlora_config.yaml
Cell 5:  Load + quantize model (BitsAndBytes 4-bit)
Cell 6:  Apply LoRA via peft
Cell 7:  Load + format dataset
Cell 8:  SFTTrainer setup
Cell 9:  trainer.train() ← the long cell
Cell 10: Save adapter to /kaggle/working/outputs/
Cell 11: Quick inference sanity check (3 prompts)
```

Key cell to add before training — GPU memory check:
```python
import torch
assert torch.cuda.is_available(), "No GPU! Enable GPU in Kaggle Settings"
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
# Expected: Tesla T4, 15.8 GB
```

---

## Step 8 — `notebooks/03_inference.ipynb` (45 min)

Loads the saved adapter and runs your 20 typed prompts systematically for human eval.

```python
EVAL_PROMPTS = [
    ("以金庸武侠风格，写一段两位高手在客栈初次相遇的场景", ""),
    ("以金庸风格写一段江湖儿女的离别场景", "主角：令狐冲，情人：任盈盈"),
    # ... one per template type
]

results = []
for instruction, context in EVAL_PROMPTS:
    output = generate(instruction, context)
    results.append({"instruction": instruction, "output": output, "score": None})

# Save for scoring
import json
with open("eval_results.jsonl", "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
```

Then score each output using the 5-dimension rubric from Phase 5 of the original plan.

---

## Full Sequence & Time Estimates

```
TODAY (local in Cursor)
├── Step 1: Fill .cursor/rules/         30 min  ← do this NOW
├── Step 2: configs/qlora_config.yaml   15 min
├── Step 3: scripts/clean_text.py        2 hrs
├── Step 4: Download raw data           20 min  (parallel)
└── Step 5: build_instructions.py        2 hrs
                                        ──────
                                        ~5 hrs total local

NEXT SESSION (write notebooks locally, run on Kaggle)
├── Step 6: 01_data_prep.ipynb          30 min write / 20 min run
├── Step 7: 02_train.ipynb               1 hr write / 3 hrs run
└── Step 8: 03_inference.ipynb          45 min write / 30 min run
                                        ──────
                                        ~6 hrs total

ITERATION (after first eval results)
└── Improve data → retrain → re-eval   2–3 hrs/cycle
```

---

## Right Now — The Single Next Action

Open Cursor, go to Agent mode (`Cmd+Shift+I`), and say:

> "Create `.cursor/rules/ml-python.mdc`, `.cursor/rules/huggingface.mdc`, and `.cursor/rules/jinyong-project.mdc` with the content from our conversation. Then create `configs/qlora_config.yaml`. Then scaffold `scripts/clean_text.py` as a complete, runnable implementation."

That one Agent prompt gets you Steps 1–3 in one shot. Once `clean_text.py` runs clean against your raw data, everything else has solid ground to build on.