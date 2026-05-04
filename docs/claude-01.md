Let me review the actual files before answering.Good — all files are in context. Here's my full assessment across your 4 questions, plus a flag on the single most important change for RTX 4090.

---

## Critical Fix First: RTX 4090 ≠ T4 Config

Your `qlora_config.yaml` still has T4-era settings. Change these **before running**:

```yaml
# qlora_config.yaml — RTX 4090 RTX overrides

model:
  bnb_4bit_compute_dtype: "bfloat16"   # was "float16" — 4090 has native bf16

training:
  fp16: false                           # was true
  bf16: true                            # was false
  per_device_train_batch_size: 4        # was 2 — 4090 has headroom
  gradient_accumulation_steps: 4        # was 8 — keep effective batch = 16
```

bf16 on Ampere (4090) is faster, numerically stabler for Chinese LLMs, and avoids the fp16 overflow risk at lr=2e-4.

---

## 1. Script Execution Order on AutoDL

```
Step 1  clean_text.py          data/raw/*.txt → data/processed/
Step 2  generate_typed_pairs.py               → data/instructions/typed_pairs.jsonl
Step 3  build_instructions.py                 → data/instructions/jinyong_sft.jsonl
Step 4  train.py                              → outputs/jinyong-qlora/adapter/
```

Exact commands:

```bash
cd /root/autodl-tmp/jinyong-finetune

# Step 1 — clean novels
python scripts/clean_text.py --src data/raw --dst data/processed --dry-run
python scripts/clean_text.py --src data/raw --dst data/processed

# Step 2 — generate typed pairs via Claude API
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/generate_typed_pairs.py claude \
    --output data/instructions/typed_pairs.jsonl \
    --per-template 20 \
    --dry-run   # verify first, remove flag to actually write

# Step 3 — build combined JSONL
python scripts/build_instructions.py \
    --typed-jsonl data/instructions/typed_pairs.jsonl \
    --stats       # verify counts, then remove --dry-run
    
# Step 4 — train (background, log to file)
nohup python scripts/train.py --config configs/qlora_config.yaml \
    > train.log 2>&1 &

tail -f train.log
```

---

## 2. generate_typed_pairs.py — What It Does & How to Use

**What it does:** Calls Claude API (claude-haiku-4-5) with each of 20 wuxia scene templates × N variation hints → outputs real AI-written paragraphs as `instruction`/`output` pairs. This gives the model genuine instruction-following examples, not just continuation windows.

**Two flags you'll actually use:**

```bash
# Dry run — 1 sample per template, prints output, costs ~$0.01
python scripts/generate_typed_pairs.py claude --dry-run

# Full run — e.g. many samples × templates in claude bucket — tune --per-template
python scripts/generate_typed_pairs.py claude \
    --output data/instructions/typed_pairs.jsonl \
    --per-template 20 \
    --sleep 0.5   # increase if you hit rate limits
```

**One bug to fix** — the model string:

```python
# generate_typed_pairs.py (_generate_claude_one)
# current:
model="claude-haiku-4-5",
# correct current model string:
model="claude-haiku-4-5-20251001",
```

Check with: `python -c "import anthropic; print(anthropic.__version__)"` — if it's ≥ 0.40, use the full versioned string.

---

## 3. What to Watch for to Make the LoRA Chat Work

These are the root causes of the previous failure (model reproducing original text instead of following instructions) and what ensures the fix holds:

**A. packing=false** — Already fixed in your config. Verify it stays false. This was the primary bug.

**B. System prompt consistency** — Must be identical in training and inference. Yours is in `qlora_config.yaml → data.system_prompt`. The Modelfile and inference code must copy it verbatim:

```
你是一位精通金庸武侠风格的写作助手。请根据用户的要求，创作符合金庸武侠小说风格的原创内容。文笔典雅，人物鲜明，江湖气息浓厚。不要复述原著情节，创作全新场景。
```

**C. Eval loss as early warning** — With `eval_steps: 100` and 5% val split, watch for eval loss diverging from train loss after step ~200. If it does, your typed pairs are too few — add more via `--per-template 40`.

**D. Token length sanity check** — `train.py` already prints sample token lengths. Confirm none are truncated — 300-char Chinese chunks should be ~400-650 tokens, well within 1024.

**E. TRL version compatibility** — One latent issue in `train.py`:

```python
# train.py — SFTConfig uses deprecated parameter name in TRL ≥ 0.12
# Change:
evaluation_strategy="steps",
# To:
eval_strategy="steps",
```

Check with `pip show trl | grep Version` and update accordingly.

---

## 4. Code Improvements Worth Making

| Issue | Where | Fix |
|---|---|---|
| bf16/fp16 flip | qlora_config.yaml | Already described above |
| Model string | generate_typed_pairs.py | Use `claude-haiku-4-5-20251001` |
| `evaluation_strategy` → `eval_strategy` | train.py L189 | TRL ≥ 0.12 deprecation |
| `dataset_text_field` deprecated | train.py L195 | Use `formatting_func` in TRL ≥ 0.13 |
| No eval JSONL output | train.py | Save val split to disk for manual inspection after training |
| generate_typed_pairs doesn't deduplicate | generate_typed_pairs.py | Add a set-based dedup before writing JSONL |

The two that matter most for output quality: bf16 switch (training stability) and `eval_strategy` rename (otherwise eval silently doesn't run and you're flying blind on overfitting).