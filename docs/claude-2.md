Q:
in cursor, what tools/plugins/skills you recommend for this development? such as openspec/bmad-method/spec-kit, cursorrules, etc?

A: Good question, and worth being opinionated here — a lot of these tools are designed for software product development, not ML research. I'll tell you what actually fits this project vs what's hype for your use case.

---

## Honest Assessment First

| Tool | Fit for this project | Verdict |
|---|---|---|
| **Cursor Rules** | High — shapes every AI interaction | ✅ Essential |
| **BMAD Method** | Low — designed for app/product PRDs | ❌ Skip |
| **OpenSpec / Spec-Kit** | Low — API/product spec tooling | ❌ Skip |
| **MCP Servers** | Medium — 1-2 specific ones useful | ⚠️ Selective |
| **Cursor Notepads** | High — perfect for experiment tracking | ✅ Essential |
| **`@docs` indexing** | High — HuggingFace docs inline | ✅ Essential |

BMAD and OpenSpec shine for multi-agent product builds with PRDs and user stories. Your project is an experiment pipeline — what it needs is *precision AI context*, not project management scaffolding.

---

## 1. Cursor Rules — Most Important Setup

Create `.cursor/rules/` directory (new format, preferred over legacy `.cursorrules`):

```
.cursor/
└── rules/
    ├── ml-python.mdc        # General ML Python conventions
    ├── huggingface.mdc      # HF/PEFT/TRL patterns
    └── jinyong-project.mdc  # Project-specific context
```

**`ml-python.mdc`**
```markdown
---
description: ML Python conventions for this project
globs: ["**/*.py", "**/*.ipynb"]
alwaysApply: true
---

# ML Python Rules

- Always use explicit dtype (torch.float16, torch.bfloat16) — never assume defaults
- Wrap all model.generate() calls in torch.no_grad()
- Use tqdm for any loop over dataset batches
- Add GPU memory checks before large operations:
  `print(torch.cuda.memory_summary())` 
- Prefer f-strings. No % formatting.
- Type-hint all function signatures
- Never use `.cuda()` directly — always `device_map="auto"` or `.to(model.device)`
- Log shapes when debugging: print(f"{tensor.shape=}")
```

**`huggingface.mdc`**
```markdown
---
description: HuggingFace, PEFT, TRL patterns
globs: ["**/*.py", "**/*.ipynb"]
alwaysApply: true
---

# HuggingFace Stack Rules

## Versions pinned in this project
- transformers==4.44.0
- peft==0.12.0  
- trl==0.10.1
- bitsandbytes==0.43.3

## Patterns
- Always call `prepare_model_for_kbit_training()` before applying LoraConfig
- SFTTrainer takes `dataset_text_field` OR `formatting_func`, not both
- tokenizer.padding_side = "right" for causal LM training
- Always set `trust_remote_code=True` for Qwen models
- Use `packing=True` in SFTConfig for short-sequence efficiency
- Save adapters with `model.save_pretrained()` — this saves LoRA only, not full weights

## Common errors to avoid
- OOM: reduce per_device_train_batch_size first, then max_seq_length
- T4 does NOT support bf16 — always fp16=True, bf16=False
- Qwen tokenizer needs explicit pad_token: `tokenizer.pad_token = tokenizer.eos_token`
```

**`jinyong-project.mdc`**
```markdown
---
description: Jin Yong fine-tuning project context
globs: ["**/*"]
alwaysApply: true
---

# Project Context

Fine-tuning Qwen2.5-7B-Instruct with QLoRA on Kaggle T4 (16GB VRAM)
to generate Chinese wuxia fiction in Jin Yong's style (金庸武侠风格).

## Stack
- Base model: Qwen/Qwen2.5-7B-Instruct
- Training: QLoRA (4-bit) via PEFT + TRL SFTTrainer
- Platform: Kaggle free tier (T4 GPU, ~30hr/week)
- Data: ~3,500 instruction pairs in JSONL (instruction/input/output)

## Key constraints
- Max VRAM budget: 16GB (T4)
- Max seq length: 1024 (safe) or 2048 (push)
- Training runs: 2-3hr sessions max before Kaggle timeout

## Data paths
- Raw novels: data/raw/*.txt
- Cleaned JSONL: data/instructions/jinyong_sft.jsonl
- Adapter outputs: outputs/jinyong-qlora-adapter/

## Chinese text handling
- All text is UTF-8
- Use ensure_ascii=False in all json.dumps()
- Preserve fullwidth punctuation: 。，！？「」『』
- repetition_penalty=1.15 is required for Chinese generation to prevent loops
```

---

## 2. `@docs` Indexing — Inline Docs Without Leaving Cursor

In Cursor Settings → Features → Docs, add these:

| Name | URL to index |
|---|---|
| HuggingFace PEFT | `https://huggingface.co/docs/peft` |
| TRL SFTTrainer | `https://huggingface.co/docs/trl` |
| Qwen2.5 docs | `https://qwen.readthedocs.io` |
| BitsAndBytes | `https://huggingface.co/docs/bitsandbytes` |

Then in chat: `@docs PEFT how do I merge LoRA adapter into base model weights` — gets you accurate, version-aware answers instead of hallucinated API calls.

---

## 3. Cursor Notepads — Experiment Tracking

Cursor Notepads (Ctrl+Shift+N) act as persistent context you can `@mention` in any chat. Create these three:

**`@notepad:runs`** — training run log
```markdown
# Training Runs

## Run 001 — 2025-01-15
- Dataset: 3,200 pairs, continuation only
- Config: r=64, lr=2e-4, bs=2, grad_acc=8, epochs=2
- Time: 2h 47min
- Loss: 1.82 → 0.94
- Notes: Fight scenes generic, dialogue good

## Run 002 — planned
- Dataset: +300 typed scene pairs
- Change: add gate_proj/up_proj to LoRA targets
- Hypothesis: broader FFN coverage improves vocabulary
```

**`@notepad:eval`** — sample outputs + scores
```markdown
# Eval Samples

## Prompt: 客栈相遇
Score: 3/5 — correct register, but 降龙十八掌 referenced incorrectly
Output: [paste raw output]

## Prompt: 离别场景
Score: 4/5 — emotional restraint is very Jin Yong
```

**`@notepad:errors`** — error → fix log
```markdown
# Errors & Fixes

## CUDA OOM during training
Fix: reduced max_seq_length 2048→1024, batch 4→2

## SFTTrainer KeyError 'text'
Fix: dataset needs 'text' field, not 'instruction'+'output' raw
Add formatting_func or format_prompt map step first
```

Then in Cursor chat: `@notepad:runs my loss isn't converging, here's my config...` — it has full project history.

---

## 4. MCP Servers — Just Two Worth Adding

**GitHub MCP** — useful for syncing your Kaggle↔local workflow:
```json
// .cursor/mcp.json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "your_token" }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem",
               "/path/to/jinyong-finetune"]
    }
  }
}
```

The filesystem MCP lets Cursor Agent actually read your JSONL files and reason about data quality — e.g. `"look at data/instructions/jinyong_sft.jsonl and tell me if the output field lengths are reasonable for training"`.

Skip: Kaggle MCP (doesn't exist yet), WandB MCP (overkill for personal use).

---

## 5. Workflow Habits in Cursor

**Use Agent mode (`Cmd+Shift+I`) for:**
- Refactoring data prep scripts across multiple files
- "Clean up all the Chinese text processing in scripts/ and add error handling for encoding issues"

**Use Inline (`Cmd+K`) for:**
- Tweaking training configs
- Fixing a specific function
- Adding logging to a training loop

**Use Chat (`Cmd+L`) + `@file` for:**
- Debugging — paste error + `@file:scripts/train.py` — it has full context
- `@file:data/instructions/jinyong_sft.jsonl` + "does this format look correct for SFTTrainer?"

---

## Recommended First Hour in Cursor

```bash
# 1. Init repo
mkdir jinyong-finetune && cd jinyong-finetune
git init

# 2. Create the rules directory
mkdir -p .cursor/rules

# 3. Create the three .mdc files above
# (paste the content from above or ask Cursor Agent to scaffold them)

# 4. Open Cursor settings → Docs → add the 4 HuggingFace doc URLs

# 5. Create your first Notepad: @notepad:runs
```

After that setup, Cursor will behave like a collaborator who already knows your model, your constraints, your stack versions, and your Chinese text requirements — rather than giving you generic Python advice on every prompt.

Want me to generate the full contents of all three `.mdc` files as ready-to-paste files?