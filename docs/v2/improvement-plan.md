# V2 Quality Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an evaluation-driven, data-balanced, deduplicated, prompt-rich, post-SFT DPO-enhanced, and streamable serving pipeline for Jin Yong style generation.

**Architecture:** Add a post-train evaluation harness as a hard quality gate, rebalance SFT data toward typed instruction pairs, deduplicate continuation data with MinHash LSH, introduce a built-in domain prompt template library, then add a second-stage DPO training pass and a FastAPI SSE server for downstream video/front-end streaming integration.

**Tech Stack:** Python 3.12, TRL/Transformers/PEFT, OpenAI GPT-4o judge API, datasketch MinHash LSH, FastAPI + Uvicorn + SSE, JSONL artifacts.

---

## Scope and Priorities

- **P0 (must-have):** automated eval harness, dataset rebalance (`--per-template`), deduplication.
- **P1 (high impact):** built-in Jin Yong prompt template library, FastAPI streaming server.
- **P2 (quality jump):** DPO pass after SFT with GPT-4o pairwise ranking.

Implementation status sync (2026-05-05, local Mac-first):
- `[x]` completed in repo
- `[ ]` pending runtime verification on local/AutoDL

---

## File Structure Plan

- Create: `scripts/eval/eval_rubric.py`
- Create: `scripts/eval/prompts_v2_typed20.jsonl`
- Create: `scripts/eval/judge_gpt4o.py`
- Create: `scripts/eval/eval_results_schema.md`
- Create: `scripts/data/dedup_pairs.py`
- Create: `configs/prompt_templates_jinyong.yaml`
- Create: `scripts/infer/prompt_library.py`
- Create: `scripts/dpo/build_preference_pairs.py`
- Create: `scripts/train/train_dpo.py`
- Create: `scripts/server/stream_api.py`
- Modify: `scripts/gen/generate_typed_pairs.py`
- Modify: `scripts/data/build_instructions.py`
- Modify: `configs/qlora_config.yaml`
- Modify: `requirements.txt`
- Modify: `docs/TYPED_PAIRS_PIPELINE.md`
- Modify: `README.md`

---

### Task 1: Add Automated 5-Dimension Evaluation Harness (Regression Gate)

**Files:**
- Create: `scripts/eval/eval_rubric.py`
- Create: `scripts/eval/prompts_v2_typed20.jsonl`
- Create: `scripts/eval/judge_gpt4o.py`
- Create: `scripts/eval/eval_results_schema.md`
- Modify: `README.md`

- [x] **Step 1: Add the fixed 20 typed prompts benchmark set**

Create `scripts/eval/prompts_v2_typed20.jsonl` with stable IDs:

```json
{"id":"typed_001","instruction":"...","input":"...","category":"dialogue"}
{"id":"typed_002","instruction":"...","input":"...","category":"battle"}
```

- [x] **Step 2: Implement model generation runner for benchmark prompts**

Add `scripts/eval/eval_rubric.py` to:
- load adapter/base from config,
- run generation on all 20 prompts with deterministic seed,
- write raw generations to `outputs/eval/<run_id>/generations.jsonl`.

```python
def run_eval_prompts(model, tokenizer, prompts: list[dict], seed: int) -> list[dict]:
    # torch.no_grad is mandatory for generation eval runs
    with torch.no_grad():
        ...
```

- [x] **Step 3: Implement GPT-4o rubric judging (5 dimensions)**

In `scripts/eval/judge_gpt4o.py`, score each sample on:
- style fidelity
- instruction following
- coherence
- vivid imagery
- originality

Return integer 1-5 per dimension and overall mean.

- [x] **Step 4: Persist trend-friendly evaluation outputs**

Append one line per prompt to `outputs/eval/eval_results.jsonl`:

```json
{"run_id":"2026-05-05_sft_v2","prompt_id":"typed_001","scores":{"style":4,"follow":5,"coherence":4,"imagery":4,"originality":4},"avg":4.2,"judge_model":"gpt-4o"}
```

Also save run summary to `outputs/eval/<run_id>/summary.json`.

- [x] **Step 5: Add regression check command**

Add a CLI mode:

```bash
python scripts/eval/eval_rubric.py --config configs/qlora_config.yaml --run-id sft_run_2026_05_05 --gate-min-avg 4.0 --gate-max-drop 0.2
```

Gate rule:
- fail if current run average < `gate-min-avg`
- fail if any dimension drops > `gate-max-drop` against previous run

- [x] **Step 6: Document evaluation workflow**

Update `README.md` with a post-train section:

```bash
python scripts/eval/eval_rubric.py --config configs/qlora_config.yaml --run-id <run_id>
```

Expected: generates per-prompt scores + pass/fail gate status.

- [x] **Step 7: Commit**

```bash
git add scripts/eval README.md
git commit -m "feat: add automated GPT-4o rubric evaluation and regression gating"
```

---

### Task 2: Rebalance Dataset by Increasing Typed Pair Coverage (`--per-template`)

**Files:**
- Modify: `scripts/gen/generate_typed_pairs.py`
- Modify: `scripts/data/build_instructions.py`
- Modify: `configs/qlora_config.yaml`
- Modify: `docs/TYPED_PAIRS_PIPELINE.md`

- [x] **Step 1: Add config-driven default for typed sample volume**

In `configs/qlora_config.yaml`, add:

```yaml
data:
  typed_pairs:
    per_template: 50
    min_ratio_vs_continuation: 0.30
```

- [x] **Step 2: Wire `generate_typed_pairs.py` to config fallback**

If CLI `--per-template` absent, read from YAML:

```python
per_template = args.per_template if args.per_template is not None else cfg.data.typed_pairs.per_template
```

- [x] **Step 3: Add ratio warning in dataset build stage**

In `scripts/data/build_instructions.py`, print and gate ratio:

```python
typed_ratio = typed_count / max(total_count, 1)
if typed_ratio < min_ratio:
    raise ValueError(f"typed ratio too low: {typed_ratio:.3f} < {min_ratio:.3f}")
```

- [x] **Step 4: Expose ratio stats in `--stats` output**

Include:
- continuation count
- typed count
- typed/total ratio
- recommendation message when below threshold

- [x] **Step 5: Update pipeline docs with target values**

In `docs/TYPED_PAIRS_PIPELINE.md`, add target:
- `--per-template >= 50`
- typed ratio target `>= 30%`

- [ ] **Step 6: Verification run**

```bash
python scripts/gen/generate_typed_pairs.py claude --per-template 50 --dry-run
python scripts/data/build_instructions.py --stats --dry-run
```

Expected: typed ratio meets threshold in stats output.

- [x] **Step 7: Commit**

```bash
git add scripts/gen/generate_typed_pairs.py scripts/data/build_instructions.py configs/qlora_config.yaml docs/TYPED_PAIRS_PIPELINE.md
git commit -m "feat: rebalance typed pair proportion with config-driven per-template defaults"
```

---

### Task 3: Deduplicate Continuation Pairs with MinHash LSH

**Files:**
- Create: `scripts/data/dedup_pairs.py`
- Modify: `scripts/data/build_instructions.py`
- Modify: `requirements.txt`
- Modify: `README.md`

- [x] **Step 1: Add dependency**

Add to `requirements.txt`:

```txt
datasketch
```

- [x] **Step 2: Implement near-duplicate detector**

In `scripts/data/dedup_pairs.py`, MinHash each continuation sample (instruction+input+output text signature), then query LSH to keep first unique representative.

```python
from datasketch import MinHash, MinHashLSH

def text_minhash(text: str, num_perm: int = 128) -> MinHash:
    ...
```

- [x] **Step 3: Add dedup CLI options in build script**

In `scripts/data/build_instructions.py`:
- `--dedup-continuation`
- `--dedup-threshold 0.85`

When enabled, dedup only continuation pairs before merge with typed pairs.

- [x] **Step 4: Emit dedup report**

Write `outputs/data/dedup_report.json`:

```json
{"before":3142,"after":2468,"removed":674,"removed_ratio":0.214}
```

- [ ] **Step 5: Verification run**

```bash
python scripts/data/build_instructions.py --dedup-continuation --dedup-threshold 0.85 --stats
```

Expected: 15-25% continuation reduction (dataset dependent), no schema break.

- [x] **Step 6: Update docs**

Add section in `README.md` for dedup flag and expected quality impact.

- [x] **Step 7: Commit**

```bash
git add scripts/data/dedup_pairs.py scripts/data/build_instructions.py requirements.txt README.md
git commit -m "feat: add MinHash LSH deduplication for continuation training pairs"
```

---

### Task 4: Add Built-in Professional Jin Yong Prompt Template Library

**Files:**
- Create: `configs/prompt_templates_jinyong.yaml`
- Create: `scripts/infer/prompt_library.py`
- Modify: `scripts/infer/inference.py`
- Modify: `README.md`

- [x] **Step 1: Create categorized template config**

`configs/prompt_templates_jinyong.yaml` with categories:
- 人物对白
- 江湖旁白
- 章节开篇
- 打斗场面
- 诗词偈语

Each template includes:
- `id`
- `category`
- `instruction_template`
- `input_template`
- `usage_notes`

- [x] **Step 2: Implement template loader and renderer**

In `scripts/infer/prompt_library.py`:

```python
def load_prompt_templates(path: str) -> dict[str, list[dict]]:
    ...

def render_prompt(template_id: str, slots: dict[str, str]) -> tuple[str, str]:
    ...
```

- [x] **Step 3: Add inference CLI options**

In `scripts/infer/inference.py`:
- `--template-id`
- `--template-slots-json`
- `--list-templates`

If template mode used, auto-build instruction/input from template.

- [x] **Step 4: Add fallback behavior**

When template fields missing, fail with clear error listing required slots.

- [ ] **Step 5: Verification run**

```bash
python scripts/infer/inference.py --list-templates
python scripts/infer/inference.py --template-id battle_opening_01 --template-slots-json '{"hero":"郭靖","villain":"欧阳锋","location":"华山绝顶"}'
```

Expected: prompt renders correctly and generation succeeds.

- [x] **Step 6: Document template usage**

Add quickstart examples to `README.md`.

- [x] **Step 7: Commit**

```bash
git add configs/prompt_templates_jinyong.yaml scripts/infer/prompt_library.py scripts/infer/inference.py README.md
git commit -m "feat: add categorized Jin Yong prompt template library for inference"
```

---

### Task 5: Add DPO Pass After SFT (Preferred vs Rejected Pairs)

**Files:**
- Create: `scripts/dpo/build_preference_pairs.py`
- Create: `scripts/train/train_dpo.py`
- Modify: `configs/qlora_config.yaml`
- Modify: `README.md`

- [x] **Step 1: Add DPO config block**

In `configs/qlora_config.yaml`:

```yaml
dpo:
  enabled: false
  beta: 0.1
  samples_per_prompt: 2
  judge_model: gpt-4o
  prompt_set: scripts/eval/prompts_v2_typed20.jsonl
```

- [x] **Step 2: Build candidate completions per prompt**

`scripts/dpo/build_preference_pairs.py`:
- run SFT model and sample 2 completions per prompt,
- send pair to GPT-4o pairwise ranker,
- save DPO dataset JSONL with `prompt`, `chosen`, `rejected`.

- [x] **Step 3: Add DPO trainer script**

`scripts/train/train_dpo.py`:
- load SFT adapter as policy init,
- load preference dataset,
- run TRL DPO training,
- save DPO adapter separately to `outputs/jinyong-dpo/adapter`.

- [x] **Step 4: Add post-DPO mandatory eval**

After DPO training, run Task 1 eval harness and compare to SFT baseline.

- [ ] **Step 5: Verification commands**

```bash
python scripts/dpo/build_preference_pairs.py --config configs/qlora_config.yaml --max-prompts 20
python scripts/train/train_dpo.py --config configs/qlora_config.yaml
python scripts/eval/eval_rubric.py --config configs/qlora_config.yaml --run-id dpo_run_001
```

Expected: style and instruction-following averages improve vs SFT baseline.

- [x] **Step 6: Document SFT->DPO workflow**

In `README.md`, add staged training section:
1) SFT
2) preference pair construction
3) DPO
4) evaluation gate

- [x] **Step 7: Commit**

```bash
git add scripts/dpo scripts/train/train_dpo.py configs/qlora_config.yaml README.md
git commit -m "feat: add post-SFT DPO pipeline with GPT-4o ranked preference pairs"
```

---

### Task 6: Add FastAPI Streaming SSE Server Wrapping Ollama

**Files:**
- Create: `scripts/server/stream_api.py`
- Modify: `requirements.txt`
- Modify: `README.md`

- [x] **Step 1: Add server dependencies**

In `requirements.txt` ensure:

```txt
fastapi
uvicorn
httpx
sse-starlette
```

- [x] **Step 2: Implement SSE proxy endpoint**

`scripts/server/stream_api.py`:
- endpoint `POST /v1/generate/stream`
- forward prompt to Ollama local API
- stream chunks as SSE events:
  - `event: token`
  - `data: {"text":"..."}`
  - final `event: done`

- [x] **Step 3: Add health and non-stream endpoint**

Endpoints:
- `GET /healthz`
- `POST /v1/generate` (non-stream full text response)

- [x] **Step 4: Add model config**

Read from env:
- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`
- `SERVER_HOST`
- `SERVER_PORT`

No hardcoded absolute paths.

- [ ] **Step 5: Verification run**

```bash
python scripts/server/stream_api.py
curl -N -X POST http://127.0.0.1:8000/v1/generate/stream -H "Content-Type: application/json" -d '{"prompt":"以金庸风格写一段华山夜战"}'
```

Expected: incremental SSE token events emitted in real time.

- [x] **Step 6: Document frontend/video integration**

In `README.md`, add snippets for:
- browser `EventSource` consumption
- video pipeline consumer polling/stream parse

- [x] **Step 7: Commit**

```bash
git add scripts/server/stream_api.py requirements.txt README.md
git commit -m "feat: add FastAPI SSE streaming server for Ollama-backed generation"
```

---

## Cross-Task Acceptance Criteria

- `eval_results.jsonl` exists and is appended after every retrain.
- Quality gate blocks regressions automatically.
- Typed pair ratio is enforced and visible in stats.
- Deduplication report is generated with measurable reduction.
- Prompt library is browsable and usable via inference CLI.
- DPO adapter is produced separately and evaluated against SFT baseline.
- Streaming API emits real-time SSE tokens consumable by frontend/video pipeline.

---

## 4-Week Delivery Sequence

- **Week 1:** Task 1 (eval harness) + Task 2 (rebalance).
- **Week 2:** Task 3 (dedup) + Task 4 (prompt templates).
- **Week 3:** Task 5 (DPO pass end-to-end).
- **Week 4:** Task 6 (FastAPI SSE server) + integration polish.

---

## Risks and Mitigations

- **Judge model variance (GPT-4o):** fix rubric prompt, set temperature=0, store judge prompt version.
- **API cost growth (eval + DPO ranking):** cap prompt count in dev mode, batch requests, cache judged pairs.
- **Over-regularization from dedup:** keep threshold configurable and compare eval deltas before/after.
- **DPO instability:** start with conservative `beta=0.1`, short runs, always compare against SFT baseline.
- **Streaming timeout issues:** add request timeout/retry and heartbeat SSE events.

---

## Minimal Command Checklist (End-to-End)

```bash
python scripts/data/build_instructions.py --dedup-continuation --stats
python scripts/train/train.py --config configs/qlora_config.yaml
python scripts/eval/eval_rubric.py --config configs/qlora_config.yaml --run-id sft_latest
python scripts/dpo/build_preference_pairs.py --config configs/qlora_config.yaml --max-prompts 20
python scripts/train/train_dpo.py --config configs/qlora_config.yaml
python scripts/eval/eval_rubric.py --config configs/qlora_config.yaml --run-id dpo_latest
python scripts/server/stream_api.py
```
