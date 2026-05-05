# Eval Results Schema

`outputs/eval/eval_results.jsonl` stores per-prompt judge scores (append-only):

```json
{
  "timestamp": "2026-05-05T19:00:00+00:00",
  "run_id": "sft_run_2026_05_05",
  "prompt_id": "typed_001",
  "category": "dialogue",
  "judge_model": "gpt-4o",
  "scores": {
    "style_fidelity": 4,
    "instruction_following": 5,
    "coherence": 4,
    "imagery": 4,
    "originality": 4
  },
  "avg": 4.2,
  "brief": "文风稳健，画面感足。"
}
```

`outputs/eval/summary_history.jsonl` stores one summary per run:

```json
{
  "timestamp": "2026-05-05T19:05:00+00:00",
  "run_id": "sft_run_2026_05_05",
  "judge_model": "gpt-4o",
  "count": 20,
  "overall_avg": 4.16,
  "dimension_avg": {
    "style_fidelity": 4.25,
    "instruction_following": 4.1,
    "coherence": 4.2,
    "imagery": 4.1,
    "originality": 4.15
  },
  "gate_passed": true,
  "gate_failures": []
}
```

