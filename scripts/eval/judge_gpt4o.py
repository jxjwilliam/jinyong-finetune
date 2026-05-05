from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

DIMENSIONS: tuple[str, ...] = (
    "style_fidelity",
    "instruction_following",
    "coherence",
    "imagery",
    "originality",
)


@dataclass(frozen=True)
class JudgeConfig:
    model: str = "gpt-4o"
    api_base: str = "https://api.openai.com/v1"
    timeout_seconds: int = 60
    temperature: float = 0.0


def _build_judge_prompt(sample: dict[str, Any]) -> str:
    rubric = (
        "你是严格的中文武侠写作评审。请根据给定的 instruction/input/output，"
        "按 5 个维度 1-5 分打分（5 为最好）："
        "style_fidelity（是否有金庸风格）、instruction_following（是否遵循任务）、"
        "coherence（逻辑与叙事连贯）、imagery（画面感）、originality（原创性）。"
        "只返回 JSON，不要解释。JSON 格式："
        '{"scores":{"style_fidelity":1,"instruction_following":1,"coherence":1,"imagery":1,"originality":1},"brief":"<=30字简评"}。'
    )
    return (
        f"{rubric}\n\n"
        f"instruction:\n{sample['instruction']}\n\n"
        f"input:\n{sample.get('input', '')}\n\n"
        f"output:\n{sample['output']}\n"
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Judge response is not valid JSON object: {text[:120]}")
    return json.loads(stripped[start : end + 1])


def _clamp_score(value: Any) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        score = 1
    return max(1, min(5, score))


def judge_one(sample: dict[str, Any], cfg: JudgeConfig) -> dict[str, Any]:
    import requests

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is required for GPT judge.")

    payload = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "messages": [
            {"role": "system", "content": "你是评分器。输出严格 JSON。"},
            {"role": "user", "content": _build_judge_prompt(sample)},
        ],
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(
        f"{cfg.api_base.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=cfg.timeout_seconds,
    )
    resp.raise_for_status()
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    parsed = _extract_json_object(content)

    raw_scores = parsed.get("scores", {})
    scores = {dim: _clamp_score(raw_scores.get(dim, 1)) for dim in DIMENSIONS}
    avg = round(sum(scores.values()) / len(DIMENSIONS), 4)
    return {
        "scores": scores,
        "avg": avg,
        "brief": str(parsed.get("brief", "")),
    }

