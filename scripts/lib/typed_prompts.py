"""Shared typed-scene prompts for Jin Yong–style generation.

Single source of truth: ``configs/jinyong_template.json`` (``id``, ``type``, ``template``).
The ``template`` string is sent verbatim as the user ``instruction``; 【slots】 are
interpreted by the LLM without a separate filler step.

Partition ``id`` ranges across backends so each LLM owns a disjoint slice (no duplicate
instructions across generators when following defaults):

  claude    1–20
  deepseek 21–40
  kimi     41–60
  minimax  61–80
  glm      81–100
"""
from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_TEMPLATE_JSON = Path("configs/jinyong_template.json")

PROVIDER_BUCKETS: dict[str, tuple[int, int]] = {
    "claude": (1, 20),
    "deepseek": (21, 40),
    "kimi": (41, 60),
    "minimax": (61, 80),
    "glm": (81, 100),
}

SYSTEM_PROMPT_JINYONG_TYPED = (
    "你是金庸式武侠小说的写作专家。"
    "请严格按照金庸的写作风格创作：文笔典雅简练，情节紧凑，"
    "人物性格鲜明，融汇历史背景，富有江湖气息。"
    "每次创作约180-220字，完全原创，禁止引用或复述金庸原著中的具体情节。"
    "只输出小说正文，不要标题、序号或解释。"
)

VARIATION_HINTS: tuple[str, ...] = (
    "场景发生在一个雨夜的古镇",
    "背景是北方大漠",
    "故事发生在江南水乡",
    "场景在高山雪峰之上",
    "背景是一座破败的古寺",
    "场景在繁华的武林大会现场",
    "背景是一艘江湖帮派的大船上",
    "场景在幽深的山洞之中",
    "故事发生在风雪交加的边疆驿站",
    "背景是一处幽静的竹林小院",
)


@dataclass(frozen=True)
class TypedScene:
    id: int
    scene_type: str
    instruction: str


def template_json_path(config_path: Path | str | None = None) -> Path:
    if config_path is None:
        return DEFAULT_TEMPLATE_JSON
    return Path(config_path)


def load_typed_scenes(config_path: Path | str | None = None) -> list[TypedScene]:
    path = template_json_path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Typed template config not found: {path}")
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected JSON array in {path}")
    scenes: list[TypedScene] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        sid = row.get("id")
        stype = row.get("type", "")
        tmpl = row.get("template", "")
        if isinstance(sid, int) and isinstance(tmpl, str) and tmpl.strip():
            scenes.append(
                TypedScene(id=sid, scene_type=str(stype), instruction=tmpl.strip())
            )
    scenes.sort(key=lambda s: s.id)
    return scenes


def scenes_for_bucket(
    bucket: str,
    *,
    scenes: list[TypedScene] | None = None,
    config_path: Path | str | None = None,
) -> list[TypedScene]:
    if bucket not in PROVIDER_BUCKETS:
        known = ", ".join(sorted(PROVIDER_BUCKETS))
        raise ValueError(f"Unknown bucket {bucket!r}. Choose one of: {known}")
    lo, hi = PROVIDER_BUCKETS[bucket]
    all_scenes = scenes if scenes is not None else load_typed_scenes(config_path)
    return [s for s in all_scenes if lo <= s.id <= hi]


def scenes_for_provider_slug(
    slug: str,
    *,
    scenes: list[TypedScene] | None = None,
    config_path: Path | str | None = None,
) -> list[TypedScene]:
    """Map OpenAI-compat provider slug (e.g. ``deepseek``) to its id bucket."""
    key = slug.strip().lower()
    mapping = {
        "deepseek": "deepseek",
        "kimi": "kimi",
        "minimax": "minimax",
        "glm": "glm",
    }
    if key not in mapping:
        raise ValueError(
            f"No template bucket for provider slug {slug!r}. "
            f"Expected one of {sorted(mapping)}."
        )
    return scenes_for_bucket(mapping[key], scenes=scenes, config_path=config_path)


def instruction_strings(config_path: Path | str | None = None) -> list[str]:
    return [s.instruction for s in load_typed_scenes(config_path)]


def typed_user_turn(instruction: str, variation_hint: str) -> str:
    """User message body (instruction + scene hint) for typed-scene generation."""
    return f"{instruction}。\n（场景提示：{variation_hint}）"


def each_typed_sample(
    scenes: Sequence[TypedScene],
    *,
    samples_per_scene: int,
    hints: Sequence[str] | None = None,
) -> Iterator[tuple[int, TypedScene, int, str]]:
    """Yield ``(scene_index, scene, sample_index_within_scene, variation_hint)``."""
    h = tuple(hints) if hints is not None else VARIATION_HINTS
    n_hints = len(h) or 1
    for t_idx, scene in enumerate(scenes):
        for i in range(samples_per_scene):
            hint = h[(t_idx * samples_per_scene + i) % n_hints]
            yield t_idx, scene, i, hint
