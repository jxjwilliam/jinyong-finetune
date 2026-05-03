"""generate_more_types_pairs.py

Generate {instruction, input, output} JSONL rows using Chinese LLM APIs (OpenAI-compatible)
so `build_instructions.py --typed-jsonl` can merge them like `generate_typed_pairs.py`.

Reads API keys from the repo-root `.env` (via python-dotenv) or existing process env.

Expected env vars (any missing provider is skipped unless you restrict with --providers):
  DEEPSEEK_API_KEY  (aliases: DEEPSEEK_APPI_KEY — common typo)
  KIMI_API_KEY      Moonshot/Kimi OpenAI-compat
  MINIMAX_API_KEY
  GLM_API_KEY       Zhipu ChatGLM OpenAI-compat

Optional overrides:
  DEEPSEEK_BASE_URL / DEEPSEEK_MODEL
  KIMI_BASE_URL / KIMI_MODEL
  MINIMAX_BASE_URL / MINIMAX_MODEL   (China default in code: https://api.minimaxi.com/v1; intl: https://api.minimax.io/v1)
  GLM_BASE_URL / GLM_MODEL

Usage:
  pip install openai python-dotenv
  python scripts/generate_more_types_pairs.py --dry-run
  python scripts/generate_more_types_pairs.py \\
      --providers deepseek,kimi \\
      --output data/instructions/more_types_pairs.jsonl \\
      --per-template 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[misc, assignment]

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("pip install openai")


# ── Prompts/templates (must stay aligned with scripts/generate_typed_pairs.py) ──

TYPED_TEMPLATES: tuple[str, ...] = (
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
)

SYSTEM = (
    "你是金庸式武侠小说的写作专家。"
    "请严格按照金庸的写作风格创作：文笔典雅简练，情节紧凑，"
    "人物性格鲜明，融汇历史背景，富有江湖气息。"
    "每次创作约180-220字，完全原创，禁止引用或复述金庸原著中的具体情节。"
    "只输出小说正文，不要标题、序号或解释。"
)

VARIATION_HINTS = [
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
]


@dataclass(frozen=True)
class Provider:
    slug: str
    key_env: tuple[str, ...]
    base_url: str
    model: str
    base_env: str | None = None  # optional override env for base URL
    model_env: str | None = None  # optional override env for model id


DEFAULT_PROVIDERS: tuple[Provider, ...] = (
    Provider(
        slug="deepseek",
        key_env=("DEEPSEEK_API_KEY", "DEEPSEEK_APPI_KEY"),
        base_env="DEEPSEEK_BASE_URL",
        base_url="https://api.deepseek.com",
        model_env="DEEPSEEK_MODEL",
        model="deepseek-chat",
    ),
    Provider(
        slug="kimi",
        key_env=("KIMI_API_KEY",),
        base_env="KIMI_BASE_URL",
        base_url="https://api.moonshot.cn/v1",
        model_env="KIMI_MODEL",
        model="moonshot-v1-8k",
    ),
    Provider(
        slug="minimax",
        key_env=("MINIMAX_API_KEY",),
        base_env="MINIMAX_BASE_URL",
        # China (国内) OpenAI-compatible entrypoint; intl keys use https://api.minimax.io/v1 via .env
        base_url="https://api.minimaxi.com/v1",
        model_env="MINIMAX_MODEL",
        model="MiniMax-M2.5",
    ),
    Provider(
        slug="glm",
        key_env=("GLM_API_KEY",),
        base_env="GLM_BASE_URL",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        model_env="GLM_MODEL",
        model="glm-4-flash",
    ),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_env_file() -> None:
    dot = repo_root() / ".env"
    if load_dotenv is None:
        if dot.is_file():
            print(
                "[warn] python-dotenv missing; skipping .env — install with `pip install python-dotenv` "
                "or export keys manually.",
                file=sys.stderr,
            )
        return
    if dot.is_file():
        load_dotenv(dot)


def first_env(*names: str) -> str | None:
    for n in names:
        v = os.getenv(n.strip())
        if v and v.strip():
            return v.strip()
    return None


def resolve_provider_key(p: Provider) -> str | None:
    return first_env(*p.key_env)


def resolve_base_url(p: Provider) -> str:
    if p.base_env:
        u = os.getenv(p.base_env, "").strip()
        if u:
            return u.rstrip("/")
    return p.base_url.rstrip("/")


def resolve_model(p: Provider) -> str:
    if p.model_env:
        m = os.getenv(p.model_env, "").strip()
        if m:
            return m
    return p.model


def chat_completion(
    *,
    api_key: str,
    base_url: str,
    model: str,
    system: str,
    user_msg: str,
    max_tokens: int,
    temperature: float,
) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url)
    rsp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    choice = rsp.choices[0].message
    text = getattr(choice, "content", None) or ""
    if isinstance(text, str):
        out = text.strip()
    elif isinstance(text, list):
        # Some APIs return multimodal-ish chunks
        bits: list[str] = []
        for part in text:
            if hasattr(part, "text") and getattr(part, "text", None):
                bits.append(str(part.text))
        out = "".join(bits).strip()
    else:
        out = ""
    return out


def parse_args() -> argparse.Namespace:
    avail = ",".join(p.slug for p in DEFAULT_PROVIDERS)
    parser = argparse.ArgumentParser(
        description=(
            "Generate typed pairs JSONL using DeepSeek/Kimi/MiniMax/GLM (OpenAI-compatible). "
            f"Reads keys from `.env`; providers: {avail}."
        )
    )
    parser.add_argument(
        "--providers",
        type=str,
        default="deepseek,kimi,minimax,glm",
        help=f"Comma list of providers to run (subset of {avail})",
    )
    parser.add_argument(
        "--output",
        default="data/instructions/more_types_pairs.jsonl",
        help="Destination JSONL",
    )
    parser.add_argument("--per-template", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.35, help="Throttle between HTTP calls")
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def providers_from_arg(arg: str) -> list[Provider]:
    want = {s.strip().lower() for s in arg.split(",") if s.strip()}
    by_slug = {p.slug: p for p in DEFAULT_PROVIDERS}
    unk = want - set(by_slug)
    if unk:
        raise SystemExit(f"Unknown provider(s): {unk}. Choose from {sorted(by_slug)}.")
    return [by_slug[s] for s in sorted(by_slug) if s in want]


def main() -> None:
    args = parse_args()
    load_env_file()

    targets = providers_from_arg(args.providers)
    ready: list[tuple[Provider, str, Callable[[str, str], str]]] = []

    for p in targets:
        key = resolve_provider_key(p)
        if not key:
            print(f"[skip] {p.slug}: no API key ({', '.join(p.key_env)})")
            continue
        base_url = resolve_base_url(p)
        model = resolve_model(p)

        def make_call(prov=p, k=key, b=base_url, m=model) -> Callable[[str, str], str]:
            def inner(template: str, hint: str) -> str:
                user_msg = f"{template}。\n（场景提示：{hint}）"
                return chat_completion(
                    api_key=k,
                    base_url=b,
                    model=m,
                    system=SYSTEM,
                    user_msg=user_msg,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )

            return inner

        ready.append((p, model, make_call()))

    if not ready:
        raise SystemExit(
            "No providers configured. Set DEEPSEEK_API_KEY / KIMI_API_KEY / "
            "MINIMAX_API_KEY / GLM_API_KEY in `.env` or the environment."
        )

    print("Active providers:")
    for p, model, _ in ready:
        print(f"  - {p.slug}  model={model}  base={resolve_base_url(p)}")

    n_per = 1 if args.dry_run else args.per_template
    pairs: list[dict[str, str]] = []
    n_hints = len(VARIATION_HINTS)

    for p, model, call in ready:
        for t_idx, template in enumerate(TYPED_TEMPLATES):
            print(f"\n[{p.slug}] [{t_idx + 1}/{len(TYPED_TEMPLATES)}] {template}")
            for i in range(n_per):
                hint = VARIATION_HINTS[(t_idx * n_per + i) % max(n_hints, 1)]
                try:
                    output = call(template, hint)
                except Exception as e:
                    print(f"  [error] {p.slug} sample {i + 1}: {e}")
                    time.sleep(2.0)
                    continue
                if not output:
                    print(f"  [warn] empty output {p.slug} sample {i + 1}; skipping")
                    time.sleep(args.sleep)
                    continue
                pairs.append({"instruction": template, "input": "", "output": output})
                if args.dry_run or i == 0:
                    print(f"  sample {i + 1}: {output[:80]}...")
                time.sleep(args.sleep)

    if args.dry_run:
        print(f"\n[dry-run] collected {len(pairs)} previews; rerun without --dry-run to save.")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in pairs:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(pairs):,} pairs → {out_path}")
    print(f"Merge: python scripts/build_instructions.py --typed-jsonl {out_path} --stats")


if __name__ == "__main__":
    main()
