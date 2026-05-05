"""Unified typed-scene JSONL generator (disjoint template buckets).

Templates: ``configs/jinyong_template.json``. Outputs rows compatible with
``build_instructions.py --typed-jsonl``.

Subcommands:

  claude — Anthropic Messages API (typical bucket ``claude``, ids 1–20).

  openai — OpenAI-compatible chat APIs (DeepSeek, Kimi, MiniMax, GLM); keys in ``.env``.

Examples::

    export ANTHROPIC_API_KEY=...
    python scripts/gen/generate_typed_pairs.py claude --output data/instructions/typed_pairs.jsonl --dry-run

    pip install openai python-dotenv
    python scripts/gen/generate_typed_pairs.py openai --providers deepseek,kimi --output data/instructions/more_types_pairs.jsonl

See ``docs/TYPED_PAIRS_PIPELINE.md``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_repo_root = Path(__file__).resolve().parents[2]
_lib = _repo_root / "scripts" / "lib"
if str(_lib) not in sys.path:
    sys.path.insert(0, str(_lib))

from instruction_jsonl import (
    count_nonempty_jsonl_lines,
    typed_pair_dict,
    write_jsonl_line,
    write_jsonl_rows,
)
from typed_prompts import (
    PROVIDER_BUCKETS,
    SYSTEM_PROMPT_JINYONG_TYPED,
    each_typed_sample,
    load_typed_scenes,
    scenes_for_bucket,
    scenes_for_provider_slug,
    typed_user_turn,
)


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ModuleNotFoundError:
        return {}
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def resolve_per_template(args: argparse.Namespace) -> int:
    if args.per_template is not None:
        return int(args.per_template)
    cfg = load_yaml(Path(args.config))
    data_cfg = cfg.get("data", {})
    typed_cfg = data_cfg.get("typed_pairs", {})
    value = int(typed_cfg.get("per_template", 10))
    print(f"[config] --per-template not provided, using data.typed_pairs.per_template={value}")
    return value

# --- Claude backend -----------------------------------------------------------

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[misc, assignment]


def _generate_claude_one(
    client: "anthropic.Anthropic", instruction: str, hint: str
) -> str:
    prompt = typed_user_turn(instruction, hint)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        system=SYSTEM_PROMPT_JINYONG_TYPED,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def run_claude(args: argparse.Namespace) -> None:
    if anthropic is None:
        raise SystemExit("pip install anthropic")
    client = anthropic.Anthropic()

    scenes = scenes_for_bucket(args.bucket, config_path=args.templates_config)
    if not scenes:
        raise SystemExit(
            f"No templates loaded for bucket {args.bucket!r} — check {args.templates_config}"
        )

    print(f"Bucket {args.bucket!r} ({PROVIDER_BUCKETS[args.bucket]}): {len(scenes)} template(s)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    append_fh = None
    if not args.dry_run:
        prev_n = count_nonempty_jsonl_lines(out_path)
        if prev_n:
            print(f"[resume] Output file already has {prev_n:,} JSONL rows; appending new pairs.")
        append_fh = out_path.open("a", encoding="utf-8")

    per_template = resolve_per_template(args)
    n_per = 1 if args.dry_run else per_template
    previews = 0
    written_session = 0

    try:
        for t_idx, scene, sample_i, hint in each_typed_sample(
            scenes, samples_per_scene=n_per
        ):
            instruction = scene.instruction
            if sample_i == 0:
                print(
                    f"\n[id {scene.id}] [{t_idx + 1}/{len(scenes)}] ({scene.scene_type}) "
                    f"{instruction[:72]}…"
                )

            try:
                output = _generate_claude_one(client, instruction, hint)
            except Exception as e:
                print(f"  [error] sample {sample_i + 1}: {e}")
                time.sleep(2)
                continue

            previews += 1
            pair = typed_pair_dict(instruction, output)
            if append_fh is not None:
                write_jsonl_line(append_fh, pair)
                append_fh.flush()
                written_session += 1

            if args.dry_run or sample_i == 0:
                print(f"  Sample {sample_i + 1}: {output[:80]}...")

            time.sleep(args.sleep)
    finally:
        if append_fh is not None:
            append_fh.close()

    if args.dry_run:
        print(f"\n[dry-run] Would write {previews:,} pairs. Re-run without --dry-run to save.")
        return

    print(f"\nWrote {written_session:,} new typed pairs this run → {out_path}")
    print(
        "Next: python scripts/data/build_instructions.py --typed-jsonl "
        f"{out_path} --stats\n"
        "      (repeat --typed-jsonl for each generator output before --stats)"
    )


def parse_claude(subparsers) -> None:
    p = subparsers.add_parser(
        "claude",
        help="Anthropic API (default bucket ids 1–20)",
    )
    p.add_argument("--output", default="data/instructions/typed_pairs.jsonl")
    p.add_argument(
        "--templates-config",
        default="configs/jinyong_template.json",
        help="JSON array of {id,type,template}.",
    )
    p.add_argument(
        "--bucket",
        default="claude",
        choices=sorted(PROVIDER_BUCKETS.keys()),
        help="Template id partition (default claude → ids 1–20).",
    )
    p.add_argument("--per-template", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sleep", type=float, default=0.3)


# --- OpenAI-compatible backends ----------------------------------------------

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[misc, assignment]

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[misc, assignment]


@dataclass(frozen=True)
class OpenAiCompatProvider:
    slug: str
    key_env: tuple[str, ...]
    base_url: str
    model: str
    base_env: str | None = None
    model_env: str | None = None


DEFAULT_OPENAI_PROVIDERS: tuple[OpenAiCompatProvider, ...] = (
    OpenAiCompatProvider(
        slug="deepseek",
        key_env=("DEEPSEEK_API_KEY", "DEEPSEEK_APPI_KEY"),
        base_env="DEEPSEEK_BASE_URL",
        base_url="https://api.deepseek.com",
        model_env="DEEPSEEK_MODEL",
        model="deepseek-chat",
    ),
    OpenAiCompatProvider(
        slug="kimi",
        key_env=("KIMI_API_KEY",),
        base_env="KIMI_BASE_URL",
        base_url="https://api.moonshot.cn/v1",
        model_env="KIMI_MODEL",
        model="moonshot-v1-8k",
    ),
    OpenAiCompatProvider(
        slug="minimax",
        key_env=("MINIMAX_API_KEY",),
        base_env="MINIMAX_BASE_URL",
        base_url="https://api.minimaxi.com/v1",
        model_env="MINIMAX_MODEL",
        model="MiniMax-M2.5",
    ),
    OpenAiCompatProvider(
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
                "[warn] python-dotenv missing; skipping .env — install with "
                "`pip install python-dotenv` or export keys manually.",
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


def resolve_provider_key(p: OpenAiCompatProvider) -> str | None:
    return first_env(*p.key_env)


def resolve_base_url(p: OpenAiCompatProvider) -> str:
    if p.base_env:
        u = os.getenv(p.base_env, "").strip()
        if u:
            return u.rstrip("/")
    return p.base_url.rstrip("/")


def resolve_model(p: OpenAiCompatProvider) -> str:
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
        bits: list[str] = []
        for part in text:
            if hasattr(part, "text") and getattr(part, "text", None):
                bits.append(str(part.text))
        out = "".join(bits).strip()
    else:
        out = ""
    return out


def providers_from_arg(arg: str) -> list[OpenAiCompatProvider]:
    want = {s.strip().lower() for s in arg.split(",") if s.strip()}
    by_slug = {p.slug: p for p in DEFAULT_OPENAI_PROVIDERS}
    unk = want - set(by_slug)
    if unk:
        raise SystemExit(f"Unknown provider(s): {unk}. Choose from {sorted(by_slug)}.")
    return [by_slug[s] for s in sorted(by_slug) if s in want]


def run_openai(args: argparse.Namespace) -> None:
    if OpenAI is None:
        raise SystemExit("pip install openai")
    load_env_file()

    all_scenes = load_typed_scenes(args.templates_config)
    targets = providers_from_arg(args.providers)
    ready: list[tuple[OpenAiCompatProvider, str, Callable[[str, str], str]]] = []

    for p in targets:
        key = resolve_provider_key(p)
        if not key:
            print(f"[skip] {p.slug}: no API key ({', '.join(p.key_env)})")
            continue
        base_url = resolve_base_url(p)
        model = resolve_model(p)

        def make_call(prov=p, k=key, b=base_url, m=model) -> Callable[[str, str], str]:
            def inner(instruction: str, hint: str) -> str:
                user_msg = typed_user_turn(instruction, hint)
                return chat_completion(
                    api_key=k,
                    base_url=b,
                    model=m,
                    system=SYSTEM_PROMPT_JINYONG_TYPED,
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
        prev = scenes_for_provider_slug(p.slug, scenes=all_scenes)
        print(f"  - {p.slug}  model={model}  templates={len(prev)} ids")

    per_template = resolve_per_template(args)
    n_per = 1 if args.dry_run else per_template
    pairs: list[dict[str, str]] = []

    for p, _model, call in ready:
        scenes = scenes_for_provider_slug(p.slug, scenes=all_scenes)
        if not scenes:
            print(f"[warn] {p.slug}: no templates in bucket — skipping")
            continue
        for t_idx, scene, sample_i, hint in each_typed_sample(
            scenes, samples_per_scene=n_per
        ):
            instruction = scene.instruction
            if sample_i == 0:
                print(
                    f"\n[{p.slug}] id={scene.id} [{t_idx + 1}/{len(scenes)}] ({scene.scene_type}) "
                    f"{instruction[:60]}…"
                )
            try:
                output = call(instruction, hint)
            except Exception as e:
                print(f"  [error] {p.slug} sample {sample_i + 1}: {e}")
                time.sleep(2.0)
                continue
            if not output:
                print(f"  [warn] empty output {p.slug} sample {sample_i + 1}; skipping")
                time.sleep(args.sleep)
                continue
            pairs.append(typed_pair_dict(instruction, output))
            if args.dry_run or sample_i == 0:
                print(f"  sample {sample_i + 1}: {output[:80]}...")
            time.sleep(args.sleep)

    if args.dry_run:
        print(f"\n[dry-run] collected {len(pairs)} previews; rerun without --dry-run to save.")
        return

    out_path = Path(args.output)
    n_saved = write_jsonl_rows(out_path, pairs)
    print(f"\nSaved {n_saved:,} pairs → {out_path}")
    print(
        "Merge example:\n"
        f"  python scripts/data/build_instructions.py --typed-jsonl {out_path} "
        "--typed-jsonl data/instructions/typed_pairs.jsonl --stats"
    )


def parse_openai(subparsers) -> None:
    avail = ",".join(p.slug for p in DEFAULT_OPENAI_PROVIDERS)
    p = subparsers.add_parser(
        "openai",
        help=f"OpenAI-compatible APIs ({avail})",
    )
    p.add_argument(
        "--providers",
        type=str,
        default="deepseek,kimi,minimax,glm",
        help=f"Comma-separated subset of {avail}",
    )
    p.add_argument("--templates-config", default="configs/jinyong_template.json")
    p.add_argument(
        "--output",
        default="data/instructions/more_types_pairs.jsonl",
    )
    p.add_argument("--per-template", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sleep", type=float, default=0.35)
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.7)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate typed Jin Yong scene pairs → JSONL. "
            "Use subcommand `claude` or `openai`."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/qlora_config.yaml",
        help="Config path used for per-template fallback.",
    )
    sub = parser.add_subparsers(dest="backend", required=True)
    parse_claude(sub)
    parse_openai(sub)
    args = parser.parse_args()

    if args.backend == "claude":
        run_claude(args)
    elif args.backend == "openai":
        run_openai(args)
    else:
        raise SystemExit(f"Unknown backend {args.backend!r}")


if __name__ == "__main__":
    main()
