"""generate_typed_pairs.py

Generates real instruction→output pairs for each TYPED_TEMPLATE using the
Claude API.  Outputs a JSONL file that build_instructions.py can ingest via
--typed-jsonl.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/generate_typed_pairs.py \
        --output data/instructions/typed_pairs.jsonl \
        --per-template 20 \
        --dry-run         # print 1 sample per template, don't write

Cost estimate: 20 templates × 20 samples × ~300 output tokens ≈ 120k tokens
               ≈ $0.18 at claude-haiku-3 pricing (fast + cheap)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    import anthropic
except ImportError:
    raise SystemExit("pip install anthropic")

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

# Vary the scene slightly each call to avoid repetition
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


def generate_one(client: anthropic.Anthropic, template: str, hint: str) -> str:
    prompt = f"{template}。\n（场景提示：{hint}）"
    response = client.messages.create(
        model="claude-haiku-4-5",   # fast + cheap; swap to claude-sonnet-4-5 for higher quality
        max_tokens=400,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate typed Jin Yong pairs via Claude API")
    p.add_argument("--output", default="data/instructions/typed_pairs.jsonl")
    p.add_argument("--per-template", type=int, default=20,
                   help="Number of samples per template (default 20 → 400 total)")
    p.add_argument("--dry-run", action="store_true",
                   help="Generate 1 sample per template and print, do not write")
    p.add_argument("--sleep", type=float, default=0.3,
                   help="Seconds to sleep between API calls (rate limit guard)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env

    n_per = 1 if args.dry_run else args.per_template
    pairs: list[dict] = []
    n_hints = len(VARIATION_HINTS)

    for t_idx, template in enumerate(TYPED_TEMPLATES):
        print(f"\n[{t_idx + 1}/{len(TYPED_TEMPLATES)}] {template}")
        for i in range(n_per):
            hint = VARIATION_HINTS[(t_idx * n_per + i) % n_hints]
            try:
                output = generate_one(client, template, hint)
            except Exception as e:
                print(f"  [error] sample {i+1}: {e}")
                time.sleep(2)
                continue

            pair = {"instruction": template, "input": "", "output": output}
            pairs.append(pair)

            if args.dry_run or i == 0:
                print(f"  Sample {i+1}: {output[:80]}...")

            time.sleep(args.sleep)

    if args.dry_run:
        print(f"\n[dry-run] Would write {len(pairs)} pairs. Re-run without --dry-run to save.")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(pairs):,} typed pairs → {out_path}")
    print(f"Next: python scripts/build_instructions.py --typed-jsonl {out_path} --stats")


if __name__ == "__main__":
    main()
