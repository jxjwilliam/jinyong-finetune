from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

# Ensure repo root is importable when running as:
# python3 scripts/tts/batch_tts_minimax.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tts import tts_minimax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch convert markdown stories to MP3 with MiniMax TTS."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="docs/stories",
        help="Directory containing .md files (default: docs/stories).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/tts",
        help="Directory for output .mp3 files (default: outputs/tts).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/minimax-t2a.json",
        help="MiniMax JSON config path (default: configs/minimax-t2a.json).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Parallel workers for async processing (default: 1).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output MP3 files.",
    )
    return parser.parse_args()


def _resolve_runtime(config_path: Path) -> dict[str, object]:
    config = tts_minimax._load_json_config(config_path)
    defaults = tts_minimax._runtime_defaults(config)
    api_key = tts_minimax._load_minimax_api_key()
    return {
        "api_key": api_key,
        "model": str(defaults["model"]),
        "voice_id": str(defaults["voice_id"]),
        "speed": float(defaults["speed"]),
        "volume": float(defaults["volume"]),
        "pitch": int(defaults["pitch"]),
        "sample_rate": int(defaults["sample_rate"]),
        "audio_format": str(defaults["audio_format"]),
        "bitrate": int(defaults.get("bitrate", 128000)),
        "channel": int(defaults.get("channel", 1)),
    }


def _synthesize_with_voice_fallback(
    text: str,
    runtime: dict[str, object],
) -> bytes:
    try:
        return tts_minimax.synthesize(
            text=text,
            api_key=str(runtime["api_key"]),
            model=str(runtime["model"]),
            voice_id=str(runtime["voice_id"]),
            speed=float(runtime["speed"]),
            volume=float(runtime["volume"]),
            pitch=int(runtime["pitch"]),
            sample_rate=int(runtime["sample_rate"]),
            bitrate=int(runtime["bitrate"]),
            audio_format=str(runtime["audio_format"]),
            channel=int(runtime["channel"]),
            stream=False,
            timeout_seconds=120,
        )
    except RuntimeError as exc:
        message = str(exc)
        should_retry = (
            "code 2054" in message
            and str(runtime["voice_id"]) != tts_minimax.DEFAULT_VOICE_ID
        )
        if not should_retry:
            raise
        print(
            (
                f"[batch_tts_minimax] Voice '{runtime['voice_id']}' unavailable; "
                f"retrying with '{tts_minimax.DEFAULT_VOICE_ID}'."
            )
        )
        return tts_minimax.synthesize(
            text=text,
            api_key=str(runtime["api_key"]),
            model=str(runtime["model"]),
            voice_id=tts_minimax.DEFAULT_VOICE_ID,
            speed=float(runtime["speed"]),
            volume=float(runtime["volume"]),
            pitch=int(runtime["pitch"]),
            sample_rate=int(runtime["sample_rate"]),
            bitrate=int(runtime["bitrate"]),
            audio_format=str(runtime["audio_format"]),
            channel=int(runtime["channel"]),
            stream=False,
            timeout_seconds=120,
        )


async def _process_one(
    md_path: Path,
    output_dir: Path,
    runtime: dict[str, object],
    sem: asyncio.Semaphore,
    overwrite: bool,
) -> None:
    async with sem:
        out_path = output_dir / f"{md_path.stem}.mp3"
        if out_path.exists() and not overwrite:
            print(f"[skip] {out_path} exists")
            return

        text = md_path.read_text(encoding="utf-8").strip()
        if not text:
            print(f"[skip] {md_path} is empty")
            return

        audio_bytes = await asyncio.to_thread(_synthesize_with_voice_fallback, text, runtime)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(audio_bytes)
        print(f"[ok] {md_path.name} -> {out_path}")


async def run_batch(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    config_path = Path(args.config)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    md_files = sorted(input_dir.glob("*.md"))
    if not md_files:
        print(f"No markdown files found in: {input_dir}")
        return 0

    runtime = _resolve_runtime(config_path)
    sem = asyncio.Semaphore(max(1, int(args.concurrency)))
    tasks = [
        _process_one(
            md_path=md_path,
            output_dir=output_dir,
            runtime=runtime,
            sem=sem,
            overwrite=bool(args.overwrite),
        )
        for md_path in md_files
    ]
    await asyncio.gather(*tasks)
    print(f"Done. Processed {len(md_files)} file(s).")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(run_batch(args))
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[batch_tts_minimax] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
