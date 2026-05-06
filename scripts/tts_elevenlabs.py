from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

DEFAULT_VOICE_ID = "Xb7hH8MSUJpSbSDYk0k2"
DEFAULT_MODEL_ID = "eleven_multilingual_v2"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'").strip('"')
    return values


def _load_elevenlabs_api_key() -> str:
    env_value = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if env_value:
        return env_value

    env_path = Path(".env")
    values = _read_env_file(env_path)
    file_value = values.get("ELEVENLABS_API_KEY", "").strip()
    if file_value:
        return file_value

    raise EnvironmentError(
        "ELEVENLABS_API_KEY is missing. Set it in environment or .env file."
    )


def _resolve_input_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text.strip()

    if args.input_file:
        content = Path(args.input_file).read_text(encoding="utf-8")
        return content.strip()

    if not sys.stdin.isatty():
        return sys.stdin.read().strip()

    raise ValueError("No input text found. Use --text, --input-file, or stdin.")


def _default_output_path() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/tts") / f"tts_{ts}.mp3"


def _build_payload(text: str, model_id: str, stability: float, similarity_boost: float) -> dict[str, Any]:
    return {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
            # Keep style moderate so Chinese wuxia narration sounds expressive but stable.
            "style": 0.45,
            "use_speaker_boost": True,
        },
    }


def synthesize(
    *,
    text: str,
    api_key: str,
    voice_id: str,
    model_id: str,
    output_format: str,
    timeout_seconds: int,
    stability: float,
    similarity_boost: float,
) -> bytes:
    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
        f"?output_format={output_format}"
    )
    payload = _build_payload(
        text=text,
        model_id=model_id,
        stability=stability,
        similarity_boost=similarity_boost,
    )
    resp = requests.post(
        url,
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        json=payload,
        timeout=timeout_seconds,
    )
    resp.raise_for_status()
    return resp.content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Chinese TTS audio from text using ElevenLabs."
    )
    parser.add_argument("--text", type=str, default="", help="Input text directly.")
    parser.add_argument(
        "--input-file",
        type=str,
        default="",
        help="Read input text from a UTF-8 file.",
    )
    parser.add_argument(
        "--voice-id",
        type=str,
        default=DEFAULT_VOICE_ID,
        help="ElevenLabs voice ID (default is Chinese-friendly).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="ElevenLabs model ID.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default=DEFAULT_OUTPUT_FORMAT,
        help="ElevenLabs output format, e.g. mp3_44100_128.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output audio file path. If omitted, writes bytes to stdout.",
    )
    parser.add_argument(
        "--save-default",
        action="store_true",
        help="Save to outputs/tts/tts_<timestamp>.mp3 when --output is omitted.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--stability",
        type=float,
        default=0.42,
        help="Voice stability [0,1]. Lower is more expressive.",
    )
    parser.add_argument(
        "--similarity-boost",
        type=float,
        default=0.78,
        help="Voice similarity boost [0,1].",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        text = _resolve_input_text(args)
        if not text:
            raise ValueError("Input text is empty after trimming.")

        api_key = _load_elevenlabs_api_key()
        audio_bytes = synthesize(
            text=text,
            api_key=api_key,
            voice_id=args.voice_id,
            model_id=args.model_id,
            output_format=args.output_format,
            timeout_seconds=args.timeout_seconds,
            stability=args.stability,
            similarity_boost=args.similarity_boost,
        )

        output_path = args.output.strip()
        if output_path:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(audio_bytes)
            print(f"Saved audio to: {target}", file=sys.stderr)
            return 0

        if args.save_default:
            target = _default_output_path()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(audio_bytes)
            print(f"Saved audio to: {target}", file=sys.stderr)
            return 0

        sys.stdout.buffer.write(audio_bytes)
        sys.stdout.buffer.flush()
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[tts_elevenlabs] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
