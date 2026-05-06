from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

DEFAULT_MODEL = "speech-2.8-hd"
DEFAULT_VOICE_ID = "Chinese (Mandarin)_Lyrical_Voice"
DEFAULT_OUTPUT_FORMAT = "mp3"
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_BITRATE = 128000
DEFAULT_CONFIG_PATH = "configs/minimax-t2a.json"


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


def _load_minimax_api_key() -> str:
    env_value = os.environ.get("MINIMAX_API_KEY", "").strip()
    if env_value:
        return env_value

    env_path = Path(".env")
    values = _read_env_file(env_path)
    file_value = values.get("MINIMAX_API_KEY", "").strip()
    if file_value:
        return file_value

    raise EnvironmentError(
        "MINIMAX_API_KEY is missing. Set it in environment or .env file."
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
    return Path("outputs/tts") / f"minimax_{ts}.mp3"


def _load_json_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    return raw


def _config_narrator_voice_id(config: dict[str, Any]) -> str | None:
    voices = config.get("voices")
    if not isinstance(voices, dict):
        return None
    narrator = voices.get("narrator")
    if not isinstance(narrator, list) or not narrator:
        return None
    first_voice = narrator[0]
    if not isinstance(first_voice, dict):
        return None
    voice_id = first_voice.get("id")
    return voice_id if isinstance(voice_id, str) and voice_id.strip() else None


def _runtime_defaults(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = config.get("model")
    params_cfg = config.get("params")

    preferred_model = (
        model_cfg.get("preferred")
        if isinstance(model_cfg, dict) and isinstance(model_cfg.get("preferred"), str)
        else DEFAULT_MODEL
    )
    voice_id = _config_narrator_voice_id(config) or DEFAULT_VOICE_ID
    speed = (
        float(params_cfg.get("speed"))
        if isinstance(params_cfg, dict) and params_cfg.get("speed") is not None
        else 1.0
    )
    volume = (
        float(params_cfg.get("vol"))
        if isinstance(params_cfg, dict) and params_cfg.get("vol") is not None
        else 1.0
    )
    pitch = (
        int(params_cfg.get("pitch"))
        if isinstance(params_cfg, dict) and params_cfg.get("pitch") is not None
        else 0
    )
    audio_format = (
        params_cfg.get("audio_format")
        if isinstance(params_cfg, dict) and isinstance(params_cfg.get("audio_format"), str)
        else DEFAULT_OUTPUT_FORMAT
    )
    sample_rate = (
        int(params_cfg.get("sample_rate"))
        if isinstance(params_cfg, dict) and params_cfg.get("sample_rate") is not None
        else DEFAULT_SAMPLE_RATE
    )
    return {
        "model": preferred_model,
        "voice_id": voice_id,
        "speed": speed,
        "volume": volume,
        "pitch": pitch,
        "audio_format": audio_format,
        "sample_rate": sample_rate,
        "bitrate": DEFAULT_BITRATE,
        "channel": 1,
    }


def _build_payload(
    text: str,
    model: str,
    voice_id: str,
    speed: float,
    volume: float,
    pitch: int,
    sample_rate: int,
    bitrate: int,
    output_format: str,
    channel: int,
    stream: bool,
) -> dict[str, Any]:
    return {
        "model": model,
        "text": text,
        "stream": stream,
        "output_format": "hex",  # decode locally instead of fetching a URL
        "language_boost": "auto",
        "voice_setting": {
            "voice_id": voice_id,
            "speed": speed,
            "vol": volume,
            "pitch": pitch,
        },
        "audio_setting": {
            "sample_rate": sample_rate,
            "bitrate": bitrate,
            "format": output_format,
            "channel": channel,
        },
    }


def synthesize(
    *,
    text: str,
    api_key: str,
    model: str,
    voice_id: str,
    speed: float,
    volume: float,
    pitch: int,
    sample_rate: int,
    bitrate: int,
    audio_format: str,
    channel: int,
    stream: bool,
    timeout_seconds: int,
) -> bytes:
    url = "https://api.minimaxi.com/v1/t2a_v2"
    payload = _build_payload(
        text=text,
        model=model,
        voice_id=voice_id,
        speed=speed,
        volume=volume,
        pitch=pitch,
        sample_rate=sample_rate,
        bitrate=bitrate,
        output_format=audio_format,
        channel=channel,
        stream=stream,
    )
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout_seconds,
    )
    resp.raise_for_status()
    body = resp.json()

    # Check response status
    base_resp = body.get("base_resp", {})
    status_code = base_resp.get("status_code", -1)
    if status_code != 0:
        raise RuntimeError(
            f"MiniMax API error (code {status_code}): {base_resp.get('status_msg', 'unknown')}"
        )

    # Decode hex-encoded audio
    data = body.get("data", {})
    audio_hex = data.get("audio", "")
    if not audio_hex:
        raise RuntimeError("No audio data in MiniMax response")

    audio_bytes = bytes.fromhex(audio_hex)
    return audio_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Chinese TTS audio from text using MiniMax T2A."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to MiniMax T2A JSON config (default: configs/minimax-t2a.json).",
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
        default="",
        help="MiniMax voice ID (default: Chinese (Mandarin)_Lyrical_Voice for wuxia).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="MiniMax speech model (default: speech-2.8-hd).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=-1.0,
        help="Speech speed [0.5, 2] (default: 1.0).",
    )
    parser.add_argument(
        "--volume",
        type=float,
        default=-1.0,
        help="Speech volume (0, 10] (default: 1.0).",
    )
    parser.add_argument(
        "--pitch",
        type=int,
        default=999,
        help="Speech pitch adjustment [-12, 12] (default: 0).",
    )
    parser.add_argument(
        "--audio-format",
        type=str,
        default="",
        choices=["mp3", "pcm", "flac", "wav"],
        help="Output audio format (default: mp3). Note: wav is non-streaming only.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=0,
        help="Sample rate (default: 44100).",
    )
    parser.add_argument(
        "--bitrate",
        type=int,
        default=DEFAULT_BITRATE,
        help="Bitrate for mp3 (default: 128000).",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=1,
        choices=[1, 2],
        help="Audio channels: 1=mono, 2=stereo (default: 1).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming output (default: false).",
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
        help="Deprecated: output now defaults to outputs/tts/minimax_<timestamp>.mp3.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="HTTP request timeout in seconds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = _load_json_config(Path(args.config))
        defaults = _runtime_defaults(config)
        text = _resolve_input_text(args)
        if not text:
            raise ValueError("Input text is empty after trimming.")

        api_key = _load_minimax_api_key()
        resolved_model = args.model.strip() or defaults["model"]
        resolved_voice_id = args.voice_id.strip() or defaults["voice_id"]
        resolved_speed = args.speed if args.speed >= 0 else defaults["speed"]
        resolved_volume = args.volume if args.volume >= 0 else defaults["volume"]
        resolved_pitch = args.pitch if args.pitch != 999 else defaults["pitch"]
        resolved_sample_rate = args.sample_rate if args.sample_rate > 0 else defaults["sample_rate"]
        resolved_audio_format = args.audio_format.strip() or defaults["audio_format"]

        try:
            audio_bytes = synthesize(
                text=text,
                api_key=api_key,
                model=resolved_model,
                voice_id=resolved_voice_id,
                speed=resolved_speed,
                volume=resolved_volume,
                pitch=resolved_pitch,
                sample_rate=resolved_sample_rate,
                bitrate=args.bitrate,
                audio_format=resolved_audio_format,
                channel=args.channel,
                stream=args.stream,
                timeout_seconds=args.timeout_seconds,
            )
        except RuntimeError as exc:
            message = str(exc)
            should_retry_with_default_voice = (
                "code 2054" in message and resolved_voice_id != DEFAULT_VOICE_ID
            )
            if not should_retry_with_default_voice:
                raise

            print(
                (
                    f"[tts_minimax] Voice '{resolved_voice_id}' is unavailable; "
                    f"retrying with fallback '{DEFAULT_VOICE_ID}'."
                ),
                file=sys.stderr,
            )
            audio_bytes = synthesize(
                text=text,
                api_key=api_key,
                model=resolved_model,
                voice_id=DEFAULT_VOICE_ID,
                speed=resolved_speed,
                volume=resolved_volume,
                pitch=resolved_pitch,
                sample_rate=resolved_sample_rate,
                bitrate=args.bitrate,
                audio_format=resolved_audio_format,
                channel=args.channel,
                stream=args.stream,
                timeout_seconds=args.timeout_seconds,
            )

        output_path = args.output.strip()
        target = Path(output_path) if output_path else _default_output_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(audio_bytes)
        print(f"Saved audio to: {target}", file=sys.stderr)
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[tts_minimax] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
