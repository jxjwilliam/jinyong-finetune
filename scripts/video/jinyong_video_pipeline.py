from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from scripts.tts import tts_minimax


@dataclass(slots=True)
class PromptTranslationResult:
    video_prompt: str
    image_prompt: str
    recommended_model: str
    camera: str
    mood: str
    duration: int


def _extract_json_object(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("Translator response does not contain a JSON object.")
    return text[start : end + 1]


def parse_translation_result(raw: str) -> PromptTranslationResult:
    obj = json.loads(_extract_json_object(raw))
    if not isinstance(obj, dict):
        raise ValueError("Translation result must be a JSON object.")
    required = ["video_prompt", "image_prompt", "recommended_model", "camera", "mood", "duration"]
    for key in required:
        if key not in obj:
            raise ValueError(f"Missing key in translation result: {key}")
    duration = int(obj["duration"])
    if duration not in (5, 10):
        raise ValueError("duration must be 5 or 10 seconds.")
    return PromptTranslationResult(
        video_prompt=str(obj["video_prompt"]).strip(),
        image_prompt=str(obj["image_prompt"]).strip(),
        recommended_model=str(obj["recommended_model"]).strip(),
        camera=str(obj["camera"]).strip(),
        mood=str(obj["mood"]).strip(),
        duration=duration,
    )


def extract_job_id(body: dict[str, Any]) -> str:
    candidates = [body.get("id"), body.get("job_id")]
    data = body.get("data")
    if isinstance(data, dict):
        candidates.append(data.get("id"))
        candidates.append(data.get("job_id"))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    raise ValueError(f"Could not parse job id from response: {body}")


def extract_video_url(body: dict[str, Any]) -> str:
    output = body.get("output")
    if isinstance(output, dict):
        direct = output.get("url")
        if isinstance(direct, str) and direct.strip():
            return direct
        nested = output.get("video")
        if isinstance(nested, dict):
            nested_url = nested.get("url")
            if isinstance(nested_url, str) and nested_url.strip():
                return nested_url
    direct = body.get("video_url")
    if isinstance(direct, str) and direct.strip():
        return direct
    raise ValueError(f"Could not parse output video url from response: {body}")


def ollama_generate_paragraph(scene_setup: str, *, model: str, base_url: str, timeout_seconds: int = 120) -> str:
    prompt = (
        "你是金庸风格的武侠小说作家。每次只写一段，约180-240字。"
        "文笔典雅，富有画面感，适合转化为视频场景。"
        "不要分点，不要解释，不要现代词汇。\n\n"
        f"场景设定：{scene_setup}"
    )
    resp = requests.post(
        f"{base_url.rstrip('/')}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout_seconds,
    )
    resp.raise_for_status()
    body = resp.json()
    text = str(body.get("response", "")).strip()
    if not text:
        raise RuntimeError(f"Ollama returned empty response: {body}")
    return text


def translate_to_video_prompt(
    paragraph: str,
    *,
    provider: str,
    translator_model: str | None = None,
    timeout_seconds: int = 120,
) -> PromptTranslationResult:
    instruction = (
        "You are a cinematic prompt engineer specializing in wuxia visuals.\n"
        "Convert the Chinese paragraph to JSON with fields:\n"
        "video_prompt, image_prompt, recommended_model, camera, mood, duration.\n"
        "Constraints: recommended_model in [kling_3, sora_2_pro, veo_3_quality, veo_3_fast], "
        "duration in [5, 10], and return JSON only."
    )
    user_content = f"Chinese paragraph:\n{paragraph}"

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is required for anthropic provider.")
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": translator_model or os.getenv("VIDEO_TRANSLATOR_MODEL", "claude-sonnet-4-20250514"),
                "max_tokens": 700,
                "messages": [{"role": "user", "content": f"{instruction}\n\n{user_content}"}],
            },
            timeout=timeout_seconds,
        )
        resp.raise_for_status()
        body = resp.json()
        content = body.get("content", [])
        text = ""
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = str(item.get("text", ""))
                    break
        return parse_translation_result(text)

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is required for openai provider.")
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": translator_model or os.getenv("VIDEO_TRANSLATOR_MODEL", "gpt-4o-mini"),
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": user_content},
                ],
            },
            timeout=timeout_seconds,
        )
        resp.raise_for_status()
        body = resp.json()
        text = body["choices"][0]["message"]["content"]
        return parse_translation_result(text)

    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        model = translator_model or os.getenv("VIDEO_TRANSLATOR_MODEL", "qwen2.5:7b-instruct")
        prompt = f"{instruction}\n\n{user_content}"
        resp = requests.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout_seconds,
        )
        if resp.status_code >= 400:
            detail = resp.text.strip()
            raise RuntimeError(
                "Ollama translator request failed: "
                f"HTTP {resp.status_code} from {base_url}/api/generate. "
                f"Model='{model}'. Body={detail}"
            )
        body = resp.json()
        return parse_translation_result(str(body.get("response", "")))

    raise ValueError(f"Unsupported provider: {provider}")


def generate_video_with_nanobanana(
    prompt_data: PromptTranslationResult,
    output_path: Path,
    *,
    poll_interval_seconds: int = 10,
    timeout_seconds: int = 1800,
) -> Path:
    api_key = os.getenv("NANOBANANA_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("NANOBANANA_API_KEY is missing.")
    base_url = os.getenv("NANOBANANA_BASE_URL", "https://api.nanobanana.io").rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "prompt": prompt_data.video_prompt,
        "model": prompt_data.recommended_model or "kling_3",
        "duration": prompt_data.duration,
        "resolution": "1080p",
        "camera_movement": prompt_data.camera,
    }
    submit_resp = requests.post(
        f"{base_url}/v1/video/generate",
        headers=headers,
        json=payload,
        timeout=120,
    )
    submit_resp.raise_for_status()
    job_id = extract_job_id(submit_resp.json())

    started = time.time()
    while True:
        status_resp = requests.get(f"{base_url}/v1/jobs/{job_id}", headers=headers, timeout=120)
        status_resp.raise_for_status()
        body = status_resp.json()
        status = str(body.get("status", "")).lower()
        if status == "completed":
            video_url = extract_video_url(body)
            download_resp = requests.get(video_url, timeout=300)
            download_resp.raise_for_status()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(download_resp.content)
            return output_path
        if status == "failed":
            raise RuntimeError(f"NanoBanana job failed: {body}")
        if time.time() - started > timeout_seconds:
            raise TimeoutError(f"NanoBanana job timeout after {timeout_seconds}s: {job_id}")
        time.sleep(poll_interval_seconds)


def synthesize_narration(paragraph: str, output_path: Path, config_path: Path) -> Path:
    config = tts_minimax._load_json_config(config_path)
    defaults = tts_minimax._runtime_defaults(config)
    api_key = tts_minimax._load_minimax_api_key()
    audio_bytes = tts_minimax.synthesize(
        text=paragraph,
        api_key=api_key,
        model=str(defaults["model"]),
        voice_id=str(defaults["voice_id"]),
        speed=float(defaults["speed"]),
        volume=float(defaults["volume"]),
        pitch=int(defaults["pitch"]),
        sample_rate=int(defaults["sample_rate"]),
        bitrate=int(defaults["bitrate"]),
        audio_format=str(defaults["audio_format"]),
        channel=int(defaults["channel"]),
        stream=False,
        timeout_seconds=120,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(audio_bytes)
    return output_path


def merge_video_and_audio(video_path: Path, audio_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jin Yong paragraph to video pipeline.")
    parser.add_argument("--scene-setup", type=str, default="", help="Scene setup for Ollama paragraph generation.")
    parser.add_argument("--paragraph", type=str, default="", help="Use an existing Chinese paragraph directly.")
    parser.add_argument("--paragraph-file", type=str, default="", help="Path to paragraph text file.")
    parser.add_argument("--output-dir", type=str, default="outputs/video", help="Directory for generated artifacts.")
    parser.add_argument(
        "--translator-provider",
        type=str,
        default="anthropic",
        choices=["anthropic", "openai", "ollama"],
        help="Provider to translate paragraph into cinematic prompt JSON.",
    )
    parser.add_argument(
        "--translator-model",
        type=str,
        default="",
        help="Model for translator provider (overrides VIDEO_TRANSLATOR_MODEL env).",
    )
    parser.add_argument("--ollama-model", type=str, default="jinyong", help="Local Ollama model for paragraph step.")
    parser.add_argument("--ollama-base-url", type=str, default="http://127.0.0.1:11434", help="Local Ollama base URL.")
    parser.add_argument("--skip-video", action="store_true", help="Skip NanoBanana generation (prompt only).")
    parser.add_argument("--with-audio", action="store_true", help="Generate Minimax narration and mux with clip.")
    parser.add_argument("--tts-config", type=str, default="configs/minimax-t2a.json", help="MiniMax TTS config path.")
    return parser.parse_args()


def _load_paragraph(args: argparse.Namespace) -> str:
    if args.paragraph.strip():
        return args.paragraph.strip()
    if args.paragraph_file:
        return Path(args.paragraph_file).read_text(encoding="utf-8").strip()
    if args.scene_setup.strip():
        return ollama_generate_paragraph(
            args.scene_setup.strip(),
            model=args.ollama_model.strip(),
            base_url=args.ollama_base_url.strip(),
        )
    raise ValueError("Provide --paragraph, --paragraph-file, or --scene-setup.")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paragraph = _load_paragraph(args)
    (output_dir / "paragraph.txt").write_text(paragraph, encoding="utf-8")

    prompt_data = translate_to_video_prompt(
        paragraph,
        provider=args.translator_provider,
        translator_model=args.translator_model.strip() or None,
    )
    (output_dir / "prompt_data.json").write_text(
        json.dumps(asdict(prompt_data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.skip_video:
        print(f"Prompt data written to: {output_dir / 'prompt_data.json'}")
        return 0

    clip_path = generate_video_with_nanobanana(prompt_data, output_dir / "clip.mp4")
    print(f"Video clip written to: {clip_path}")

    if args.with_audio:
        audio_path = synthesize_narration(paragraph, output_dir / "narration.mp3", Path(args.tts_config))
        final_path = merge_video_and_audio(clip_path, audio_path, output_dir / "final_with_narration.mp4")
        print(f"Final narrated video written to: {final_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

