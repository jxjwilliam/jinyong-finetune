from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stitch multiple mp4 clips into one video via ffmpeg concat.")
    parser.add_argument("--input-dir", type=str, default="outputs/video", help="Directory containing clip_*.mp4 files.")
    parser.add_argument("--pattern", type=str, default="clip_*.mp4", help="Glob pattern for clip files.")
    parser.add_argument("--output", type=str, default="outputs/video/full_scene.mp4", help="Output mp4 path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    clips = sorted(input_dir.glob(args.pattern))
    if not clips:
        raise FileNotFoundError(f"No clips found in {input_dir} with pattern {args.pattern}")

    concat_file = input_dir / "clips_concat.txt"
    concat_lines = [f"file '{clip.resolve()}'" for clip in clips]
    concat_file.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c",
        "copy",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    print(f"Stitched {len(clips)} clip(s) into: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

