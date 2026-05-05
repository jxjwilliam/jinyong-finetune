from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert merged HF model to GGUF for Ollama deployment.")
    parser.add_argument("--config", default="configs/qlora_config.yaml", help="Path to config file.")
    parser.add_argument("--model-dir", default=None, help="Path to merged HF model directory.")
    parser.add_argument("--llama-cpp-dir", default=None, help="Path to llama.cpp directory (contains convert_hf_to_gguf.py).")
    parser.add_argument(
        "--quantize",
        default="f16",
        help="convert_hf_to_gguf --outtype (default f16 per docs/LORA_TO_GGUF_GUIDE.md; use q4_k_m only if you skip llama-quantize).",
    )
    parser.add_argument("--skip-convert", action="store_true", help="Skip conversion, only build Ollama Modelfile.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))

    model_dir = Path(args.model_dir) if args.model_dir else Path(config["training"]["output_dir"]) / "merged"
    if not model_dir.exists():
        raise FileNotFoundError(f"Merged model directory not found: {model_dir}")

    llama_cpp_dir = Path(args.llama_cpp_dir) if args.llama_cpp_dir else Path.home() / "llama.cpp"
    if not llama_cpp_dir.exists():
        raise FileNotFoundError(
            f"llama.cpp not found at {llama_cpp_dir}. Clone with: git clone https://github.com/ggerganov/llama.cpp.git"
        )

    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found in {llama_cpp_dir}")

    gguf_output = model_dir / f"jinyong-qwen2.5-7b-{args.quantize}.gguf"

    if not args.skip_convert:
        print(f"Converting {model_dir} to GGUF ({args.quantize})...")
        subprocess.run(
            [
                "python3", str(convert_script),
                str(model_dir),
                "--outtype", args.quantize,
                "--outfile", str(gguf_output),
            ],
            check=True,
        )
        print(f"GGUF model saved to: {gguf_output}")

    modelfile = model_dir / "Modelfile"
    with open(modelfile, "w", encoding="utf-8") as f:
        f.write(f"FROM ./{gguf_output.name}\n")
        f.write(f"SYSTEM \"{config['data'].get('system_prompt', '你是一位精通金庸武侠风格的写作助手。')}\"\n")
    print(f"Ollama Modelfile created at: {modelfile}")
    print(f"To build Ollama model, run: ollama create jinyong-qwen -f {modelfile}")


if __name__ == "__main__":
    main()
