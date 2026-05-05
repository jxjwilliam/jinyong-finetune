#!/usr/bin/env python3
"""Upload the Jin Yong raw text bundle to a Hugging Face *dataset* repo.

Default source is local ``data/raw`` (same files as the Kaggle dataset). Optional
``--from-kaggle`` downloads ``evilpsycho42/jinyong-wuxia`` into a temp dir first.

Requires: pip install huggingface_hub pyyaml
Optional Kaggle path: pip install kaggle and ``~/.kaggle/kaggle.json``

Auth: huggingface-cli login   or   export HF_TOKEN=...

Run from repo root:
  python scripts/hub/upload_raw_corpus_hf.py --dry-run
  python scripts/hub/upload_raw_corpus_hf.py --repo-id jxjwilliam/jinyong-wuxia
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

KAGGLE_DATASET = "evilpsycho42/jinyong-wuxia"
KAGGLE_URL = "https://www.kaggle.com/datasets/evilpsycho42/jinyong-wuxia"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dataset_readme_markdown(*, repo_id: str) -> str:
    return f"""---
language:
- zh
tags:
- wuxia
- jin-yong
- chinese
- corpus
license: other
size_categories:
- 10M<n<100M
---

# Jin Yong wuxia raw text bundle

UTF-8 text files used for cleaning and instruction building in the Jin Yong fine-tuning pipeline.

## Provenance

The file set matches the public Kaggle dataset **[{KAGGLE_DATASET}]({KAGGLE_URL})** (same layout as this repo’s ``data/raw`` when synced from Kaggle).

## Loading from the Hub

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="{repo_id}", filename="jinyong.txt", repo_type="dataset")
```

Or clone the dataset repo with ``git`` / the Hugging Face CLI.

## License and use

Original Jin Yong (金庸) novels are copyrighted. Use this corpus only where your jurisdiction and the **Kaggle dataset license** allow. This Hub copy is for **research and training reproducibility**; it does not grant additional rights beyond those of the upstream source.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Upload raw corpus text files to a Hugging Face dataset repo.",
    )
    p.add_argument(
        "--config",
        default="configs/qlora_config.yaml",
        help="YAML config (uses data.raw_txt_dir when --folder is omitted).",
    )
    p.add_argument(
        "--folder",
        default=None,
        help="Override local folder to upload (default: data.raw_txt_dir from config).",
    )
    p.add_argument(
        "--repo-id",
        default="jxjwilliam/jinyong-wuxia",
        help="Target HF dataset repo id (namespace/name).",
    )
    p.add_argument(
        "--from-kaggle",
        action="store_true",
        help=f"Download {KAGGLE_DATASET} with the Kaggle API, then upload (ignores local --folder).",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private (only when the repo is first created).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved paths and exit (no download, no Hub API).",
    )
    p.add_argument(
        "--commit-message",
        default="Upload Jin Yong wuxia raw text bundle (Kaggle-aligned)",
        help="Git commit message for the Hub upload.",
    )
    return p.parse_args()


def resolve_local_folder(root: Path, args: argparse.Namespace) -> Path:
    if args.folder:
        p = Path(args.folder)
        return p.resolve() if p.is_absolute() else (root / p).resolve()
    config_path = root / args.config if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(config_path)
    data_cfg = cfg.get("data") or {}
    rel = data_cfg.get("raw_txt_dir", "data/raw")
    return (root / rel).resolve()


def copy_corpus_tree(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise FileNotFoundError(f"Not a directory: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.iterdir()):
        if path.is_file():
            shutil.copy2(path, dst / path.name)
        elif path.is_dir():
            shutil.copytree(path, dst / path.name, dirs_exist_ok=True)


def normalize_kaggle_download_root(tmp: Path) -> Path:
    """Kaggle may unzip flat or into a single subfolder; prefer a dir that has .txt files."""
    if any(tmp.glob("*.txt")):
        return tmp
    for child in sorted(tmp.iterdir()):
        if child.is_dir() and any(child.glob("*.txt")):
            return child
    return tmp


def download_kaggle_corpus() -> Path:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise RuntimeError("pip install kaggle and configure ~/.kaggle/kaggle.json") from e

    tmp = Path(tempfile.mkdtemp(prefix="jinyong-kaggle-"))
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(tmp), unzip=True)
    return tmp


def main() -> int:
    args = parse_args()
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Missing dependency: pip install huggingface_hub", file=sys.stderr)
        return 1

    root = Path(__file__).resolve().parents[2]
    kaggle_tmp: Path | None = None

    if args.from_kaggle:
        if args.dry_run:
            print(f"Would download Kaggle dataset {KAGGLE_DATASET} to a temp dir, then upload.")
            print(f"HF dataset repo: {args.repo_id}")
            return 0
        try:
            kaggle_tmp = download_kaggle_corpus()
            source_dir = normalize_kaggle_download_root(kaggle_tmp)
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Kaggle download failed: {e}", file=sys.stderr)
            return 1
    else:
        source_dir = resolve_local_folder(root, args)

    if not source_dir.is_dir():
        print(f"Corpus directory not found: {source_dir}", file=sys.stderr)
        return 1
    txt_files = list(source_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files under {source_dir}", file=sys.stderr)
        return 1

    print(f"Source: {source_dir} ({len(txt_files)} top-level .txt file(s))")
    print(f"Repo: {args.repo_id} (repo_type=dataset)")

    if args.dry_run:
        print("Dry run: no upload.")
        return 0

    with tempfile.TemporaryDirectory(prefix="jinyong-hf-dataset-") as staging_str:
        staging = Path(staging_str)
        copy_corpus_tree(source_dir, staging)
        (staging / "README.md").write_text(
            dataset_readme_markdown(repo_id=args.repo_id),
            encoding="utf-8",
        )
        print(f"Staged upload: {staging} ({len(list(staging.iterdir()))} entries)")

        api = HfApi()
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=str(staging),
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=args.commit_message,
        )

    if kaggle_tmp is not None:
        shutil.rmtree(kaggle_tmp, ignore_errors=True)

    print(f"Done: https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
