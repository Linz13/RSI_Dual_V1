#!/usr/bin/env python3
"""Read-only deployment and identity checks for InstructTTSEval."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

from common import (
    DATASET_REVISION,
    LANGUAGES,
    OFFICIAL_COMMIT,
    load_samples,
    sha256_file,
)


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
SHARED_BENCHMARK_ROOT = BENCHMARK_ROOT.parent
CAPTION_ROOT = SHARED_BENCHMARK_ROOT.parent
sys.path.insert(0, str(SHARED_BENCHMARK_ROOT))
from model_adapter_utils import describe_adapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, action="append", default=[])
    parser.add_argument(
        "--manifest-dir", type=Path, default=BENCHMARK_ROOT / "data/manifests"
    )
    parser.add_argument(
        "--data-metadata", type=Path, default=BENCHMARK_ROOT / "data/metadata.json"
    )
    return parser.parse_args()


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def main() -> int:
    args = parse_args()
    for package in ("qwen-tts", "torch", "soundfile", "peft", "numpy", "tqdm"):
        print(f"[CHECK] {package}={package_version(package)}")
    commit = subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={BENCHMARK_ROOT}",
            f"--git-dir={BENCHMARK_ROOT / '.git'}",
            f"--work-tree={BENCHMARK_ROOT}",
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(f"Official commit mismatch: {commit} != {OFFICIAL_COMMIT}")
    print(f"[CHECK] official_commit={commit}")

    metadata = json.loads(args.data_metadata.read_text(encoding="utf-8"))
    if metadata.get("dataset_revision") != DATASET_REVISION:
        raise RuntimeError("Dataset revision mismatch")
    samples = load_samples(args.manifest_dir.resolve(), LANGUAGES)
    language_counts = {
        language: sum(row["language"] == language for row in samples)
        for language in LANGUAGES
    }
    if language_counts != {"en": 1000, "zh": 1000}:
        raise RuntimeError(f"Unexpected split counts: {language_counts}")
    for language in LANGUAGES:
        source = Path(metadata["source"][language]["path"])
        manifest = Path(metadata["manifests"][language]["path"])
        if sha256_file(source) != metadata["source"][language]["sha256"]:
            raise RuntimeError(f"Source hash mismatch: {source}")
        if sha256_file(manifest) != metadata["manifests"][language]["sha256"]:
            raise RuntimeError(f"Manifest hash mismatch: {manifest}")
    print(f"[CHECK] dataset_revision={DATASET_REVISION} counts={language_counts}")

    model_path = args.model_path.resolve()
    for name in ("config.json", "model.safetensors"):
        path = model_path / name
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(path)
    print(f"[CHECK] base_model={model_path}")
    for adapter_dir in args.adapter_dir:
        descriptor = describe_adapter(adapter_dir, model_path)
        print(
            f"[CHECK] adapter={descriptor['path']} "
            f"sha256={descriptor['weights_sha256']}"
        )
    print("[DONE] Repository, data, base model, and adapters passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
