#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPO_URL = "https://github.com/NKU-HLT/EmotionTalk.git"
DEFAULT_COMMIT = "cb8397e226ce7c1fccee41ea21161a3f98f578e1"
DEFAULT_HF_REVISION = "adbc17fc944e8cf2873643906160c6ca0259ab61"


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official EmotionTalk speech/text sources.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--github-commit", default=DEFAULT_COMMIT)
    parser.add_argument("--hf-revision", default=DEFAULT_HF_REVISION)
    args = parser.parse_args()
    source = args.root.resolve() / "source"
    repo = source / "official_repo"
    hf = source / "hf"
    source.mkdir(parents=True, exist_ok=True)
    hf.mkdir(parents=True, exist_ok=True)

    if not (repo / ".git").exists():
        run(["git", "clone", "--filter=blob:none", "--no-checkout", REPO_URL, str(repo)])
    run(["git", "config", "--global", "--add", "safe.directory", str(repo)])
    run(["git", "fetch", "origin", args.github_commit], cwd=repo)
    run(["git", "sparse-checkout", "init", "--cone"], cwd=repo)
    run(["git", "sparse-checkout", "set", "EmotionTalk/dataset/mm-process"], cwd=repo)
    run(["git", "checkout", "--detach", args.github_commit], cwd=repo)

    run([
        "hf", "download", "BAAI/Emotiontalk", "Audio.tar", "Text.tar",
        "--repo-type", "dataset", "--revision", args.hf_revision,
        "--local-dir", str(hf),
    ])
    print("Sources downloaded. Run scripts/prepare_benchmark.py next.")


if __name__ == "__main__":
    main()
