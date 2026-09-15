from __future__ import annotations

from pathlib import Path
from typing import Any

from .constants import FRAMEWORK_VERSION, TRAJECTORY_VERSION
from .io import atomic_json, hash_path


def checkpoint_record(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    target = Path(path).resolve()
    digest = hash_path(target)
    if digest is None:
        raise FileNotFoundError(f"Checkpoint does not exist: {target}")
    return {"path": str(target), "sha256": digest}


def commit_round(
    run_dir: str | Path, round_index: int, *, caption_checkpoint: str, tts_checkpoint: str,
    input_caption_checkpoint: str | None, input_tts_checkpoint: str | None,
) -> dict[str, Any]:
    record = {
        "framework_version": FRAMEWORK_VERSION,
        "trajectory_version": TRAJECTORY_VERSION,
        "round": int(round_index),
        "same_round_start": True,
        "input_captioner": checkpoint_record(input_caption_checkpoint),
        "input_tts": checkpoint_record(input_tts_checkpoint),
        "captioner": checkpoint_record(caption_checkpoint),
        "tts": checkpoint_record(tts_checkpoint),
    }
    root = Path(run_dir)
    atomic_json(root / f"round_{round_index:03d}" / "commit.json", record)
    atomic_json(root / "latest.json", record)
    return record
