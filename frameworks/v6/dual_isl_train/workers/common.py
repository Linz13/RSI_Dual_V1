from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from dual_isl_train.config import load_config
from dual_isl_train.constants import CHECKPOINT_METADATA, FRAMEWORK_VERSION, TRAJECTORY_VERSION
from dual_isl_train.io import atomic_json, read_jsonl, write_jsonl

WORKER_STARTED = time.perf_counter()


def worker_parser(description: str, actions: list[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("action", choices=actions)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-in", default="")
    parser.add_argument("--target-checkpoint", default="")
    parser.add_argument("--checkpoint-out", default="")
    parser.add_argument("--training-phase", default="")
    return parser


def load_job(args: argparse.Namespace, *, initialize_torch: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = load_config(args.config)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    seed = int(config["run"].get("seed", 42)) + local_rank
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    if initialize_torch:
        try:
            import torch
            if int(os.environ.get("WORLD_SIZE", "1")) > 1 and torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            if torch.cuda.is_available():
                from dual_isl_train.config_v6 import set_gpu_budget
                set_gpu_budget(torch, local_rank, config.get("gpu_memory_gib"))
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        except ImportError:
            pass
    return config, list(read_jsonl(args.input))


def save_checkpoint_marker(path: str, role: str, checkpoint_in: str, details: dict[str, Any] | None = None) -> None:
    if not path:
        return
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    (target / "dual_isl_train_checkpoint.json").write_text(json.dumps({
        "framework_version": FRAMEWORK_VERSION, "trajectory_version": TRAJECTORY_VERSION,
        "role": role, "parent": checkpoint_in or None, **(details or {}),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_checkpoint_version(path: str | Path) -> None:
    checkpoint = Path(path)
    if not checkpoint:
        return
    metadata = checkpoint / CHECKPOINT_METADATA
    mock_metadata = checkpoint / "dual_isl_train_checkpoint.json"
    selected = metadata if metadata.is_file() else mock_metadata if mock_metadata.is_file() else None
    if selected is None:
        raise RuntimeError(f"Checkpoint {checkpoint} is not a DualISL-Train checkpoint")
    value = json.loads(selected.read_text(encoding="utf-8"))
    if value.get("framework_version") != FRAMEWORK_VERSION:
        raise RuntimeError(
            f"Checkpoint {checkpoint} has framework_version={value.get('framework_version')!r}; "
            f"expected {FRAMEWORK_VERSION!r}"
        )
    if value.get("trajectory_version") != TRAJECTORY_VERSION:
        raise RuntimeError(
            f"Checkpoint {checkpoint} has trajectory_version={value.get('trajectory_version')!r}; "
            f"expected {TRAJECTORY_VERSION!r}"
        )


def write_output(path: str, rows: list[dict[str, Any]], extra_metrics: dict[str, Any] | None = None) -> None:
    write_jsonl(path, rows)
    torch = sys.modules.get("torch")
    peak_gpu_bytes = int(torch.cuda.max_memory_allocated()) if torch is not None and torch.cuda.is_available() else 0
    atomic_json(str(path) + ".metrics.json", {
        "elapsed_seconds": time.perf_counter() - WORKER_STARTED,
        "gpu_peak_memory_bytes": peak_gpu_bytes,
        "output_rows": len(rows),
        **(extra_metrics or {}),
    })

