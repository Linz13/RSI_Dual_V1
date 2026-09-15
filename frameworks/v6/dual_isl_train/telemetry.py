from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .io import append_jsonl


class MetricLogger:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.path = self.directory / "metrics.jsonl"
        self.writer = None

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(self.directory / "tensorboard"))
        except Exception:  # noqa: BLE001
            self.writer = None
        return self

    def log(self, step: int, **values: Any) -> None:
        append_jsonl(self.path, {"step": step, **values})
        if self.writer is not None:
            for key, value in values.items():
                if isinstance(value, (int, float, bool)):
                    self.writer.add_scalar(key, float(value), step)

    def __exit__(self, *_args):
        if self.writer is not None:
            self.writer.close()


class ProgressLogger:
    """Immediately flushed JSONL heartbeat stream for one distributed rank."""

    def __init__(self, directory: str | Path, rank: int):
        self.directory = Path(directory)
        self.rank = int(rank)
        self.path = self.directory / f"rank_{self.rank:03d}.jsonl"

    def log(self, event: str, **values: Any) -> None:
        append_jsonl(self.path, {
            "timestamp": time.time(), "rank": self.rank, "event": event, **values,
        })
