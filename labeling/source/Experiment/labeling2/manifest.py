from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Sample:
    sample_id: str
    audio_path: str
    dataset: str = ""
    metadata: dict[str, Any] | None = None
    transcript: str | None = None
    language_hint: str | None = None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: manifest row must be an object")
            rows.append(value)
    return rows


def load_manifest(path: Path, *, check_audio: bool = True) -> list[Sample]:
    samples: list[Sample] = []
    seen: set[str] = set()
    for index, row in enumerate(read_jsonl(path), 1):
        sample_id = str(row.get("sample_id", "")).strip()
        audio = Path(str(row.get("audio_path", "")).strip()).expanduser()
        if not sample_id or not str(row.get("audio_path", "")).strip():
            raise ValueError(f"{path}:{index}: sample_id and audio_path are required")
        if not audio.is_absolute():
            audio = path.parent / audio
        audio = audio.resolve()
        if sample_id in seen:
            raise ValueError(f"{path}:{index}: duplicate sample_id={sample_id!r}")
        if check_audio and not audio.is_file():
            raise FileNotFoundError(f"{path}:{index}: audio not found: {audio}")
        metadata = row.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError(f"{path}:{index}: metadata must be an object")
        samples.append(Sample(
            sample_id=sample_id,
            audio_path=str(audio),
            dataset=str(row.get("dataset", "")),
            metadata=metadata or {},
            transcript=(str(row["transcript"]).strip() if row.get("transcript") is not None else None),
            language_hint=(str(row["language_hint"]).strip() if row.get("language_hint") else None),
        ))
        seen.add(sample_id)
    if not samples:
        raise ValueError(f"Manifest is empty: {path}")
    return samples


def select_samples(samples: list[Sample], start_index: int = 0, max_samples: int = 0) -> list[Sample]:
    if start_index < 0 or max_samples < 0:
        raise ValueError("start_index and max_samples must be >= 0")
    selected = samples[start_index:]
    return selected[:max_samples] if max_samples else selected


def sample_json(sample: Sample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id, "audio_path": sample.audio_path, "dataset": sample.dataset,
        "metadata": sample.metadata or {}, "transcript": sample.transcript,
        "language_hint": sample.language_hint,
    }
