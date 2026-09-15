#!/usr/bin/env python3
"""Shared, side-effect-free helpers for the local InstructTTSEval deployment."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


OFFICIAL_COMMIT = "b7e4120c7cee179a3ce1b99819cf13d8e6f199ce"
DATASET_REVISION = "b12cdf288e78cfddb1d1975fd0e05d6fd61ac0d2"
TASKS = ("APS", "DSD", "RP")
LANGUAGES = ("en", "zh")
QWEN_LANGUAGE = {"en": "English", "zh": "Chinese"}
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def validate_sample(sample: dict[str, Any], language: str | None = None) -> None:
    required = ("id", "language", "text", *TASKS)
    missing = [name for name in required if name not in sample]
    if missing:
        raise ValueError(f"Sample is missing fields {missing}: {sample.get('id')!r}")
    sample_id = sample["id"]
    if not isinstance(sample_id, str) or not SAFE_ID.fullmatch(sample_id):
        raise ValueError(f"Unsafe or invalid sample id: {sample_id!r}")
    if sample["language"] not in LANGUAGES:
        raise ValueError(f"Invalid language for {sample_id}: {sample['language']!r}")
    if language is not None and sample["language"] != language:
        raise ValueError(
            f"Language mismatch for {sample_id}: {sample['language']} != {language}"
        )
    for field in ("text", *TASKS):
        if not isinstance(sample[field], str) or not sample[field].strip():
            raise ValueError(f"Empty or non-string {field} for {sample_id}")


def load_samples(manifest_dir: Path, languages: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for language in languages:
        if language not in LANGUAGES:
            raise ValueError(f"Unsupported language: {language}")
        split_rows = read_jsonl(manifest_dir / f"{language}.jsonl")
        for row in split_rows:
            validate_sample(row, language)
            if row["id"] in seen:
                raise ValueError(f"Duplicate sample id: {row['id']}")
            seen.add(row["id"])
        rows.extend(split_rows)
    return rows


def selected_samples(
    rows: list[dict[str, Any]], num_samples_per_language: int | None
) -> list[dict[str, Any]]:
    if num_samples_per_language is None:
        return rows
    if num_samples_per_language <= 0:
        raise ValueError("--num-samples-per-language must be positive")
    selected: list[dict[str, Any]] = []
    for language in LANGUAGES:
        language_rows = [row for row in rows if row["language"] == language]
        selected.extend(language_rows[:num_samples_per_language])
    return selected


def audio_relative_path(language: str, task: str, sample_id: str) -> Path:
    if language not in LANGUAGES or task not in TASKS:
        raise ValueError(f"Invalid audio key: {language}/{task}/{sample_id}")
    if not SAFE_ID.fullmatch(sample_id):
        raise ValueError(f"Unsafe sample id: {sample_id!r}")
    return Path("audios") / language / task / f"{sample_id}.wav"


def record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return str(record["language"]), str(record["id"]), str(record["task"])


def load_latest_records(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not path.is_file():
        return {}
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        if all(name in row for name in ("language", "id", "task")):
            latest[record_key(row)] = row
    return latest


class JsonlCheckpointWriter:
    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def __enter__(self) -> "JsonlCheckpointWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8", buffering=1)
        return self

    def write(self, payload: dict[str, Any]) -> None:
        if self.handle is None:
            raise RuntimeError("Checkpoint writer is not open")
        self.handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.handle is not None:
            self.handle.close()

