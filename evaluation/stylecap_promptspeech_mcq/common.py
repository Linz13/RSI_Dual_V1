#!/usr/bin/env python3
"""Shared constants and helpers for the StyleCap/PromptSpeech MCQ benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
SOURCES_DIR = ROOT / "sources"
DOWNLOADS_DIR = SOURCES_DIR / "downloads"
DATA_DIR = ROOT / "data"
AUDIO_DIR = ROOT / "audio"

PROMPTSPEECH_URL = (
    "https://speechresearch.github.io/dataset/promptspeech/Real_training.zip"
)
PROMPTSPEECH_SHA256 = (
    "f1a250714495d8992ebce8a9ce3cc24207fc5328aaf9bbf780bfc1e063dae87d"
)
STYLECAP_SPLIT_URL = (
    "https://ntt-hilab-gensp.github.io/icassp2024stylecap/"
    "train_dev_test_set.zip"
)
STYLECAP_SPLIT_SHA256 = (
    "ddaa37bf855ae59bf40a5a48dcb7021ff4d75b42556d4c2515c69b8bb8c3efb4"
)
LIBRITTS_TEST_CLEAN_URL = "https://www.openslr.org/resources/60/test-clean.tar.gz"
LIBRITTS_TEST_CLEAN_MD5 = "7bed3bdb047c4c197f1ad3bc412db59f"
LIBRITTS_TEST_CLEAN_BYTES = 1_230_670_113

PROMPTSPEECH_CSV_SHA256 = (
    "816e51ea540f1aca6d5e309d9bbcd60b9b2ae93314999ccf70c70a3f2878c793"
)
STYLECAP_CSV_SHA256 = {
    "train": "dd34b3b36bef7d3cb31a5122692af57462fe1764144a635dbe4db0cff9de67f2",
    "dev": "9fbffac395dde95bafb57e3de2ca5d9b69ab803f9fa6677248d2d35190368248",
    "test": "e7204ad2cbc243cffac48f77fdfd43d6fc3f84181ec287efe07334967fc5ddd6",
}
EXPECTED_SPLITS = {
    "train": {"rows": 24_953, "speakers": 1_113},
    "dev": {"rows": 857, "speakers": 40},
    "test": {"rows": 778, "speakers": 38},
}
EXPECTED_METADATA_ROWS = 26_588
EXPECTED_QUESTIONS = 3_112

PROMPTSPEECH_CSV = SOURCES_DIR / "Real_training.csv"
STYLECAP_CSV = {
    split: SOURCES_DIR / f"stylecap_{split}_set.csv"
    for split in EXPECTED_SPLITS
}
TEST_AUDIO_JSONL = DATA_DIR / "test_audio.jsonl"
BENCHMARK_JSONL = DATA_DIR / "benchmark.jsonl"
SUMMARY_JSON = DATA_DIR / "summary.json"
SOURCE_MANIFEST_JSON = SOURCES_DIR / "source_manifest.json"

TASK_ORDER = ("gender", "pitch", "speaking_speed", "volume")
TASK_SPECS: dict[str, dict[str, Any]] = {
    "gender": {
        "question": "请听这段语音。说话者的性别是？",
        "choices": {"A": "Male", "B": "Female"},
        "labels": ("male", "female"),
    },
    "pitch": {
        "question": "请听这段语音。说话者整体的音高属于哪一类？",
        "choices": {"A": "Low", "B": "Normal", "C": "High"},
        "labels": ("low", "normal", "high"),
    },
    "speaking_speed": {
        "question": "请听这段语音。整体语速属于哪一类？",
        "choices": {"A": "Slow", "B": "Normal", "C": "Fast"},
        "labels": ("slow", "normal", "fast"),
    },
    "volume": {
        "question": "请听这段语音。整体说话音量属于哪一类？",
        "choices": {"A": "Low", "B": "Normal", "C": "High"},
        "labels": ("low", "normal", "high"),
    },
}
EXPECTED_DISTRIBUTIONS = {
    "gender": {"male": 266, "female": 512},
    "pitch": {"low": 266, "normal": 285, "high": 227},
    "speaking_speed": {"slow": 278, "normal": 263, "fast": 237},
    "volume": {"low": 318, "normal": 234, "high": 226},
}


def digest_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_bytes(data: bytes, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    digest.update(data)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    atomic_write_bytes(path, payload.encode("utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n" for row in rows
    )
    atomic_write_bytes(path, payload.encode("utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def speaker_id_from_audio_id(audio_id: str) -> str:
    speaker_id, separator, _ = audio_id.partition("_")
    if not separator or not speaker_id:
        raise ValueError(f"invalid LibriTTS audio ID: {audio_id!r}")
    return speaker_id


def normalize_metadata_row(row: dict[str, str]) -> dict[str, str]:
    required = ("item_name", "spk_id", "gender", "pitch", "speaking_speed", "energy")
    missing = [field for field in required if not str(row.get(field, "")).strip()]
    if missing:
        raise ValueError(f"metadata row is missing fields {missing}: {row}")
    gender = {"M": "male", "F": "female"}.get(row["gender"].strip())
    if gender is None:
        raise ValueError(f"unsupported PromptSpeech gender: {row['gender']!r}")
    attributes = {
        "gender": gender,
        "pitch": row["pitch"].strip(),
        "speaking_speed": row["speaking_speed"].strip(),
        # PromptSpeech names this acoustic field "energy". StyleCap calls the
        # corresponding categorical factor "volume".
        "volume": row["energy"].strip(),
    }
    for task, label in attributes.items():
        if label not in TASK_SPECS[task]["labels"]:
            raise ValueError(f"unsupported {task} label {label!r} for {row['item_name']}")
    return attributes


def answer_for_label(task: str, label: str) -> str:
    spec = TASK_SPECS[task]
    for answer, choice in spec["choices"].items():
        if choice.casefold() == label.casefold():
            return answer
    raise ValueError(f"no answer mapping for {task}={label!r}")


def make_question_records(audio: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task in TASK_ORDER:
        label = audio["attributes"][task]
        spec = TASK_SPECS[task]
        records.append(
            {
                "question_id": f"stylecap-{audio['audio_id']}-{task}",
                "audio_id": audio["audio_id"],
                "speaker_id": audio["speaker_id"],
                "audio_path": audio["audio_path"],
                "relative_audio_path": audio["relative_audio_path"],
                "task": task,
                "question": spec["question"],
                "choices": dict(spec["choices"]),
                "answer": answer_for_label(task, label),
                "label": label,
            }
        )
    return records


def distributions(audio_rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counters = {task: Counter() for task in TASK_ORDER}
    for row in audio_rows:
        for task in TASK_ORDER:
            counters[task][row["attributes"][task]] += 1
    return {
        task: {label: counters[task][label] for label in TASK_SPECS[task]["labels"]}
        for task in TASK_ORDER
    }
