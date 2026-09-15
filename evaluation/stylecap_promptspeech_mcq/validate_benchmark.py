#!/usr/bin/env python3
"""Strictly validate the official split, audio manifest, and generated MCQs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import (
    BENCHMARK_JSONL,
    EXPECTED_DISTRIBUTIONS,
    EXPECTED_METADATA_ROWS,
    EXPECTED_QUESTIONS,
    EXPECTED_SPLITS,
    PROMPTSPEECH_CSV,
    ROOT,
    STYLECAP_CSV,
    SUMMARY_JSON,
    TASK_ORDER,
    TASK_SPECS,
    TEST_AUDIO_JSONL,
    answer_for_label,
    distributions,
    read_csv,
    read_jsonl,
    speaker_id_from_audio_id,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_sources() -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    metadata_rows = read_csv(PROMPTSPEECH_CSV)
    require(
        len(metadata_rows) == EXPECTED_METADATA_ROWS,
        f"PromptSpeech metadata rows: expected {EXPECTED_METADATA_ROWS}, got {len(metadata_rows)}",
    )
    metadata = {row["item_name"]: row for row in metadata_rows}
    require(len(metadata) == len(metadata_rows), "duplicate PromptSpeech item_name")

    split_ids: dict[str, set[str]] = {}
    split_speakers: dict[str, set[str]] = {}
    test_rows: list[dict[str, str]] = []
    for split, expected in EXPECTED_SPLITS.items():
        rows = read_csv(STYLECAP_CSV[split])
        ids = [row["file_id"] for row in rows]
        speakers = {speaker_id_from_audio_id(audio_id) for audio_id in ids}
        require(len(rows) == expected["rows"], f"{split}: expected {expected['rows']} rows")
        require(len(ids) == len(set(ids)), f"{split}: duplicate file_id")
        require(
            len(speakers) == expected["speakers"],
            f"{split}: expected {expected['speakers']} speakers, got {len(speakers)}",
        )
        split_ids[split] = set(ids)
        split_speakers[split] = speakers
        if split == "test":
            test_rows = rows

    require(not (split_ids["train"] & split_ids["dev"]), "train/dev ID overlap")
    require(not (split_ids["train"] & split_ids["test"]), "train/test ID overlap")
    require(not (split_ids["dev"] & split_ids["test"]), "dev/test ID overlap")
    require(not (split_speakers["train"] & split_speakers["dev"]), "train/dev speaker overlap")
    require(not (split_speakers["train"] & split_speakers["test"]), "train/test speaker overlap")
    require(not (split_speakers["dev"] & split_speakers["test"]), "dev/test speaker overlap")
    union = split_ids["train"] | split_ids["dev"] | split_ids["test"]
    require(union == set(metadata), "StyleCap split union does not equal PromptSpeech metadata IDs")

    for row in test_rows:
        audio_id = row["file_id"]
        require(audio_id in metadata, f"unmatched test ID: {audio_id}")
        speaker = speaker_id_from_audio_id(audio_id)
        require(metadata[audio_id]["spk_id"] == speaker, f"speaker mismatch: {audio_id}")
        expected_suffix = f"/{speaker}/"
        relative = row["relative_path_within_libritts"]
        require(relative.startswith("LibriTTS/test-clean/"), f"non-test-clean path: {relative}")
        require(expected_suffix in relative, f"speaker absent from path: {relative}")
        require(relative.endswith(f"/{audio_id}.wav"), f"audio ID/path mismatch: {relative}")
    return metadata, test_rows


def validate_audio_manifest(test_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    audio_rows = read_jsonl(TEST_AUDIO_JSONL)
    require(len(audio_rows) == 778, f"expected 778 audio rows, got {len(audio_rows)}")
    require(len({row["audio_id"] for row in audio_rows}) == 778, "duplicate audio_id")
    require(len({row["speaker_id"] for row in audio_rows}) == 38, "expected 38 speakers")
    require(
        [row["audio_id"] for row in audio_rows] == [row["file_id"] for row in test_rows],
        "test_audio.jsonl does not preserve official StyleCap test order",
    )
    expected_paths = {
        row["file_id"]: row["relative_path_within_libritts"] for row in test_rows
    }
    seen_paths: set[str] = set()
    for row in audio_rows:
        audio_id = row["audio_id"]
        require(row["speaker_id"] == speaker_id_from_audio_id(audio_id), f"speaker mismatch: {audio_id}")
        require(set(row["attributes"]) == set(TASK_ORDER), f"incomplete attributes: {audio_id}")
        for task in TASK_ORDER:
            require(
                row["attributes"][task] in TASK_SPECS[task]["labels"],
                f"invalid {task} label for {audio_id}",
            )
        require(
            row["source_relative_audio_path"] == expected_paths[audio_id],
            f"official path mismatch: {audio_id}",
        )
        path = Path(row["audio_path"])
        require(path.is_absolute(), f"audio_path is not absolute: {path}")
        require(path.is_file(), f"missing audio: {path}")
        require(path.stat().st_size > 0, f"empty audio: {path}")
        expected_local = (ROOT / row["relative_audio_path"]).resolve()
        require(path.resolve() == expected_local, f"absolute/relative path mismatch: {audio_id}")
        require(
            path.as_posix().endswith(expected_paths[audio_id]),
            f"local/official path mismatch: {audio_id}",
        )
        require(str(path) not in seen_paths, f"duplicate local audio path: {path}")
        seen_paths.add(str(path))
    require(distributions(audio_rows) == EXPECTED_DISTRIBUTIONS, "attribute distributions changed")
    return audio_rows


def validate_questions(audio_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    questions = read_jsonl(BENCHMARK_JSONL)
    require(len(questions) == EXPECTED_QUESTIONS, f"expected 3112 questions, got {len(questions)}")
    require(
        len({row["question_id"] for row in questions}) == EXPECTED_QUESTIONS,
        "duplicate question_id",
    )
    by_audio: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in questions:
        by_audio[row["audio_id"]].append(row)
    require(set(by_audio) == {row["audio_id"] for row in audio_rows}, "question/audio ID mismatch")

    expected_by_audio = {row["audio_id"]: row for row in audio_rows}
    for audio in audio_rows:
        audio_id = audio["audio_id"]
        rows = by_audio[audio_id]
        require(len(rows) == 4, f"{audio_id}: expected four questions")
        require([row["task"] for row in rows] == list(TASK_ORDER), f"{audio_id}: task order changed")
        for row in rows:
            task = row["task"]
            label = audio["attributes"][task]
            require(row["question_id"] == f"stylecap-{audio_id}-{task}", "question ID mismatch")
            require(row["speaker_id"] == audio["speaker_id"], "question speaker mismatch")
            require(row["audio_path"] == audio["audio_path"], "question audio path mismatch")
            require(row["relative_audio_path"] == audio["relative_audio_path"], "relative path mismatch")
            require(row["question"] == TASK_SPECS[task]["question"], "question text mismatch")
            require(row["choices"] == TASK_SPECS[task]["choices"], "choice table mismatch")
            require(row["label"] == label, "question label mismatch")
            require(row["answer"] == answer_for_label(task, label), "answer/label mismatch")
            require(
                row["choices"][row["answer"]].casefold() == row["label"].casefold(),
                "answer does not select label",
            )
    return questions


def main() -> None:
    _, test_rows = validate_sources()
    audio_rows = validate_audio_manifest(test_rows)
    questions = validate_questions(audio_rows)
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    require(summary["audio_count"] == 778, "summary audio_count mismatch")
    require(summary["speaker_count"] == 38, "summary speaker_count mismatch")
    require(summary["question_count"] == 3112, "summary question_count mismatch")
    require(summary["distributions"] == EXPECTED_DISTRIBUTIONS, "summary distributions mismatch")
    print("Validation passed")
    print(f"  Test audio: {len(audio_rows)}")
    print(f"  Speakers: {len({row['speaker_id'] for row in audio_rows})}")
    print(f"  Questions: {len(questions)}")
    for task in TASK_ORDER:
        print(f"  {task}: {EXPECTED_DISTRIBUTIONS[task]}")


if __name__ == "__main__":
    main()
