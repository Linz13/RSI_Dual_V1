#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from common import ROOT, TASKS, read_jsonl, stable_hash


FORBIDDEN_INFERENCE_FIELDS = {
    "transcript", "speaker_caption", "style_caption", "emotion_caption",
    "overall_captions", "reference", "references", "caption", "captions",
}


def split_ids(npz_path: Path, key: str) -> list[str]:
    with np.load(npz_path, allow_pickle=True) as archive:
        corpus = archive[key].item()
    return list(corpus)


def nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def split_matches(candidate_ids: list[str], official_ids: list[str]) -> bool:
    return candidate_ids == official_ids and set(candidate_ids) == set(official_ids)


def validate(root: Path, check_audio: bool = True) -> dict:
    errors: list[str] = []
    references_path = root / "data/test_references.jsonl"
    inference_path = root / "data/test_inference.jsonl"
    provenance_path = root / "data/provenance.json"
    refs = read_jsonl(references_path)
    requests = read_jsonl(inference_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    npz = root / "source/official_repo/EmotionTalk/dataset/mm-process/audio_label.npz"
    official_train = split_ids(npz, "train_corpus")
    official_val = split_ids(npz, "val_corpus")
    official_test = split_ids(npz, "test_corpus")

    if len(refs) != 1929:
        errors.append(f"references count={len(refs)}, expected 1929")
    if len(requests) != 7716:
        errors.append(f"inference count={len(requests)}, expected 7716")
    ids = [row.get("id") for row in refs]
    if len(ids) != len(set(ids)):
        errors.append("reference IDs are not unique")
    if not split_matches(ids, official_test):
        errors.append("reference IDs or order differ from official test_corpus")
    if provenance.get("split_ordered_id_sha256") != stable_hash(official_test):
        errors.append("provenance split ID hash mismatch")

    sessions = Counter(str(uid).split("_")[0] for uid in ids)
    if dict(sessions) != {"G00003": 793, "G00015": 1136}:
        errors.append(f"unexpected test session composition: {dict(sessions)}")
    train_speakers = {uid.split("_")[2] for uid in official_train}
    val_speakers = {uid.split("_")[2] for uid in official_val}
    test_speakers = {uid.split("_")[2] for uid in official_test}
    if train_speakers & test_speakers:
        errors.append(f"train/test speaker overlap: {sorted(train_speakers & test_speakers)}")
    if provenance.get("speaker_overlap", {}).get("validation_test") != sorted(val_speakers & test_speakers):
        errors.append("provenance validation/test speaker overlap does not match official IDs")

    request_groups: dict[str, list[dict]] = defaultdict(list)
    expected_fields = {"id", "task", "audio_path", "prompt"}
    for index, row in enumerate(requests, 1):
        extra_forbidden = FORBIDDEN_INFERENCE_FIELDS & row.keys()
        if extra_forbidden:
            errors.append(f"inference line {index} leaks fields: {sorted(extra_forbidden)}")
        if set(row) != expected_fields:
            errors.append(f"inference line {index} fields={sorted(row)}, expected={sorted(expected_fields)}")
        request_groups[str(row.get("id"))].append(row)
    if set(request_groups) != set(official_test):
        errors.append("inference ID set differs from official test_corpus")
    for uid in official_test:
        tasks = [row.get("task") for row in request_groups.get(uid, [])]
        if Counter(tasks) != Counter(TASKS):
            errors.append(f"{uid}: task coverage is {tasks}")

    audio_checked = 0
    for row in refs:
        uid = row.get("id")
        for key in ("speaker_caption", "style_caption", "emotion_caption"):
            if not nonempty_text(row.get(key)):
                errors.append(f"{uid}: empty {key}")
        overall = row.get("overall_captions")
        if not isinstance(overall, list) or len(overall) != 5 or not all(nonempty_text(x) for x in overall):
            errors.append(f"{uid}: overall_captions must contain exactly five nonempty strings")
        filename_speaker = str(uid).split("_")[2]
        if row.get("speaker_id") != filename_speaker:
            errors.append(f"{uid}: speaker_id mismatch")
        if check_audio:
            audio_path = root / str(row.get("audio_path"))
            if not audio_path.is_file():
                errors.append(f"{uid}: missing audio {audio_path}")
                continue
            try:
                info = sf.info(audio_path)
                if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
                    errors.append(f"{uid}: invalid audio metadata {info}")
                else:
                    with sf.SoundFile(audio_path) as handle:
                        handle.read(min(16, len(handle)), dtype="float32", always_2d=True)
                    audio_checked += 1
            except Exception as error:
                errors.append(f"{uid}: audio decode failed: {error}")
    if errors:
        preview = "\n".join(f"- {message}" for message in errors[:30])
        suffix = f"\n... and {len(errors) - 30} more" if len(errors) > 30 else ""
        raise ValueError(f"Benchmark validation failed with {len(errors)} error(s):\n{preview}{suffix}")
    return {
        "status": "ok", "references": len(refs), "inference_requests": len(requests),
        "audio_decoded": audio_checked, "test_sessions": dict(sorted(sessions.items())),
        "train_test_speaker_overlap": sorted(train_speakers & test_speakers),
        "validation_test_speaker_overlap": sorted(val_speakers & test_speakers),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--skip-audio", action="store_true", help="Only for unit/debug checks; full validation must decode audio.")
    args = parser.parse_args()
    report = validate(args.root.resolve(), check_audio=not args.skip_audio)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
