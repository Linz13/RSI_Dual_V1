#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import shutil
import subprocess
import sys
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import TASKS, sha256_file, stable_hash, write_json, write_jsonl  # noqa: E402


CAPTION_KEYS = ("spe_cap", "style_cap", "emo_cap", "caption_1", "caption_2", "caption_3", "caption_4", "caption_5")


def utterance_id(path_value: str) -> str:
    return Path(path_value).stem


def parse_caption_content(value: str) -> dict[str, str]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, dict) or any(key not in parsed for key in CAPTION_KEYS):
        raise ValueError("Missing required caption field")
    if any(not isinstance(parsed[key], str) or not parsed[key].strip() for key in CAPTION_KEYS):
        raise ValueError("Caption fields must be nonempty strings")
    return {key: parsed[key] for key in CAPTION_KEYS}


def resolve_speaker(uid: str, metadata: dict) -> tuple[str, str]:
    filename_speaker = uid.split("_")[2]
    official_speaker = str(metadata.get("speaker_id", ""))
    if not official_speaker:
        return filename_speaker, "derived_from_official_filename"
    speaker_id = official_speaker.zfill(2)
    if speaker_id != filename_speaker:
        raise ValueError(f"Speaker mismatch for {uid}: Text.tar={speaker_id}, filename={filename_speaker}")
    return speaker_id, "Text.tar:speaker_id"


def read_split(npz_path: Path, key: str) -> tuple[list[str], dict]:
    with np.load(npz_path, allow_pickle=True) as archive:
        value = archive[key]
        corpus = value.item()
    if not isinstance(corpus, dict):
        raise TypeError(f"{npz_path}:{key} is not a dictionary")
    return list(corpus), corpus


def load_audio_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["file_name", "emotion", "content"]:
            raise ValueError(f"Unexpected audio.csv fields: {reader.fieldnames}")
        for row in reader:
            uid = utterance_id(row["file_name"])
            if uid in rows:
                raise ValueError(f"Duplicate audio.csv ID: {uid}")
            try:
                parsed = parse_caption_content(row["content"])
            except ValueError as error:
                raise ValueError(f"Invalid caption content for {uid}: {error}") from error
            rows[uid] = {**row, "captions": parsed}
    return rows


def load_transcripts(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["name", "emotion", "chinese"]:
            raise ValueError(f"Unexpected transcription.csv fields: {reader.fieldnames}")
        for row in reader:
            uid = utterance_id(row["name"])
            if uid in rows:
                raise ValueError(f"Duplicate transcription ID: {uid}")
            rows[uid] = row["chinese"]
    return rows


def load_text_metadata(text_tar: Path, wanted: set[str]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    with tarfile.open(text_tar, "r:*") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            uid = Path(member.name).stem
            if uid not in wanted:
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise OSError(f"Cannot read {member.name}")
            result[uid] = json.load(io.TextIOWrapper(stream, encoding="utf-8"))
    missing = wanted - result.keys()
    if missing:
        raise ValueError(f"Text.tar is missing {len(missing)} test IDs; first: {sorted(missing)[:5]}")
    return result


def extract_test_audio(audio_tar: Path, output_root: Path, wanted: set[str]) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    existing = {path.stem: path for path in output_root.rglob("*.wav") if path.stem in wanted}
    missing = wanted - existing.keys()
    if not missing:
        return existing
    print(f"Extracting {len(missing)} test WAV files from {audio_tar} ...", flush=True)
    with tarfile.open(audio_tar, "r:*") as archive:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith(".wav"):
                continue
            uid = Path(member.name).stem
            if uid not in missing:
                continue
            target = expected_audio_path(output_root, uid)
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise OSError(f"Cannot read {member.name}")
            with target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            existing[uid] = target
            missing.remove(uid)
            if not missing:
                break
    if missing:
        raise ValueError(f"Audio.tar is missing {len(missing)} test WAV files; first: {sorted(missing)[:5]}")
    return existing


def expected_audio_path(output_root: Path, uid: str) -> Path:
    parts = uid.split("_")
    if len(parts) != 4:
        raise ValueError(f"Unexpected utterance ID: {uid}")
    return output_root / parts[0] / f"{parts[0]}_{parts[1]}" / f"{parts[0]}_{parts[1]}_{parts[2]}" / f"{uid}.wav"


def git_revision(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        # The benchmark is often prepared from a shared volume whose checkout
        # owner differs from the current user. Git's dubious-ownership guard
        # then blocks read-only metadata access. Resolve HEAD directly without
        # changing global Git configuration or trusting another repository.
        git_dir = repo / ".git"
        head_path = git_dir / "HEAD"
        if not head_path.is_file():
            raise RuntimeError(f"Cannot read Git revision from {repo}") from error
        head = head_path.read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        ref_path = git_dir / ref
        if ref_path.is_file():
            return ref_path.read_text(encoding="utf-8").strip()
        packed_refs = git_dir / "packed-refs"
        if packed_refs.is_file():
            prefix = "^{}"
            for line in packed_refs.read_text(encoding="utf-8").splitlines():
                if line and not line.startswith("#") and not line.startswith(prefix):
                    commit, packed_ref = line.split(" ", 1)
                    if packed_ref == ref:
                        return commit
        raise RuntimeError(f"Cannot resolve Git HEAD ref {ref} in {repo}") from error


def hf_resolved_revision(hf_root: Path) -> str | None:
    metadata = hf_root / ".cache/huggingface/download/Text.tar.metadata"
    if not metadata.is_file():
        return None
    lines = metadata.read_text(encoding="utf-8").splitlines()
    return lines[0] if lines else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the official EmotionTalk speech-captioning test benchmark.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--skip-audio-extraction", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    mm = root / "source/official_repo/EmotionTalk/dataset/mm-process"
    npz = mm / "audio_label.npz"
    audio_csv = mm / "audio.csv"
    transcript_csv = mm / "transcription.csv"
    text_tar = root / "source/hf/Text.tar"
    audio_tar = root / "source/hf/Audio.tar"
    required = [npz, audio_csv, transcript_csv, text_tar]
    if not args.skip_audio_extraction:
        required.append(audio_tar)
    absent = [str(path) for path in required if not path.is_file()]
    if absent:
        raise FileNotFoundError("Missing official sources:\n" + "\n".join(absent))

    train_ids, _ = read_split(npz, "train_corpus")
    val_ids, _ = read_split(npz, "val_corpus")
    test_ids, test_labels = read_split(npz, "test_corpus")
    if len(test_ids) != 1929:
        raise ValueError(f"Official test_corpus has {len(test_ids)} IDs, expected 1929")
    wanted = set(test_ids)
    audio_rows = load_audio_rows(audio_csv)
    transcripts = load_transcripts(transcript_csv)
    text_metadata = load_text_metadata(text_tar, wanted)
    if args.skip_audio_extraction:
        audio_paths = {path.stem: path for path in (root / "audio/test").rglob("*.wav")}
    else:
        audio_paths = extract_test_audio(audio_tar, root / "audio/test", wanted)

    prompts = json.loads((root / "config/prompts.json").read_text(encoding="utf-8"))
    references = []
    inference = []
    for uid in test_ids:
        if uid not in audio_rows or uid not in transcripts:
            raise ValueError(f"Official CSV join failed for {uid}")
        row = audio_rows[uid]
        meta = text_metadata[uid]
        speaker_id, speaker_source = resolve_speaker(uid, meta)
        captions = row["captions"]
        relative_audio = audio_paths.get(uid)
        if relative_audio is None:
            relative_value = str(expected_audio_path(root / "audio/test", uid).relative_to(root))
        else:
            relative_value = str(relative_audio.relative_to(root))
        reference = {
            "id": uid,
            "speaker_id": speaker_id,
            "speaker_id_source": speaker_source,
            "session_id": uid.split("_")[0],
            "audio_path": relative_value,
            "transcript": transcripts[uid],
            "emotion_label": row["emotion"],
            "speaker_caption": captions["spe_cap"],
            "style_caption": captions["style_cap"],
            "emotion_caption": captions["emo_cap"],
            "overall_captions": [captions[f"caption_{index}"] for index in range(1, 6)],
        }
        references.append(reference)
        for task in TASKS:
            inference.append({"id": uid, "task": task, "audio_path": relative_value, "prompt": prompts[task]})

    write_jsonl(root / "data/test_references.jsonl", references)
    write_jsonl(root / "data/test_inference.jsonl", inference)

    train_speakers = sorted({uid.split("_")[2] for uid in train_ids})
    val_speakers = sorted({uid.split("_")[2] for uid in val_ids})
    test_speakers = sorted({uid.split("_")[2] for uid in test_ids})
    sources = [npz, audio_csv, transcript_csv, text_tar]
    if audio_tar.is_file():
        sources.append(audio_tar)
    provenance = {
        "benchmark": "EmotionTalk speech-only Emotional Speaker Style Captioning (Table 5)",
        "official_github": "https://github.com/NKU-HLT/EmotionTalk",
        "official_github_commit": git_revision(root / "source/official_repo"),
        "official_huggingface": "BAAI/Emotiontalk",
        "huggingface_resolved_revision": hf_resolved_revision(root / "source/hf"),
        "sources": {str(path.relative_to(root)): {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sources},
        "split_authority": "source/official_repo/EmotionTalk/dataset/mm-process/audio_label.npz:test_corpus",
        "split_ordered_id_sha256": stable_hash(test_ids),
        "split_counts": {"train": len(train_ids), "validation": len(val_ids), "test": len(test_ids)},
        "test_session_counts": dict(sorted(Counter(uid.split("_")[0] for uid in test_ids).items())),
        "speaker_sets": {"train": train_speakers, "validation": val_speakers, "test": test_speakers},
        "speaker_overlap": {
            "train_test": sorted(set(train_speakers) & set(test_speakers)),
            "train_validation": sorted(set(train_speakers) & set(val_speakers)),
            "validation_test": sorted(set(val_speakers) & set(test_speakers)),
        },
        "field_mapping": {
            "speaker_caption": "audio.csv content.spe_cap",
            "style_caption": "audio.csv content.style_cap",
            "emotion_caption": "audio.csv content.emo_cap",
            "overall_captions": "audio.csv content.caption_1..caption_5",
            "transcript": "transcription.csv chinese",
            "speaker_id": "Text.tar per-sample JSON speaker_id (checked against filename third token)",
        },
        "observed_source_schema": {
            "audio.csv": ["file_name", "emotion", "content"],
            "transcription.csv": ["name", "emotion", "chinese"],
            "Text.tar JSON": sorted(text_metadata[test_ids[0]].keys()),
        },
        "notes": [
            "Captions are preserved verbatim from audio.csv; no rewriting, deduplication, or regeneration is applied.",
            "Transcript is analysis-only and is absent from test_inference.jsonl.",
            "Official validation and test speaker tokens overlap; the official split is retained unchanged.",
        ],
    }
    write_json(root / "data/provenance.json", provenance)
    print(f"Wrote {len(references)} references and {len(inference)} inference requests.")


if __name__ == "__main__":
    main()
