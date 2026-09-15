#!/usr/bin/env python3
"""Download official sources and build the StyleCap 778-audio MCQ benchmark."""

from __future__ import annotations

import argparse
import csv
import io
import os
import shutil
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    AUDIO_DIR,
    BENCHMARK_JSONL,
    DATA_DIR,
    DOWNLOADS_DIR,
    EXPECTED_DISTRIBUTIONS,
    EXPECTED_METADATA_ROWS,
    EXPECTED_QUESTIONS,
    EXPECTED_SPLITS,
    LIBRITTS_TEST_CLEAN_BYTES,
    LIBRITTS_TEST_CLEAN_MD5,
    LIBRITTS_TEST_CLEAN_URL,
    PROMPTSPEECH_CSV,
    PROMPTSPEECH_CSV_SHA256,
    PROMPTSPEECH_SHA256,
    PROMPTSPEECH_URL,
    ROOT,
    SOURCE_MANIFEST_JSON,
    STYLECAP_CSV,
    STYLECAP_CSV_SHA256,
    STYLECAP_SPLIT_SHA256,
    STYLECAP_SPLIT_URL,
    SUMMARY_JSON,
    TASK_ORDER,
    TEST_AUDIO_JSONL,
    atomic_write_bytes,
    digest_bytes,
    digest_file,
    distributions,
    make_question_records,
    normalize_metadata_row,
    speaker_id_from_audio_id,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--libritts-root",
        type=Path,
        help=(
            "Reuse an existing official LibriTTS tree. The path may point to the "
            "directory containing LibriTTS, to LibriTTS itself, or to test-clean."
        ),
    )
    return parser.parse_args()


def _verify(path: Path, algorithm: str, expected: str) -> bool:
    if not path.is_file():
        return False
    actual = digest_file(path, algorithm)
    if actual != expected:
        raise ValueError(
            f"checksum mismatch for {path}: expected {algorithm}={expected}, got {actual}"
        )
    return True


def download(
    url: str,
    destination: Path,
    *,
    algorithm: str,
    expected_digest: str,
    expected_size: int | None = None,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _verify(destination, algorithm, expected_digest):
        print(f"[OK] verified cached download: {destination}", flush=True)
        return destination

    partial = destination.with_name(destination.name + ".part")
    if expected_size is not None and partial.is_file() and partial.stat().st_size > expected_size:
        raise ValueError(f"partial download is larger than expected: {partial}")
    if expected_size is not None and partial.is_file() and partial.stat().st_size == expected_size:
        if not _verify(partial, algorithm, expected_digest):
            raise AssertionError("unreachable")
        os.replace(partial, destination)
        print(f"[OK] promoted completed partial download: {destination}", flush=True)
        return destination

    for attempt in range(1, 9):
        existing = partial.stat().st_size if partial.is_file() else 0
        headers = {"User-Agent": "stylecap-promptspeech-benchmark/1.0"}
        if existing:
            headers["Range"] = f"bytes={existing}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status = getattr(response, "status", response.getcode())
                append = bool(existing and status == 206)
                mode = "ab" if append else "wb"
                if existing and not append:
                    existing = 0
                downloaded = existing
                next_report = downloaded + 64 * 1024 * 1024
                print(
                    f"[DOWNLOAD] {url} -> {partial} (resume={downloaded} bytes)",
                    flush=True,
                )
                with partial.open(mode) as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if downloaded >= next_report:
                            print(f"  downloaded {downloaded / 1e9:.2f} GB", flush=True)
                            next_report += 64 * 1024 * 1024
            if expected_size is not None and partial.stat().st_size != expected_size:
                raise OSError(
                    f"downloaded size {partial.stat().st_size}, expected {expected_size}"
                )
            if not _verify(partial, algorithm, expected_digest):
                raise AssertionError("unreachable")
            os.replace(partial, destination)
            print(f"[OK] downloaded and verified: {destination}", flush=True)
            return destination
        except (OSError, urllib.error.URLError) as error:
            if attempt == 8:
                raise RuntimeError(f"download failed after {attempt} attempts: {url}") from error
            print(f"[RETRY {attempt}/8] {error}", flush=True)
            time.sleep(min(20, attempt * 2))
    raise AssertionError("unreachable")


def extract_metadata(prompt_zip: Path, split_zip: Path) -> None:
    with zipfile.ZipFile(prompt_zip) as archive:
        payload = archive.read("Real_training.csv")
    actual = digest_bytes(payload)
    if actual != PROMPTSPEECH_CSV_SHA256:
        raise ValueError(f"unexpected Real_training.csv SHA-256: {actual}")
    atomic_write_bytes(PROMPTSPEECH_CSV, payload)

    with zipfile.ZipFile(split_zip) as archive:
        for split, destination in STYLECAP_CSV.items():
            payload = archive.read(f"train_dev_test_set/{split}_set.csv")
            actual = digest_bytes(payload)
            if actual != STYLECAP_CSV_SHA256[split]:
                raise ValueError(f"unexpected StyleCap {split} CSV SHA-256: {actual}")
            atomic_write_bytes(destination, payload)
    print("[OK] extracted and pinned official metadata CSV files", flush=True)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def source_path(root: Path, official_relative_path: str) -> Path:
    relative = Path(official_relative_path)
    suffix_after_libritts = Path(*relative.parts[1:])
    suffix_after_test_clean = Path(*relative.parts[2:])
    candidates = (
        root / relative,
        root / suffix_after_libritts,
        root / suffix_after_test_clean,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"cannot locate {official_relative_path} below --libritts-root={root}"
    )


def materialize_from_existing(root: Path, relative_paths: list[str]) -> None:
    root = root.expanduser().resolve()
    for index, relative_path in enumerate(relative_paths, 1):
        source = source_path(root, relative_path)
        destination = AUDIO_DIR / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and destination.stat().st_size == source.stat().st_size:
            continue
        temporary = destination.with_name(destination.name + ".part")
        if temporary.exists():
            temporary.unlink()
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        os.replace(temporary, destination)
        if index % 100 == 0:
            print(f"  materialized {index}/{len(relative_paths)} WAV files", flush=True)


def materialize_from_archive(archive_path: Path, relative_paths: list[str]) -> None:
    required = set(relative_paths)
    found: set[str] = set()
    for relative_path in relative_paths:
        destination = AUDIO_DIR / relative_path
        if destination.is_file() and destination.stat().st_size > 0:
            found.add(relative_path)

    print(
        f"[EXTRACT] scanning official LibriTTS archive; cached={len(found)}, "
        f"needed={len(required)}",
        flush=True,
    )
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if name not in required:
                continue
            if not member.isfile():
                raise ValueError(f"expected regular WAV member: {member.name}")
            found.add(name)
            destination = AUDIO_DIR / name
            if destination.is_file() and destination.stat().st_size > 0:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot extract tar member: {member.name}")
            temporary = destination.with_name(destination.name + ".part")
            with source, temporary.open("wb") as handle:
                shutil.copyfileobj(source, handle, length=1024 * 1024)
            os.replace(temporary, destination)
            if len(found) % 100 == 0:
                print(f"  found {len(found)}/{len(required)} selected WAV files", flush=True)
    missing = sorted(required - found)
    if missing:
        raise ValueError(f"LibriTTS archive is missing {len(missing)} selected WAVs: {missing[:3]}")
    print(f"[OK] materialized {len(required)} selected WAV files", flush=True)


def build_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metadata_rows = csv_rows(PROMPTSPEECH_CSV)
    if len(metadata_rows) != EXPECTED_METADATA_ROWS:
        raise ValueError(f"expected {EXPECTED_METADATA_ROWS} metadata rows, got {len(metadata_rows)}")
    metadata = {row["item_name"]: row for row in metadata_rows}
    if len(metadata) != len(metadata_rows):
        raise ValueError("PromptSpeech metadata has duplicate item_name values")
    test_rows = csv_rows(STYLECAP_CSV["test"])
    if len(test_rows) != EXPECTED_SPLITS["test"]["rows"]:
        raise ValueError(f"expected 778 StyleCap test rows, got {len(test_rows)}")

    audio_rows: list[dict[str, Any]] = []
    for split_row in test_rows:
        audio_id = split_row["file_id"]
        if audio_id not in metadata:
            raise ValueError(f"StyleCap test ID is absent from PromptSpeech metadata: {audio_id}")
        meta = metadata[audio_id]
        speaker_id = speaker_id_from_audio_id(audio_id)
        if meta["spk_id"] != speaker_id:
            raise ValueError(f"speaker ID mismatch for {audio_id}")
        official_relative_path = split_row["relative_path_within_libritts"]
        local_path = (AUDIO_DIR / official_relative_path).resolve()
        relative_audio_path = local_path.relative_to(ROOT.resolve()).as_posix()
        audio_rows.append(
            {
                "audio_id": audio_id,
                "speaker_id": speaker_id,
                "split": "stylecap_speaker_open_test",
                "audio_path": str(local_path),
                "relative_audio_path": relative_audio_path,
                "source_relative_audio_path": official_relative_path,
                "attributes": normalize_metadata_row(meta),
            }
        )
    if len({row["audio_id"] for row in audio_rows}) != len(audio_rows):
        raise ValueError("StyleCap test contains duplicate audio IDs")
    question_rows = [
        question
        for audio in audio_rows
        for question in make_question_records(audio)
    ]
    if len(question_rows) != EXPECTED_QUESTIONS:
        raise ValueError(f"expected {EXPECTED_QUESTIONS} questions, got {len(question_rows)}")
    return audio_rows, question_rows


def main() -> None:
    args = parse_args()
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    prompt_zip = download(
        PROMPTSPEECH_URL,
        DOWNLOADS_DIR / "Real_training.zip",
        algorithm="sha256",
        expected_digest=PROMPTSPEECH_SHA256,
    )
    split_zip = download(
        STYLECAP_SPLIT_URL,
        DOWNLOADS_DIR / "train_dev_test_set.zip",
        algorithm="sha256",
        expected_digest=STYLECAP_SPLIT_SHA256,
    )
    extract_metadata(prompt_zip, split_zip)
    audio_rows, question_rows = build_rows()
    relative_paths = [row["source_relative_audio_path"] for row in audio_rows]

    libritts_archive: Path | None = None
    if args.libritts_root:
        materialize_from_existing(args.libritts_root, relative_paths)
        audio_source = {"mode": "existing_tree", "path": str(args.libritts_root.resolve())}
    else:
        libritts_archive = download(
            LIBRITTS_TEST_CLEAN_URL,
            DOWNLOADS_DIR / "test-clean.tar.gz",
            algorithm="md5",
            expected_digest=LIBRITTS_TEST_CLEAN_MD5,
            expected_size=LIBRITTS_TEST_CLEAN_BYTES,
        )
        materialize_from_archive(libritts_archive, relative_paths)
        audio_source = {
            "mode": "official_archive",
            "path": str(libritts_archive.resolve()),
            "md5": LIBRITTS_TEST_CLEAN_MD5,
        }

    for row in audio_rows:
        path = Path(row["audio_path"])
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"missing or empty materialized audio: {path}")

    observed_distributions = distributions(audio_rows)
    if observed_distributions != EXPECTED_DISTRIBUTIONS:
        raise ValueError(
            f"unexpected test distributions: expected={EXPECTED_DISTRIBUTIONS}, "
            f"got={observed_distributions}"
        )
    write_jsonl(TEST_AUDIO_JSONL, audio_rows)
    write_jsonl(BENCHMARK_JSONL, question_rows)

    split_speakers = {
        split: len({speaker_id_from_audio_id(row["file_id"]) for row in csv_rows(path)})
        for split, path in STYLECAP_CSV.items()
    }
    source_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "promptspeech_real_training": {
                "url": PROMPTSPEECH_URL,
                "archive_sha256": PROMPTSPEECH_SHA256,
                "csv_sha256": PROMPTSPEECH_CSV_SHA256,
            },
            "stylecap_speaker_open_split": {
                "url": STYLECAP_SPLIT_URL,
                "archive_sha256": STYLECAP_SPLIT_SHA256,
                "csv_sha256": STYLECAP_CSV_SHA256,
            },
            "libritts_test_clean": {
                "url": LIBRITTS_TEST_CLEAN_URL,
                "archive_md5": LIBRITTS_TEST_CLEAN_MD5,
                "license": "CC BY 4.0",
                **audio_source,
            },
        },
    }
    write_json(SOURCE_MANIFEST_JSON, source_manifest)
    summary = {
        "benchmark": "stylecap-promptspeech-speaker-open-mcq-v1",
        "audio_count": len(audio_rows),
        "speaker_count": len({row["speaker_id"] for row in audio_rows}),
        "question_count": len(question_rows),
        "questions_per_audio": len(TASK_ORDER),
        "split_counts": {split: values["rows"] for split, values in EXPECTED_SPLITS.items()},
        "split_speaker_counts": split_speakers,
        "distributions": observed_distributions,
        "benchmark_jsonl_sha256": digest_file(BENCHMARK_JSONL),
        "test_audio_jsonl_sha256": digest_file(TEST_AUDIO_JSONL),
    }
    write_json(SUMMARY_JSON, summary)
    print(
        f"[DONE] audio={len(audio_rows)} speakers={summary['speaker_count']} "
        f"questions={len(question_rows)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
