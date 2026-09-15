#!/usr/bin/env python3
"""Stream VoxCeleb2 WebDataset shards and retain ParaSpeechCaps test clips only."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
TEST_CSV = ROOT / "data" / "test.csv"
RAW_OUTPUT = ROOT / "audio" / ".download_cache" / "voxceleb2_raw"
MARKERS = ROOT / "audio" / ".download_cache" / "voxceleb2_shards_done"
MIRROR = "gaunernst/voxceleb2-dev-wds"
REVISION = "main"
SHARD_COUNT = 779


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def targets() -> dict[str, list[int]]:
    with TEST_CSV.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, list[int]] = {}
    for row in rows:
        relative = row["relative_audio_path"]
        if row["source"] != "voxceleb" or not relative.startswith("voxceleb2/"):
            continue
        parts = relative.split("/")
        key = "/".join(parts[-3:-1] + [parts[-1].replace("_voicefixer.wav", ".m4a")])
        result.setdefault(key, []).append(int(row["benchmark_index"]))
    return result


def shard_url(index: int) -> str:
    return (
        f"https://huggingface.co/datasets/{MIRROR}/resolve/{REVISION}/"
        f"voxceleb2-dev-{index:04d}.tar"
    )


def scan_shard(index: int, wanted: dict[str, list[int]]) -> tuple[int, list[int], str]:
    marker = MARKERS / f"{index:04d}.done"
    if marker.is_file():
        return index, [], "existing"

    last_error: Exception | None = None
    for attempt in range(8):
        try:
            with requests.get(shard_url(index), stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                response.raw.decode_content = True
                extracted: list[int] = []
                with tarfile.open(fileobj=response.raw, mode="r|") as archive:
                    for member in archive:
                        name = member.name.removeprefix("./")
                        indices = wanted.get(name)
                        if not indices or not member.isfile():
                            continue
                        source = archive.extractfile(member)
                        if source is None:
                            raise RuntimeError(f"Could not extract {member.name}")
                        first_path = RAW_OUTPUT / f"{indices[0]:03d}.m4a"
                        temporary = first_path.with_suffix(".part")
                        with temporary.open("wb") as handle:
                            shutil.copyfileobj(source, handle)
                        temporary.replace(first_path)
                        for duplicate_index in indices[1:]:
                            duplicate_path = RAW_OUTPUT / f"{duplicate_index:03d}.m4a"
                            shutil.copyfile(first_path, duplicate_path)
                        extracted.extend(indices)
                marker.touch()
                return index, extracted, "downloaded"
        except Exception as exc:
            last_error = exc
            time.sleep(min(60, 3 * (attempt + 1)))
    return index, [], f"failed: {last_error!r}"


def main() -> None:
    args = parse_args()
    wanted = targets()
    RAW_OUTPUT.mkdir(parents=True, exist_ok=True)
    MARKERS.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    completed = 0
    failures: list[tuple[int, str]] = []
    newly_extracted: list[int] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(scan_shard, index, wanted)
            for index in range(SHARD_COUNT)
        ]
        for future in as_completed(futures):
            index, extracted, status = future.result()
            with lock:
                completed += 1
                newly_extracted.extend(extracted)
                if status.startswith("failed"):
                    failures.append((index, status))
                if extracted:
                    print(
                        f"[MATCH] shard={index:04d} benchmark_indices={sorted(extracted)}",
                        flush=True,
                    )
                if completed % 25 == 0 or completed == SHARD_COUNT:
                    print(
                        f"[PROGRESS] {completed}/{SHARD_COUNT} shards; "
                        f"raw_files={len(list(RAW_OUTPUT.glob('*.m4a')))}; "
                        f"failures={len(failures)}",
                        flush=True,
                    )

    found = sorted(int(path.stem) for path in RAW_OUTPUT.glob("*.m4a"))
    expected = sorted(index for indices in wanted.values() for index in indices)
    report = {
        "mirror": MIRROR,
        "shards": SHARD_COUNT,
        "expected_indices": expected,
        "found_indices": found,
        "missing_indices": sorted(set(expected) - set(found)),
        "failed_shards": failures,
    }
    (RAW_OUTPUT / "extraction_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[SUMMARY] found={len(found)}/{len(expected)} "
        f"missing={report['missing_indices']} failed_shards={len(failures)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
