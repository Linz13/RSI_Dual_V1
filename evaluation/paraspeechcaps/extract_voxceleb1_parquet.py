#!/usr/bin/env python3
"""Extract the ParaSpeechCaps VoxCeleb1 clips from cached Parquet shards."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import pyarrow.parquet as pq
import requests
import soundfile as sf


ROOT = Path(__file__).resolve().parent
TEST_CSV = ROOT / "data" / "test.csv"
OUTPUT_DIR = ROOT / "audio" / "test"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
CACHE_DIR = ROOT / "audio" / ".download_cache" / "danjacobellis_parquet"
RAW_DIR = ROOT / "audio" / ".download_cache" / "voxceleb1_raw"
REPO = "danjacobellis/vox_celeb_v1"
PARQUET_ENDPOINT = "https://datasets-server.huggingface.co/parquet"


def targets() -> dict[str, list[int]]:
    with TEST_CSV.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, list[int]] = {}
    for row in rows:
        relative = row["relative_audio_path"]
        if row["source"] != "voxceleb" or not relative.startswith("voxceleb1/"):
            continue
        parts = relative.split("/")
        key = "vox_celeb_v1/dev/wav/" + "/".join(
            parts[-3:-1] + [parts[-1].replace("_voicefixer.wav", ".wav")]
        )
        result.setdefault(key, []).append(int(row["benchmark_index"]))
    return result


def convert(raw_path: Path, output_path: Path) -> None:
    temporary = output_path.with_suffix(".part.wav")
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-loglevel", "error",
            "-y",
            "-i", str(raw_path),
            "-c:a", "pcm_s16le",
            str(temporary),
        ],
        check=True,
    )
    temporary.replace(output_path)


def main() -> None:
    response = requests.get(PARQUET_ENDPOINT, params={"dataset": REPO}, timeout=60)
    response.raise_for_status()
    remote_files = response.json()["parquet_files"]
    local_files: list[tuple[dict, Path]] = []
    for item in remote_files:
        path = CACHE_DIR / f"{item['split']}-{item['filename']}"
        if not path.is_file() or path.stat().st_size != item["size"]:
            raise RuntimeError(
                f"Incomplete Parquet shard: {path} "
                f"({path.stat().st_size if path.exists() else 0}/{item['size']} bytes)"
            )
        local_files.append((item, path))

    wanted = targets()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    found: dict[int, str] = {}
    for item, path in local_files:
        parquet = pq.ParquetFile(path)
        for row_group in range(parquet.num_row_groups):
            paths = parquet.read_row_group(row_group, columns=["path"])["path"].to_pylist()
            if not any(value in wanted for value in paths):
                continue
            table = parquet.read_row_group(row_group, columns=["path", "opus"])
            for row in table.to_pylist():
                indices = wanted.get(row["path"])
                if not indices:
                    continue
                payload = row["opus"]["bytes"]
                if not payload:
                    raise RuntimeError(f"No audio bytes for {row['path']}")
                for index in indices:
                    raw_path = RAW_DIR / f"{index:03d}.opus"
                    raw_path.write_bytes(payload)
                    output_path = OUTPUT_DIR / f"{index:03d}.wav"
                    convert(raw_path, output_path)
                    found[index] = row["path"]
                    print(f"[OK] {index:03d} {row['path']}", flush=True)

    expected = sorted(index for indices in wanted.values() for index in indices)
    missing = sorted(set(expected) - set(found))
    if missing:
        raise RuntimeError(f"VoxCeleb1 targets missing from mirror: {missing}")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest_by_index = {
        int(item["benchmark_index"]): item for item in manifest["samples"]
    }
    for index, mirror_id in found.items():
        output_path = OUTPUT_DIR / f"{index:03d}.wav"
        info = sf.info(output_path)
        manifest_by_index[index].update(
            status="downloaded",
            mirror=REPO,
            mirror_id=mirror_id,
            audio_file=output_path.name,
            sample_rate=info.samplerate,
            duration_seconds=round(info.duration, 6),
            preprocessing=(
                "raw mirror audio; ParaSpeechCaps loudness normalization and "
                "VoiceFixer still required"
            ),
        )
    counts: dict[str, int] = {}
    for item in manifest["samples"]:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    manifest["status_counts"] = counts
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[SUMMARY] extracted={len(found)} status={counts}", flush=True)


if __name__ == "__main__":
    main()
