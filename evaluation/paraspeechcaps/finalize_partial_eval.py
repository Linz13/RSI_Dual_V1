#!/usr/bin/env python3
"""Finalize the locally available ParaSpeechCaps evaluation subset.

This script converts already extracted VoxCeleb2 M4A files, extracts exact
VoxCeleb1 matches from complete cached Parquet shards, updates the download
manifest, and writes a CSV containing only samples whose audio is available.
It does not perform any further network downloads.
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
TEST_CSV = ROOT / "data" / "test.csv"
OUTPUT_DIR = ROOT / "audio" / "test"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
SUBSET_CSV = ROOT / "data" / "test_available.csv"
V1_CACHE = ROOT / "audio" / ".download_cache" / "danjacobellis_parquet"
V1_RAW = ROOT / "audio" / ".download_cache" / "voxceleb1_raw"
V2_RAW = ROOT / "audio" / ".download_cache" / "voxceleb2_raw"
FFMPEG = (
    WORKSPACE
    / "miniconda3/envs/qwen3-tts/lib/python3.12/site-packages/"
      "imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)


def convert(source: Path, target: Path) -> None:
    temporary = target.with_suffix(".part.wav")
    subprocess.run(
        [
            str(FFMPEG), "-loglevel", "error", "-y", "-i", str(source),
            "-c:a", "pcm_s16le", str(temporary),
        ],
        check=True,
    )
    temporary.replace(target)


def load_rows() -> list[dict[str, str]]:
    with TEST_CSV.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def v1_targets(rows: list[dict[str, str]]) -> dict[str, list[int]]:
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


def update_audio_entry(
    entry: dict, index: int, mirror: str, mirror_id: str, preprocessing: str
) -> None:
    output = OUTPUT_DIR / f"{index:03d}.wav"
    info = sf.info(output)
    entry.update(
        status="downloaded",
        audio_file=output.name,
        mirror=mirror,
        mirror_id=mirror_id,
        sample_rate=info.samplerate,
        duration_seconds=round(info.duration, 6),
        preprocessing=preprocessing,
    )


def extract_v1(
    rows: list[dict[str, str]], manifest_by_index: dict[int, dict]
) -> set[int]:
    wanted = v1_targets(rows)
    found: set[int] = set()
    V1_RAW.mkdir(parents=True, exist_ok=True)
    for path in sorted(V1_CACHE.glob("*.parquet")):
        try:
            parquet = pq.ParquetFile(path)
        except Exception:
            continue
        for row_group in range(parquet.num_row_groups):
            paths = parquet.read_row_group(row_group, columns=["path"])["path"].to_pylist()
            if not any(value in wanted for value in paths):
                continue
            table = parquet.read_row_group(row_group, columns=["path", "opus"])
            for item in table.to_pylist():
                indices = wanted.get(item["path"])
                if not indices:
                    continue
                payload = item["opus"]["bytes"]
                if not payload:
                    continue
                for index in indices:
                    raw = V1_RAW / f"{index:03d}.opus"
                    raw.write_bytes(payload)
                    convert(raw, OUTPUT_DIR / f"{index:03d}.wav")
                    update_audio_entry(
                        manifest_by_index[index], index,
                        "danjacobellis/vox_celeb_v1", item["path"],
                        "raw mirror audio; ParaSpeechCaps loudness normalization "
                        "and VoiceFixer still required",
                    )
                    found.add(index)
                    print(f"[V1] {index:03d} {item['path']}", flush=True)
    return found


def convert_v2(
    rows_by_index: dict[int, dict[str, str]], manifest_by_index: dict[int, dict]
) -> set[int]:
    found: set[int] = set()
    for raw in sorted(V2_RAW.glob("*.m4a")):
        index = int(raw.stem)
        output = OUTPUT_DIR / f"{index:03d}.wav"
        convert(raw, output)
        relative = rows_by_index[index]["relative_audio_path"]
        mirror_id = relative.removeprefix("voxceleb2/dev/aac/").replace(
            "_voicefixer.wav", ".m4a"
        )
        update_audio_entry(
            manifest_by_index[index], index,
            "gaunernst/voxceleb2-dev-wds", mirror_id,
            "raw mirror audio; ParaSpeechCaps loudness normalization and "
            "VoiceFixer still required",
        )
        found.add(index)
        print(f"[V2] {index:03d} {mirror_id}", flush=True)
    return found


def validate_and_write_subset(
    rows: list[dict[str, str]], manifest: dict, manifest_by_index: dict[int, dict]
) -> tuple[int, float]:
    available: list[dict[str, str]] = []
    duration = 0.0
    for row in rows:
        index = int(row["benchmark_index"])
        entry = manifest_by_index[index]
        audio = OUTPUT_DIR / f"{index:03d}.wav"
        if entry.get("status") not in {"downloaded", "linked"} or not audio.exists():
            continue
        info = sf.info(audio)
        if info.frames <= 0 or info.samplerate <= 0:
            raise RuntimeError(f"Invalid audio: {audio}")
        duration += info.duration
        output_row = dict(row)
        output_row["local_audio_path"] = str(audio.resolve())
        available.append(output_row)

    counts: dict[str, int] = {}
    for item in manifest["samples"]:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    manifest["status_counts"] = counts
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with SUBSET_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(available)
    return len(available), duration


def main() -> None:
    if not FFMPEG.is_file():
        raise RuntimeError(f"ffmpeg not found: {FFMPEG}")
    rows = load_rows()
    rows_by_index = {int(row["benchmark_index"]): row for row in rows}
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest_by_index = {
        int(item["benchmark_index"]): item for item in manifest["samples"]
    }
    v1 = extract_v1(rows, manifest_by_index)
    v2 = convert_v2(rows_by_index, manifest_by_index)
    count, duration = validate_and_write_subset(rows, manifest, manifest_by_index)
    print(
        f"[SUMMARY] v1_added_or_replaced={len(v1)} v2_added={len(v2)} "
        f"available={count}/246 duration={duration:.1f}s subset={SUBSET_CSV}",
        flush=True,
    )


if __name__ == "__main__":
    main()
