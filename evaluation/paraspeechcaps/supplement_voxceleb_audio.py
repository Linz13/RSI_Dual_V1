#!/usr/bin/env python3
"""Supplement ParaSpeechCaps test audio from path-preserving VoxCeleb mirrors."""

from __future__ import annotations

import csv
import io
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
import requests
import soundfile as sf


ROOT = Path(__file__).resolve().parent
TEST_CSV = ROOT / "data" / "test.csv"
OUTPUT_DIR = ROOT / "audio" / "test"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
PARQUET_ENDPOINT = "https://datasets-server.huggingface.co/parquet"

MIRRORS = {
    "voxceleb1": "humanify/voxceleb1_dev",
    "voxceleb2": "humanify/voxceleb2_dev",
}


def benchmark_filename(relative_path: str) -> str:
    parts = relative_path.split("/")
    return "_".join(
        [parts[-3], parts[-2], parts[-1].replace("_voicefixer.wav", ".wav")]
    )


def download(url: str, output_path: Path) -> None:
    temporary = output_path.with_suffix(".part")
    last_error: Exception | None = None
    for attempt in range(8):
        try:
            with requests.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        handle.write(chunk)
            temporary.replace(output_path)
            return
        except Exception as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def main() -> None:
    with TEST_CSV.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest_by_index = {
        str(item["benchmark_index"]): item for item in manifest["samples"]
    }

    matches: dict[str, tuple[str, str]] = {}
    for corpus, repo in MIRRORS.items():
        targets = {
            benchmark_filename(row["relative_audio_path"]): row["benchmark_index"]
            for row in rows
            if row["source"] == "voxceleb"
            and row["relative_audio_path"].startswith(corpus + "/")
        }
        response = requests.get(
            PARQUET_ENDPOINT, params={"dataset": repo}, timeout=60
        )
        response.raise_for_status()
        parquet_url = response.json()["parquet_files"][0]["url"]
        parquet_response = requests.get(parquet_url, timeout=120)
        parquet_response.raise_for_status()
        metadata = pq.read_table(
            io.BytesIO(parquet_response.content),
            columns=["query_audio_path", "document_audio_path"],
        ).to_pylist()
        for item in metadata:
            for column in ("query_audio_path", "document_audio_path"):
                relative = item[column]
                index = targets.get(Path(relative).name)
                if index is not None:
                    matches[index] = (repo, relative)

    for index, (repo, relative) in sorted(matches.items(), key=lambda pair: int(pair[0])):
        output_path = OUTPUT_DIR / f"{int(index):03d}.wav"
        if not output_path.exists():
            url = f"https://huggingface.co/datasets/{repo}/resolve/main/{relative}"
            download(url, output_path)
            print(f"[OK] {int(index):03d} {repo}/{relative}", flush=True)
        info = sf.info(output_path)
        entry = manifest_by_index[index]
        entry.update(
            status="downloaded",
            mirror=repo,
            mirror_id=relative,
            audio_file=output_path.name,
            sample_rate=info.samplerate,
            duration_seconds=round(info.duration, 6),
            preprocessing=(
                "raw mirror audio; ParaSpeechCaps loudness normalization and "
                "VoiceFixer still required"
            ),
        )
        MANIFEST_PATH.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    counts: dict[str, int] = {}
    for item in manifest["samples"]:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    manifest["status_counts"] = counts
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[SUMMARY] matched={len(matches)} status={counts}", flush=True)


if __name__ == "__main__":
    main()
