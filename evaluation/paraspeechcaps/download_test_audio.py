#!/usr/bin/env python3
"""Materialize the 246 ParaSpeechCaps test clips in one directory.

EARS clips are linked from the existing local copy. Expresso and VoxCeleb clips
are matched by their original corpus identifiers against public Hugging Face
audio mirrors, then downloaded one clip at a time. A manifest records the exact
source and whether the downloaded waveform still needs ParaSpeechCaps-specific
preprocessing.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
from pathlib import Path
from typing import Any
import aiohttp
import fsspec
import pyarrow.parquet as pq
import requests
import soundfile as sf


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "data" / "test.csv"
DEFAULT_OUTPUT = ROOT / "audio" / "test"
CACHE_DIR = ROOT / "audio" / ".download_cache"
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
PARQUET_ENDPOINT = "https://datasets-server.huggingface.co/parquet"

EXPRESSO_REPO = "shangeth/expresso"
VOXCELEB_REPO = "TwinkStart/VoxCeleb"

EXPRESSO_CONV_RE = re.compile(
    r"(?P<source>.+)_channel(?P<channel>[12])_segment_"
    r"(?P<start>[0-9.]+)_(?P<end>[0-9.]+)\.wav$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scan-concurrency", type=int, default=4)
    parser.add_argument("--download-concurrency", type=int, default=8)
    parser.add_argument("--scan-only", action="store_true")
    return parser.parse_args()


def load_benchmark(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 246:
        raise ValueError(f"Expected 246 benchmark rows, found {len(rows)}")
    return rows


async def fetch_json(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    params: dict[str, Any],
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(12):
        try:
            async with semaphore:
                async with session.get(ROWS_ENDPOINT, params=params) as response:
                    if response.status == 429:
                        retry_after = float(response.headers.get("Retry-After", 0) or 0)
                        await asyncio.sleep(max(retry_after, min(30.0, 2.0 * (attempt + 1))))
                        continue
                    if response.status != 200:
                        raise RuntimeError(
                            f"HTTP {response.status}: {(await response.text())[:300]}"
                        )
                    return await response.json()
        except Exception as exc:
            last_error = exc
            await asyncio.sleep(min(30.0, 1.5 * (attempt + 1)))
    raise RuntimeError(f"Failed rows request {params}: {last_error}")


async def scan_split(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    repo: str,
    config: str,
    split: str,
) -> list[dict[str, Any]]:
    cache_path = CACHE_DIR / f"{repo.replace('/', '--')}--{config}--{split}.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"[CACHE] {repo} {config}/{split}: {len(cached)} rows", flush=True)
        return cached
    response = requests.get(PARQUET_ENDPOINT, params={"dataset": repo}, timeout=60)
    response.raise_for_status()
    files = [
        item for item in response.json()["parquet_files"]
        if item["config"] == config and item["split"] == split
    ]
    if repo == EXPRESSO_REPO and config == "read":
        columns = ["id"]
    elif repo == EXPRESSO_REPO:
        columns = ["id", "source_file_id", "channel", "start_s", "end_s"]
    else:
        columns = ["WavPath"]

    def read_index(url: str) -> list[dict[str, Any]]:
        with fsspec.open(url, "rb", block_size=1024 * 1024) as handle:
            table = pq.read_table(handle, columns=columns)
        return table.to_pylist()

    async def read_file(url: str) -> list[dict[str, Any]]:
        async with semaphore:
            for attempt in range(5):
                try:
                    return await asyncio.to_thread(read_index, url)
                except Exception:
                    if attempt == 4:
                        raise
                    await asyncio.sleep(2.0 * (attempt + 1))
        raise AssertionError("unreachable")

    file_rows = await asyncio.gather(*(read_file(item["url"]) for item in files))
    rows = []
    global_index = 0
    for current in file_rows:
        for row in current:
            rows.append({"row_idx": global_index, "row": row})
            global_index += 1
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(rows), encoding="utf-8")
    print(f"[SCAN] {repo} {config}/{split}: {len(rows)} rows", flush=True)
    return rows


async def add_audio_urls(
    discovered: dict[str, Any],
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
) -> None:
    async def hydrate(info: dict[str, Any]) -> None:
        result = await fetch_json(
            session,
            semaphore,
            {
                "dataset": info["mirror"],
                "config": info["mirror_config"],
                "split": info["mirror_split"],
                "offset": info["mirror_row_idx"],
                "length": 1,
            },
        )
        if len(result["rows"]) != 1:
            raise RuntimeError(f"Mirror row lookup returned {len(result['rows'])} rows")
        mirror_row = result["rows"][0]["row"]
        returned_id = mirror_row.get("id", mirror_row.get("WavPath"))
        if returned_id != info["mirror_id"]:
            raise RuntimeError(
                f"Mirror row mismatch: expected {info['mirror_id']!r}, "
                f"received {returned_id!r}"
            )
        info["url"] = audio_url(mirror_row)

    # The rows endpoint is deliberately queried in small batches. Launching one
    # request per clip at once causes server-side 429 throttling.
    items = list(discovered.values())
    for start in range(0, len(items), 2):
        await asyncio.gather(*(hydrate(info) for info in items[start:start + 2]))
        await asyncio.sleep(0.25)


def benchmark_targets(rows: list[dict[str, str]]) -> dict[str, Any]:
    expresso_read: dict[str, str] = {}
    expresso_conversational: list[dict[str, Any]] = []
    voxceleb: dict[str, str] = {}

    for row in rows:
        index = row["benchmark_index"]
        relative = row["relative_audio_path"]
        if row["source"] == "expresso":
            basename = Path(relative).name
            if "/read/" in relative:
                expresso_read[Path(basename).stem] = index
            else:
                match = EXPRESSO_CONV_RE.fullmatch(basename)
                if not match:
                    raise ValueError(f"Unrecognized Expresso path: {relative}")
                expresso_conversational.append(
                    {
                        "index": index,
                        "source_file_id": match.group("source"),
                        "channel": int(match.group("channel")),
                        "start": float(match.group("start")),
                        "end": float(match.group("end")),
                    }
                )
        elif row["source"] == "voxceleb":
            parts = relative.split("/")
            corpus = parts[0]
            stem = Path(parts[-1]).name.removesuffix("_voicefixer.wav")
            suffix = ".wav" if corpus == "voxceleb1" else ".m4a"
            mirror_path = "/".join(parts[-4:-1] + [stem + suffix])
            voxceleb[mirror_path] = index

    return {
        "expresso_read": expresso_read,
        "expresso_conversational": expresso_conversational,
        "voxceleb": voxceleb,
    }


def audio_url(row: dict[str, Any]) -> str:
    audio = row["audio"]
    if not audio or not audio[0].get("src"):
        raise ValueError(f"No audio URL in mirror row: {row}")
    return str(audio[0]["src"])


async def discover(rows: list[dict[str, str]], concurrency: int) -> dict[str, Any]:
    targets = benchmark_targets(rows)
    timeout = aiohttp.ClientTimeout(total=120)
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        scans = []
        for config in ("read", "conversational"):
            for split in ("train", "dev", "test"):
                scans.append(
                    await scan_split(
                        session, semaphore, EXPRESSO_REPO, config, split
                    )
                )
        for split in ("voxceleb1", "voxceleb2"):
            scans.append(
                await scan_split(
                    session, semaphore, VOXCELEB_REPO, "default", split
                )
            )

    found: dict[str, dict[str, Any]] = {}
    expresso_rows = scans[:6]
    for split_name, mirror_rows in zip(
        ("read/train", "read/dev", "read/test",
         "conversational/train", "conversational/dev", "conversational/test"),
        expresso_rows,
    ):
        for wrapped in mirror_rows:
            mirror = wrapped["row"]
            if split_name.startswith("read/"):
                index = targets["expresso_read"].get(mirror["id"])
                if index is not None:
                    found[index] = {
                        "mirror": EXPRESSO_REPO,
                        "mirror_config": "read",
                        "mirror_split": split_name.split("/", 1)[1],
                        "mirror_row_idx": wrapped["row_idx"],
                        "mirror_id": mirror["id"],
                        "preprocessing": "mirror-provided VAD segment; ParaSpeechCaps loudness normalization not verified",
                    }
            else:
                for target in targets["expresso_conversational"]:
                    if (
                        mirror["source_file_id"] == target["source_file_id"]
                        and int(mirror["channel"]) == target["channel"]
                        and abs(float(mirror["start_s"]) - target["start"]) < 0.011
                        and abs(float(mirror["end_s"]) - target["end"]) < 0.011
                    ):
                        found[target["index"]] = {
                            "mirror": EXPRESSO_REPO,
                            "mirror_config": "conversational",
                            "mirror_split": split_name.split("/", 1)[1],
                            "mirror_row_idx": wrapped["row_idx"],
                            "mirror_id": mirror["id"],
                            "preprocessing": "mirror-provided VAD segment; ParaSpeechCaps loudness normalization not verified",
                        }

    for split_name, mirror_rows in zip(("voxceleb1", "voxceleb2"), scans[6:]):
        for wrapped in mirror_rows:
            mirror = wrapped["row"]
            index = targets["voxceleb"].get(mirror["WavPath"])
            if index is not None:
                found[index] = {
                    "mirror": VOXCELEB_REPO,
                    "mirror_config": "default",
                    "mirror_split": split_name,
                    "mirror_row_idx": wrapped["row_idx"],
                    "mirror_id": mirror["WavPath"],
                    "preprocessing": "raw mirror audio; ParaSpeechCaps loudness normalization and VoiceFixer still required",
                }
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=concurrency)) as session:
        await add_audio_urls(found, session, asyncio.Semaphore(concurrency))
    return found


async def download_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    url: str,
    output_path: Path,
) -> None:
    temporary = output_path.with_suffix(".part")
    for attempt in range(5):
        try:
            async with semaphore:
                async with session.get(url) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as handle:
                        async for chunk in response.content.iter_chunked(1024 * 1024):
                            handle.write(chunk)
            temporary.replace(output_path)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 4:
                raise
            await asyncio.sleep(1.5 * (attempt + 1))


async def materialize(
    rows: list[dict[str, str]],
    discovered: dict[str, Any],
    output_dir: Path,
    concurrency: int,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    downloads: list[tuple[dict[str, Any], str, Path]] = []

    for row in rows:
        index = row["benchmark_index"]
        output_path = output_dir / f"{int(index):03d}.wav"
        entry: dict[str, Any] = {
            "benchmark_index": int(index),
            "source": row["source"],
            "relative_audio_path": row["relative_audio_path"],
            "audio_file": output_path.name,
            "status": "missing",
        }
        if output_path.is_file() or output_path.is_symlink():
            entry["status"] = "existing"
        elif row["source"] == "ears":
            source = (ROOT / "audio" / "ears" / row["relative_audio_path"]).resolve()
            if source.is_file():
                output_path.symlink_to(os.path.relpath(source, output_dir))
                entry.update(status="linked", source_file=str(source))
        elif index in discovered:
            info = discovered[index]
            entry.update(info)
            downloads.append((entry, info["url"], output_path))
        manifest.append(entry)

    timeout = aiohttp.ClientTimeout(total=300)
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def run_download(item: tuple[dict[str, Any], str, Path]) -> None:
            entry, url, output_path = item
            try:
                await download_one(session, semaphore, url, output_path)
                info = sf.info(output_path)
                entry.update(
                    status="downloaded",
                    sample_rate=info.samplerate,
                    duration_seconds=round(info.duration, 6),
                )
                print(f"[OK] {entry['benchmark_index']:03d} {entry['source']}", flush=True)
            except Exception as exc:
                entry.update(status="failed", error=repr(exc))
                print(f"[ERROR] {entry['benchmark_index']:03d}: {exc!r}", flush=True)

        await asyncio.gather(*(run_download(item) for item in downloads))
    return manifest


def write_manifest(output_dir: Path, manifest: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for item in manifest:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    payload = {
        "benchmark": "ParaSpeechCaps main evaluation test set",
        "expected_count": 246,
        "status_counts": counts,
        "samples": manifest,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[SUMMARY] {counts}", flush=True)


async def async_main() -> None:
    args = parse_args()
    rows = load_benchmark(args.csv)
    discovered = await discover(rows, args.scan_concurrency)
    print(f"[MATCH] public mirrors matched {len(discovered)} non-EARS rows", flush=True)
    if args.scan_only:
        expected = {r["benchmark_index"] for r in rows if r["source"] != "ears"}
        print(f"[MISSING] {sorted(expected - set(discovered), key=int)}")
        return
    manifest = await materialize(
        rows, discovered, args.output_dir, args.download_concurrency
    )
    write_manifest(args.output_dir, manifest)


if __name__ == "__main__":
    asyncio.run(async_main())
