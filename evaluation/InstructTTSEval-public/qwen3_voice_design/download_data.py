#!/usr/bin/env python3
"""Resumable parallel Range downloader for pinned Hugging Face dataset files."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=20)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def auth_headers() -> dict[str, str]:
    try:
        from huggingface_hub import get_token

        token = get_token()
    except Exception:
        token = None
    return {"Authorization": f"Bearer {token}"} if token else {}


def download_range(
    url: str,
    part_path: Path,
    start: int,
    end: int,
    total: int,
    attempts: int,
) -> Path:
    expected = end - start + 1
    headers = auth_headers()
    for attempt in range(1, attempts + 1):
        current = part_path.stat().st_size if part_path.is_file() else 0
        if current == expected:
            return part_path
        if current > expected:
            raise RuntimeError(f"Oversized chunk: {part_path}")
        range_start = start + current
        request_headers = {**headers, "Range": f"bytes={range_start}-{end}"}
        try:
            with requests.get(
                url,
                headers=request_headers,
                stream=True,
                allow_redirects=True,
                timeout=(30, 30),
            ) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(
                        f"Server ignored Range request for {range_start}-{end}: "
                        f"HTTP {response.status_code}"
                    )
                content_range = response.headers.get("Content-Range", "")
                expected_prefix = f"bytes {range_start}-{end}/{total}"
                if content_range != expected_prefix:
                    raise RuntimeError(
                        f"Unexpected Content-Range {content_range!r}; "
                        f"expected {expected_prefix!r}"
                    )
                with part_path.open("ab") as handle:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            handle.write(block)
                    handle.flush()
                    os.fsync(handle.fileno())
        except Exception as exc:
            if attempt >= attempts:
                raise RuntimeError(
                    f"Chunk {start}-{end} failed after {attempts} attempts"
                ) from exc
            print(
                f"[RETRY] chunk={start}-{end} attempt={attempt}/{attempts} "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(min(attempt * 2, 20))
    raise AssertionError("unreachable")


def main() -> int:
    args = parse_args()
    if args.workers <= 0 or args.attempts <= 0 or args.expected_bytes <= 0:
        raise ValueError("workers, attempts, and expected bytes must be positive")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and output.stat().st_size == args.expected_bytes:
        actual = sha256_file(output)
        if actual == args.expected_sha256:
            print(f"[SKIP] Verified existing file: {output}")
            return 0
        print(f"[WARN] Existing complete-size file has wrong SHA-256: {output}")

    chunk_dir = output.with_name(output.name + ".chunks")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = (args.expected_bytes + args.workers - 1) // args.workers
    chunks = []
    for index in range(args.workers):
        start = index * chunk_size
        if start >= args.expected_bytes:
            break
        end = min(args.expected_bytes - 1, start + chunk_size - 1)
        chunks.append((index, start, end, chunk_dir / f"{index:03d}.part"))
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = {
            executor.submit(
                download_range,
                args.url,
                path,
                start,
                end,
                args.expected_bytes,
                args.attempts,
            ): (index, start, end)
            for index, start, end, path in chunks
        }
        for future in as_completed(futures):
            index, start, end = futures[future]
            future.result()
            print(f"[CHUNK][OK] index={index} bytes={start}-{end}", flush=True)

    assembling = output.with_name(output.name + ".assembling")
    digest = hashlib.sha256()
    with assembling.open("wb") as destination:
        for _, start, end, path in chunks:
            if path.stat().st_size != end - start + 1:
                raise RuntimeError(f"Incomplete chunk during assembly: {path}")
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    destination.write(block)
                    digest.update(block)
        destination.flush()
        os.fsync(destination.fileno())
    if assembling.stat().st_size != args.expected_bytes:
        raise RuntimeError("Assembled file size mismatch")
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != args.expected_sha256:
        raise RuntimeError(
            f"Assembled SHA-256 mismatch: {actual_sha256} != {args.expected_sha256}"
        )
    os.replace(assembling, output)
    shutil.rmtree(chunk_dir)
    print(f"[DONE] Downloaded and verified {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
