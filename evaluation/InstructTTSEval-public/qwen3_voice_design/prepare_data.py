#!/usr/bin/env python3
"""Materialize lightweight, pinned EN/ZH manifests from official Parquet files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from common import (
    DATASET_REVISION,
    LANGUAGES,
    OFFICIAL_COMMIT,
    TASKS,
    atomic_write_json,
    atomic_write_jsonl,
    sha256_file,
    validate_sample,
)


EXPECTED_ROWS = {"en": 1000, "zh": 1000}
EXPECTED_BYTES = {"en": 347_034_080, "zh": 287_103_717}
SOURCE_BLOB_IDS = {
    "en": "cd3b457b62f6ba829c3363b9c17e06d6257fa171",
    "zh": "4f690ed5e6bb215faf8a88ae327163498fa4978c",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=root / "data/source")
    parser.add_argument("--manifest-dir", type=Path, default=root / "data/manifests")
    parser.add_argument("--metadata", type=Path, default=root / "data/metadata.json")
    return parser.parse_args()


def convert_split(path: Path, language: str) -> list[dict]:
    if path.stat().st_size != EXPECTED_BYTES[language]:
        raise ValueError(
            f"Unexpected {language} Parquet size: {path.stat().st_size}; "
            f"expected {EXPECTED_BYTES[language]}"
        )
    table = pq.read_table(path, columns=["id", "text", *TASKS])
    rows = []
    for raw in table.to_pylist():
        row = {"id": raw["id"], "language": language, "text": raw["text"]}
        row.update({task: raw[task] for task in TASKS})
        validate_sample(row, language)
        rows.append(row)
    if len(rows) != EXPECTED_ROWS[language]:
        raise ValueError(
            f"Unexpected {language} row count: {len(rows)}; "
            f"expected {EXPECTED_ROWS[language]}"
        )
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate ids in {language} split")
    return rows


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    manifest_dir = args.manifest_dir.resolve()
    metadata = {
        "benchmark": "InstructTTSEval",
        "official_commit": OFFICIAL_COMMIT,
        "dataset_revision": DATASET_REVISION,
        "source": {},
        "manifests": {},
    }
    all_ids: set[str] = set()
    for language in LANGUAGES:
        source = source_dir / f"{language}.parquet"
        if not source.is_file():
            raise FileNotFoundError(source)
        rows = convert_split(source, language)
        overlap = all_ids.intersection(row["id"] for row in rows)
        if overlap:
            raise ValueError(f"Cross-split duplicate ids: {sorted(overlap)[:5]}")
        all_ids.update(row["id"] for row in rows)
        output = manifest_dir / f"{language}.jsonl"
        atomic_write_jsonl(output, rows)
        metadata["source"][language] = {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
            "huggingface_blob_id": SOURCE_BLOB_IDS[language],
        }
        metadata["manifests"][language] = {
            "path": str(output.resolve()),
            "rows": len(rows),
            "sha256": sha256_file(output),
        }
    atomic_write_json(args.metadata.resolve(), metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
