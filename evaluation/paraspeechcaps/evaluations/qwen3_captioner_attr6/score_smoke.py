#!/usr/bin/env python3
"""Score a one-sample Attr6 smoke run without changing the full-run protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ALL_FIELDS, DEFAULT_RUN_ROOT, load_schema, read_jsonl, sha256_file
from score import latest_predictions, score_samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_RUN_ROOT / "smoke_score"
    )
    args = parser.parse_args()

    manifest = read_jsonl(args.manifest.resolve())
    if len(manifest) != 1:
        raise ValueError(f"smoke scoring requires exactly one manifest row, found {len(manifest)}")
    predictions = latest_predictions(args.predictions.resolve())
    scored = score_samples(manifest, predictions)
    row = scored[0]
    if row["status"] != "success":
        raise RuntimeError(f"smoke prediction is not successful: {row['status']} {row['error']}")

    summary = {
        "smoke": True,
        "protocol": load_schema()["protocol"],
        "samples": 1,
        "sample_id": row["sample_id"],
        "status": row["status"],
        "sample_score": row["sample_score"],
        "field_scores": {field: row[f"score_{field}"] for field in ALL_FIELDS},
        "manifest_sha256": sha256_file(args.manifest.resolve()),
        "predictions_sha256": sha256_file(args.predictions.resolve()),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "scored_sample.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
