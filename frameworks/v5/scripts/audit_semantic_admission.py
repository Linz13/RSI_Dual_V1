"""Read-only CPU audit of stored Captioner group JSONL; does not recompute rewards."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from dual_isl_train.dual_space import reparse_synth_candidate
from dual_isl_train.io import read_jsonl, sha256_file
from dual_isl_train.semantic_admission import ADMISSION_VERSION


def audit(path: Path) -> dict:
    counts, repairs, rejections = Counter(), Counter(), Counter()
    for group in read_jsonl(path):
        counts["groups"] += 1
        raw_usable = safe_usable = 0
        for original in group.get("candidates", []):
            item = reparse_synth_candidate(original)
            counts["candidates"] += 1
            for key in ("raw_schema_valid", "normalized_schema_valid", "semantic_input_valid", "trajectory_valid"):
                counts[key] += bool(item.get(key))
            counts["safely_repaired"] += bool(item["semantic_input_valid"] and not item["raw_schema_valid"])
            raw_usable += bool(item.get("raw_schema_valid") and item.get("trajectory_valid"))
            safe_usable += bool(item["semantic_input_valid"] and item.get("trajectory_valid"))
            if item["semantic_input_valid"]:
                repairs.update(item["safe_normalization_rules"])
            else:
                rejections.update(item["semantic_rejection_reasons"])
        counts["groups_with_2_raw_valid_trajectories"] += raw_usable >= 2
        counts["groups_with_2_safe_valid_trajectories"] += safe_usable >= 2
    return {"source": str(path.resolve()), "source_sha256": sha256_file(path),
            "counts": dict(counts), "accepted_repair_rules": dict(repairs.most_common()),
            "top_rejections": dict(rejections.most_common(20))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, help="New JSON report; existing files are never overwritten")
    args = parser.parse_args()
    result = {
        "admission_version": ADMISSION_VERSION,
        "scope": "Historical text admission only. Newly admitted candidates need fresh current/anchor/ASR scoring; counts are NOT semantic GRPO groups or a forecast for the new prompt.",
        "files": [audit(path) for path in args.input],
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
        print(args.output.resolve())
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
