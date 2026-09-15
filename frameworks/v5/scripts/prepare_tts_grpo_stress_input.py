from __future__ import annotations

import argparse
from pathlib import Path

from dual_isl_train.io import atomic_json, read_jsonl, write_jsonl


def group_cost(group: dict) -> int:
    return sum(
        len(candidate.get("codec_codes") or candidate.get("old_main_logprobs") or [])
        for candidate in group.get("candidates", [])
        if not candidate.get("skip_update")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the heaviest TTS GRPO groups for a DDP stress test")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--world-size", required=True, type=int)
    args = parser.parse_args()
    if args.world_size < 2:
        raise SystemExit("world-size must be >= 2")
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    rows = [row for row in read_jsonl(source) if group_cost(row) > 0]
    if len(rows) < args.world_size:
        raise SystemExit(f"Need at least {args.world_size} usable groups, found {len(rows)}")
    selected = sorted(rows, key=lambda row: (-group_cost(row), str(row.get("id", ""))))[:args.world_size]
    write_jsonl(output, selected)
    report = {
        "source": str(source),
        "output": str(output),
        "world_size": args.world_size,
        "selected": [
            {
                "id": str(row.get("id")),
                "codec_frames": group_cost(row),
                "candidate_frames": [
                    len(candidate.get("codec_codes") or candidate.get("old_main_logprobs") or [])
                    for candidate in row.get("candidates", [])
                    if not candidate.get("skip_update")
                ],
            }
            for row in selected
        ],
    }
    atomic_json(str(output) + ".report.json", report)


if __name__ == "__main__":
    main()
