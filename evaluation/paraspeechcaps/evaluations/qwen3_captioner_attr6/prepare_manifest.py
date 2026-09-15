#!/usr/bin/env python3
"""Build and validate the 140-unique-audio normalized Attr6 manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import soundfile as sf

from common import (
    MANIFEST_META_PATH,
    MANIFEST_PATH,
    SCHEMA_PATH,
    TEST_AUDIO_DIR,
    TEST_AVAILABLE_CSV,
    TEST_FULL_CSV,
    load_schema,
    sha256_file,
    write_jsonl,
)


def parse_list(value: str) -> list[str]:
    return [str(item) for item in json.loads(value)] if value.strip() else []


def normalized_gt(row: dict[str, str], schema: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    accents = set(schema["single_choice"]["accent"])
    intrinsic = [item for item in parse_list(row["intrinsic_tags"]) if item not in accents]
    situational = parse_list(row["situational_tags"])
    actions: list[str] = []
    if "enunciated" in situational:
        situational = [item for item in situational if item != "enunciated"]
        if "enunciated" not in intrinsic:
            intrinsic.append("enunciated")
        actions.append("moved enunciated from situational_traits to intrinsic_traits")
    repairs = schema["normalization"]["explicit_gt_repairs"].get(row["benchmark_index"], [])
    for label in repairs:
        if label not in intrinsic:
            intrinsic.append(label)
            actions.append(
                f"added {label} from tag_of_interest/reference audit for benchmark_index={row['benchmark_index']}"
            )
    intrinsic_order = {
        item: index for index, item in enumerate(schema["multi_choice"]["intrinsic_traits"])
    }
    situational_order = {
        item: index for index, item in enumerate(schema["multi_choice"]["situational_traits"])
    }
    return (
        {
            "gender": row["gender"],
            "pitch": row["pitch"],
            "speaking_rate": row["speaking_rate"],
            "accent": row["accent"],
            "intrinsic_traits": sorted(set(intrinsic), key=intrinsic_order.__getitem__),
            "situational_traits": sorted(set(situational), key=situational_order.__getitem__),
        },
        actions,
    )


def build_samples(
    csv_path: Path,
    schema: dict[str, Any],
    audio_dir: Path = TEST_AUDIO_DIR,
) -> list[dict[str, Any]]:
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for row in rows:
        grouped.setdefault(row["relative_audio_path"], []).append(row)

    samples: list[dict[str, Any]] = []
    for relative_audio_path, group in grouped.items():
        audio_paths = [
            str((audio_dir / f"{int(row['benchmark_index']):03d}.wav").resolve())
            for row in group
        ]
        audio_path = audio_paths[0]
        normalized = [normalized_gt(row, schema) for row in group]
        first_gt = normalized[0][0]
        if any(gt != first_gt for gt, _ in normalized[1:]):
            raise ValueError(
                f"duplicate audio has inconsistent normalized GT: {audio_path} "
                f"indices={[row['benchmark_index'] for row in group]}"
            )
        sources = {row["source"] for row in group}
        if len(sources) != 1:
            raise ValueError(f"duplicate audio metadata mismatch: {audio_path}")
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        info = sf.info(path)
        if info.frames <= 0 or info.samplerate <= 0:
            raise ValueError(f"invalid audio: {path}")
        indices = sorted(int(row["benchmark_index"]) for row in group)
        actions = sorted({action for _, values in normalized for action in values})
        samples.append(
            {
                "sample_id": f"psc-{indices[0]:03d}",
                "benchmark_indices": indices,
                "audio_path": audio_path,
                "relative_audio_path": relative_audio_path,
                "duplicate_local_audio_paths": audio_paths,
                "source": group[0]["source"],
                "duration_seconds": round(info.duration, 6),
                "sample_rate": info.samplerate,
                "gt": first_gt,
                "normalization_actions": actions,
                "raw_annotations": [
                    {
                        "benchmark_index": int(row["benchmark_index"]),
                        "intrinsic_tags": parse_list(row["intrinsic_tags"]),
                        "situational_tags": parse_list(row["situational_tags"]),
                        "tag_of_interest": row["tag_of_interest"],
                    }
                    for row in group
                ],
            }
        )
    return sorted(samples, key=lambda row: row["sample_id"])


def validate_samples(samples: list[dict[str, Any]], schema: dict[str, Any]) -> None:
    if len(samples) != 140:
        raise ValueError(f"expected 140 unique audio samples, found {len(samples)}")
    if len({row["sample_id"] for row in samples}) != len(samples):
        raise ValueError("duplicate sample_id")
    if len({row["audio_path"] for row in samples}) != len(samples):
        raise ValueError("duplicate audio_path remains after grouping")
    candidates = {**schema["single_choice"], **schema["multi_choice"]}
    for sample in samples:
        gt = sample["gt"]
        for field in ("gender", "pitch", "speaking_rate", "accent"):
            if gt[field] not in candidates[field]:
                raise ValueError(f"invalid GT {field} in {sample['sample_id']}: {gt[field]}")
        for field in ("intrinsic_traits", "situational_traits"):
            invalid = sorted(set(gt[field]) - set(candidates[field]))
            if invalid:
                raise ValueError(f"invalid GT {field} in {sample['sample_id']}: {invalid}")
    repaired = {
        index
        for sample in samples
        for index in sample["benchmark_indices"]
        if any("tag_of_interest/reference audit" in x for x in sample["normalization_actions"])
    }
    if repaired != {65, 68, 69}:
        raise ValueError(f"unexpected explicit GT repair set: {sorted(repaired)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=TEST_AVAILABLE_CSV)
    parser.add_argument("--audio-dir", type=Path, default=TEST_AUDIO_DIR)
    parser.add_argument("--output", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--meta-output", type=Path, default=MANIFEST_META_PATH)
    args = parser.parse_args()
    schema = load_schema()
    samples = build_samples(args.input.resolve(), schema, args.audio_dir.resolve())
    validate_samples(samples, schema)
    write_jsonl(args.output.resolve(), samples)
    sources: dict[str, int] = {}
    for sample in samples:
        sources[sample["source"]] = sources.get(sample["source"], 0) + 1
    meta = {
        "protocol": schema["protocol"],
        "input_rows": sum(len(sample["benchmark_indices"]) for sample in samples),
        "unique_audio": len(samples),
        "sources": sources,
        "duration_seconds": round(sum(float(sample["duration_seconds"]) for sample in samples), 6),
        "schema_sha256": sha256_file(SCHEMA_PATH),
        "test_available_csv_sha256": sha256_file(args.input.resolve()),
        "test_full_csv_sha256": sha256_file(TEST_FULL_CSV),
        "normalization_action_counts": {
            "move_enunciated": sum(
                any("moved enunciated" in action for action in sample["normalization_actions"])
                for sample in samples
            ),
            "explicit_gt_repair": sum(
                any("tag_of_interest/reference audit" in action for action in sample["normalization_actions"])
                for sample in samples
            ),
        },
    }
    args.meta_output.parent.mkdir(parents=True, exist_ok=True)
    args.meta_output.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
