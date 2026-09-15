#!/usr/bin/env python3
"""Run six independent closed-set Attr6 questions per audio sample."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BENCHMARK_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BENCHMARK_ROOT))
from model_adapter_utils import describe_adapter, sha256_json  # noqa: E402

from common import (  # noqa: E402
    ALL_FIELDS,
    DEFAULT_MIDASHENG_MODEL_DIR,
    DEFAULT_MODEL_DIR,
    MANIFEST_PATH,
    append_jsonl,
    load_schema,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_jsonl,
)
from content_scheme_a import (  # noqa: E402
    PROTOCOL,
    build_all_field_prompts,
    parse_field_response,
)
from run_qwen3_captioner import infer_one, load_captioner  # noqa: E402


TERMINAL_FIELD_STATUSES = {"success", "unparsed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--backend", choices=("qwen3", "midasheng", "qwen25"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--attn-backend",
        choices=("flash_attention_2", "sdpa"),
        default="sdpa",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def latest_field_records(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id", ""))
        field = str(row.get("field", ""))
        if sample_id and field in ALL_FIELDS:
            latest[(sample_id, field)] = row
    return latest


def evaluation_identity(
    args: argparse.Namespace,
    manifest_path: Path,
    selected: list[dict[str, Any]],
    prompts: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    model_dir = args.model_dir.expanduser().resolve()
    adapter = describe_adapter(args.adapter_dir, model_dir)
    payload = {
        "protocol": PROTOCOL,
        "candidate_name": args.candidate_name,
        "backend": args.backend,
        "model_dir": str(model_dir),
        "adapter": adapter,
        "manifest_sha256": sha256_file(manifest_path),
        "selected_sample_ids_sha256": sha256_json([row["sample_id"] for row in selected]),
        "field_prompt_sha256": {
            field: sha256_text(prompt) for field, prompt in prompts.items()
        },
        "max_new_tokens": args.max_new_tokens,
        "attn_backend": args.attn_backend,
        "do_sample": False,
    }
    return sha256_json(payload), payload


def aggregate_predictions(
    selected: list[dict[str, Any]],
    field_records: dict[tuple[str, str], dict[str, Any]],
    identity_sha256: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in selected:
        fields = {
            field: field_records.get(
                (sample["sample_id"], field),
                {
                    "field": field,
                    "status": "missing",
                    "prediction": None,
                    "error": "missing field prediction",
                },
            )
            for field in ALL_FIELDS
        }
        rows.append(
            {
                "protocol": PROTOCOL,
                "sample_id": sample["sample_id"],
                "benchmark_indices": sample["benchmark_indices"],
                "audio_path": sample["audio_path"],
                "source": sample["source"],
                "evaluation_identity_sha256": identity_sha256,
                "fields": fields,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    if args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative")
    manifest_path = args.manifest.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    if args.adapter_dir is not None and not args.adapter_dir.expanduser().resolve().is_dir():
        raise FileNotFoundError(args.adapter_dir)

    manifest = read_jsonl(manifest_path)
    if not manifest:
        raise ValueError(f"manifest is empty: {manifest_path}")
    selected = manifest[: args.max_samples] if args.max_samples else manifest
    schema = load_schema()
    prompts = build_all_field_prompts(schema)
    identity_sha256, identity = evaluation_identity(
        args, manifest_path, selected, prompts
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_manifest_path = output_dir / "selected_manifest.jsonl"
    field_records_path = output_dir / "field_records.jsonl"
    predictions_path = output_dir / "predictions.jsonl"
    existing_outputs = [
        path
        for path in (
            field_records_path,
            predictions_path,
            output_dir / "run_metadata.json",
        )
        if path.exists()
    ]
    if existing_outputs and not args.resume:
        raise RuntimeError(
            "Scheme A output already exists. Refusing to mix or overwrite it without "
            f"--resume: {[str(path) for path in existing_outputs]}"
        )
    write_jsonl(selected_manifest_path, selected)

    existing = latest_field_records(field_records_path) if args.resume else {}
    mismatched = [
        key
        for key, row in existing.items()
        if row.get("evaluation_identity_sha256") != identity_sha256
    ]
    if mismatched:
        raise RuntimeError(
            "Refusing --resume because existing Scheme A records belong to a different "
            f"evaluation identity; examples={mismatched[:3]}"
        )

    tasks = [
        (sample, field)
        for sample in selected
        for field in ALL_FIELDS
        if existing.get((sample["sample_id"], field), {}).get("status")
        not in TERMINAL_FIELD_STATUSES
    ]
    print(
        f"protocol={PROTOCOL} candidate={args.candidate_name} samples={len(selected)} "
        f"fields_per_sample={len(ALL_FIELDS)} total_fields={len(selected) * len(ALL_FIELDS)} "
        f"pending_fields={len(tasks)} cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
        flush=True,
    )
    started = time.perf_counter()
    bundle: dict[str, Any] | None = None
    if tasks:
        load_started = time.perf_counter()
        bundle = load_captioner(args)
        print(f"model_loaded_seconds={time.perf_counter() - load_started:.2f}", flush=True)
        if bundle["adapter"] is not None:
            print(
                "adapter_loaded="
                + json.dumps(bundle["adapter"], ensure_ascii=False, sort_keys=True),
                flush=True,
            )

    for ordinal, (sample, field) in enumerate(tasks, 1):
        field_started = time.perf_counter()
        raw = ""
        parsed: dict[str, Any] = {
            "status": "error",
            "prediction": None,
            "parse_mode": "inference_error",
        }
        error = ""
        try:
            assert bundle is not None
            raw = str(infer_one(bundle, sample["audio_path"], prompts[field], args) or "")
            parsed = parse_field_response(field, raw, schema)
        except Exception as exc:  # keep a resumable field-level checkpoint
            error = repr(exc)
        elapsed = time.perf_counter() - field_started
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": PROTOCOL,
            "candidate_name": args.candidate_name,
            "backend": args.backend,
            "sample_id": sample["sample_id"],
            "benchmark_indices": sample["benchmark_indices"],
            "audio_path": sample["audio_path"],
            "source": sample["source"],
            "field": field,
            "response_text": raw,
            **parsed,
            "error": error,
            "elapsed_seconds": round(elapsed, 6),
            "prompt_sha256": sha256_text(prompts[field]),
            "evaluation_identity_sha256": identity_sha256,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
        }
        append_jsonl(field_records_path, record)
        existing[(sample["sample_id"], field)] = record
        print(
            f"[{ordinal}/{len(tasks)}] {sample['sample_id']} field={field} "
            f"status={record['status']} parse={record['parse_mode']} seconds={elapsed:.2f}",
            flush=True,
        )
        if error:
            print(error, flush=True)

    latest = latest_field_records(field_records_path)
    predictions = aggregate_predictions(selected, latest, identity_sha256)
    write_jsonl(predictions_path, predictions)
    status_counts = Counter(
        prediction["fields"][field].get("status", "missing")
        for prediction in predictions
        for field in ALL_FIELDS
    )
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL,
        "candidate_name": args.candidate_name,
        "backend": args.backend,
        "model_dir": str(model_dir),
        "adapter": identity["adapter"],
        "manifest_sha256": sha256_file(manifest_path),
        "selected_manifest_sha256": sha256_file(selected_manifest_path),
        "evaluation_identity_sha256": identity_sha256,
        "evaluation_identity": identity,
        "python": sys.version,
        "platform": platform.platform(),
        "selected_samples": len(selected),
        "expected_field_records": len(selected) * len(ALL_FIELDS),
        "merged_field_records": sum(status_counts.values()),
        "field_status_counts": dict(status_counts),
        "wall_seconds": round(time.perf_counter() - started, 6),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    if status_counts.get("error", 0) or status_counts.get("missing", 0):
        print(
            "[RETRY] Inference errors or missing fields remain; rerun the same command with --resume.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
