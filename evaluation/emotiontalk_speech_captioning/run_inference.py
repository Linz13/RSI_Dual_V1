#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from common import ROOT, TASKS, read_jsonl, sha256_file, stable_hash, write_json, write_jsonl
from backends.base import CaptionBackend, CaptionRequest
from model_adapter_utils import describe_adapter


def parse_tasks(value: str) -> tuple[str, ...]:
    tasks = TASKS if value == "all" else tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = set(tasks) - set(TASKS)
    if invalid or not tasks:
        raise argparse.ArgumentTypeError(f"invalid task selection: {value}")
    return tasks


def load_backend(name: str, **kwargs: object) -> CaptionBackend:
    if name == "qwen25":
        from backends.qwen25_omni import Qwen25OmniBackend
        return Qwen25OmniBackend(**kwargs)
    if name == "mock":
        from backends.mock import MockBackend
        return MockBackend(**kwargs)
    if name == "qwen3_omni":
        from backends.qwen3_omni import Qwen3OmniBackend
        return Qwen3OmniBackend(**kwargs)
    if name == "midasheng":
        from backends.midasheng import MiDashengBackend
        return MiDashengBackend(**kwargs)
    if ":" not in name:
        raise ValueError("Custom backend must use module:Class syntax")
    module_name, class_name = name.split(":", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(**kwargs)


def select_requests(rows: list[dict], tasks: tuple[str, ...], max_samples: int | None) -> list[dict]:
    selected_ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        uid = row["id"]
        if uid not in seen:
            if max_samples is not None and len(selected_ids) >= max_samples:
                continue
            selected_ids.append(uid)
            seen.add(uid)
    allowed_ids = set(selected_ids)
    return [row for row in rows if row["id"] in allowed_ids and row["task"] in tasks]


def validate_resume_identity(events: list[dict], run_id: str) -> None:
    if any(event.get("run_id") != run_id for event in events):
        raise ValueError("Resume identity mismatch: model, prompt, manifest, task selection, or decoding changed")


def generate_batch_resilient(
    backend: CaptionBackend, requests: list[CaptionRequest]
) -> list[tuple[str | None, dict[str, str] | None]]:
    """Generate a batch and recursively isolate sample-specific failures."""
    try:
        predictions = backend.generate_batch(requests)
        if len(predictions) != len(requests):
            raise RuntimeError(
                f"backend output count mismatch: {len(predictions)} != {len(requests)}"
            )
        return [(prediction.strip(), None) for prediction in predictions]
    except Exception as error:
        detail = {
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    # Leave the exception scope before retrying. This releases references held
    # by its traceback, which matters when the original failure was CUDA OOM.
    if len(requests) > 1:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        midpoint = len(requests) // 2
        print(
            f"[batch split] size={len(requests)} "
            f"error={detail['error_type']}: {detail['error']}",
            flush=True,
        )
        return generate_batch_resilient(
            backend, requests[:midpoint]
        ) + generate_batch_resilient(backend, requests[midpoint:])
    return [(None, detail)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pluggable speech-captioning inference.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/test_inference.jsonl")
    parser.add_argument("--backend", default="mock")
    parser.add_argument("--tasks", type=parse_tasks, default=TASKS)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--model-path", default="/data/L202500147/Caption/models/Qwen3-Omni-30B-A3B-Captioner")
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--attn-backend", choices=("sdpa", "flash_attention_2"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    manifest = args.manifest.resolve()
    root = ROOT
    rows = read_jsonl(manifest)
    selected = select_requests(rows, args.tasks, args.max_samples)
    if not selected:
        raise ValueError("No inference requests selected")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "run_metadata.json"
    events_path = output_dir / "events.jsonl"
    predictions_path = output_dir / "predictions.jsonl"

    adapter_dir = args.adapter_dir.resolve() if args.adapter_dir is not None else None
    adapter = describe_adapter(adapter_dir, args.model_path) if adapter_dir is not None else None
    pre_identity: dict[str, object] = {
        "backend": args.backend,
        "model_path": str(Path(args.model_path).resolve()),
        "max_new_tokens": args.max_new_tokens,
        "manifest_sha256": sha256_file(manifest),
        "selected_prompt_hash": stable_hash([(row["task"], row["prompt"]) for row in selected]),
        "tasks": list(args.tasks),
        "max_samples": args.max_samples,
    }
    if adapter is not None:
        pre_identity["adapter"] = adapter
    if args.attn_backend is not None:
        pre_identity["attn_backend"] = args.attn_backend
    if args.batch_size != 1:
        pre_identity["batch_size"] = args.batch_size
    run_id = stable_hash(pre_identity)
    prior_events = read_jsonl(events_path) if events_path.exists() else []
    if prior_events and not args.resume:
        raise FileExistsError(f"{events_path} exists; use --resume or a new output directory")
    validate_resume_identity(prior_events, run_id)
    completed = {(event["id"], event["task"]): event for event in prior_events if event.get("status") == "ok"}
    pending_rows = [row for row in selected if (row["id"], row["task"]) not in completed]

    old_metadata = None
    if metadata_path.exists() and args.resume:
        old_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if old_metadata.get("run_id") != run_id:
            raise ValueError("run_metadata.json identity mismatch")

    backend = None
    if pending_rows:
        backend = load_backend(
            args.backend,
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
            adapter_dir=str(adapter_dir) if adapter_dir is not None else None,
            attn_backend=args.attn_backend,
        )
        metadata = {
            "run_id": run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "request_count": len(selected),
            "request_schema": ["id", "task", "audio_path", "prompt"],
            "prediction_schema": ["id", "task", "prediction"],
            "identity": pre_identity,
            "backend_identity": backend.identity(),
        }
    elif old_metadata is None:
        raise FileNotFoundError(
            f"{metadata_path} is required for a fully completed resumed run"
        )

    if old_metadata is None:
        write_json(metadata_path, metadata)

    with events_path.open("a", encoding="utf-8") as event_file:
        for offset in range(0, len(pending_rows), args.batch_size):
            batch_rows = pending_rows[offset:offset + args.batch_size]
            requests = [CaptionRequest(
                id=row["id"], task=row["task"],
                audio_path=str((root / row["audio_path"]).resolve()), prompt=row["prompt"],
            ) for row in batch_rows]
            end = min(offset + len(requests), len(pending_rows))
            print(
                f"[{offset + 1}-{end}/{len(pending_rows)}] "
                f"batch={len(requests)} {requests[0].id}/{requests[0].task}",
                flush=True,
            )
            assert backend is not None
            results = generate_batch_resilient(backend, requests)
            for request, (prediction, error) in zip(requests, results, strict=True):
                key = (request.id, request.task)
                if error is None and prediction:
                    event = {
                        "run_id": run_id, "status": "ok", "id": request.id,
                        "task": request.task, "prediction": prediction,
                    }
                    completed[key] = event
                else:
                    detail = error or {
                        "error_type": "RuntimeError",
                        "error": "backend returned an empty prediction",
                        "traceback": "",
                    }
                    event = {
                        "run_id": run_id, "status": "error", "id": request.id,
                        "task": request.task, **detail,
                    }
                event_file.write(
                    json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                event_file.flush()

    if not pending_rows:
        for index, row in enumerate(selected, 1):
            print(f"[{index}/{len(selected)}] resume skip {row['id']}/{row['task']}", flush=True)

    ordered_predictions = []
    for row in selected:
        event = completed.get((row["id"], row["task"]))
        if event:
            ordered_predictions.append({"id": event["id"], "task": event["task"], "prediction": event["prediction"]})
    write_jsonl(predictions_path, ordered_predictions)
    failures = len(selected) - len(ordered_predictions)
    print(f"Saved {len(ordered_predictions)} predictions to {predictions_path}; failures={failures}")
    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
