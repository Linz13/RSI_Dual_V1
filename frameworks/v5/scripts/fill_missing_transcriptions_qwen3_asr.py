#!/usr/bin/env python3
"""Fill blank source transcriptions with local Qwen3-ASR, with resumable progress."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from tqdm.auto import tqdm


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def semantic(row: dict[str, Any]) -> dict[str, Any]:
    try:
        value = row["Target_JSON_Schema"]["semantic_content"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Row {row.get('sample_id')} has no Target_JSON_Schema.semantic_content"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(f"Row {row.get('sample_id')} semantic_content is not an object")
    return value


def role_index(data_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for role in ("paired", "audio_only", "caption_only"):
        for row in read_jsonl(data_dir / f"{role}.jsonl"):
            sample_id = str(row.get("id") or "").strip()
            if not sample_id or sample_id in result:
                raise ValueError(f"Invalid or duplicate partition ID: {sample_id!r}")
            result[sample_id] = role
    return result


def resolve_audio(data_dir: Path, row: dict[str, Any]) -> Path:
    name = Path(str(row.get("audio_path") or "")).name
    path = data_dir / "audio" / name
    if not name or not path.is_file():
        raise FileNotFoundError(
            f"Local audio for {row.get('sample_id')} does not exist: {path}"
        )
    return path


def task_rows(
    rows: list[dict[str, Any]], roles: dict[str, str]
) -> list[tuple[dict[str, Any], str]]:
    tasks: list[tuple[dict[str, Any], str]] = []
    for row in rows:
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id or sample_id not in roles:
            raise ValueError(f"Label ID is missing from partitions: {sample_id!r}")
        transcription = semantic(row).get("transcription")
        if not isinstance(transcription, str) or not transcription.strip():
            tasks.append((row, roles[sample_id]))
    return tasks


def forced_language(row: dict[str, Any]) -> str | None:
    language = str(semantic(row).get("language") or "").strip()
    return language if language in {"Chinese", "English"} else None


def load_progress(
    path: Path, expected_meta: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    expected_header = {"type": "meta", **expected_meta}
    if not path.exists():
        append_jsonl(path, expected_header)
        return {}
    entries = read_jsonl(path)
    if not entries or entries[0] != expected_header:
        raise RuntimeError(
            f"Progress metadata does not match this experiment: {path}. "
            "Move the old progress file aside before starting a different experiment."
        )
    latest: dict[str, dict[str, Any]] = {}
    for entry in entries[1:]:
        if entry.get("type") == "result" and entry.get("sample_id"):
            latest[str(entry["sample_id"])] = entry
    return latest


def chunks(values: Sequence[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def transcribe_batch(model, data_dir: Path, batch, max_new_tokens: int):
    del max_new_tokens  # Configured once when the model wrapper is created.
    audio_paths = [str(resolve_audio(data_dir, row)) for row, _role in batch]
    languages = [forced_language(row) for row, _role in batch]
    results = model.transcribe(
        audio=audio_paths,
        language=languages,
        return_time_stamps=False,
    )
    if len(results) != len(batch):
        raise RuntimeError(f"Expected {len(batch)} results, received {len(results)}")
    return results


def result_entry(
    row: dict[str, Any], role: str, audio_path: Path, result
) -> dict[str, Any]:
    text = str(result.text or "").strip()
    if not text:
        raise RuntimeError("Qwen3-ASR returned an empty transcription")
    return {
        "type": "result",
        "status": "ok",
        "sample_id": str(row["sample_id"]),
        "role": role,
        "audio_file": audio_path.name,
        "source_language": semantic(row).get("language"),
        "detected_language": result.language,
        "transcription": text,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    progress_path = args.progress.expanduser().resolve()
    report_path = args.report.expanduser().resolve()

    if output_path == input_path:
        raise ValueError("Output must differ from input; the source JSONL is read-only")
    required_model_files = (
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )
    missing_model_files = [
        name for name in required_model_files if not (model_path / name).is_file()
    ]
    if missing_model_files:
        raise FileNotFoundError(
            f"Qwen3-ASR model is incomplete at {model_path}; missing {missing_model_files}"
        )

    rows = read_jsonl(input_path)
    roles = role_index(data_dir)
    if len(rows) != len(roles):
        raise ValueError(f"Label/partition count mismatch: {len(rows)} != {len(roles)}")
    ids = [str(row.get("sample_id") or "") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample_id values in label file")
    for row in rows:
        resolve_audio(data_dir, row)

    tasks = task_rows(rows, roles)
    language_counts = Counter(forced_language(row) or "auto" for row, _role in tasks)
    meta = {
        "algorithm_version": 1,
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "model": str(model_path),
        "model_index_sha256": sha256(model_path / "model.safetensors.index.json"),
        "max_new_tokens": args.max_new_tokens,
        "language_policy": "force source Chinese/English; otherwise auto-detect",
        "existing_transcription_policy": "preserve exactly; fill blanks only",
    }

    audit = {
        **meta,
        "input_rows": len(rows),
        "existing_transcriptions": len(rows) - len(tasks),
        "blank_transcriptions": len(tasks),
        "target_tasks": len(tasks),
        "target_role_counts": dict(Counter(role for _row, role in tasks)),
        "target_language_counts": dict(language_counts),
    }
    if args.audit_only:
        return {**audit, "status": "audit_only"}

    progress = load_progress(progress_path, meta)
    successful = {
        sample_id: value
        for sample_id, value in progress.items()
        if value.get("status") == "ok" and str(value.get("transcription") or "").strip()
    }
    pending = [
        task for task in tasks if str(task[0]["sample_id"]) not in successful
    ]
    if args.max_new_items:
        pending = pending[: args.max_new_items]

    if pending:
        import torch
        from qwen_asr import Qwen3ASRModel

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the selected Python environment")
        print(
            f"Loading Qwen3-ASR from {model_path} on {args.device}; "
            f"pending={len(pending)}, batch_size={args.batch_size}",
            flush=True,
        )
        model = Qwen3ASRModel.from_pretrained(
            str(model_path),
            dtype=torch.bfloat16,
            device_map=args.device,
            local_files_only=True,
            max_inference_batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        bar = tqdm(
            total=len(pending),
            desc="Qwen3-ASR transcription",
            unit="clip",
            dynamic_ncols=True,
        )

        def save_success(batch, results) -> None:
            for (row, role), result in zip(batch, results):
                sample_id = str(row["sample_id"])
                audio_path = resolve_audio(data_dir, row)
                try:
                    entry = result_entry(row, role, audio_path, result)
                    successful[sample_id] = entry
                    bar.set_postfix(role=role, id=sample_id[-12:])
                except Exception as exc:  # noqa: BLE001
                    entry = {
                        "type": "result",
                        "status": "error",
                        "sample_id": sample_id,
                        "role": role,
                        "audio_file": audio_path.name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    bar.set_postfix(error=sample_id[-12:])
                append_jsonl(progress_path, entry)
                bar.update(1)

        for batch in chunks(pending, args.batch_size):
            try:
                save_success(
                    batch,
                    transcribe_batch(model, data_dir, batch, args.max_new_tokens),
                )
            except Exception as batch_exc:  # noqa: BLE001
                print(
                    f"Batch failed ({type(batch_exc).__name__}: {batch_exc}); "
                    "retrying each sample individually.",
                    flush=True,
                )
                torch.cuda.empty_cache()
                for task in batch:
                    try:
                        save_success(
                            [task],
                            transcribe_batch(model, data_dir, [task], args.max_new_tokens),
                        )
                    except Exception as exc:  # noqa: BLE001
                        row, role = task
                        sample_id = str(row["sample_id"])
                        audio_path = resolve_audio(data_dir, row)
                        entry = {
                            "type": "result",
                            "status": "error",
                            "sample_id": sample_id,
                            "role": role,
                            "audio_file": audio_path.name,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        append_jsonl(progress_path, entry)
                        bar.set_postfix(error=sample_id[-12:])
                        bar.update(1)
        bar.close()

    unresolved = [
        str(row["sample_id"])
        for row, _role in tasks
        if str(row["sample_id"]) not in successful
    ]
    report: dict[str, Any] = {
        **audit,
        "output": str(output_path),
        "progress": str(progress_path),
        "completed_tasks": len(tasks) - len(unresolved),
        "unresolved_tasks": unresolved,
    }
    if unresolved:
        report["status"] = "incomplete"
        atomic_write_json(report_path, report)
        raise RuntimeError(
            f"Qwen3-ASR experiment is incomplete: {len(unresolved)} tasks remain. "
            f"Re-run the same command to resume; report: {report_path}"
        )

    output_rows = json.loads(json.dumps(rows, ensure_ascii=False))
    filled = 0
    for row in output_rows:
        values = semantic(row)
        if isinstance(values.get("transcription"), str) and values["transcription"].strip():
            continue
        result = successful[str(row["sample_id"])]
        values["transcription"] = result["transcription"]
        filled += 1

    remaining_blank = sum(
        not bool(str(semantic(row).get("transcription") or "").strip())
        for row in output_rows
    )
    if filled != len(tasks) or remaining_blank:
        raise RuntimeError(
            f"Output validation failed: filled={filled}, expected={len(tasks)}, "
            f"remaining_blank={remaining_blank}"
        )

    atomic_write_jsonl(output_path, output_rows)
    report.update(
        {
            "status": "complete",
            "qwen3_asr_transcriptions_filled": filled,
            "remaining_blank_in_output": remaining_blank,
            "output_sha256": sha256(output_path),
        }
    )
    atomic_write_json(report_path, report)
    return report


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    caption_root = project.parent
    shared_root = caption_root.parent
    data_dir = caption_root / "training_data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=data_dir / "labels_open_resolved_no_environment_check_trans.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=data_dir
        / "labels_open_resolved_no_environment_check_trans_qwen3_asr_completed.jsonl",
    )
    parser.add_argument("--data-dir", type=Path, default=data_dir)
    parser.add_argument(
        "--model",
        type=Path,
        default=shared_root / "Models/Qwen3-ASR/ckpt/Qwen3-ASR-1.7B",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--progress",
        type=Path,
        default=project / "runs/qwen3_asr_transcription_completion.progress.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=project / "reports/qwen3_asr_transcription_completion.json",
    )
    parser.add_argument(
        "--max-new-items",
        type=int,
        default=0,
        help="Debug only; 0 processes every pending task",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate inputs and show task counts without loading the model",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1 or args.max_new_items < 0:
        parser.error(
            "--batch-size and --max-new-tokens must be positive; "
            "--max-new-items must be non-negative"
        )
    result = run(args)
    if args.audit_only:
        atomic_write_json(args.report.expanduser().resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
