#!/usr/bin/env python3
"""Fill blank source transcriptions with local Whisper, with resumable progress."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


TRAINING_CAPTION_ROLES = {"paired", "caption_only"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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
        raise ValueError(f"Row {row.get('sample_id')} has no Target_JSON_Schema.semantic_content") from exc
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
        raise FileNotFoundError(f"Local audio for {row.get('sample_id')} does not exist: {path}")
    return path


def normalize_detected_language(value: Any) -> str | None:
    text = str(value or "").strip().casefold()
    if text == "en" or text.startswith("en-") or text == "english":
        return "English"
    if text == "zh" or text.startswith("zh-") or text in {"chinese", "mandarin"}:
        return "Chinese"
    return None


def load_audio(path: Path):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        divisor = math.gcd(int(sample_rate), 16000)
        audio = resample_poly(audio, 16000 // divisor, int(sample_rate) // divisor).astype(np.float32)
    return audio


def detect_english_chinese(model, audio) -> tuple[str, dict[str, float]]:
    """Use Whisper probabilities, constrained to the experiment's two languages."""
    import whisper

    mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio), n_mels=model.dims.n_mels).to(model.device)
    _tokens, probabilities = model.detect_language(mel)
    selected = "zh" if float(probabilities.get("zh", 0.0)) >= float(probabilities.get("en", 0.0)) else "en"
    return selected, {"en": float(probabilities.get("en", 0.0)), "zh": float(probabilities.get("zh", 0.0))}


def task_rows(
    rows: list[dict[str, Any]], roles: dict[str, str], scope: str,
) -> list[tuple[dict[str, Any], str, bool, bool]]:
    tasks = []
    for row in rows:
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id or sample_id not in roles:
            raise ValueError(f"Label ID is missing from partitions: {sample_id!r}")
        values = semantic(row)
        transcription = values.get("transcription")
        missing_text = not isinstance(transcription, str) or not transcription.strip()
        needs_language = values.get("language") == "other"
        if not (missing_text or needs_language):
            continue
        role = roles[sample_id]
        if scope == "training-required" and role not in TRAINING_CAPTION_ROLES:
            continue
        tasks.append((row, role, missing_text, needs_language))
    return tasks


def load_progress(path: Path, expected_meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        append_jsonl(path, {"type": "meta", **expected_meta})
        return {}
    entries = read_jsonl(path)
    if not entries or entries[0] != {"type": "meta", **expected_meta}:
        raise RuntimeError(f"Progress metadata does not match this experiment: {path}")
    latest: dict[str, dict[str, Any]] = {}
    for entry in entries[1:]:
        if entry.get("type") == "result" and entry.get("sample_id"):
            latest[str(entry["sample_id"])] = entry
    return latest


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    data_dir = args.data_dir.resolve()
    model_path = args.model.resolve()
    progress_path = args.progress.resolve()
    report_path = args.report.resolve()
    if output_path == input_path:
        raise ValueError("Output must differ from input; the source JSONL is read-only")
    if not model_path.is_file():
        raise FileNotFoundError(f"Whisper checkpoint does not exist: {model_path}")
    rows = read_jsonl(input_path)
    roles = role_index(data_dir)
    if len(rows) != len(roles):
        raise ValueError(f"Label/partition count mismatch: {len(rows)} != {len(roles)}")
    ids = [str(row.get("sample_id") or "") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample_id values in label file")
    for row in rows:
        resolve_audio(data_dir, row)
    tasks = task_rows(rows, roles, args.scope)
    meta = {
        "algorithm_version": 2,
        "input": str(input_path), "input_sha256": sha256(input_path),
        "model": str(model_path), "model_size": model_path.stat().st_size,
        "scope": args.scope, "beam_size": args.beam_size,
        "condition_on_previous_text": False,
    }
    if args.audit_only:
        return {
            **meta, "status": "audit_only", "input_rows": len(rows),
            "existing_transcriptions": sum(
                bool(str(semantic(row).get("transcription") or "").strip()) for row in rows
            ),
            "blank_transcriptions": sum(
                not bool(str(semantic(row).get("transcription") or "").strip()) for row in rows
            ),
            "target_tasks": len(tasks),
            "scope_role_counts": dict(Counter(task[1] for task in tasks)),
            "missing_transcription_tasks": sum(task[2] for task in tasks),
            "language_detection_tasks": sum(task[3] for task in tasks),
        }
    progress = load_progress(progress_path, meta)
    successful = {sample_id: value for sample_id, value in progress.items() if value.get("status") == "ok"}
    pending = [task for task in tasks if str(task[0]["sample_id"]) not in successful]
    if args.max_new_items:
        pending = pending[:args.max_new_items]

    if pending:
        import torch
        import whisper

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        model = whisper.load_model(str(model_path), device=args.device, download_root=str(model_path.parent))
        bar = tqdm(pending, total=len(pending), desc="Whisper transcription", unit="clip", dynamic_ncols=True)
        for row, role, missing_text, needs_language in bar:
            sample_id = str(row["sample_id"])
            source_language = semantic(row).get("language")
            forced_language = "zh" if source_language == "Chinese" else "en" if source_language == "English" else None
            audio_path = resolve_audio(data_dir, row)
            try:
                audio = load_audio(audio_path)
                constrained_probabilities = None
                if needs_language:
                    forced_language, constrained_probabilities = detect_english_chinese(model, audio)
                result = model.transcribe(
                    audio, language=forced_language, fp16=args.device.startswith("cuda"),
                    temperature=0.0, beam_size=args.beam_size,
                    condition_on_previous_text=False, word_timestamps=False,
                )
                text = str(result.get("text") or "").strip()
                detected = normalize_detected_language(result.get("language"))
                if missing_text and not text:
                    raise RuntimeError("Whisper returned an empty transcription")
                if needs_language and detected not in {"English", "Chinese"}:
                    raise RuntimeError(f"Whisper language was not English/Chinese: {result.get('language')!r}")
                entry = {
                    "type": "result", "status": "ok", "sample_id": sample_id, "role": role,
                    "audio_file": audio_path.name, "transcription": text,
                    "detected_language": detected, "filled_transcription": missing_text,
                    "repaired_language_other": needs_language,
                }
                if constrained_probabilities is not None:
                    entry["whisper_language_probabilities_en_zh"] = constrained_probabilities
                successful[sample_id] = entry
                bar.set_postfix(role=role, id=sample_id[-12:])
            except Exception as exc:  # noqa: BLE001
                entry = {
                    "type": "result", "status": "error", "sample_id": sample_id, "role": role,
                    "audio_file": audio_path.name, "error": f"{type(exc).__name__}: {exc}",
                }
                bar.set_postfix(error=sample_id[-12:])
            append_jsonl(progress_path, entry)

    unresolved = [str(task[0]["sample_id"]) for task in tasks if str(task[0]["sample_id"]) not in successful]
    report: dict[str, Any] = {
        **meta, "output": str(output_path), "progress": str(progress_path),
        "input_rows": len(rows), "original_transcriptions_preserved": sum(
            bool(str(semantic(row).get("transcription") or "").strip()) for row in rows
        ),
        "target_tasks": len(tasks), "completed_tasks": len(tasks) - len(unresolved),
        "unresolved_tasks": unresolved, "scope_role_counts": dict(Counter(task[1] for task in tasks)),
    }
    if unresolved:
        atomic_write_json(report_path, report)
        raise RuntimeError(
            f"Whisper experiment is incomplete: {len(unresolved)} tasks remain. "
            f"Re-run the same command to resume; report: {report_path}"
        )

    output_rows = json.loads(json.dumps(rows))
    filled = 0
    language_repairs = 0
    for row in output_rows:
        sample_id = str(row["sample_id"])
        result = successful.get(sample_id)
        if result is None:
            continue
        values = semantic(row)
        if result["filled_transcription"]:
            values["transcription"] = result["transcription"]
            filled += 1
        if result["repaired_language_other"]:
            values["language"] = result["detected_language"]
            language_repairs += 1
    atomic_write_jsonl(output_path, output_rows)
    report.update({
        "status": "complete", "whisper_transcriptions_filled": filled,
        "language_other_repaired": language_repairs, "output_sha256": sha256(output_path),
        "remaining_blank_in_output": sum(
            not bool(str(semantic(row).get("transcription") or "").strip()) for row in output_rows
        ),
    })
    atomic_write_json(report_path, report)
    return report


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    caption_root = project.parent
    data_dir = caption_root / "training_data"
    input_path = data_dir / "labels_open_resolved_no_environment_check_trans.jsonl"
    output_path = data_dir / "labels_open_resolved_no_environment_check_trans_whisper_completed.jsonl"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=input_path)
    parser.add_argument("--output", type=Path, default=output_path)
    parser.add_argument("--data-dir", type=Path, default=data_dir)
    parser.add_argument("--model", type=Path, default=caption_root / "models/whisper/large-v3-turbo.pt")
    parser.add_argument("--scope", choices=("all", "training-required"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--progress", type=Path, default=project / "runs/whisper_transcription_completion.progress.jsonl")
    parser.add_argument("--report", type=Path, default=project / "reports/whisper_transcription_completion.json")
    parser.add_argument("--max-new-items", type=int, default=0, help="Debug only; 0 processes every pending task")
    parser.add_argument("--audit-only", action="store_true", help="Validate inputs and show the task count without loading Whisper")
    args = parser.parse_args()
    if args.beam_size < 1 or args.max_new_items < 0:
        parser.error("--beam-size must be >=1 and --max-new-items must be >=0")
    result = run(args)
    if args.audit_only:
        atomic_write_json(args.report.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
