#!/usr/bin/env python3
"""Audit or build the portable, no-environment DualISL training_data_v2.

Transcript files are applied in command-line order.  The first value for an ID
wins, so pass original transcripts before Whisper fallback transcripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


ROLES = ("paired", "audio_only", "caption_only")
MIN_DURATION = 1.2
MAX_DURATION = 30.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_language(value: Any) -> str | None:
    text = str(value or "").strip().casefold().replace("_", "-")
    if text in {"english", "en", "en-us", "en-gb"} or text.startswith("en-"):
        return "English"
    if text in {"chinese", "zh", "zh-cn", "zh-tw", "mandarin"} or text.startswith("zh-"):
        return "Chinese"
    return None


def transcript_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        if isinstance(value.get("rows"), list):
            return value["rows"]
        return [
            ({"id": key, **item} if isinstance(item, dict) else {"id": key, "transcript": item})
            for key, item in value.items()
        ]
    raise ValueError(f"Unsupported transcript container in {path}")


def load_transcripts(
    paths: list[Path],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    lookup: dict[str, dict[str, Any]] = {}
    ignored_conflicts: list[dict[str, Any]] = []
    language_fallbacks: list[dict[str, Any]] = []
    for path in paths:
        for row in transcript_rows(path):
            nested = row.get("Target_JSON_Schema") or {}
            nested_semantic = nested.get("semantic_content", {}) if isinstance(nested, dict) else {}
            sample_id = str(row.get("id") or row.get("sample_id") or row.get("group_id") or "").strip()
            audio_name = Path(str(row.get("audio_path") or row.get("audio") or "")).name
            text = str(
                row.get("transcript") or row.get("transcription") or row.get("text")
                or nested_semantic.get("transcript") or nested_semantic.get("transcription") or ""
            ).strip()
            if not text or text.casefold() == "unknown":
                continue
            item = {
                "transcript": text,
                "language": normalize_language(
                    row.get("detected_language") or row.get("language") or nested_semantic.get("language")
                ),
                "source": str(path.resolve()),
            }
            keys = [key for key in (sample_id, audio_name, Path(audio_name).stem if audio_name else "") if key]
            if not keys:
                raise ValueError(f"Transcript row in {path} has neither ID nor audio path")
            language_fallback_applied = False
            for key in keys:
                previous = lookup.get(key)
                if previous is None:
                    lookup[key] = item
                    continue
                if previous["language"] is None and item["language"] in {"English", "Chinese"}:
                    previous["language"] = item["language"]
                    language_fallback_applied = True
                if previous["transcript"] != text:
                    ignored_conflicts.append({"key": key, "kept": previous["source"], "ignored": str(path.resolve())})
            if language_fallback_applied:
                language_fallbacks.append({
                    "key": sample_id or keys[0],
                    "language": item["language"],
                    "source": str(path.resolve()),
                })
    return lookup, ignored_conflicts, language_fallbacks


def audio_info(path: Path) -> tuple[float, int, int]:
    import soundfile as sf

    info = sf.info(str(path))
    return float(info.duration), int(info.samplerate), int(info.channels)


def resolve_audio(source_dir: Path, row: dict[str, Any]) -> Path:
    name = Path(str(row.get("audio_path") or "")).name
    if not name:
        raise ValueError(f"Audio row {row.get('id')} has no audio_path")
    path = source_dir / "audio" / name
    if not path.is_file():
        raise FileNotFoundError(f"Audio for {row.get('id')} does not exist: {path}")
    return path


def audit(source_dir: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    role_rows = {role: read_jsonl(source_dir / f"{role}.jsonl") for role in ROLES}
    report: dict[str, Any] = {
        "source_dir": str(source_dir.resolve()),
        "policy": {"min_duration": MIN_DURATION, "max_duration": MAX_DURATION, "action": "remove_without_trimming"},
        "input_counts": {role: len(rows) for role, rows in role_rows.items()},
        "input_manifest_sha256": {role: sha256(source_dir / f"{role}.jsonl") for role in ROLES},
        "source_audio_file_count": len(list((source_dir / "audio").glob("*"))),
        "accepted_counts": {},
        "excluded_audio": [],
        "audio_warnings": [],
    }
    for role, rows in role_rows.items():
        accepted = 0
        for row in rows:
            if role == "caption_only":
                accepted += 1
                continue
            path = resolve_audio(source_dir, row)
            duration, sample_rate, channels = audio_info(path)
            if not MIN_DURATION <= duration <= MAX_DURATION:
                report["excluded_audio"].append({
                    "role": role, "id": row.get("id"), "file": path.name,
                    "duration": duration, "reason": "outside_duration_range",
                })
                continue
            accepted += 1
            if channels != 1:
                report["audio_warnings"].append({"role": role, "id": row.get("id"), "channels": channels})
            if sample_rate < 8000:
                report["audio_warnings"].append({"role": role, "id": row.get("id"), "sample_rate": sample_rate})
        report["accepted_counts"][role] = accepted
    report["excluded_count"] = len(report["excluded_audio"])
    return report, role_rows


def source_caption(row: dict[str, Any], transcript_lookup: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    caption = json.loads(json.dumps(row.get("caption")))
    if not isinstance(caption, dict):
        raise ValueError(f"Caption row {row.get('id')} has no caption object")
    caption.pop("environment", None)
    semantic = caption.setdefault("semantic_content", {})
    existing = str(semantic.get("transcript") or "").strip()
    audio_name = Path(str(row.get("audio_path") or "")).name
    match = next((transcript_lookup[key] for key in (str(row.get("id")), audio_name, Path(audio_name).stem) if key in transcript_lookup), None)
    if not existing or existing.casefold() == "unknown":
        if match is None:
            raise ValueError(f"Missing transcript for source-domain caption {row.get('id')}")
        semantic["transcript"] = match["transcript"]
    language = str(semantic.get("language") or "unknown")
    if language == "other":
        if match is None or match["language"] not in {"English", "Chinese"}:
            raise ValueError(f"Caption {row.get('id')} needs Whisper English/Chinese language detection")
        semantic["language"] = match["language"]
    paralinguistic = caption.get("paralinguistic", {})
    repair = {"neutral_intensity": False, "language_other": language == "other"}
    if paralinguistic.get("emotion") == "neutral" and paralinguistic.get("emotion_intensity") in {"medium", "high"}:
        paralinguistic["emotion_intensity"] = "low"
        repair["neutral_intensity"] = True
    return caption, repair


def build(source_dir: Path, output_dir: Path, transcript_paths: list[Path], force: bool) -> dict[str, Any]:
    report, role_rows = audit(source_dir)
    transcripts, conflicts, language_fallbacks = load_transcripts(transcript_paths)
    report["transcript_files"] = [str(path.resolve()) for path in transcript_paths]
    report["transcript_conflicts_ignored"] = conflicts
    report["transcript_language_fallbacks"] = language_fallbacks
    parent = output_dir.resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and not force:
        raise FileExistsError(f"Output already exists (use --force to replace): {output_dir}")
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=parent))
    repair_counts: Counter[str] = Counter()
    try:
        output_rows: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLES}
        copied_audio: set[str] = set()
        for role, rows in role_rows.items():
            for raw in rows:
                row = dict(raw)
                source_audio: Path | None = None
                if role != "caption_only":
                    source_audio = resolve_audio(source_dir, row)
                    duration, _, _ = audio_info(source_audio)
                    if not MIN_DURATION <= duration <= MAX_DURATION:
                        continue
                    row["audio_path"] = f"audio/{source_audio.name}"
                    if source_audio.name not in copied_audio:
                        (staging / "audio").mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_audio, staging / "audio" / source_audio.name)
                        copied_audio.add(source_audio.name)
                else:
                    row["audio_path"] = ""
                row.pop("codec_path", None)
                if role in {"paired", "caption_only"}:
                    row["caption"], repairs = source_caption(row, transcripts)
                    row["known_fields"] = [
                        "semantic_content.language", "semantic_content.transcript", "semantic_content.topic", "semantic_content.intent",
                        "speaker_profile.gender", "speaker_profile.age", "speaker_profile.timbre", "speaker_profile.accent",
                        "paralinguistic.speaking_rate", "paralinguistic.pitch_level", "paralinguistic.volume_level",
                        "paralinguistic.emotion", "paralinguistic.emotion_intensity", "paralinguistic.emphasis.level",
                        "paralinguistic.emphasis.emphasized_text", "paralinguistic.prosody", "paralinguistic.pause",
                        "paralinguistic.nonverbal_vocalization",
                    ]
                    repair_counts.update(key for key, applied in repairs.items() if applied)
                else:
                    row.pop("caption", None)
                    row.pop("known_fields", None)
                output_rows[role].append(row)
        for role, rows in output_rows.items():
            write_jsonl(staging / f"{role}.jsonl", rows)
        smoke = staging / "smoke"
        smoke_rows_by_role: dict[str, list[dict[str, Any]]] = {}
        for role, rows in output_rows.items():
            smoke_rows = []
            for row in rows[:8]:
                smoke_row = dict(row)
                if smoke_row.get("audio_path"):
                    # Smoke manifests live one level below the full manifests,
                    # so their relative paths must step back to the shared audio dir.
                    smoke_row["audio_path"] = f"../{smoke_row['audio_path']}"
                smoke_rows.append(smoke_row)
            smoke_rows_by_role[role] = smoke_rows
            write_jsonl(smoke / f"{role}.jsonl", smoke_rows)
        report["output_counts"] = {role: len(rows) for role, rows in output_rows.items()}
        report["smoke_counts"] = {role: len(rows) for role, rows in smoke_rows_by_role.items()}
        report["repairs"] = dict(sorted(repair_counts.items()))
        report["copied_audio_files"] = len(copied_audio)
        report["manifest_sha256"] = {role: sha256(staging / f"{role}.jsonl") for role in ROLES}
        report["smoke_manifest_sha256"] = {role: sha256(smoke / f"{role}.jsonl") for role in ROLES}
        (staging / "migration_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from dual_isl_train.data import validate_manifest_roles
        from dual_isl_train.schema import canonical_source_caption

        for role in ("paired", "caption_only"):
            for row in output_rows[role]:
                canonical_source_caption(row["caption"])
        validation = validate_manifest_roles({role: staging / f"{role}.jsonl" for role in ROLES}, hash_audio=True)
        if not validation["ok"]:
            raise RuntimeError(f"Built manifests failed cross-role validation: {validation['errors'][:3]}")
        report["cross_role_validation"] = validation
        smoke_validation = validate_manifest_roles(
            {role: smoke / f"{role}.jsonl" for role in ROLES}, hash_audio=True,
        )
        missing_smoke_audio = [
            {"role": role, "id": row["id"], "audio_path": row["audio_path"]}
            for role, rows in smoke_rows_by_role.items()
            for row in rows
            if row.get("audio_path") and not (smoke / row["audio_path"]).resolve().is_file()
        ]
        if not smoke_validation["ok"] or missing_smoke_audio:
            raise RuntimeError(
                "Built smoke manifests failed validation: "
                f"errors={smoke_validation['errors'][:3]}, missing_audio={missing_smoke_audio[:3]}"
            )
        report["smoke_cross_role_validation"] = smoke_validation
        (staging / "migration_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if output_dir.exists():
            backup = output_dir.with_name(output_dir.name + ".previous")
            if backup.exists():
                raise FileExistsError(f"Refusing to overwrite existing backup: {backup}")
            output_dir.rename(backup)
        staging.rename(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=project_root.parent / "training_data")
    parser.add_argument("--output-dir", type=Path, default=project_root.parent / "training_data_v2")
    parser.add_argument("--transcripts", type=Path, action="append", default=[])
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        report, _ = audit(args.source_dir.resolve())
    else:
        if not args.transcripts:
            parser.error("build requires at least one --transcripts file; use --audit-only while transcripts are pending")
        report = build(args.source_dir.resolve(), args.output_dir.resolve(), args.transcripts, args.force)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
