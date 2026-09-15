from __future__ import annotations

import wave
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .constants import SOURCE_FIELDS
from .io import atomic_json, read_jsonl, sha256_file, stable_hash, write_jsonl
from .schema import canonical_source_caption, get_path, source_validation_errors

MODALITIES = {"paired", "audio_only", "caption_only"}
SPLITS = {"train", "validation", "test"}


@dataclass
class ValidationReport:
    rows: int = 0
    valid_rows: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    modality_counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "rows": self.rows, "valid_rows": self.valid_rows,
            "modality_counts": self.modality_counts, "errors": self.errors, "warnings": self.warnings,
        }


def _audio_info(path: Path) -> tuple[float, int, int]:
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as handle:
                rate = handle.getframerate()
                frames = handle.getnframes()
                return frames / float(rate), rate, handle.getnchannels()
        except wave.Error:
            # IEEE-float WAV is valid but unsupported by Python's stdlib wave
            # reader. Fall through to libsndfile instead of rewriting audio.
            pass
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ValueError(f"soundfile is required to inspect non-WAV audio: {path}") from exc
    info = sf.info(str(path))
    return float(info.duration), int(info.samplerate), int(info.channels)


def normalize_record(row: dict[str, Any], *, manifest_dir: str | Path | None = None) -> dict[str, Any]:
    record = {
        "id": str(row.get("id") or row.get("sample_id") or "").strip(),
        "modality": str(row.get("modality") or "").strip(),
        "split": str(row.get("split") or "train").strip(),
        "source": str(row.get("source") or row.get("dataset") or "unknown").strip(),
        "group_id": str(row.get("group_id") or row.get("speaker_id") or row.get("id") or row.get("sample_id") or "").strip(),
        "audio_path": str(row.get("audio_path") or "").strip(),
        "codec_path": str(row.get("codec_path") or "").strip(),
        "caption": row.get("caption"),
        "known_fields": list(row.get("known_fields") or []),
        "metadata": dict(row.get("metadata") or {}),
    }
    if not record["modality"]:
        record["modality"] = "paired" if record["audio_path"] and record["caption"] else "audio_only" if record["audio_path"] else "caption_only"
    if record["caption"] is not None:
        record["caption"] = canonical_source_caption(record["caption"])
        if not record["known_fields"]:
            record["known_fields"] = list(SOURCE_FIELDS)
    if manifest_dir is not None:
        root = Path(manifest_dir)
        for key in ("audio_path", "codec_path"):
            if record[key] and not Path(record[key]).is_absolute():
                record[key] = str((root / record[key]).resolve())
    return record


def validate_records(
    rows: Iterable[dict[str, Any]], *, require_audio: bool = True, inspect_audio: bool = True,
    min_duration: float = 1.2, max_duration: float = 30.0, hash_audio: bool = False,
) -> ValidationReport:
    report = ValidationReport()
    ids: dict[str, int] = {}
    audio_seen: dict[str, str] = {}
    group_splits: defaultdict[str, set[str]] = defaultdict(set)
    counts: defaultdict[str, int] = defaultdict(int)
    for index, raw in enumerate(rows, 1):
        report.rows += 1
        errors: list[str] = []
        try:
            row = normalize_record(raw)
        except (TypeError, ValueError) as exc:
            report.errors.append({"row": index, "id": raw.get("id"), "errors": [str(exc)]})
            continue
        sample_id = row["id"]
        if not sample_id:
            errors.append("missing id")
        elif sample_id in ids:
            errors.append(f"duplicate id; first seen at row {ids[sample_id]}")
        else:
            ids[sample_id] = index
        if row["modality"] not in MODALITIES:
            errors.append(f"invalid modality: {row['modality']!r}")
        if row["split"] not in SPLITS:
            errors.append(f"invalid split: {row['split']!r}")
        needs_audio = row["modality"] in {"paired", "audio_only"}
        needs_caption = row["modality"] in {"paired", "caption_only"}
        if needs_audio and not row["audio_path"]:
            errors.append("audio_path is required")
        if needs_caption and row["caption"] is None:
            errors.append("caption is required")
        if row["caption"] is not None:
            errors.extend(source_validation_errors(row["caption"]))
            invalid_fields = sorted(set(row["known_fields"]) - set(SOURCE_FIELDS))
            if invalid_fields:
                errors.append(f"unknown known_fields: {invalid_fields}")
            language = get_path(row["caption"], "semantic_content.language")
            transcript = str(get_path(row["caption"], "semantic_content.transcript", "")).strip().lower()
            if language not in {"English", "Chinese"}:
                errors.append(f"training is limited to English/Chinese, got {language!r}")
            if row["modality"] == "caption_only" and transcript in {"", "unknown"}:
                errors.append("caption_only requires a non-unknown transcript")
        if row["audio_path"]:
            path = Path(row["audio_path"])
            if require_audio and not path.is_file():
                errors.append(f"audio does not exist: {path}")
            elif path.is_file():
                identity = sha256_file(path) if hash_audio else str(path.resolve())
                if identity in audio_seen and audio_seen[identity] != sample_id:
                    errors.append(f"duplicate audio with id {audio_seen[identity]}")
                else:
                    audio_seen[identity] = sample_id
                if inspect_audio:
                    try:
                        duration, sample_rate, channels = _audio_info(path)
                        if not min_duration <= duration <= max_duration:
                            errors.append(f"duration {duration:.3f}s outside [{min_duration}, {max_duration}]")
                        if channels != 1:
                            report.warnings.append({"row": index, "id": sample_id, "warning": f"audio has {channels} channels"})
                        if sample_rate < 8000:
                            errors.append(f"sample rate too low: {sample_rate}")
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"cannot inspect audio: {exc}")
        group_splits[row["group_id"]].add(row["split"])
        counts[row["modality"]] += 1
        if errors:
            report.errors.append({"row": index, "id": sample_id, "errors": errors})
        else:
            report.valid_rows += 1
    for group_id, splits in sorted(group_splits.items()):
        if group_id and len(splits) > 1:
            report.errors.append({"group_id": group_id, "errors": [f"group leaks across splits: {sorted(splits)}"]})
    report.modality_counts = dict(sorted(counts.items()))
    return report


def _base_record(raw: dict[str, Any], modality: str, split: str) -> dict[str, Any]:
    sample_id = str(raw.get("id") or raw.get("sample_id") or "").strip()
    return {
        "id": sample_id, "modality": modality, "split": str(raw.get("split") or split),
        "source": str(raw.get("source") or raw.get("dataset") or f"{modality}_manifest"),
        "group_id": str(raw.get("group_id") or raw.get("speaker_id") or sample_id),
        "audio_path": "", "codec_path": "", "caption": None, "known_fields": [],
        "metadata": dict(raw.get("metadata") or {}),
    }


def convert_audio_manifest(input_path: str | Path, output_path: str | Path, *, split: str = "train") -> int:
    rows = []
    for raw in read_jsonl(input_path):
        row = _base_record(raw, "audio_only", split)
        row["audio_path"] = str(raw.get("audio_path") or raw.get("sampled_audio_path") or raw.get("audio") or "")
        row["metadata"]["oracle_transcript"] = raw.get("transcript")
        rows.append(row)
    write_jsonl(output_path, rows)
    return len(rows)


def convert_caption_manifest(input_path: str | Path, output_path: str | Path, *, split: str = "train") -> int:
    rows = []
    for raw in read_jsonl(input_path):
        row = _base_record(raw, "caption_only", split)
        row["caption"] = canonical_source_caption(raw.get("caption") or raw.get("Target_JSON_Schema") or raw)
        row["known_fields"] = list(raw.get("known_fields") or SOURCE_FIELDS)
        rows.append(row)
    write_jsonl(output_path, rows)
    return len(rows)


def convert_paired_manifest(input_path: str | Path, output_path: str | Path, *, split: str = "train") -> int:
    rows = []
    for raw in read_jsonl(input_path):
        row = _base_record(raw, "paired", split)
        row["audio_path"] = str(raw.get("audio_path") or raw.get("audio") or "")
        row["caption"] = canonical_source_caption(raw.get("caption") or raw.get("Target_JSON_Schema") or raw.get("label"))
        row["known_fields"] = list(raw.get("known_fields") or SOURCE_FIELDS)
        rows.append(row)
    write_jsonl(output_path, rows)
    return len(rows)


def load_records(path: str | Path, modalities: set[str] | None = None, split: str | None = None) -> list[dict[str, Any]]:
    manifest = Path(path).resolve()
    records = [normalize_record(row, manifest_dir=manifest.parent) for row in read_jsonl(manifest)]
    if modalities is not None:
        records = [row for row in records if row["modality"] in modalities]
    if split is not None:
        records = [row for row in records if row["split"] == split]
    return records


def validate_manifest_roles(
    manifests: dict[str, str | Path], *, require_disjoint: bool = True, hash_audio: bool = False,
) -> dict[str, Any]:
    """Validate role and held-out manifests together, not one file at a time.

    Leakage is checked at sample-id, group-id and resolved-audio identity.  Caption
    text is deliberately not used as an identity because common transcripts such as
    short greetings can legitimately occur in different utterances.
    """
    expected = {
        "paired": ({"paired"}, "train"),
        "audio_only": ({"audio_only"}, "train"),
        "caption_only": ({"caption_only"}, "train"),
        "validation": ({"paired"}, "validation"),
        "test": ({"paired"}, "test"),
    }
    rows_by_role: dict[str, list[dict[str, Any]]] = {}
    errors: list[dict[str, Any]] = []
    for role, path in manifests.items():
        rows = load_records(path)
        rows_by_role[role] = rows
        allowed_modalities, expected_split = expected.get(role, (MODALITIES, "train"))
        for row in rows:
            if row["modality"] not in allowed_modalities:
                errors.append({"role": role, "id": row["id"], "error": f"unexpected modality {row['modality']}"})
            if row["split"] != expected_split:
                errors.append({"role": role, "id": row["id"], "error": f"expected split {expected_split}, got {row['split']}"})

    identity_maps: dict[str, dict[str, set[str]]] = {
        "id": defaultdict(set), "group_id": defaultdict(set), "audio": defaultdict(set),
    }
    ids_within_role: defaultdict[str, dict[str, str]] = defaultdict(dict)
    audio_within_role: defaultdict[str, dict[str, str]] = defaultdict(dict)
    for role, rows in rows_by_role.items():
        for row in rows:
            if row["id"] in ids_within_role[role]:
                errors.append({"role": role, "id": row["id"], "error": "duplicate id within manifest"})
            ids_within_role[role][row["id"]] = row["id"]
            identity_maps["id"][row["id"]].add(role)
            if row["group_id"]:
                identity_maps["group_id"][row["group_id"]].add(role)
            if row["audio_path"]:
                audio_path = Path(row["audio_path"])
                audio_identity = sha256_file(audio_path) if hash_audio and audio_path.is_file() else str(audio_path.resolve())
                previous_id = audio_within_role[role].get(audio_identity)
                if previous_id is not None and previous_id != row["id"]:
                    errors.append({"role": role, "id": row["id"], "other_id": previous_id, "error": "duplicate audio within manifest"})
                audio_within_role[role][audio_identity] = row["id"]
                identity_maps["audio"][audio_identity].add(role)
    overlaps: dict[str, list[dict[str, Any]]] = {}
    for kind, mapping in identity_maps.items():
        overlaps[kind] = [
            {"identity": identity, "roles": sorted(roles)}
            for identity, roles in sorted(mapping.items()) if len(roles) > 1
        ]
        if require_disjoint:
            errors.extend({"identity_type": kind, **item, "error": "cross-manifest leakage"} for item in overlaps[kind])
    counts = {role: len(rows) for role, rows in rows_by_role.items()}
    return {"ok": not errors, "counts": counts, "overlaps": overlaps, "errors": errors}


def partition_paired_manifest(
    input_path: str | Path, output_dir: str | Path, *, seed: int = 42,
    paired_groups: int = 45, audio_groups: int = 70, caption_groups: int = 69,
    validation_groups: int = 25, test_groups: int = 25,
) -> dict[str, Any]:
    """Create mutually exclusive role/split manifests from paired seed data."""
    source_rows = load_records(input_path, {"paired"})
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        grouped[row["group_id"]].append(row)
    groups = sorted(grouped)
    random.Random(seed).shuffle(groups)
    requested = {
        "paired": paired_groups, "audio_only": audio_groups, "caption_only": caption_groups,
        "validation": validation_groups, "test": test_groups,
    }
    if any(count < 0 for count in requested.values()):
        raise ValueError("Partition group counts must be non-negative")
    if sum(requested.values()) != len(groups):
        raise ValueError(f"Partition counts sum to {sum(requested.values())}, but source has {len(groups)} groups")
    root = Path(output_dir)
    outputs: dict[str, str] = {}
    cursor = 0
    for role, count in requested.items():
        selected_groups = groups[cursor:cursor + count]
        cursor += count
        rows: list[dict[str, Any]] = []
        for group_id in selected_groups:
            for source in grouped[group_id]:
                row = dict(source)
                row["metadata"] = {**row.get("metadata", {}), "partition_seed": seed, "partition_role": role}
                if role == "audio_only":
                    row.update(modality="audio_only", split="train", caption=None, known_fields=[])
                elif role == "caption_only":
                    row.update(modality="caption_only", split="train", audio_path="", codec_path="")
                elif role in {"validation", "test"}:
                    row.update(modality="paired", split=role)
                else:
                    row.update(modality="paired", split="train")
                rows.append(row)
        path = root / f"{role}.jsonl"
        write_jsonl(path, rows)
        outputs[role] = str(path.resolve())
    report = validate_manifest_roles(outputs, hash_audio=False)
    report.update({"seed": seed, "source": str(Path(input_path).resolve()), "source_hash": stable_hash(source_rows), "paths": outputs})
    if not report["ok"]:
        raise RuntimeError(f"Generated partitions failed leakage validation: {report['errors'][:3]}")
    atomic_json(root / "partition_report.json", report)
    return report
