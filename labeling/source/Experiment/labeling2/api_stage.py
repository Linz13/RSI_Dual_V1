from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .manifest import Sample
from .pipeline import MODEL_FIELDS, latest_by_id
from .target_schema import get_path, set_path, set_transcript


API_MODELS = ("gemini", "qwen35")
LOCAL_MODELS = ("qwen3_captioner", "kimi_audio", "step_audio_r1_1")
EXPERTS = ("volume", "emotion", "accent_en", "accent_zh", "rate_en", "rate_zh")

FIELD_SOURCES: dict[str, tuple[str, ...]] = {
    field: tuple(model for model in API_MODELS if field in MODEL_FIELDS[model])
    for field in sorted(set(MODEL_FIELDS["gemini"]) | set(MODEL_FIELDS["qwen35"]))
}

PENDING_GPU_FIELDS = {
    "speaker_profile.gender": ("qwen3_captioner",),
    "speaker_profile.age": ("qwen3_captioner",),
    "paralinguistic.emotion_intensity": ("qwen3_captioner",),
    "paralinguistic.emphasis.level": ("qwen3_captioner", "kimi_audio"),
    "paralinguistic.emphasis.emphasized_text": ("qwen3_captioner", "kimi_audio"),
    "paralinguistic.nonverbal_vocalization": ("qwen3_captioner", "kimi_audio", "step_audio_r1_1"),
    "environment.recording_quality": ("qwen3_captioner",),
}

GPU_ONLY_FIELDS: dict[str, dict[str, Any]] = {
    "speaker_profile.accent": {"required_expert": ["accent_en", "accent_zh"]},
    "paralinguistic.speaking_rate": {"required_expert": ["rate_en", "rate_zh"]},
    "paralinguistic.volume_level": {"required_expert": ["volume"]},
    "paralinguistic.emotion": {"required_expert": ["emotion"]},
}

OPEN_API_FIELDS = {
    "paralinguistic.prosody",
    "environment.background_sound_events",
    "environment.acoustic_scene",
}


def empty_api_target() -> dict[str, Any]:
    return {
        "semantic_content": {"language": None, "topic": None, "intent": None},
        "speaker_profile": {"gender": None, "age": None, "timbre": None, "accent": None},
        "paralinguistic": {
            "speaking_rate": None,
            "pitch_level": None,
            "volume_level": None,
            "emotion": None,
            "emotion_intensity": None,
            "emphasis": {"level": None, "emphasized_text": None},
            "prosody": None,
            "pause": None,
            "nonverbal_vocalization": None,
        },
        "environment": {
            "background_sound_events": None,
            "recording_quality": None,
            "acoustic_scene": None,
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prediction_value(row: dict[str, Any], field: str) -> Any:
    if row.get("status") not in {"success", "partial"}:
        return None
    if field in (row.get("field_errors") or {}):
        return None
    return get_path(row.get("parsed") or {}, field)


def finalize_api_stage(samples: list[Sample], run_dir: Path) -> dict[str, Any]:
    predictions = {
        model: latest_by_id(run_dir / "raw_predictions" / f"{model}.jsonl")
        for model in API_MODELS
    }
    decisions = latest_by_id(run_dir / "open_resolution" / "decisions.jsonl")
    labels: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()

    for sample in samples:
        target = empty_api_target()
        field_status: dict[str, dict[str, Any]] = {}
        pending_gpu_fields: list[str] = []
        not_run_fields: list[str] = []
        api_error_fields: list[str] = []
        completed_backends = [
            model for model in API_MODELS
            if predictions[model].get(sample.sample_id, {}).get("status") in {"success", "partial"}
        ]

        sample_decisions = (decisions.get(sample.sample_id) or {}).get("fields", {})
        for field, sources in FIELD_SOURCES.items():
            evidence = []
            for model in sources:
                row = predictions[model].get(sample.sample_id, {})
                value = _prediction_value(row, field)
                evidence.append({
                    "model": model,
                    "status": row.get("status", "missing"),
                    "attempt": row.get("attempt"),
                    "value": value,
                    "error": row.get("error", ""),
                    "field_error": (row.get("field_errors") or {}).get(field, ""),
                })

            decision = sample_decisions.get(field, {}) if field in OPEN_API_FIELDS else {}
            if decision.get("status") in {"resolved", "resolved_empty"} and decision.get("passed"):
                value = decision.get("value")
                status = "api_consensus"
                meta = {"status": status, "value": value, "decision": decision, "sources": list(sources)}
            else:
                available = [(row["model"], row["value"]) for row in evidence if row["value"] is not None]
                if field in OPEN_API_FIELDS and len(sources) > 1:
                    value = None
                    status = "api_consensus_pending" if available else "api_error"
                    meta = {"status": status, "value": value, "candidates": evidence, "decision": decision}
                elif available:
                    model, value = available[0]
                    if field in PENDING_GPU_FIELDS:
                        status = "api_provisional"
                        pending_gpu_fields.append(field)
                        meta = {
                            "status": status,
                            "value": value,
                            "source_model": model,
                            "pending_gpu_consensus": True,
                            "required_backend": list(PENDING_GPU_FIELDS[field]),
                        }
                    else:
                        status = "api_final"
                        meta = {"status": status, "value": value, "source_model": model}
                else:
                    value = None
                    status = "api_error"
                    meta = {"status": status, "value": value, "candidates": evidence}
            set_path(target, field, value)
            field_status[field] = meta
            status_counts[status] += 1
            field_counts[field] += int(value is not None)
            if status in {"api_error", "api_consensus_pending"}:
                api_error_fields.append(field)

        for field, requirement in GPU_ONLY_FIELDS.items():
            set_path(target, field, None)
            field_status[field] = {"status": "not_run", "value": None, **requirement}
            not_run_fields.append(field)
            status_counts["not_run"] += 1

        set_transcript(target, sample.transcript)

        label = {
            "schema_version": "labeling2.api_stage.v1",
            "sample_id": sample.sample_id,
            "audio_path": sample.audio_path,
            "dataset": sample.dataset,
            "Target_JSON_Schema": target,
            "completed_backends": completed_backends,
            "not_run_backends": [*LOCAL_MODELS, *EXPERTS],
            "pending_gpu_fields": sorted(pending_gpu_fields),
            "not_run_fields": sorted(not_run_fields),
            "api_error_fields": sorted(set(api_error_fields)),
        }
        labels.append(label)
        provenance_rows.append({
            "schema_version": "labeling2.api_stage.v1",
            "sample_id": sample.sample_id,
            "fields": field_status,
        })
        if pending_gpu_fields or not_run_fields or api_error_fields:
            review_rows.append({
                "schema_version": "labeling2.api_stage.v1",
                "sample_id": sample.sample_id,
                "pending_gpu_fields": sorted(pending_gpu_fields),
                "not_run_fields": sorted(not_run_fields),
                "api_error_fields": sorted(set(api_error_fields)),
            })

    final_dir = run_dir / "final"
    label_path = final_dir / "labels_api_stage.jsonl"
    provenance_path = final_dir / "provenance_api_stage.jsonl"
    review_path = final_dir / "review_queue_api_stage.jsonl"
    _write_jsonl(label_path, labels)
    _write_jsonl(provenance_path, provenance_rows)
    _write_jsonl(review_path, review_rows)
    summary = {
        "schema_version": "labeling2.api_stage.v1",
        "samples": len(samples),
        "completed_api_backends": list(API_MODELS),
        "not_run_backends": [*LOCAL_MODELS, *EXPERTS],
        "status_counts": dict(status_counts),
        "field_non_null_counts": dict(sorted(field_counts.items())),
        "labels": str(label_path.resolve()),
        "provenance": str(provenance_path.resolve()),
        "review_queue": str(review_path.resolve()),
    }
    (run_dir / "api_stage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
