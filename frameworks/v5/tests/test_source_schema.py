from __future__ import annotations

import json

import pytest

from dual_isl_train.constants import SOURCE_FIELDS
from dual_isl_train.data import load_records
from dual_isl_train.schema import (
    canonical_source_caption,
    parse_dual_caption_text,
    source_caption_json,
)


def test_source_schema_keeps_topic_and_intent_but_forbids_environment(sample_caption):
    source = {key: value for key, value in sample_caption.items() if key != "environment"}
    canonical = canonical_source_caption(source)
    assert canonical["semantic_content"]["topic"] == "testing"
    assert canonical["semantic_content"]["intent"] == "statement"
    assert "environment" not in canonical
    assert len(SOURCE_FIELDS) == 18
    full = dict(source)
    full["environment"] = {
        "background_sound_events": ["none"],
        "recording_quality": "good",
        "acoustic_scene": "studio",
    }
    with pytest.raises(ValueError, match="environment"):
        canonical_source_caption(full)


def test_source_caption_round_trip_does_not_restore_environment(sample_caption):
    source = {key: value for key, value in sample_caption.items() if key != "environment"}
    serialized = source_caption_json(source)
    parsed, errors = parse_dual_caption_text(serialized)
    assert not errors
    assert parsed == source
    assert "environment" not in parsed


def test_source_parser_rejects_list_typed_enums_without_crashing(sample_caption):
    source = {key: value for key, value in sample_caption.items() if key != "environment"}
    source["paralinguistic"]["emotion"] = "neutral"
    source["paralinguistic"]["emotion_intensity"] = ["high"]
    source["paralinguistic"]["emphasis"] = {
        "level": ["none"],
        "emphasized_text": ["testing"],
    }

    parsed, errors = parse_dual_caption_text(json.dumps({"Target_JSON_Schema": source}))

    assert parsed is None
    assert errors


def test_manifest_relative_audio_paths_resolve_from_manifest_directory(tmp_path, sample_caption):
    audio = tmp_path / "audio" / "example.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"placeholder")
    source = {key: value for key, value in sample_caption.items() if key != "environment"}
    manifest = tmp_path / "paired.jsonl"
    manifest.write_text(json.dumps({
        "id": "relative", "modality": "paired", "split": "train",
        "audio_path": "audio/example.wav", "caption": source,
    }) + "\n", encoding="utf-8")
    rows = load_records(manifest)
    assert rows[0]["audio_path"] == str(audio.resolve())
