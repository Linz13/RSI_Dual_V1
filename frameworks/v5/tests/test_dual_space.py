from __future__ import annotations

import json

from dual_isl_train.constants import SYNTHESIZABLE_FIELDS
from dual_isl_train.dual_space import (
    EXCLUDED_FIELDS, parse_synth_caption_text_detailed, project_synth_caption,
    synth_caption_json, synth_validation_errors,
)


def test_psyn_is_exactly_the_16_tts_supported_fields(sample_caption):
    projected = project_synth_caption(sample_caption)
    assert len(SYNTHESIZABLE_FIELDS) == 16
    assert set(projected) == {"semantic_content", "speaker_profile", "paralinguistic"}
    assert "topic" not in projected["semantic_content"]
    assert "intent" not in projected["semantic_content"]
    assert "environment" not in projected
    assert set(EXCLUDED_FIELDS) == {
        "semantic_content.topic", "semantic_content.intent",
        "environment.background_sound_events", "environment.recording_quality",
        "environment.acoustic_scene",
    }


def test_lenient_psyn_normalization_keeps_raw_audit(sample_caption):
    projected = project_synth_caption(sample_caption)
    projected["speaker_profile"]["age"] = "young adult"
    projected["paralinguistic"]["speaking_rate"] = "normal"
    projected["paralinguistic"].pop("pitch_level")
    projected["environment"] = {"acoustic_scene": "street"}
    raw = json.dumps({"Target_JSON_Schema": projected})
    caption, errors, metadata = parse_synth_caption_text_detailed(raw)
    assert not errors
    assert caption is not None
    assert caption["speaker_profile"]["age"] == "adult"
    assert caption["paralinguistic"]["speaking_rate"] == "moderate"
    assert caption["paralinguistic"]["pitch_level"] == "unknown"
    assert "environment" not in caption
    assert metadata["raw_schema_valid"] is False
    assert metadata["normalized_schema_valid"] is True
    assert metadata["normalization_rules"]
    assert not synth_validation_errors(caption)
    assert json.loads(synth_caption_json(caption))["Target_JSON_Schema"] == caption


def test_psyn_parser_repairs_list_typed_enums_without_crashing(sample_caption):
    projected = project_synth_caption(sample_caption)
    projected["paralinguistic"]["emotion"] = "neutral"
    projected["paralinguistic"]["emotion_intensity"] = ["high"]
    projected["paralinguistic"]["emphasis"] = {
        "level": ["none"],
        "emphasized_text": ["testing"],
    }

    caption, errors, metadata = parse_synth_caption_text_detailed(json.dumps({
        "Target_JSON_Schema": projected,
    }))

    assert not errors
    assert caption is not None
    assert caption["paralinguistic"]["emotion_intensity"] == "unknown"
    assert caption["paralinguistic"]["emphasis"] == {
        "level": "emphasized",
        "emphasized_text": ["testing"],
    }
    assert metadata["raw_schema_valid"] is False
    assert metadata["normalized_schema_valid"] is True
    assert "paralinguistic.emotion_intensity:invalid_enum_type_to_unknown" in metadata["normalization_rules"]
    assert "paralinguistic.emphasis.level:invalid_enum_type_to_unknown" in metadata["normalization_rules"]
