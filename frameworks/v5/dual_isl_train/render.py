from __future__ import annotations

import json
from typing import Any

from .schema import get_path, source_schema_document
from .dual_space import project_synth_caption, synth_schema_document


def _usable(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip()) and value.strip().lower() not in {"unknown", "unspecified", "none"}
    if isinstance(value, list):
        return any(_usable(item) for item in value)
    return True


def _add(parts: list[str], label: str, value: Any) -> None:
    if not _usable(value):
        return
    text = ", ".join(str(item).strip() for item in value if _usable(item)) if isinstance(value, list) else str(value).strip()
    if text:
        parts.append(f"{label}: {text}")


def _tts_caption(caption: dict[str, Any]) -> dict[str, Any]:
    return project_synth_caption(caption)


def render_qwen_instruct(caption: dict[str, Any], enabled_fields: set[str] | None = None) -> str:
    caption = _tts_caption(caption)
    parts: list[str] = []
    for label, field in (
        ("gender", "speaker_profile.gender"), ("age", "speaker_profile.age"),
        ("accent", "speaker_profile.accent"), ("voice timbre", "speaker_profile.timbre"),
        ("emotion", "paralinguistic.emotion"), ("emotion intensity", "paralinguistic.emotion_intensity"),
        ("speaking rate", "paralinguistic.speaking_rate"), ("pitch", "paralinguistic.pitch_level"),
        ("volume", "paralinguistic.volume_level"), ("emphasis", "paralinguistic.emphasis.level"),
        ("emphasized words", "paralinguistic.emphasis.emphasized_text"),
        ("prosody", "paralinguistic.prosody"), ("pauses", "paralinguistic.pause"),
        ("nonverbal vocalizations", "paralinguistic.nonverbal_vocalization"),
    ):
        if enabled_fields is None or field in enabled_fields:
            _add(parts, label, get_path(caption, field))
    return "; ".join(parts) or "Speak naturally in a neutral voice."


def render_qwen_request(caption: dict[str, Any], enabled_fields: set[str] | None = None) -> dict[str, str]:
    caption = _tts_caption(caption)
    text = str(get_path(caption, "semantic_content.transcript", "")).strip()
    if not text or text.lower() == "unknown":
        raise ValueError("Qwen VoiceDesign requires a non-unknown semantic_content.transcript")
    language = str(get_path(caption, "semantic_content.language", "Auto"))
    if language not in {"English", "Chinese"}:
        language = "Auto"
    return {"text": text, "instruct": render_qwen_instruct(caption, enabled_fields), "language": language}


def caption_prompt() -> str:
    compact_schema = json.dumps(source_schema_document(), ensure_ascii=False, separators=(",", ":"))
    return (
        "Listen to the speech and return exactly one valid JSON object with the key "
        "Target_JSON_Schema and no prose. Include a verbatim semantic_content.transcript and every "
        "required source field. Do not add an environment field. Use only allowed closed-vocabulary "
        "values and use unknown only when evidence "
        "is unavailable. List-valued fields must always be JSON arrays. The authoritative "
        f"dual_isl_train.v3.source-no-environment JSON Schema is: {compact_schema}"
    )


def synth_caption_example() -> dict[str, Any]:
    """A complete, schema-valid shape example, never used as a training label."""
    return {"Target_JSON_Schema": {
        "semantic_content": {"language": "English", "transcript": "This is an example sentence."},
        "speaker_profile": {
            "gender": "female", "age": "adult", "timbre": "bright", "accent": "US English",
        },
        "paralinguistic": {
            "speaking_rate": "moderate", "pitch_level": "medium", "volume_level": "medium",
            "emotion": "neutral", "emotion_intensity": "low",
            "emphasis": {"level": "none", "emphasized_text": []},
            "prosody": "steady intonation", "pause": "a brief pause at the end",
            "nonverbal_vocalization": ["none"],
        },
    }}


def synth_caption_prompt() -> str:
    compact_schema = json.dumps(synth_schema_document(), ensure_ascii=False, separators=(",", ":"))
    example = json.dumps(synth_caption_example(), ensure_ascii=False, separators=(",", ":"))
    return (
        "Listen to the speech and return exactly one valid JSON object with the key "
        "Target_JSON_Schema and no prose. Include a verbatim semantic_content.transcript and every "
        "required field in the synthesizable P_syn schema. Do not add topic, intent, or environment "
        "fields. Use only allowed closed-vocabulary values and use unknown only when evidence is "
        "unavailable. List-valued fields must always be JSON arrays. "
        f"The authoritative P_syn JSON Schema is: {compact_schema}\n"
        "The following fictional example demonstrates the output STRUCTURE ONLY. "
        "It is not a description of the provided audio. Replace the transcript and all "
        "attribute values with evidence from the current audio; do not copy the example answers. "
        "semantic_content, speaker_profile and paralinguistic must be sibling objects inside "
        "Target_JSON_Schema. nonverbal_vocalization and emphasized_text must be JSON arrays. "
        "For no nonverbal vocalization use [\"none\"], not \"none\" or []; "
        "for no emphasized words use []. Do not output the JSON Schema definition itself.\n"
        f"Complete output example: {example}\n"
        "Now describe the actual audio using this structure and close every object and array."
    )


dual_caption_prompt = synth_caption_prompt
