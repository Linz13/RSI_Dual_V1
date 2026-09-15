from __future__ import annotations

from copy import deepcopy
from typing import Any

SCHEMA_VERSION = "labeling2.v1"

CLOSED_VALUES: dict[str, set[str]] = {
    "semantic_content.language": {"English", "Chinese", "other", "unknown"},
    "semantic_content.intent": {
        "statement", "question", "request", "command", "suggestion", "greeting",
        "thanks", "apology", "acknowledgment", "refusal", "other", "unknown",
    },
    "speaker_profile.gender": {"male", "female", "unknown"},
    "speaker_profile.age": {"child", "teenager", "adult", "elderly", "unknown"},
    "speaker_profile.timbre": {
        "neutral", "breathy", "rough", "creaky", "strained", "nasal", "bright",
        "dark", "other", "unknown",
    },
    "speaker_profile.accent": {
        "US English", "England English", "Australian English", "Indian English",
        "Canadian English", "Bermudian English", "Scottish English", "African English",
        "Irish English", "New Zealand English", "Welsh English", "Malaysian English",
        "Philippine English", "Singapore English", "Hong Kong English",
        "South Atlantic English", "Mandarin", "Jiang-Huai Mandarin", "Jiao-Liao Mandarin",
        "Ji-Lu Mandarin", "Lan-Yin Mandarin", "Southwestern Mandarin",
        "Zhongyuan Mandarin", "Cantonese", "other", "unknown",
    },
    "paralinguistic.speaking_rate": {"slow", "moderate", "fast", "unknown"},
    "paralinguistic.pitch_level": {"low", "medium", "high", "unknown"},
    "paralinguistic.volume_level": {"low", "medium", "high", "unknown"},
    "paralinguistic.emotion": {
        "neutral", "happy", "sad", "angry", "fearful", "surprised", "disgusted",
        "other", "unknown",
    },
    "paralinguistic.emotion_intensity": {"none", "low", "medium", "high", "unknown"},
    "paralinguistic.emphasis.level": {"none", "emphasized", "heavily emphasized", "unknown"},
    "environment.recording_quality": {"bad", "poor", "fair", "good", "excellent", "unknown"},
}

EVENT_VALUES = {
    "none", "laughter", "sigh", "crying", "sobbing", "cough", "throat clearing",
    "audible breath", "sneeze", "sniff", "yawn", "scream", "filled pause",
    "agreement sound", "other", "unknown",
}

LIST_FIELDS = {"paralinguistic.nonverbal_vocalization", "environment.background_sound_events"}
OPEN_FIELDS = {
    "semantic_content.topic", "paralinguistic.emphasis.emphasized_text",
    "paralinguistic.prosody", "paralinguistic.pause",
    "environment.background_sound_events", "environment.acoustic_scene",
}
OPEN_STRING_FIELDS = {
    "semantic_content.topic", "paralinguistic.prosody", "paralinguistic.pause",
    "environment.acoustic_scene",
}


def empty_target() -> dict[str, Any]:
    """Create a schema-shaped object with safe unknown defaults."""
    return {
        "semantic_content": {"language": "unknown", "topic": "unknown", "intent": "unknown"},
        "speaker_profile": {
            "gender": "unknown", "age": "unknown", "timbre": "unknown", "accent": "unknown",
        },
        "paralinguistic": {
            "speaking_rate": "unknown", "pitch_level": "unknown", "volume_level": "unknown",
            "emotion": "unknown", "emotion_intensity": "unknown",
            "emphasis": {"level": "unknown", "emphasized_text": []},
            "prosody": "unknown", "pause": "unknown", "nonverbal_vocalization": ["unknown"],
        },
        "environment": {
            "background_sound_events": ["unknown"], "recording_quality": "unknown",
            "acoustic_scene": "unknown",
        },
    }


def get_path(obj: dict[str, Any], dotted: str) -> Any:
    current: Any = obj
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def set_path(obj: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    current = obj
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def set_transcript(target: dict[str, Any], transcript: str | None) -> None:
    """Insert ground-truth transcript immediately before intent in final labels."""
    semantic = target.setdefault("semantic_content", {})
    reordered: dict[str, Any] = {}
    inserted = False
    for key, value in semantic.items():
        if key == "intent":
            reordered["transcript"] = transcript if transcript is not None else ""
            inserted = True
        if key != "transcript":
            reordered[key] = value
    if not inserted:
        reordered["transcript"] = transcript if transcript is not None else ""
    target["semantic_content"] = reordered


def normalize_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()]
    if isinstance(value, (tuple, list)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def validate_target(target: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "semantic_content.language", "semantic_content.topic", "semantic_content.intent",
        "speaker_profile.gender", "speaker_profile.age", "speaker_profile.timbre", "speaker_profile.accent",
        "paralinguistic.speaking_rate", "paralinguistic.pitch_level", "paralinguistic.volume_level",
        "paralinguistic.emotion", "paralinguistic.emotion_intensity", "paralinguistic.emphasis.level",
        "paralinguistic.emphasis.emphasized_text", "paralinguistic.prosody", "paralinguistic.pause",
        "paralinguistic.nonverbal_vocalization", "environment.background_sound_events",
        "environment.recording_quality", "environment.acoustic_scene",
    }
    for field in required:
        if get_path(target, field) is None:
            errors.append(f"missing:{field}")
    transcript = get_path(target, "semantic_content.transcript")
    if transcript is not None and not isinstance(transcript, str):
        errors.append(f"invalid_string:semantic_content.transcript={transcript!r}")
    for field, allowed in CLOSED_VALUES.items():
        value = get_path(target, field)
        if not isinstance(value, str) or value not in allowed:
            errors.append(f"invalid_closed_value:{field}={value!r}")
    for field in OPEN_STRING_FIELDS:
        value = get_path(target, field)
        if not isinstance(value, str):
            errors.append(f"invalid_string:{field}={value!r}")
    for field in LIST_FIELDS:
        value = get_path(target, field)
        if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
            errors.append(f"invalid_list:{field}")
            continue
        if len(set(value)) != len(value):
            errors.append(f"duplicate_list_value:{field}")
        if field == "paralinguistic.nonverbal_vocalization":
            invalid = set(value) - EVENT_VALUES
            if invalid:
                errors.append(f"invalid_event_value:{sorted(invalid)!r}")
        if "none" in value and len(value) > 1:
            errors.append(f"none_not_exclusive:{field}")
        if "unknown" in value and len(value) > 1:
            errors.append(f"unknown_not_exclusive:{field}")
    emphasized = get_path(target, "paralinguistic.emphasis.emphasized_text")
    if not isinstance(emphasized, list) or any(not isinstance(item, str) for item in emphasized):
        errors.append("invalid_list:paralinguistic.emphasis.emphasized_text")
    return errors


def canonical_target(value: Any) -> dict[str, Any] | None:
    """Accept either the wrapper or the inner Target_JSON_Schema object."""
    if not isinstance(value, dict):
        return None
    if "Target_JSON_Schema" in value and isinstance(value["Target_JSON_Schema"], dict):
        value = value["Target_JSON_Schema"]
    if not isinstance(value, dict):
        return None
    candidate = deepcopy(value)
    return candidate if not validate_target(candidate) else None
