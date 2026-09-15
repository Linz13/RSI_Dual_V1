from __future__ import annotations

import json
import re
from copy import deepcopy
from functools import lru_cache
from importlib.resources import files
from typing import Any

from jsonschema import Draft202012Validator

from .constants import ALL_FIELDS, ENVIRONMENT_FIELDS, SOURCE_FIELDS, WRAPPER_KEY


@lru_cache(maxsize=1)
def schema_document() -> dict[str, Any]:
    resource = files("dual_isl_train.schemas").joinpath("target_schema.v3.json")
    return json.loads(resource.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def validator() -> Draft202012Validator:
    return Draft202012Validator(schema_document())


@lru_cache(maxsize=1)
def dual_schema_document() -> dict[str, Any]:
    """Schema emitted inside dual loops; environment is intentionally absent."""
    document = deepcopy(schema_document())
    caption = document["properties"][WRAPPER_KEY]
    caption["required"] = [name for name in caption["required"] if name != "environment"]
    caption["properties"].pop("environment", None)
    document["$id"] = "dual_isl_train.v3.dual-no-environment"
    return document


@lru_cache(maxsize=1)
def dual_validator() -> Draft202012Validator:
    return Draft202012Validator(dual_schema_document())


# The source-domain training labels use the same no-environment contract as the
# outer dual-loop caption, while still retaining semantic topic and intent.
source_schema_document = dual_schema_document
source_validator = dual_validator


def wrap_caption(value: dict[str, Any]) -> dict[str, Any]:
    return value if WRAPPER_KEY in value else {WRAPPER_KEY: value}


def unwrap_caption(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Caption must be a JSON object")
    inner = value.get(WRAPPER_KEY, value)
    if not isinstance(inner, dict):
        raise ValueError(f"{WRAPPER_KEY} must contain an object")
    return deepcopy(inner)


def _logic_errors(caption: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    paralinguistic = caption.get("paralinguistic", {})
    if not isinstance(paralinguistic, dict):
        return errors
    emphasis = paralinguistic.get("emphasis", {})
    emphasis_level = emphasis.get("level") if isinstance(emphasis, dict) else None
    if (
        isinstance(emphasis, dict)
        and isinstance(emphasis_level, str)
        and emphasis_level in {"none", "unknown"}
        and emphasis.get("emphasized_text") not in ([], None)
    ):
        errors.append("paralinguistic.emphasis.emphasized_text must be [] when level is none/unknown")
    emotion = paralinguistic.get("emotion")
    intensity = paralinguistic.get("emotion_intensity")
    if emotion == "neutral" and isinstance(intensity, str) and intensity not in {"none", "low", "unknown"}:
        errors.append("neutral emotion cannot have medium/high intensity")
    return errors


def validation_errors(value: Any) -> list[str]:
    try:
        wrapped = wrap_caption(unwrap_caption(value))
    except ValueError as exc:
        return [str(exc)]
    errors = []
    for error in sorted(validator().iter_errors(wrapped), key=lambda item: list(item.absolute_path)):
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        errors.append(f"{location}: {error.message}")
    errors.extend(_logic_errors(wrapped[WRAPPER_KEY]))
    return errors


def canonical_caption(value: Any) -> dict[str, Any]:
    inner = unwrap_caption(value)
    errors = validation_errors(inner)
    if errors:
        raise ValueError("Invalid caption: " + "; ".join(errors))
    return inner


def source_validation_errors(value: Any) -> list[str]:
    try:
        wrapped = wrap_caption(unwrap_caption(value))
    except ValueError as exc:
        return [str(exc)]
    errors = []
    for error in sorted(source_validator().iter_errors(wrapped), key=lambda item: list(item.absolute_path)):
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        errors.append(f"{location}: {error.message}")
    errors.extend(_logic_errors(wrapped[WRAPPER_KEY]))
    return errors


def canonical_source_caption(value: Any) -> dict[str, Any]:
    inner = unwrap_caption(value)
    errors = source_validation_errors(inner)
    if errors:
        raise ValueError("Invalid source caption: " + "; ".join(errors))
    return inner


def source_caption_json(value: Any) -> str:
    caption = canonical_source_caption(value)
    ordered = {
        "semantic_content": caption["semantic_content"],
        "speaker_profile": caption["speaker_profile"],
        "paralinguistic": caption["paralinguistic"],
    }
    return json.dumps({WRAPPER_KEY: ordered}, ensure_ascii=False, separators=(",", ":"))


def caption_json(value: Any) -> str:
    """Stable serialization; environment is deliberately last for cycle loss masking."""
    caption = canonical_caption(value)
    ordered = {
        "semantic_content": caption["semantic_content"],
        "speaker_profile": caption["speaker_profile"],
        "paralinguistic": caption["paralinguistic"],
        "environment": caption["environment"],
    }
    return json.dumps({WRAPPER_KEY: ordered}, ensure_ascii=False, separators=(",", ":"))


def environment_character_boundary(serialized: str) -> int:
    marker = ',"environment":'
    index = serialized.find(marker)
    if index < 0:
        raise ValueError("Canonical caption is missing the environment boundary")
    return index


def get_path(value: dict[str, Any], dotted: str, default: Any = None) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def set_path(value: dict[str, Any], dotted: str, item: Any) -> None:
    current = value
    parts = dotted.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = item


def empty_caption() -> dict[str, Any]:
    return {
        "semantic_content": {"language": "unknown", "transcript": "unknown", "topic": "unknown", "intent": "unknown"},
        "speaker_profile": {"gender": "unknown", "age": "unknown", "timbre": "unknown", "accent": "unknown"},
        "paralinguistic": {
            "speaking_rate": "unknown", "pitch_level": "unknown", "volume_level": "unknown",
            "emotion": "unknown", "emotion_intensity": "unknown",
            "emphasis": {"level": "unknown", "emphasized_text": []},
            "prosody": "unknown", "pause": "unknown", "nonverbal_vocalization": ["unknown"],
        },
        "environment": {"background_sound_events": ["unknown"], "recording_quality": "unknown", "acoustic_scene": "unknown"},
    }


def synthesizable_view(value: Any) -> dict[str, Any]:
    caption = canonical_caption(value)
    return {field: get_path(caption, field) for field in ALL_FIELDS if field not in ENVIRONMENT_FIELDS}


def project_caption(value: Any, enabled_fields: set[str]) -> dict[str, Any]:
    """Return a schema-valid view that exposes only evidence-backed dual fields."""
    source = canonical_caption(value)
    projected = empty_caption()
    for field in enabled_fields:
        if field not in ALL_FIELDS:
            raise ValueError(f"Unknown caption projection field: {field}")
        set_path(projected, field, deepcopy(get_path(source, field)))
    return canonical_caption(projected)


def _normalize_model_caption(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    normalized = deepcopy(value)
    try:
        inner = normalized.get(WRAPPER_KEY, normalized)
    except (AttributeError, TypeError):
        return normalized
    if not isinstance(inner, dict):
        return normalized
    paralinguistic = inner.get("paralinguistic")
    environment = inner.get("environment")
    for container, key in ((paralinguistic, "nonverbal_vocalization"), (environment, "background_sound_events")):
        if not isinstance(container, dict):
            continue
        item = container.get(key)
        # Model output often uses [] for an observed absence.  Persist the
        # schema's explicit absence sentinel rather than treating it as a
        # missing/invalid field.
        if item == []:
            container[key] = ["none"]
        if isinstance(item, str) and item.strip():
            container[key] = [item.strip()]
    enum_values = {
        "semantic_content.language": ["English", "Chinese", "other", "unknown"],
        "semantic_content.intent": schema_document()["$defs"]["intent"]["enum"],
        "speaker_profile.gender": ["male", "female", "unknown"],
        "speaker_profile.age": ["child", "teenager", "adult", "elderly", "unknown"],
        "speaker_profile.timbre": ["neutral", "breathy", "rough", "creaky", "strained", "nasal", "bright", "dark", "other", "unknown"],
        "speaker_profile.accent": schema_document()["$defs"]["accent"]["enum"],
        "paralinguistic.speaking_rate": ["slow", "moderate", "fast", "unknown"],
        "paralinguistic.pitch_level": ["low", "medium", "high", "unknown"],
        "paralinguistic.volume_level": ["low", "medium", "high", "unknown"],
        "paralinguistic.emotion": ["neutral", "happy", "sad", "angry", "fearful", "surprised", "disgusted", "other", "unknown"],
        "paralinguistic.emotion_intensity": ["none", "low", "medium", "high", "unknown"],
        "paralinguistic.emphasis.level": ["none", "emphasized", "heavily emphasized", "unknown"],
        "environment.recording_quality": ["bad", "poor", "fair", "good", "excellent", "unknown"],
    }
    for path, allowed in enum_values.items():
        item = get_path(inner, path)
        if isinstance(item, str):
            normalized_item = item.strip().casefold()
            aliases = {
                ("paralinguistic.volume_level", "moderate"): "medium",
                ("paralinguistic.emphasis.level", "emphasis"): "emphasized",
                ("paralinguistic.emphasis.level", "heavy emphasis"): "heavily emphasized",
            }
            normalized_item = aliases.get((path, normalized_item), normalized_item)
            canonical = {candidate.casefold(): candidate for candidate in allowed}.get(normalized_item)
            if canonical is not None:
                set_path(inner, path, canonical)
    emphasis = paralinguistic.get("emphasis") if isinstance(paralinguistic, dict) else None
    if isinstance(emphasis, dict) and emphasis.get("emphasized_text"):
        level = emphasis.get("level")
        if isinstance(level, str) and level in {"none", "unknown"}:
            emphasis["level"] = "emphasized"
    return normalized


def parse_caption_text(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL)
    candidates = [raw]
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        candidates.append(raw[start:end + 1])
    parse_errors: list[str] = []
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as exc:
            parse_errors.append(str(exc))
            continue
        value = _normalize_model_caption(value)
        errors = validation_errors(value)
        if not errors:
            return unwrap_caption(value), []
        parse_errors.extend(errors)
    return None, parse_errors or ["No JSON object found"]


def parse_dual_caption_text(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Parse the source no-environment schema without reintroducing environment."""
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL)
    candidates = [raw]
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        candidates.append(raw[start:end + 1])
    parse_errors: list[str] = []
    for candidate in candidates:
        try:
            value = _normalize_model_caption(json.loads(candidate))
        except json.JSONDecodeError as exc:
            parse_errors.append(str(exc))
            continue
        try:
            wrapped = wrap_caption(unwrap_caption(value))
        except ValueError as exc:
            parse_errors.append(str(exc))
            continue
        errors = []
        for error in sorted(dual_validator().iter_errors(wrapped), key=lambda item: list(item.absolute_path)):
            location = ".".join(str(part) for part in error.absolute_path) or "$"
            errors.append(f"{location}: {error.message}")
        errors.extend(_logic_errors(wrapped[WRAPPER_KEY]))
        if errors:
            parse_errors.extend(errors)
            continue
        return canonical_source_caption(wrapped[WRAPPER_KEY]), []
    return None, parse_errors or ["No JSON object found"]


def known_field_values(caption: dict[str, Any], known_fields: list[str] | None = None) -> dict[str, Any]:
    fields_to_use = known_fields or list(SOURCE_FIELDS)
    return {field: get_path(caption, field) for field in fields_to_use if get_path(caption, field) is not None}
