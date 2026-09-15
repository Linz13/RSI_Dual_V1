"""RewardV3 input admission: canonicalize representations, never fill answers.

The legacy lenient parser remains responsible for format-curriculum telemetry.
This module independently reads raw text and owns semantic input admission.
"""
from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from .constants import WRAPPER_KEY
from .dual_space import _NONVERBAL_VALUES, _SYNTH_ENUMS, synth_validation_errors
from .schema import get_path, set_path


ADMISSION_VERSION = "safe_semantic_input_v1"

# Deliberately narrower than the legacy lossy normalizer. No unsupported value
# becomes unknown; emphasis intensity and uncertain regional accents are not guessed.
SAFE_ENUM_ALIASES = {
    "speaker_profile.age": {"young adult": "adult", "middle-aged": "adult", "middle aged": "adult"},
    "speaker_profile.accent": {
        "american english": "US English", "general american": "US English",
        "standard mandarin": "Mandarin", "standard putonghua": "Mandarin",
    },
    "paralinguistic.speaking_rate": {"normal": "moderate", "average": "moderate"},
    "paralinguistic.pitch_level": {"mid": "medium", "mid-level": "medium"},
    "paralinguistic.volume_level": {
        "loud": "high", "quiet": "low", "soft": "low", "moderate": "medium",
        "normal": "medium", "average": "medium",
    },
    "paralinguistic.emotion_intensity": {"moderate": "medium"},
    "paralinguistic.emphasis.level": {"heavy emphasis": "heavily emphasized"},
}
SAFE_NONVERBAL_ALIASES = {
    "laugh": "laughter", "laughing": "laughter", "sighing": "sigh",
    "coughing": "cough", "breathing": "audible breath", "breath": "audible breath",
    "clearing throat": "throat clearing", "cleared throat": "throat clearing",
    "no vocalization": "none", "no vocalizations": "none",
    "no nonverbal vocalization": "none", "no nonverbal vocalizations": "none",
    "no observable nonverbal vocalizations": "none",
}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate_key:{key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f"nonfinite_json_constant:{value}")


def _read_object(text: Any) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("missing_raw_text")
    raw = text.strip()
    if raw.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.I | re.S)
        if not match:
            raise ValueError("incomplete_json_fence")
        raw = match.group(1).strip()
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    # Accept one object surrounded by prose, but never rescue a nested object
    # from a broken outer object, an array, or a dangling wrapper key.
    start = raw.find("{")
    if start < 0 or any(char in raw[:start] for char in '{}[]"'):
        raise ValueError("not_a_single_json_object")
    try:
        value, end = decoder.raw_decode(raw, start)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid_json:{exc.msg}") from exc
    if any(char in raw[end:] for char in '{}[]"'):
        raise ValueError("ambiguous_trailing_json")
    if not isinstance(value, dict):
        raise ValueError("not_a_json_object")
    return value


def admit_synth_text(text: Any) -> dict[str, Any]:
    """Return auditable admission metadata; input and sampled trajectory stay untouched."""
    rules: list[str] = []
    result: dict[str, Any] = {
        "semantic_admission_version": ADMISSION_VERSION,
        "semantic_input_valid": False,
        "semantic_caption": None,
        "safe_normalization_rules": rules,
        "semantic_rejection_reasons": [],
    }
    try:
        value = _read_object(text)
        if WRAPPER_KEY in value:
            # Known unused labels may be discarded. Other wrapper siblings could
            # contain conflicting answers and must not be silently ignored.
            extra = set(value) - {WRAPPER_KEY, "topic", "intent", "environment"}
            if extra:
                raise ValueError("ambiguous_wrapper_siblings:" + ",".join(sorted(extra)))
            for key in sorted(set(value) - {WRAPPER_KEY}):
                rules.append(f"$.{key}:drop_excluded_field")
            value = value[WRAPPER_KEY]
        if not isinstance(value, dict):
            raise ValueError("caption_not_an_object")
        caption = deepcopy(value)
        for key in ("topic", "intent", "environment"):
            if key in caption:
                caption.pop(key)
                rules.append(f"{key}:drop_excluded_field")
        semantic = caption.get("semantic_content")
        if isinstance(semantic, dict):
            for key in ("topic", "intent"):
                if key in semantic:
                    semantic.pop(key)
                    rules.append(f"semantic_content.{key}:drop_excluded_field")

        for field, allowed in _SYNTH_ENUMS.items():
            current = get_path(caption, field)
            if not isinstance(current, str):
                continue  # The unchanged schema reports missing/wrongly typed values.
            key = current.strip().casefold()
            canonical = {item.casefold(): item for item in allowed}.get(key)
            rule = "enum_case_whitespace"
            if canonical is None:
                canonical = SAFE_ENUM_ALIASES.get(field, {}).get(key)
                rule = "explicit_enum_alias"
            if canonical is not None and canonical != current:
                set_path(caption, field, canonical)
                rules.append(f"{field}:{rule}")

        for field in ("paralinguistic.nonverbal_vocalization", "paralinguistic.emphasis.emphasized_text"):
            current = get_path(caption, field)
            if isinstance(current, str):
                current = [current]
                rules.append(f"{field}:scalar_to_list")
                set_path(caption, field, current)
            if field == "paralinguistic.nonverbal_vocalization" and isinstance(current, list):
                allowed = _NONVERBAL_VALUES | {"none", "unknown"}
                normalized = []
                for item in current:
                    if not isinstance(item, str):
                        normalized.append(item)
                        continue
                    key = item.strip().casefold()
                    canonical = key if key in allowed else SAFE_NONVERBAL_ALIASES.get(key)
                    normalized.append(canonical if canonical is not None else item)
                if normalized != current:
                    set_path(caption, field, normalized)
                    rules.append(f"{field}:explicit_value_normalization")

        # No missing sections/fields are added; empty arrays/strings, duplicate
        # list items, unknown enum values and semantic conflicts still fail.
        errors = synth_validation_errors(caption)
        if errors:
            result["semantic_rejection_reasons"] = errors
            return result
        transcript = get_path(caption, "semantic_content.transcript")
        if not transcript.strip() or transcript.strip().casefold() == "unknown":
            raise ValueError("tts_requires_non_unknown_transcript")
        result.update(semantic_input_valid=True, semantic_caption=caption)
    except (ValueError, RecursionError) as exc:
        result["semantic_rejection_reasons"] = [str(exc)]
    return result
