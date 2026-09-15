from __future__ import annotations

import inspect
import json
import re
from copy import deepcopy
from functools import lru_cache
from typing import Any

from jsonschema import Draft202012Validator

from .io import stable_hash
from dual_isl_train.constants import ALL_FIELDS, WRAPPER_KEY  # noqa: E402
from dual_isl_train.schema import (  # noqa: E402
    _normalize_model_caption,
    canonical_caption,
    canonical_source_caption,
    empty_caption,
    get_path,
    schema_document,
    set_path,
    unwrap_caption,
    wrap_caption,
)


DUAL_SPACE_NAME = "qwen3_voice_design_v1"

TEXT_FIELD = "semantic_content.transcript"
LANGUAGE_FIELD = "semantic_content.language"
INSTRUCT_FIELDS = (
    "speaker_profile.gender",
    "speaker_profile.age",
    "speaker_profile.timbre",
    "speaker_profile.accent",
    "paralinguistic.speaking_rate",
    "paralinguistic.pitch_level",
    "paralinguistic.volume_level",
    "paralinguistic.emotion",
    "paralinguistic.emotion_intensity",
    "paralinguistic.emphasis.level",
    "paralinguistic.emphasis.emphasized_text",
    "paralinguistic.prosody",
    "paralinguistic.pause",
    "paralinguistic.nonverbal_vocalization",
)
SYNTHESIZABLE_FIELDS = (LANGUAGE_FIELD, TEXT_FIELD, *INSTRUCT_FIELDS)
SYNTHESIZABLE_FIELD_SET = frozenset(SYNTHESIZABLE_FIELDS)
EXCLUDED_FIELDS = tuple(field for field in ALL_FIELDS if field not in SYNTHESIZABLE_FIELD_SET)
FIELD_DESTINATIONS = {
    TEXT_FIELD: "text",
    LANGUAGE_FIELD: "language",
    **{field: "instruct" for field in INSTRUCT_FIELDS},
}
EXCLUSION_REASONS = {
    "semantic_content.topic": "not independently consumed by the Qwen3-TTS request",
    "semantic_content.intent": "not independently consumed by the Qwen3-TTS request",
    "environment.background_sound_events": "VoiceDesign renderer has no environment/background condition",
    "environment.recording_quality": "VoiceDesign renderer has no recording-channel condition",
    "environment.acoustic_scene": "VoiceDesign renderer has no acoustic-scene condition",
}


@lru_cache(maxsize=1)
def synth_schema_document() -> dict[str, Any]:
    """Reduced schema used for every pseudo-pair inside the dual loop."""
    document = deepcopy(schema_document())
    caption = document["properties"][WRAPPER_KEY]
    caption["required"] = ["semantic_content", "speaker_profile", "paralinguistic"]
    caption["properties"].pop("environment", None)
    semantic = caption["properties"]["semantic_content"]
    semantic["required"] = ["language", "transcript"]
    semantic["properties"] = {
        key: value for key, value in semantic["properties"].items()
        if key in {"language", "transcript"}
    }
    document["$id"] = "isl-train.synth-caption.v1"
    return document


@lru_cache(maxsize=1)
def synth_validator() -> Draft202012Validator:
    return Draft202012Validator(synth_schema_document())


def synth_validation_errors(value: Any) -> list[str]:
    try:
        wrapped = wrap_caption(unwrap_caption(value))
    except ValueError as exc:
        return [str(exc)]
    errors = []
    for error in sorted(synth_validator().iter_errors(wrapped), key=lambda item: list(item.absolute_path)):
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        errors.append(f"{location}: {error.message}")
    inner = wrapped[WRAPPER_KEY]
    paralinguistic = inner.get("paralinguistic")
    if not isinstance(paralinguistic, dict):
        return errors
    emphasis = paralinguistic.get("emphasis")
    if isinstance(emphasis, dict):
        level = emphasis.get("level")
        if (
            isinstance(level, str)
            and level in {"none", "unknown"}
            and emphasis.get("emphasized_text") not in ([], None)
        ):
            errors.append("paralinguistic.emphasis.emphasized_text must be [] when level is none/unknown")
    emotion = paralinguistic.get("emotion")
    intensity = paralinguistic.get("emotion_intensity")
    if emotion == "neutral" and isinstance(intensity, str) and intensity not in {"none", "low", "unknown"}:
        errors.append("neutral emotion cannot have medium/high intensity")
    return errors


def canonical_synth_caption(value: Any) -> dict[str, Any]:
    inner = unwrap_caption(value)
    errors = synth_validation_errors(inner)
    if errors:
        raise ValueError("Invalid synthesizable caption: " + "; ".join(errors))
    return inner


def project_synth_caption(value: Any) -> dict[str, Any]:
    """P_syn(c): retain only fields represented by the actual TTS request."""
    try:
        source = canonical_source_caption(value)
    except ValueError:
        try:
            source = canonical_caption(value)
        except ValueError:
            return canonical_synth_caption(value)
    return {
        "semantic_content": {
            "language": deepcopy(get_path(source, LANGUAGE_FIELD)),
            "transcript": deepcopy(get_path(source, TEXT_FIELD)),
        },
        "speaker_profile": deepcopy(source["speaker_profile"]),
        "paralinguistic": deepcopy(source["paralinguistic"]),
    }


def restore_full_caption(value: Any) -> dict[str, Any]:
    """Restore canonical unknowns only for internal full-schema compatibility."""
    synth = canonical_synth_caption(value)
    restored = empty_caption()
    restored["semantic_content"]["language"] = synth["semantic_content"]["language"]
    restored["semantic_content"]["transcript"] = synth["semantic_content"]["transcript"]
    restored["speaker_profile"] = deepcopy(synth["speaker_profile"])
    restored["paralinguistic"] = deepcopy(synth["paralinguistic"])
    return canonical_caption(restored)


def synth_caption_json(value: Any) -> str:
    synth = project_synth_caption(value)
    ordered = {
        "semantic_content": synth["semantic_content"],
        "speaker_profile": synth["speaker_profile"],
        "paralinguistic": synth["paralinguistic"],
    }
    return json.dumps({WRAPPER_KEY: ordered}, ensure_ascii=False, separators=(",", ":"))


_SYNTH_ENUMS = {
    "semantic_content.language": ("English", "Chinese", "other", "unknown"),
    "speaker_profile.gender": ("male", "female", "unknown"),
    "speaker_profile.age": ("child", "teenager", "adult", "elderly", "unknown"),
    "speaker_profile.timbre": ("neutral", "breathy", "rough", "creaky", "strained", "nasal", "bright", "dark", "other", "unknown"),
    "speaker_profile.accent": tuple(schema_document()["$defs"]["accent"]["enum"]),
    "paralinguistic.speaking_rate": ("slow", "moderate", "fast", "unknown"),
    "paralinguistic.pitch_level": ("low", "medium", "high", "unknown"),
    "paralinguistic.volume_level": ("low", "medium", "high", "unknown"),
    "paralinguistic.emotion": ("neutral", "happy", "sad", "angry", "fearful", "surprised", "disgusted", "other", "unknown"),
    "paralinguistic.emotion_intensity": ("none", "low", "medium", "high", "unknown"),
    "paralinguistic.emphasis.level": ("none", "emphasized", "heavily emphasized", "unknown"),
}

_SYNTH_ENUM_ALIASES = {
    "speaker_profile.age": {
        "young adult": "adult",
        "middle aged": "adult",
        "middle-aged": "adult",
    },
    "speaker_profile.timbre": {
        "clear and bright": "bright",
    },
    "speaker_profile.accent": {
        "american": "US English",
        "american english": "US English",
        "general american": "US English",
        "west african english": "African English",
        "standard mandarin": "Mandarin",
        "standard putonghua": "Mandarin",
        "mandarin with slight regional inflection": "Mandarin",
        "mandarin with standard putonghua accent": "Mandarin",
    },
    "paralinguistic.speaking_rate": {
        "normal": "moderate",
        "average": "moderate",
    },
    "paralinguistic.pitch_level": {
        "mid": "medium",
        "mid-level": "medium",
    },
    "paralinguistic.volume_level": {
        "moderate": "medium",
        "normal": "medium",
        "average": "medium",
        "loud": "high",
        "quiet": "low",
        "soft": "low",
    },
    "paralinguistic.emotion_intensity": {
        "moderate": "medium",
    },
    "paralinguistic.emphasis.level": {
        "emphasis": "emphasized",
        "low": "emphasized",
        "medium": "emphasized",
        "moderate": "emphasized",
        "high": "heavily emphasized",
        "heavy emphasis": "heavily emphasized",
    },
}

_NONVERBAL_ALIASES = {
    "uh": "filled pause",
    "um": "filled pause",
    "erm": "filled pause",
    "hmm": "filled pause",
    "breath": "audible breath",
    "breathing": "audible breath",
    "clearing throat": "throat clearing",
    "cleared throat": "throat clearing",
    "no vocalization": "none",
    "no vocalizations": "none",
    "no nonverbal vocalization": "none",
    "no nonverbal vocalizations": "none",
    "yes": "agreement sound",
    "mhm": "agreement sound",
    "mm-hmm": "agreement sound",
}

_NONVERBAL_VALUES = frozenset(
    schema_document()["$defs"]["nonverbalList"]["oneOf"][2]["items"]["enum"]
)

_SYNTH_SECTION_FIELDS = {
    "semantic_content": ("language", "transcript"),
    "speaker_profile": ("gender", "age", "timbre", "accent"),
    "paralinguistic": (
        "speaking_rate", "pitch_level", "volume_level", "emotion", "emotion_intensity",
        "emphasis", "prosody", "pause", "nonverbal_vocalization",
    ),
}


def _schema_echo_value(value: Any) -> tuple[Any, str | None]:
    if not isinstance(value, dict):
        return value, None
    if set(value) == {"const"}:
        return deepcopy(value["const"]), "schema_echo_const"
    if "oneOf" in value:
        return "unknown", "schema_echo_oneof"
    return value, None


def normalize_synth_caption(value: Any) -> tuple[Any, list[str]]:
    """Repair a JSON-like Captioner output into the canonical TTS-facing schema.

    Shared full-caption repairs run first so the same field/value pair is not
    accepted by the full parser and rejected by the P_syn parser.  P_syn then
    applies deterministic, conservative repairs: recognized aliases are mapped
    to canonical enums, unsupported conditioning values become ``unknown``, and
    missing non-semantic fields are filled from the canonical empty caption.
    The transcript remains mandatory at the sanity-gate layer.
    """
    original = deepcopy(value)
    normalized = _normalize_model_caption(value)
    inner = normalized.get(WRAPPER_KEY, normalized) if isinstance(normalized, dict) else normalized
    if not isinstance(inner, dict):
        return normalized, []
    rules: list[str] = []
    if isinstance(normalized, dict) and WRAPPER_KEY in normalized:
        for key in list(normalized):
            if key != WRAPPER_KEY:
                normalized.pop(key)
                rules.append(f"{key}:drop_outer_field")

    original_inner = original.get(WRAPPER_KEY, original) if isinstance(original, dict) else original
    if isinstance(original_inner, dict):
        missing = object()
        for field in SYNTHESIZABLE_FIELDS:
            before = get_path(original_inner, field, missing)
            after = get_path(inner, field, missing)
            if before is not missing and after is not missing and before != after:
                rules.append(f"{field}:shared_full_normalization")

    defaults = project_synth_caption(empty_caption())
    for key in list(inner):
        if key not in _SYNTH_SECTION_FIELDS:
            inner.pop(key)
            rules.append(f"{key}:drop_non_synth_field")
    for section, allowed_fields in _SYNTH_SECTION_FIELDS.items():
        current = inner.get(section)
        if not isinstance(current, dict):
            inner[section] = deepcopy(defaults[section])
            rules.append(f"{section}:fill_invalid_section")
            continue
        for key in list(current):
            if key not in allowed_fields:
                current.pop(key)
                rules.append(f"{section}.{key}:drop_non_synth_field")
        for key in allowed_fields:
            if key not in current:
                current[key] = deepcopy(defaults[section][key])
                rules.append(f"{section}.{key}:fill_unknown")

    emphasis = inner["paralinguistic"].get("emphasis")
    if isinstance(emphasis, str):
        inner["paralinguistic"]["emphasis"] = {"level": emphasis, "emphasized_text": []}
        emphasis = inner["paralinguistic"]["emphasis"]
        rules.append("paralinguistic.emphasis:scalar_to_object")
    elif not isinstance(emphasis, dict):
        inner["paralinguistic"]["emphasis"] = deepcopy(defaults["paralinguistic"]["emphasis"])
        emphasis = inner["paralinguistic"]["emphasis"]
        rules.append("paralinguistic.emphasis:fill_invalid_object")
    for key in list(emphasis):
        if key not in {"level", "emphasized_text"}:
            emphasis.pop(key)
            rules.append(f"paralinguistic.emphasis.{key}:drop_unknown_field")
    if "level" not in emphasis:
        emphasis["level"] = "unknown"
        rules.append("paralinguistic.emphasis.level:fill_unknown")
    if "emphasized_text" not in emphasis:
        emphasis["emphasized_text"] = []
        rules.append("paralinguistic.emphasis.emphasized_text:fill_empty")

    for field in SYNTHESIZABLE_FIELDS:
        current = get_path(inner, field)
        repaired, rule = _schema_echo_value(current)
        if rule:
            set_path(inner, field, repaired)
            rules.append(f"{field}:{rule}")

    for field, allowed in _SYNTH_ENUMS.items():
        current = get_path(inner, field)
        if isinstance(current, str):
            key = current.strip().casefold()
            canonical = {item.strip().casefold(): item for item in allowed}.get(key)
            if canonical is None:
                canonical = _SYNTH_ENUM_ALIASES.get(field, {}).get(key)
                if canonical is not None:
                    rules.append(f"{field}:enum_alias")
            if canonical is None:
                canonical = "unknown"
                rules.append(f"{field}:unsupported_enum_to_unknown")
            if canonical != current:
                set_path(inner, field, canonical)
                if f"{field}:enum_alias" not in rules and f"{field}:unsupported_enum_to_unknown" not in rules:
                    rules.append(f"{field}:enum_case_whitespace")
        else:
            set_path(inner, field, "unknown")
            rules.append(f"{field}:invalid_enum_type_to_unknown")

    transcript = get_path(inner, TEXT_FIELD)
    if isinstance(transcript, str):
        stripped = transcript.strip()
        if stripped != transcript:
            set_path(inner, TEXT_FIELD, stripped)
            rules.append(f"{TEXT_FIELD}:strip_whitespace")
    else:
        set_path(inner, TEXT_FIELD, "unknown")
        rules.append(f"{TEXT_FIELD}:invalid_type_to_unknown")

    for field in ("paralinguistic.prosody", "paralinguistic.pause"):
        current = get_path(inner, field)
        if isinstance(current, str):
            stripped = current.strip()
            if not stripped:
                stripped = "unknown"
                rules.append(f"{field}:empty_to_unknown")
            elif stripped != current:
                rules.append(f"{field}:strip_whitespace")
            set_path(inner, field, stripped)
        else:
            set_path(inner, field, "unknown")
            rules.append(f"{field}:invalid_type_to_unknown")

    emphasized_text = emphasis.get("emphasized_text")
    if isinstance(emphasized_text, str):
        emphasized_text = [emphasized_text]
        rules.append("paralinguistic.emphasis.emphasized_text:scalar_to_list")
    if not isinstance(emphasized_text, list):
        emphasized_text = []
        rules.append("paralinguistic.emphasis.emphasized_text:invalid_type_to_empty")
    cleaned_text: list[str] = []
    for item in emphasized_text:
        if not isinstance(item, str) or not item.strip():
            continue
        item = item.strip()
        if item.casefold() in {"none", "unknown"}:
            continue
        if item not in cleaned_text:
            cleaned_text.append(item)
    if cleaned_text != emphasis.get("emphasized_text"):
        emphasis["emphasized_text"] = cleaned_text
        rules.append("paralinguistic.emphasis.emphasized_text:clean_unique_strings")
    if emphasis["level"] in {"none", "unknown"} and cleaned_text:
        emphasis["level"] = "emphasized"
        rules.append("paralinguistic.emphasis.level:infer_from_emphasized_text")

    if (
        inner["paralinguistic"]["emotion"] == "neutral"
        and inner["paralinguistic"]["emotion_intensity"] in {"medium", "high"}
    ):
        inner["paralinguistic"]["emotion_intensity"] = "unknown"
        rules.append("paralinguistic.emotion_intensity:neutral_conflict_to_unknown")

    field = "paralinguistic.nonverbal_vocalization"
    current = get_path(inner, field)
    if isinstance(current, str):
        current = [current]
        rules.append(f"{field}:scalar_to_list")
    if not isinstance(current, list):
        current = ["unknown"]
        rules.append(f"{field}:invalid_type_to_unknown")
    cleaned: list[str] = []
    for item in current:
        if not isinstance(item, str) or not item.strip():
            continue
        key = item.strip().casefold()
        canonical = _NONVERBAL_ALIASES.get(key, key)
        if canonical not in _NONVERBAL_VALUES and canonical not in {"none", "unknown"}:
            canonical = "other"
        if canonical not in cleaned:
            cleaned.append(canonical)
    real = [item for item in cleaned if item not in {"none", "unknown"}]
    canonical_nonverbal = real or (["none"] if "none" in cleaned else ["unknown"])
    if canonical_nonverbal != get_path(inner, field):
        set_path(inner, field, canonical_nonverbal)
        rules.append(f"{field}:canonicalize_list")
    return normalized, rules


def parse_synth_caption_text_detailed(text: str) -> tuple[dict[str, Any] | None, list[str], dict[str, Any]]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL)
    candidates = [raw]
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        candidates.append(raw[start:end + 1])
    candidates = list(dict.fromkeys(candidates))
    errors: list[str] = []
    applied_rules: list[str] = []
    raw_schema_valid = False
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            continue
        raw_validation = synth_validation_errors(value)
        raw_schema_valid = raw_schema_valid or not raw_validation
        normalized, rules = normalize_synth_caption(value)
        applied_rules.extend(rule for rule in rules if rule not in applied_rules)
        validation = synth_validation_errors(normalized)
        if not validation:
            return canonical_synth_caption(normalized), [], {
                "raw_schema_valid": not raw_validation,
                "normalized_schema_valid": True,
                "normalization_rules": rules,
            }
        errors.extend(validation)
    return None, errors or ["No JSON object found"], {
        "raw_schema_valid": raw_schema_valid,
        "normalized_schema_valid": False,
        "normalization_rules": applied_rules,
    }


def parse_synth_caption_text(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    caption, errors, _ = parse_synth_caption_text_detailed(text)
    return caption, errors


def reparse_synth_candidate(candidate: Any) -> dict[str, Any] | None:
    """Reapply the current P_syn contract to a persisted rollout candidate.

    Filtering owns the final acceptance decision.  Reparsing the saved raw text
    here keeps that decision independent from the parser version used inside a
    model worker and permits CPU-only audits of already generated rollouts.
    """
    if not isinstance(candidate, dict):
        return None
    from .semantic_admission import admit_synth_text

    reparsed = deepcopy(candidate)
    raw_text = reparsed.get("raw_text")
    if not isinstance(raw_text, str):
        reparsed.update(admit_synth_text(raw_text))
        return reparsed
    caption, errors, metadata = parse_synth_caption_text_detailed(raw_text)
    reparsed.update({"caption": caption, "parse_errors": errors, **metadata})
    admission = admit_synth_text(raw_text)
    reparsed.update(admission)
    if admission["semantic_input_valid"]:
        caption = admission["semantic_caption"]
        reparsed["caption"] = caption
    if caption is not None:
        reparsed["canonical_text"] = synth_caption_json(caption)
    else:
        reparsed.pop("canonical_text", None)
    return reparsed


def _contract_fixture() -> dict[str, Any]:
    value = empty_caption()
    value["semantic_content"].update(
        language="English", transcript="A contract-test utterance.", topic="contract testing", intent="statement",
    )
    value["speaker_profile"].update(gender="male", age="adult", timbre="neutral", accent="US English")
    value["paralinguistic"].update(
        speaking_rate="moderate", pitch_level="medium", volume_level="medium",
        emotion="neutral", emotion_intensity="none", prosody="even declarative phrasing",
        pause="brief natural pauses", nonverbal_vocalization=["none"],
    )
    value["paralinguistic"]["emphasis"] = {"level": "none", "emphasized_text": []}
    value["environment"] = {
        "background_sound_events": ["none"], "recording_quality": "good", "acoustic_scene": "studio",
    }
    return canonical_caption(value)


CONTRACT_ALTERNATIVES = {
    LANGUAGE_FIELD: "Chinese",
    TEXT_FIELD: "A changed contract-test utterance.",
    "speaker_profile.gender": "female",
    "speaker_profile.age": "elderly",
    "speaker_profile.timbre": "breathy",
    "speaker_profile.accent": "England English",
    "paralinguistic.speaking_rate": "fast",
    "paralinguistic.pitch_level": "high",
    "paralinguistic.volume_level": "high",
    "paralinguistic.emotion": "happy",
    "paralinguistic.emotion_intensity": "low",
    "paralinguistic.emphasis.level": "emphasized",
    "paralinguistic.emphasis.emphasized_text": ["contract-test"],
    "paralinguistic.prosody": "rising expressive intonation",
    "paralinguistic.pause": "a long pause after the first phrase",
    "paralinguistic.nonverbal_vocalization": ["laughter"],
    "semantic_content.topic": "a changed topic",
    "semantic_content.intent": "question",
    "environment.background_sound_events": ["traffic"],
    "environment.recording_quality": "poor",
    "environment.acoustic_scene": "street",
}


def dual_subspace_report() -> dict[str, Any]:
    from .render import render_qwen_request

    fixture = _contract_fixture()
    baseline = render_qwen_request(fixture)
    checks = []
    for field in (*SYNTHESIZABLE_FIELDS, *EXCLUDED_FIELDS):
        changed = deepcopy(fixture)
        set_path(changed, field, deepcopy(CONTRACT_ALTERNATIVES[field]))
        if field == "paralinguistic.emphasis.emphasized_text":
            set_path(changed, "paralinguistic.emphasis.level", "emphasized")
        request = render_qwen_request(changed)
        destination = FIELD_DESTINATIONS.get(field)
        observed = request != baseline
        ok = observed if destination else not observed
        checks.append({
            "field": field,
            "expected_destination": destination,
            "request_changed": observed,
            "ok": ok,
        })
    schema = synth_schema_document()
    return {
        "ok": all(item["ok"] for item in checks),
        "name": DUAL_SPACE_NAME,
        "definition": "conditioning_supported",
        "empirically_validated": False,
        "evidence": {
            "renderer": "dual_isl_train.render.render_qwen_request",
            "interpretation": (
                "text, language, and instruct are demonstrably encoded; individual natural-language "
                "control fidelity is not established by this static contract"
            ),
        },
        "included_fields": [
            {"field": field, "destination": FIELD_DESTINATIONS[field]} for field in SYNTHESIZABLE_FIELDS
        ],
        "excluded_fields": [
            {"field": field, "reason": EXCLUSION_REASONS[field]} for field in EXCLUDED_FIELDS
        ],
        "contract_checks": checks,
        "hashes": {
            "field_list": stable_hash(SYNTHESIZABLE_FIELDS),
            "schema": stable_hash(schema),
            "implementation": stable_hash(inspect.getsource(project_synth_caption)),
            "renderer": stable_hash(inspect.getsource(render_qwen_request)),
        },
    }
