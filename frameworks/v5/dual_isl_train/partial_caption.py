"""V5 field admission. Never replace a sampled trajectory with repaired text."""
from __future__ import annotations

import json
import re
import unicodedata
from copy import deepcopy

from jsonschema import Draft202012Validator

from .constants import WRAPPER_KEY
from .dual_space import SYNTHESIZABLE_FIELDS, project_synth_caption, synth_schema_document
from .schema import empty_caption, get_path, set_path

VERSION = "partial_psyn_v5.1"


def known(value):
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip()) and value.strip().casefold() != "unknown"
    if isinstance(value, list):
        return not any(isinstance(v, str) and v.strip().casefold() == "unknown" for v in value)
    return True


def _object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key: " + key)
        out[key] = value
    return out


def parse_object(text):
    if not isinstance(text, str):
        raise ValueError("expected text")
    text = text.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.I)
        if not match:
            raise ValueError("incomplete code fence")
        text = match.group(1)
    def invalid_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    value = json.loads(text, object_pairs_hook=_object, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("expected one JSON object")
    if WRAPPER_KEY in value:
        value = value[WRAPPER_KEY]
        if not isinstance(value, dict):
            raise ValueError("invalid schema wrapper")
    return value


def field_schema(field):
    document = synth_schema_document()
    node = document["properties"][WRAPPER_KEY]
    for part in field.split("."):
        node = node["properties"][part]
    def resolve(value):
        if isinstance(value, dict):
            if "$ref" in value:
                ref = document
                for part in value["$ref"].removeprefix("#/").split("/"):
                    ref = ref[part]
                return resolve(ref)
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        return value
    return resolve(node)


def normalize_field(value, schema):
    if isinstance(value, str):
        value = unicodedata.normalize("NFKC", value).strip()
        choices = schema.get("enum", [])
        for choice in choices:
            if isinstance(choice, str) and value.casefold() == choice.casefold():
                return choice
    if isinstance(value, list) and "items" in schema:
        value = [normalize_field(v, schema["items"]) for v in value]
        value = list(dict.fromkeys(value)) if all(isinstance(v, str) for v in value) else value
    return value


def admit(text):
    caption = project_synth_caption(empty_caption())
    result = {"admission_version": VERSION, "caption": caption, "json_parseable": False,
              "format_score": 0.0, "valid_fields": [], "condition_fields": [],
              "field_errors": {}, "normalization_rules": [], "semantic_input_valid": False}
    try:
        raw = parse_object(text)
    except (ValueError, TypeError) as exc:
        result["parse_errors"] = [str(exc)]
        return result
    result["json_parseable"] = True
    absent = object()
    for field in SYNTHESIZABLE_FIELDS:
        original = get_path(raw, field, absent)
        if original is absent:
            result["field_errors"][field] = "missing"
            continue
        schema = field_schema(field)
        value = normalize_field(original, schema)
        errors = list(Draft202012Validator(schema).iter_errors(value))
        if field.endswith("nonverbal_vocalization") and value == []:
            value = ["none"]
            errors = []
        if isinstance(value, list) and len(value) > 1 and any(v in ("unknown", "none") for v in value):
            errors = ["mixed absence/unknown sentinel"]
        if errors:
            result["field_errors"][field] = "invalid type or value"
            continue
        result["valid_fields"].append(field)
        if value != original:
            result["normalization_rules"].append(field + ":safe_normalization")
        set_path(caption, field, deepcopy(value))
        if known(value):
            result["condition_fields"].append(field)
    # Conflicting conditions are not passed to TTS; do not invent a replacement.
    emphasis = "paralinguistic.emphasis.emphasized_text"
    if get_path(caption, "paralinguistic.emphasis.level") in ("unknown", "none"):
        set_path(caption, emphasis, [])
        if emphasis in result["condition_fields"]:
            result["condition_fields"].remove(emphasis)
    intensity = "paralinguistic.emotion_intensity"
    if get_path(caption, "paralinguistic.emotion") == "neutral" and get_path(caption, intensity) in ("medium", "high"):
        set_path(caption, intensity, "unknown")
        result["field_errors"][intensity] = "conflicts with neutral emotion"
        if intensity in result["condition_fields"]:
            result["condition_fields"].remove(intensity)
    result["format_score"] = 0.5 + 0.5 * len(result["valid_fields"]) / len(SYNTHESIZABLE_FIELDS)
    result["semantic_input_valid"] = known(get_path(caption, "semantic_content.transcript"))
    result["raw_schema_valid"] = Draft202012Validator(synth_schema_document()).is_valid({WRAPPER_KEY: raw})
    result["parse_errors"] = []
    return result
