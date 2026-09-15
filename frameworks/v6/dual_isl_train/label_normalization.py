"""Versioned field-level normalization for independent evaluators only."""
from __future__ import annotations

from copy import deepcopy
import json
import unicodedata

from jsonschema import Draft202012Validator

from .partial_caption import field_schema, normalize_field, parse_object
from .schema import get_path, set_path

VERSION = "evaluator_fields_v5.2"
EVENTS = "paralinguistic.nonverbal_vocalization"
EMPHASIS = "paralinguistic.emphasis.emphasized_text"
EVENT_ALIASES = {
    "laugh": "laughter", "laughs": "laughter", "laughing": "laughter",
    "sighs": "sigh", "sighing": "sigh", "coughs": "cough", "coughing": "cough",
    "cry": "crying", "sobs": "sobbing", "sob": "sobbing",
    "throat-clearing": "throat clearing", "clearing throat": "throat clearing",
    "audible breathing": "audible breath", "sneezing": "sneeze", "sneezes": "sneeze",
    "sniffing": "sniff", "yawning": "yawn", "screaming": "scream",
}
# Explicit out-of-ontology vocal events, never a blanket mapping for arbitrary strings.
OTHER_EVENTS = frozenset({
    "kissing sound", "kissing sounds", "kiss sound", "kissing",
    "groan", "groans", "groaning", "moan", "moans", "moaning",
    "gasp", "gasps", "gasping", "humming",
})
EVENT_VALUES = frozenset({
    "none", "unknown", "laughter", "sigh", "crying", "sobbing", "cough",
    "throat clearing", "audible breath", "sneeze", "sniff", "yawn", "scream",
    "filled pause", "agreement sound", "other",
})


def canonical(value):
    return " ".join(unicodedata.normalize("NFKC", value).strip().casefold().split())


def unknown_value(field):
    return ["unknown"] if field in (EVENTS, EMPHASIS) else "unknown"


def normalize_value(field, original):
    schema = field_schema(field)
    value = normalize_field(original, schema)
    if field == EVENTS and isinstance(value, list):
        normalized = []
        for item in value:
            if not isinstance(item, str):
                normalized.append(item)
                continue
            token = canonical(item)
            normalized.append(EVENT_ALIASES.get(token, "other" if token in OTHER_EVENTS else
                                                 token if token in EVENT_VALUES else item))
        value = list(dict.fromkeys(normalized)) if all(isinstance(v, str) for v in normalized) else normalized
        if value == []:
            value = ["none"]  # Existing evaluator convention for an explicitly empty event set.
    if field == EMPHASIS and isinstance(value, list) and len(value) == 1:
        if isinstance(value[0], str) and canonical(value[0]) == "unknown":
            value = ["unknown"]
    return value


def parse_fields(raw, fields):
    data = parse_object(raw)
    missing = object()
    for field in fields:
        node = data
        for part in field.split(".")[:-1]:
            if part not in node:
                break
            node = node[part]
            if not isinstance(node, dict):
                raise ValueError("invalid evaluator container")
    for field in fields:
        if field in data:
            if get_path(data, field, missing) is not missing:
                raise ValueError("ambiguous flat/nested evaluator field: " + field)
            set_path(data, field, data.pop(field))
    # A broken document/container is a response failure, not field uncertainty.
    for field in fields:
        node = data
        for part in field.split(".")[:-1]:
            if part not in node:
                break
            node = node[part]
            if not isinstance(node, dict):
                raise ValueError("invalid evaluator container")
    if not any(get_path(data, f, missing) is not missing for f in fields):
        raise ValueError("no requested evaluator fields")
    attrs, errors, changes, valid = {}, {}, {}, []
    for field in fields:
        original = get_path(data, field, missing)
        if original is missing:
            errors[field] = "missing"
            continue
        value = normalize_value(field, original)
        schema = field_schema(field)
        invalid = not Draft202012Validator(schema).is_valid(value)
        if field in (EVENTS, EMPHASIS) and isinstance(value, list) and len(value) > 1:
            invalid |= any(isinstance(v, str) and canonical(v) in {"none", "unknown"} for v in value)
        if invalid:
            errors[field] = "invalid type or value"
            continue
        valid.append(field)
        set_path(attrs, field, deepcopy(value))
        if value != original:
            changes[field] = {"from": original, "to": value}
    # Preserve the existing convention for emphasis conditioned on a valid none/unknown level.
    if EMPHASIS in valid and get_path(attrs, "paralinguistic.emphasis.level") in ("none", "unknown"):
        set_path(attrs, EMPHASIS, [])
    return {"attributes": attrs, "valid_fields": valid, "field_errors": errors,
            "normalizations": changes, "output_policy_version": VERSION}


def finish_fields(parsed):
    result = deepcopy(parsed["attributes"])
    for field in parsed["field_errors"]:
        set_path(result, field, unknown_value(field))
    return result


def retry_feedback(errors):
    return ("\nValidation feedback (field names only): " + json.dumps(sorted(errors)) +
            ". Return all requested fields in the original nested shape. Use exact allowed enum strings. "
            "Use other only for a clearly identified category not covered by the allowed list, and only "
            "where that enum includes other. Use unknown for uncertain attributes, [\"unknown\"] for "
            "uncertain event/text lists. Never mix unknown/none with event labels. "
            "Do not invent new category names or infer a label from the validation feedback.")
