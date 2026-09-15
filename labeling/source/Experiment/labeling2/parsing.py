from __future__ import annotations

import json
import re
import copy
from typing import Any

from .target_schema import CLOSED_VALUES, EVENT_VALUES, get_path, set_path


def balanced_json(text: str) -> list[str]:
    candidates: list[str] = []
    for match in re.finditer(r"\{", text):
        start, depth, in_string, escaped = match.start(), 0, False, False
        for pos in range(start, len(text)):
            char = text[pos]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start:pos + 1])
                    break
    return sorted(set(candidates), key=len, reverse=True)


def extract_json(text: str) -> tuple[dict[str, Any] | None, str]:
    text = str(text or "").strip()
    if not text:
        return None, "empty_response"
    candidates = [text]
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.I | re.S)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    if "</think>" in text:
        candidates.insert(0, text.rsplit("</think>", 1)[-1].strip())
    candidates.extend(balanced_json(text))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            if isinstance(value.get("Target_JSON_Schema"), dict):
                value = value["Target_JSON_Schema"]
            return value, "json"
    return None, "unparseable_json"


def _scalar(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
        return value[0].strip()
    return None


def _string_list(value: Any) -> list[str] | None:
    """Normalize recoverable model list encodings without relaxing schemas.

    Audio models commonly return a requested JSON list as either a single
    string or as a JSON-encoded string such as ``"[]"``.  Both forms are
    unambiguous, so decode them before validation and normalization.  Other
    value types remain invalid.
    """
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, list):
                value = decoded
            else:
                value = [text]
        else:
            value = [text]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return None
    return [item.strip() for item in value if item.strip()]


def expand_dotted_fields(payload: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    """Recover requested nested fields returned as top-level dotted keys."""
    result = copy.deepcopy(payload)
    for field in fields:
        if field in result and get_path(result, field) is None:
            set_path(result, field, result[field])
    return result


def _event_list(value: Any, *, closed_nonverbal: bool) -> list[str] | None:
    values = _string_list(value)
    if values is None:
        return None
    if not values:
        return ["none"]
    if not closed_nonverbal:
        return values
    normalized: list[str] = []
    for item in values:
        event = item if item in EVENT_VALUES else "other"
        if event not in normalized:
            normalized.append(event)
    return normalized


def validate_subset(payload: dict[str, Any], fields: list[str]) -> dict[str, str]:
    """Return field -> error for only the fields requested from one model."""
    payload = expand_dotted_fields(payload, fields)
    errors: dict[str, str] = {}
    for field in fields:
        value = get_path(payload, field)
        if field in {"paralinguistic.nonverbal_vocalization", "environment.background_sound_events"}:
            value = _event_list(value, closed_nonverbal=field.endswith("nonverbal_vocalization"))
            if not value:
                errors[field] = "must be a non-empty string list"
                continue
            if len(set(value)) != len(value):
                errors[field] = "duplicate values"
            if ("none" in value or "unknown" in value) and len(value) != 1:
                errors[field] = "none/unknown must be exclusive"
            continue
        if field == "paralinguistic.emphasis.emphasized_text":
            values = _string_list(value)
            if values is None:
                errors[field] = "must be a string list"
            elif len(set(values)) != len(values):
                errors[field] = "duplicate values"
            elif len(values) > 5:
                errors[field] = "must contain at most 5 items"
            continue
        scalar = _scalar(value)
        if scalar is None or not scalar:
            errors[field] = "must be a non-empty string"
            continue
        allowed = CLOSED_VALUES.get(field)
        if allowed is not None and scalar not in allowed:
            errors[field] = f"not in closed vocabulary: {scalar!r}"
    return errors


def normalize_payload(payload: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    payload = expand_dotted_fields(payload, fields)
    result: dict[str, Any] = {}
    for field in fields:
        value = get_path(payload, field)
        if field in {"paralinguistic.nonverbal_vocalization", "environment.background_sound_events", "paralinguistic.emphasis.emphasized_text"}:
            normalized = (
                _event_list(value, closed_nonverbal=field.endswith("nonverbal_vocalization"))
                if field != "paralinguistic.emphasis.emphasized_text"
                else _string_list(value)
            )
            set_path(result, field, normalized or [])
        else:
            set_path(result, field, _scalar(value) or "unknown")
    return result
