#!/usr/bin/env python3
"""Prompt and parsing helpers for the independent-field Attr6 Scheme A protocol."""

from __future__ import annotations

import re
from typing import Any

from common import MULTI_FIELDS, SINGLE_FIELDS, load_schema


PROTOCOL = "paraspeechcaps-attr6-content-scheme-a-v1"
FIELD_PREFIX = {
    "gender": "G",
    "pitch": "P",
    "speaking_rate": "R",
    "accent": "A",
    "intrinsic_traits": "I",
    "situational_traits": "S",
}
FIELD_TITLES = {
    "gender": "the speaker's perceived gender",
    "pitch": "the speaker's perceived pitch",
    "speaking_rate": "the speaker's speaking rate",
    "accent": "the speaker's perceived accent",
    "intrinsic_traits": "the speaker's intrinsic voice traits",
    "situational_traits": "the speaker's situational or expressive traits",
}


def option_table(field: str, schema: dict[str, Any] | None = None) -> dict[str, str]:
    schema = schema or load_schema()
    prefix = FIELD_PREFIX[field]
    if field in SINGLE_FIELDS:
        values = schema["single_choice"][field]
        table = {f"{prefix}{index}": value for index, value in enumerate(values, 1)}
        table[f"{prefix}0"] = "unknown"
        return table
    if field in MULTI_FIELDS:
        values = schema["multi_choice"][field]
        return {f"{prefix}{index:02d}": value for index, value in enumerate(values, 1)}
    raise ValueError(f"unsupported Scheme A field: {field}")


def build_field_prompt(field: str, schema: dict[str, Any] | None = None) -> str:
    schema = schema or load_schema()
    table = option_table(field, schema)
    common = (
        "Listen carefully to the speech audio and answer one closed-set speech-attribute "
        "question. Judge only observable acoustic properties. Do not infer the speaker's "
        "identity, and do not use the meaning of the spoken words as evidence.\n\n"
    )
    if field in SINGLE_FIELDS:
        option_lines = "\n".join(f"- {key}: {value}" for key, value in table.items())
        return (
            common
            + f"Question: Which option best describes {FIELD_TITLES[field]}?\n\n"
            + option_lines
            + "\n\nChoose exactly one option. Use the unknown option only when the audio does "
            "not provide enough evidence. Return only the option ID, with no explanation."
        )
    option_lines = "\n".join(f"- {key}: {value}" for key, value in table.items())
    return (
        common
        + f"Question: Which options clearly describe {FIELD_TITLES[field]}?\n\n"
        + option_lines
        + "\n\nSelect zero or more options; do not guess. Return only comma-separated option "
        f"IDs, for example {next(iter(table))}. If no option is clearly supported, return NONE."
    )


def build_all_field_prompts(schema: dict[str, Any] | None = None) -> dict[str, str]:
    schema = schema or load_schema()
    return {
        field: build_field_prompt(field, schema)
        for field in SINGLE_FIELDS + MULTI_FIELDS
    }


def _labels_mentioned(raw: str, labels: list[str]) -> list[str]:
    """Find canonical labels in free text without substring collisions."""
    text = str(raw or "").casefold().replace("_", "-")
    found: list[str] = []
    for label in sorted(labels, key=len, reverse=True):
        pieces = [re.escape(piece) for piece in re.split(r"[-\s]+", label.casefold())]
        pattern = r"(?<![a-z0-9-])" + r"[-\s]+".join(pieces) + r"(?![a-z0-9-])"
        if re.search(pattern, text):
            found.append(label)
    order = {label: index for index, label in enumerate(labels)}
    return sorted(set(found), key=order.__getitem__)


def parse_field_response(
    field: str,
    raw: str,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse option IDs first, then exact canonical labels as a deterministic fallback."""
    schema = schema or load_schema()
    table = option_table(field, schema)
    prefix = FIELD_PREFIX[field]
    text = str(raw or "")
    normalized_table = {key.upper(): value for key, value in table.items()}
    mentioned_ids = [
        f"{prefix}{int(number):02d}" if field in MULTI_FIELDS else f"{prefix}{int(number)}"
        for number in re.findall(
            rf"(?<![A-Za-z0-9]){re.escape(prefix)}\s*0*([0-9]+)(?![A-Za-z0-9])",
            text,
            flags=re.IGNORECASE,
        )
    ]
    valid_ids = sorted(
        {key for key in mentioned_ids if key.upper() in normalized_table},
        key=lambda key: list(normalized_table).index(key.upper()),
    )
    invalid_ids = sorted({key for key in mentioned_ids if key.upper() not in normalized_table})

    if field in SINGLE_FIELDS:
        if len(valid_ids) == 1:
            return {
                "status": "success",
                "prediction": normalized_table[valid_ids[0].upper()],
                "parse_mode": "option_id",
                "recognized_ids": valid_ids,
                "ignored_invalid_ids": invalid_ids,
            }
        if len(valid_ids) > 1:
            return {
                "status": "unparsed",
                "prediction": None,
                "parse_mode": "multiple_options",
                "recognized_ids": valid_ids,
                "ignored_invalid_ids": invalid_ids,
                "mentioned_labels": [],
            }
        labels = list(schema["single_choice"][field]) + ["unknown"]
        mentioned_labels = _labels_mentioned(text, labels)
        if len(mentioned_labels) == 1:
            return {
                "status": "success",
                "prediction": mentioned_labels[0],
                "parse_mode": "canonical_label_fallback",
                "recognized_ids": valid_ids,
                "ignored_invalid_ids": invalid_ids,
            }
        reason = "multiple_options" if len(mentioned_labels) > 1 else "no_option"
        return {
            "status": "unparsed",
            "prediction": None,
            "parse_mode": reason,
            "recognized_ids": valid_ids,
            "ignored_invalid_ids": invalid_ids,
            "mentioned_labels": mentioned_labels,
        }

    if valid_ids:
        return {
            "status": "success",
            "prediction": [normalized_table[key.upper()] for key in valid_ids],
            "parse_mode": "option_ids",
            "recognized_ids": valid_ids,
            "ignored_invalid_ids": invalid_ids,
        }
    labels = list(schema["multi_choice"][field])
    mentioned_labels = _labels_mentioned(text, labels)
    if mentioned_labels:
        return {
            "status": "success",
            "prediction": mentioned_labels,
            "parse_mode": "canonical_label_fallback",
            "recognized_ids": [],
            "ignored_invalid_ids": invalid_ids,
        }
    explicit_none = bool(
        re.search(
            r"\bnone\b|\bno\s+(?:clearly\s+supported\s+)?(?:traits|options|labels)\b|\[\s*\]",
            text,
            flags=re.IGNORECASE,
        )
    )
    if explicit_none:
        return {
            "status": "success",
            "prediction": [],
            "parse_mode": "explicit_none",
            "recognized_ids": [],
            "ignored_invalid_ids": invalid_ids,
        }
    return {
        "status": "unparsed",
        "prediction": None,
        "parse_mode": "no_option",
        "recognized_ids": [],
        "ignored_invalid_ids": invalid_ids,
    }
