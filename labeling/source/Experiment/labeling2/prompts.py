from __future__ import annotations

import json
from typing import Iterable

from .target_schema import CLOSED_VALUES, EVENT_VALUES

FIELD_DESCRIPTIONS = {
    "semantic_content.language": "language of the spoken content",
    "semantic_content.topic": "short topic noun phrase; use unknown only when inaudible",
    "semantic_content.intent": "communicative intent of the main speaker",
    "speaker_profile.gender": "perceived gender of the main speaker",
    "speaker_profile.age": "perceived age group of the main speaker",
    "speaker_profile.timbre": "dominant vocal timbre",
    "paralinguistic.pitch_level": "overall perceived pitch level",
    "paralinguistic.emotion_intensity": "intensity of the expressed emotion",
    "paralinguistic.emphasis.level": "strength of prosodic emphasis",
    "paralinguistic.emphasis.emphasized_text": "up to 5 distinct exact emphasized words or phrases; never repeat an item; use [] if none or unknown",
    "paralinguistic.prosody": "concise rhythm, intonation and phrasing description, without repeating rate, pitch, volume, emphasis or pause",
    "paralinguistic.pause": "concise pause frequency, duration, placement and function description",
    "paralinguistic.nonverbal_vocalization": "all salient nonverbal vocal events as a list",
    "environment.background_sound_events": "all salient background sound events as a concise string list",
    "environment.recording_quality": "overall recording quality",
    "environment.acoustic_scene": "specific acoustic scene or unknown",
}


def _path_set(fields: Iterable[str]) -> dict:
    root: dict = {}
    for field in fields:
        current = root
        parts = field.split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = FIELD_DESCRIPTIONS.get(field, field)
    return root


def build_prompt(model_name: str, fields: list[str], *, repair_errors: dict[str, str] | None = None) -> str:
    closed = {field: sorted(CLOSED_VALUES[field]) for field in fields if field in CLOSED_VALUES}
    if "paralinguistic.nonverbal_vocalization" in fields:
        closed["paralinguistic.nonverbal_vocalization"] = sorted(EVENT_VALUES)
    instructions = [
        "Listen to the audio and annotate only the requested fields.",
        "Return one valid JSON object and nothing else. Do not return markdown or explanations.",
        "The output must contain exactly the requested nested paths and no Target_JSON_Schema wrapper is required.",
        "Use the main/most prominent speaker when more than one speaker is present.",
        "Closed-vocabulary fields must use exactly one listed string.",
        "For event-list fields, return a non-empty list; use [\"none\"] or [\"unknown\"] exclusively.",
        "Do not use metadata or a supplied transcript; judge the audio itself.",
        f"Model backend: {model_name}.",
        "Requested field descriptions:\n" + json.dumps(_path_set(fields), ensure_ascii=False, indent=2),
    ]
    if closed:
        instructions.append("Allowed closed values:\n" + json.dumps(closed, ensure_ascii=False, indent=2))
    if model_name == "kimi_audio":
        instructions.append(
            "Kimi structured-output constraints: keep the entire response under 200 text tokens. "
            "Do not transcribe or retell the audio. emphasized_text must contain at most 5 distinct items. "
            "nonverbal_vocalization must contain distinct event labels only, with each event type at most once. "
            "emphasis.level describes prosodic emphasis only and MUST be exactly one of: "
            "none, emphasized, heavily emphasized, unknown. Words such as sobbing, crying, screaming, "
            "laughter, cough, or sigh are NEVER emphasis levels; put such events only in "
            "nonverbal_vocalization. If emphasis cannot be judged, use unknown. "
            "Before answering, silently remove duplicate emphasized_text items and keep at most the first 5. "
            "Never repeat words, phrases, list items, or JSON blocks. Stop immediately after the final closing brace."
        )
    if repair_errors:
        instructions.append(
            "Repair the invalid field paths listed below and return the complete requested object. "
            "Use nested JSON objects exactly as shown in Requested field descriptions; never use dotted keys as output keys.\n"
            + json.dumps(repair_errors, ensure_ascii=False)
        )
    return "\n\n".join(instructions)
