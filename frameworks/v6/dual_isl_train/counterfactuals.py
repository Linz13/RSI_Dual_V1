from __future__ import annotations

from copy import deepcopy
from typing import Any

from .dual_space import canonical_synth_caption
from .io import stable_hash
from .schema import get_path, set_path


# Only fields with a deterministic, schema-valid alternative participate in
# the first implementation.  Transcript/language stay fixed so the margin
# measures speaker/paralinguistic specificity rather than content changes.
COUNTERFACTUAL_VALUES: dict[str, tuple[str, ...]] = {
    "speaker_profile.gender": ("male", "female"),
    "speaker_profile.age": ("child", "teenager", "adult", "elderly"),
    "speaker_profile.timbre": (
        "neutral", "breathy", "rough", "creaky", "strained", "nasal", "bright", "dark",
    ),
    "speaker_profile.accent": (
        "US English", "England English", "Australian English", "African English",
        "Indian English", "Mandarin", "Southwestern Mandarin", "Cantonese",
    ),
    "paralinguistic.speaking_rate": ("slow", "moderate", "fast"),
    "paralinguistic.pitch_level": ("low", "medium", "high"),
    "paralinguistic.volume_level": ("low", "medium", "high"),
    "paralinguistic.emotion": (
        "neutral", "happy", "sad", "angry", "fearful", "surprised", "disgusted",
    ),
    "paralinguistic.emotion_intensity": ("none", "low", "medium", "high"),
    "paralinguistic.emphasis.level": ("none", "emphasized", "heavily emphasized"),
}


def _ordered_alternatives(caption: dict[str, Any], field: str, key: str) -> list[str]:
    current = get_path(caption, field)
    alternatives = [value for value in COUNTERFACTUAL_VALUES[field] if value != current]
    if field == "paralinguistic.emotion" and get_path(
        caption, "paralinguistic.emotion_intensity"
    ) in {"medium", "high"}:
        alternatives = [value for value in alternatives if value != "neutral"]
    if field == "paralinguistic.emotion_intensity" and get_path(
        caption, "paralinguistic.emotion"
    ) == "neutral":
        alternatives = [value for value in alternatives if value in {"none", "low"}]
    return sorted(alternatives, key=lambda value: stable_hash({"key": key, "value": value}))


def counterfactual_captions(
    caption: dict[str, Any], *, key: str, max_count: int,
) -> list[dict[str, Any]]:
    """Return deterministic one-field counterfactuals for a valid P_syn.

    Open-text/list fields are intentionally left unchanged in v1.  The selected
    closed fields rotate deterministically across candidates so the extra
    reverse-model scoring cost is bounded while coverage accumulates over data.
    """
    if max_count < 1:
        return []
    source = canonical_synth_caption(caption)
    eligible = [
        field for field in COUNTERFACTUAL_VALUES
        if get_path(source, field) not in {"unknown", "other"}
        and _ordered_alternatives(source, field, f"{key}:{field}")
    ]
    eligible.sort(key=lambda field: stable_hash({"key": key, "field": field}))
    output: list[dict[str, Any]] = []
    for field in eligible[:max_count]:
        current = get_path(source, field)
        alternative = _ordered_alternatives(source, field, f"{key}:{field}")[0]
        changed = deepcopy(source)
        set_path(changed, field, alternative)
        if field == "paralinguistic.emphasis.level" and alternative == "none":
            set_path(changed, "paralinguistic.emphasis.emphasized_text", [])
        try:
            changed = canonical_synth_caption(changed)
        except ValueError:
            continue
        output.append({
            "field": field,
            "original_value": current,
            "alternative_value": alternative,
            "caption": changed,
        })
    return output
