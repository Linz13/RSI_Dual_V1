from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

from .constants import CLOSED_FIELDS, LIST_FIELDS, SYNTHESIZABLE_FIELDS
from .schema import get_path


def normalize_text(value: Any) -> str:
    return " ".join(re.findall(r"[\w']+", str(value).lower(), flags=re.UNICODE))


def levenshtein(a: list[str], b: list[str]) -> int:
    previous = list(range(len(b) + 1))
    for index, left in enumerate(a, 1):
        current = [index]
        for right_index, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[right_index] + 1, previous[right_index - 1] + (left != right)))
        previous = current
    return previous[-1]


def error_rate(reference: str, hypothesis: str, language: str = "English") -> float:
    if language == "Chinese":
        ref = [char for char in reference if not char.isspace()]
        hyp = [char for char in hypothesis if not char.isspace()]
    else:
        ref = normalize_text(reference).split()
        hyp = normalize_text(hypothesis).split()
    return min(1.0, levenshtein(ref, hyp) / max(len(ref), 1))


def chrf(reference: str, hypothesis: str, max_order: int = 3) -> float:
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    if not ref and not hyp:
        return 1.0
    scores = []
    for order in range(1, max_order + 1):
        ref_counts = Counter(ref[i:i + order] for i in range(max(len(ref) - order + 1, 0)))
        hyp_counts = Counter(hyp[i:i + order] for i in range(max(len(hyp) - order + 1, 0)))
        overlap = sum((ref_counts & hyp_counts).values())
        precision = overlap / max(sum(hyp_counts.values()), 1)
        recall = overlap / max(sum(ref_counts.values()), 1)
        scores.append(2 * precision * recall / max(precision + recall, 1e-12))
    return sum(scores) / len(scores)


def set_f1(reference: Iterable[Any], hypothesis: Iterable[Any]) -> float:
    ref = {normalize_text(item) for item in reference if normalize_text(item)}
    hyp = {normalize_text(item) for item in hypothesis if normalize_text(item)}
    if not ref and not hyp:
        return 1.0
    return 2 * len(ref & hyp) / max(len(ref) + len(hyp), 1)


def field_score(reference: Any, hypothesis: Any, field: str) -> float:
    if field in LIST_FIELDS:
        return set_f1(reference or [], hypothesis or [])
    if field in CLOSED_FIELDS:
        return float(reference == hypothesis)
    if field == "semantic_content.transcript":
        return 1.0 - error_rate(str(reference), str(hypothesis))
    return chrf(str(reference), str(hypothesis))


def caption_similarity(reference: dict[str, Any], hypothesis: dict[str, Any], fields: Iterable[str] | None = None) -> tuple[float, dict[str, float]]:
    selected = list(SYNTHESIZABLE_FIELDS if fields is None else fields)
    per_field = {
        field: field_score(get_path(reference, field), get_path(hypothesis, field), field)
        for field in selected
        if get_path(reference, field) is not None and get_path(hypothesis, field) is not None
    }
    return (sum(per_field.values()) / len(per_field) if per_field else 0.0), per_field

