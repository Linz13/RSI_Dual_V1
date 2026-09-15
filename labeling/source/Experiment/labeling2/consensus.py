from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VoteResult:
    value: Any
    passed: bool
    required: int
    valid_votes: int
    counts: dict[str, int]
    reason: str


def required_votes(model_count: int) -> int:
    if model_count <= 0:
        return 0
    return model_count if model_count == 2 else model_count - 1


def scalar_vote(values: list[Any], *, fallback: Any = "unknown") -> VoteResult:
    n = len(values)
    required = required_votes(n)
    normalized = [str(value).strip() for value in values if isinstance(value, str) and value.strip()]
    counts = Counter(normalized)
    if not counts:
        return VoteResult(fallback, False, required, 0, {}, "no_valid_votes")
    value, count = counts.most_common(1)[0]
    passed = count >= required and required > 0
    return VoteResult(value if passed else fallback, passed, required, len(normalized), dict(counts), "consensus" if passed else "conflict")


def list_vote(values: list[list[str] | str], *, fallback: list[str] | None = None) -> VoteResult:
    """Vote independently for each event; n-1 models must support an event."""
    required = required_votes(len(values))
    normalized: list[list[str]] = []
    for value in values:
        if isinstance(value, str):
            value = [value]
        if isinstance(value, list):
            normalized.append(sorted(set(str(item).strip() for item in value if str(item).strip())))
    counts: Counter[str] = Counter(item for row in normalized for item in row)
    if not normalized:
        return VoteResult(fallback or ["unknown"], False, required, 0, {}, "no_valid_votes")
    accepted = sorted(item for item, count in counts.items() if count >= required and item not in {"unknown"})
    if "none" in accepted:
        accepted = ["none"]
    if not accepted:
        unknown_count = counts.get("unknown", 0)
        if unknown_count >= required:
            accepted = ["unknown"]
    passed = bool(accepted) and (accepted != ["unknown"] or counts.get("unknown", 0) >= required)
    return VoteResult(accepted or (fallback or ["unknown"]), passed, required, len(normalized), dict(counts), "consensus" if passed else "conflict")

