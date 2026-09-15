from __future__ import annotations

import math
from typing import Any

from .constants import TRAJECTORY_VERSION


def _finite_vector(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value)


def caption_trajectory_errors(candidate: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    token_ids = candidate.get("sampled_token_ids")
    old = candidate.get("old_token_logprobs")
    ref = candidate.get("ref_token_logprobs")
    if candidate.get("trajectory_version") != TRAJECTORY_VERSION:
        errors.append("missing or unsupported trajectory_version")
    if not isinstance(token_ids, list) or not token_ids or not all(isinstance(item, int) for item in token_ids):
        errors.append("sampled_token_ids must be a non-empty integer list")
    if not _finite_vector(old):
        errors.append("old_token_logprobs must be a finite non-empty list")
    if not _finite_vector(ref):
        errors.append("ref_token_logprobs must be a finite non-empty list")
    if isinstance(token_ids, list) and isinstance(old, list) and len(token_ids) != len(old):
        errors.append("sampled token and old log-prob lengths differ")
    if isinstance(token_ids, list) and isinstance(ref, list) and len(token_ids) != len(ref):
        errors.append("sampled token and reference log-prob lengths differ")
    if isinstance(token_ids, list) and token_ids:
        if candidate.get("terminal_token_id") != token_ids[-1]:
            errors.append("terminal_token_id does not match the last sampled token")
        if candidate.get("generated_token_count") != len(token_ids):
            errors.append("generated_token_count does not match sampled tokens")
    if candidate.get("behavior_logprob_mode") != "generation_scores":
        errors.append("caption behavior_logprob_mode must be generation_scores")
    replay_error = candidate.get("behavior_replay_max_abs_error")
    if not isinstance(replay_error, (int, float)) or not math.isfinite(float(replay_error)):
        errors.append("missing finite behavior_replay_max_abs_error")
    elif float(replay_error) > 5e-4:
        errors.append(f"generation/replay log-prob mismatch: {replay_error}")
    eos_token_id = candidate.get("eos_token_id")
    if not isinstance(eos_token_id, int):
        errors.append("missing eos_token_id")
    eos_positions = (
        [index for index, token_id in enumerate(token_ids) if token_id == eos_token_id]
        if isinstance(token_ids, list) and isinstance(eos_token_id, int) else []
    )
    terminated = bool(candidate.get("terminated_by_eos", False))
    finish_reason = candidate.get("finish_reason")
    if finish_reason not in {"eos", "length"}:
        errors.append("finish_reason must be eos or length")
    if terminated:
        if finish_reason != "eos" or eos_positions != [len(token_ids) - 1]:
            errors.append("EOS-terminated trajectory must contain exactly one terminal EOS")
        if candidate.get("post_eos_token_count") != 0:
            errors.append("EOS-terminated trajectory contains tokens after EOS")
    elif eos_positions:
        errors.append("non-terminated trajectory contains EOS before the terminal position")
    return errors


def tts_trajectory_errors(candidate: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    codes = candidate.get("codec_codes")
    old_main = candidate.get("old_main_logprobs")
    ref_main = candidate.get("ref_main_logprobs")
    old_sub = candidate.get("old_sub_logprobs")
    ref_sub = candidate.get("ref_sub_logprobs")
    if candidate.get("trajectory_version") != TRAJECTORY_VERSION:
        errors.append("missing or unsupported trajectory_version")
    if not isinstance(codes, list) or not codes or not all(isinstance(frame, list) and len(frame) == 16 for frame in codes):
        errors.append("codec_codes must be a non-empty [T,16] list")
    length = len(codes) if isinstance(codes, list) else -1
    if not _finite_vector(old_main) or len(old_main) != length:
        errors.append("old_main_logprobs must be finite with T entries")
    if not _finite_vector(ref_main) or len(ref_main) != length:
        errors.append("ref_main_logprobs must be finite with T entries")
    for name, values in (("old_sub_logprobs", old_sub), ("ref_sub_logprobs", ref_sub)):
        if not isinstance(values, list) or len(values) != length or not all(_finite_vector(frame) and len(frame) == 15 for frame in values):
            errors.append(f"{name} must be finite with shape [T,15]")
    if candidate.get("behavior_logprob_mode") != "processed_generation_scores":
        errors.append("TTS behavior_logprob_mode must be processed_generation_scores")
    main_replay_error = candidate.get("main_behavior_replay_max_abs_error")
    if not isinstance(main_replay_error, (int, float)) or not math.isfinite(float(main_replay_error)):
        errors.append("missing finite main_behavior_replay_max_abs_error")
    elif float(main_replay_error) > 5e-4:
        errors.append(f"main generation/replay log-prob mismatch: {main_replay_error}")
    replay_error = candidate.get("subtalker_behavior_replay_max_abs_error")
    if not isinstance(replay_error, (int, float)) or not math.isfinite(float(replay_error)):
        errors.append("missing finite subtalker_behavior_replay_max_abs_error")
    elif float(replay_error) > 5e-4:
        errors.append(f"subtalker generation/replay log-prob mismatch: {replay_error}")
    return errors


def mark_trajectory(candidate: dict[str, Any], kind: str) -> dict[str, Any]:
    errors = caption_trajectory_errors(candidate) if kind == "caption" else tts_trajectory_errors(candidate)
    candidate["trajectory_kind"] = kind
    candidate["trajectory_valid"] = not errors
    candidate["trajectory_errors"] = errors
    if kind == "caption":
        candidate["completion_valid"] = (
            not errors
            and bool(candidate.get("terminated_by_eos"))
            and candidate.get("finish_reason") == "eos"
            and candidate.get("post_eos_token_count") == 0
        )
    return candidate

