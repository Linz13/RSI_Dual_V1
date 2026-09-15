#!/usr/bin/env python
"""Summarize frozen dual-loop artifacts without mutating the canonical run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from dual_isl_train.dual_space import SYNTHESIZABLE_FIELDS, reparse_synth_candidate, synth_validation_errors
from dual_isl_train.io import atomic_json, read_jsonl
from dual_isl_train.rewards import fit_round0_calibration, reward_summary, score_groups
from dual_isl_train.schema import get_path


def explicit_unknown(value) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() == "unknown"
    if isinstance(value, list):
        return bool(value) and all(explicit_unknown(item) for item in value)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--diagnostic", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.run_dir).resolve()
    audio_groups = list(read_jsonl(root / "round_000/collections/round_000_audio_collection.output.jsonl"))
    caption_groups = list(read_jsonl(root / "round_000/collections/round_000_caption_tts_rollout.output.jsonl"))
    critics = {
        row["candidate_id"]: row for row in read_jsonl(
            root / "round_000/collections/round_000_caption_audio_quality.output.jsonl"
        )
    }
    diagnostic = json.loads(Path(args.diagnostic).read_text(encoding="utf-8"))
    reconstruction = {row["candidate_id"]: row for row in diagnostic["rows"]}
    merged_caption = []
    for group in caption_groups:
        merged_caption.append({
            **group,
            "candidates": [
                {
                    **candidate,
                    **critics[candidate["candidate_id"]],
                    "caption_target_logprob": reconstruction[candidate["candidate_id"]]["caption_target_logprob_macro"],
                    "caption_target_logprob_macro": reconstruction[candidate["candidate_id"]]["caption_target_logprob_macro"],
                    "caption_target_value_token_count": reconstruction[candidate["candidate_id"]]["value_token_count"],
                    "diagnostic_order_only_workaround": True,
                }
                for candidate in group["candidates"]
            ],
        })

    audio_candidates = [candidate for group in audio_groups for candidate in group["candidates"]]
    caption_candidates = [candidate for group in merged_caption for candidate in group["candidates"]]
    normalized = [reparse_synth_candidate(candidate) or candidate for candidate in audio_candidates]
    field_counts = {}
    for field in SYNTHESIZABLE_FIELDS:
        values = [get_path(candidate["caption"], field) for candidate in normalized if candidate.get("caption")]
        field_counts[field] = {
            "unknown": sum(explicit_unknown(value) for value in values),
            "total": len(values),
            "ratio": sum(explicit_unknown(value) for value in values) / len(values) if values else None,
        }
    overall_unknown = sum(value["unknown"] for value in field_counts.values())
    overall_total = sum(value["total"] for value in field_counts.values())

    calibration = fit_round0_calibration(audio_groups, merged_caption)
    scored_audio = score_groups(audio_groups, loop="audio_only", calibration=calibration, alpha=0.5)
    scored_caption = score_groups(merged_caption, loop="caption_only", calibration=calibration, alpha=0.5)
    audio_valid = [
        candidate for group in scored_audio for candidate in group["candidates"]
        if candidate.get("structural_valid")
    ]
    caption_valid = [
        candidate for group in scored_caption for candidate in group["candidates"]
        if candidate.get("structural_valid")
    ]
    state = json.loads((root / "run_state.json").read_text(encoding="utf-8"))
    result = {
        "diagnostic_only": True,
        "canonical_smoke_status": "blocked_before_training",
        "canonical_failed_stage": "round_000_caption_caption_reconstruction",
        "canonical_failure": state["stages"]["round_000_caption_caption_reconstruction"].get("error"),
        "before_any_optimizer_step": True,
        "audio_to_caption_to_tts": {
            "groups": len(audio_groups),
            "candidates": len(audio_candidates),
            "raw_schema_valid": sum(bool(candidate.get("raw_schema_valid")) for candidate in normalized),
            "normalized_schema_valid": sum(
                bool(candidate.get("caption")) and not synth_validation_errors(candidate["caption"])
                for candidate in normalized
            ),
            "trajectory_valid": sum(bool(candidate.get("trajectory_valid")) for candidate in normalized),
            "reconstruction_finite": sum(math.isfinite(float(candidate.get("tts_target_logprob", float("nan")))) for candidate in normalized),
            "groups_with_at_least_2_valid": sum(
                sum(bool(candidate.get("structural_valid")) for candidate in group["candidates"]) >= 2
                for group in scored_audio
            ),
            "grpo_and_top1": reward_summary(scored_audio),
        },
        "caption_to_tts_to_caption": {
            "groups": len(merged_caption),
            "source_schema_valid": not synth_validation_errors(merged_caption[0]["source_caption"]),
            "candidates": len(caption_candidates),
            "trajectory_valid": sum(bool(candidate.get("trajectory_valid")) for candidate in caption_candidates),
            "reconstruction_finite": sum(
                math.isfinite(float(candidate.get("caption_target_logprob_macro", float("nan"))))
                for candidate in caption_candidates
            ),
            "groups_with_at_least_2_valid": sum(
                sum(bool(candidate.get("structural_valid")) for candidate in group["candidates"]) >= 2
                for group in scored_caption
            ),
            "grpo_and_top1": reward_summary(scored_caption),
            "reconstruction_is_order_only_diagnostic": True,
        },
        "psyn_unknown": {
            "explicit_unknown": overall_unknown,
            "field_values": overall_total,
            "overall_ratio": overall_unknown / overall_total if overall_total else None,
            "per_field": field_counts,
        },
        "diagnostic_calibration": calibration,
        "diagnostic_valid_candidate_counts": {
            "audio_only": len(audio_valid), "caption_only": len(caption_valid),
        },
    }
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
