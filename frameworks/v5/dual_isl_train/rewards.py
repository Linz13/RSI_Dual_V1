from __future__ import annotations

import json
import math
from copy import deepcopy
from statistics import fmean, pstdev
from typing import Any, Iterable

from .dual_space import reparse_synth_candidate, synth_validation_errors
from .constants import CYCLE_SFT_SELECTION


CALIBRATION_VERSION = 2
LOOPS = ("audio_only", "caption_only")
COMPONENTS = ("reconstruction", "counterfactual")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _unit(value: Any) -> float:
    return min(1.0, max(0.0, float(value))) if _finite(value) else 0.0


def _logmeanexp(values: list[float]) -> float:
    if not values:
        return float("nan")
    maximum = max(values)
    return maximum + math.log(fmean(math.exp(value - maximum) for value in values))


def _counterfactual_scores(candidate: dict[str, Any], key: str) -> list[float]:
    values = []
    for item in candidate.get(key, []):
        score = item.get("reconstruction") if isinstance(item, dict) else None
        if _finite(score):
            values.append(float(score))
    return values


def _json_object_parseable(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    raw = text.strip()
    start, end = raw.find("{"), raw.rfind("}")
    candidates = [raw]
    if start >= 0 and end > start:
        candidates.append(raw[start:end + 1])
    for candidate in candidates:
        try:
            return isinstance(json.loads(candidate), dict)
        except (json.JSONDecodeError, TypeError):
            continue
    return False


def schema_progress(candidate: dict[str, Any]) -> float:
    """A deterministic curriculum score used only while raw P_syn is invalid."""
    if bool(candidate.get("raw_schema_valid")):
        return 1.0
    if bool(candidate.get("normalized_schema_valid")) or candidate.get("caption") is not None:
        repairs = len(candidate.get("normalization_rules") or [])
        return 2.0 / 3.0 + 1.0 / (3.0 * (1.0 + max(1, repairs)))
    if _json_object_parseable(candidate.get("raw_text")):
        return 1.0 / 3.0
    return 0.0


def _margin(positive: Any, negatives: list[float]) -> float:
    if not _finite(positive) or not negatives:
        return float("nan")
    return float(positive) - _logmeanexp(negatives)


def _audio_only_components(
    candidate: dict[str, Any], quality: dict[str, Any],
) -> tuple[bool, dict[str, float]]:
    reparsed = reparse_synth_candidate(candidate) or {}
    caption = reparsed.get("caption")
    normalized_valid = bool(caption is not None and not synth_validation_errors(caption))
    reconstruction = reparsed.get("tts_target_logprob")
    anchor_reconstruction = reparsed.get("anchor_tts_target_logprob")
    counterfactuals = _counterfactual_scores(reparsed, "counterfactual_reconstruction")
    anchor_counterfactuals = _counterfactual_scores(
        reparsed, "anchor_counterfactual_reconstruction"
    )
    transcript_quality = _unit(reparsed.get("asr_score"))
    semantic_valid = (
        bool(reparsed.get("semantic_input_valid"))
        and normalized_valid
        and bool(reparsed.get("trajectory_valid"))
        and _finite(reconstruction)
        and _finite(anchor_reconstruction)
        and bool(counterfactuals)
        and bool(anchor_counterfactuals)
        and _finite(reparsed.get("asr_score"))
        and transcript_quality >= float(quality.get("min_asr_score", 0.0))
    )
    return semantic_valid, {
        "schema_progress": schema_progress(reparsed),
        "transcript_quality": transcript_quality,
        "reconstruction": float(reconstruction) if _finite(reconstruction) else float("nan"),
        "counterfactual": _margin(reconstruction, counterfactuals),
        "anchor_margin": _margin(anchor_reconstruction, anchor_counterfactuals),
    }


def _caption_only_components(
    candidate: dict[str, Any], quality: dict[str, Any],
) -> tuple[bool, dict[str, float]]:
    reconstruction = candidate.get(
        "caption_target_logprob_macro", candidate.get("caption_target_logprob")
    )
    anchor_reconstruction = candidate.get(
        "anchor_caption_target_logprob_macro", candidate.get("anchor_caption_target_logprob")
    )
    counterfactuals = _counterfactual_scores(candidate, "counterfactual_reconstruction")
    anchor_counterfactuals = _counterfactual_scores(
        candidate, "anchor_counterfactual_reconstruction"
    )
    health = _unit(candidate.get("audio_health"))
    transcript_quality = _unit(candidate.get("asr_score"))
    semantic_valid = (
        bool(str(candidate.get("audio_path", "")).strip())
        and bool(candidate.get("trajectory_valid"))
        and _finite(candidate.get("audio_health"))
        and health >= float(quality.get("min_audio_health", 0.0))
        and _finite(candidate.get("asr_score"))
        and transcript_quality >= float(quality.get("min_asr_score", 0.0))
        and _finite(reconstruction)
        and _finite(anchor_reconstruction)
        and bool(counterfactuals)
        and bool(anchor_counterfactuals)
    )
    return semantic_valid, {
        "audio_health": health,
        "transcript_quality": transcript_quality,
        "reconstruction": float(reconstruction) if _finite(reconstruction) else float("nan"),
        "counterfactual": _margin(reconstruction, counterfactuals),
        "anchor_margin": _margin(anchor_reconstruction, anchor_counterfactuals),
    }


def candidate_components(
    loop: str, candidate: dict[str, Any], quality: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, float]]:
    quality = quality or {}
    if loop == "audio_only":
        return _audio_only_components(candidate, quality)
    if loop == "caption_only":
        return _caption_only_components(candidate, quality)
    raise ValueError(f"Unknown loop: {loop}")


def _iter_valid(
    groups: Iterable[dict[str, Any]], loop: str, quality: dict[str, Any],
) -> Iterable[dict[str, float]]:
    for group in groups:
        for candidate in group.get("candidates", []):
            semantic_valid, components = candidate_components(loop, candidate, quality)
            if semantic_valid:
                yield components


def _statistics(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise RuntimeError("Cannot calibrate a reward component without valid candidates")
    std = pstdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": fmean(values),
        "std": 0.0 if std < 1.0e-12 else std,
    }


def fit_round0_calibration(
    audio_groups: list[dict[str, Any]], caption_groups: list[dict[str, Any]],
    *, quality: dict[str, Any] | None = None,
    fallback_candidates: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    quality = quality or {}
    fallback_candidates = fallback_candidates or {}
    result: dict[str, Any] = {
        "version": CALIBRATION_VERSION,
        "method": "round0_dual_counterfactual_zscore",
        "fitted_round": 0,
        "frozen_across_rounds": True,
        "loops": {},
        "sources": {},
    }
    for loop, groups in (("audio_only", audio_groups), ("caption_only", caption_groups)):
        loop_quality = quality.get(loop, quality) if isinstance(quality.get(loop), dict) else quality
        valid = list(_iter_valid(groups, loop, loop_quality))
        source = "round0_generated_candidates"
        if not valid:
            valid = []
            for candidate in fallback_candidates.get(loop, []):
                semantic_valid, components = candidate_components(loop, candidate, loop_quality)
                if semantic_valid:
                    valid.append(components)
            source = "paired_counterfactual_calibration_fallback"
        result["loops"][loop] = {
            component: _statistics([float(item[component]) for item in valid])
            for component in COMPONENTS
        }
        result["sources"][loop] = {
            "kind": source,
            "semantic_valid_candidates": len(valid),
        }
    return result


def validate_calibration(calibration: dict[str, Any]) -> None:
    if calibration.get("version") != CALIBRATION_VERSION:
        raise ValueError("Unsupported reward calibration version")
    if calibration.get("method") != "round0_dual_counterfactual_zscore":
        raise ValueError("Unsupported reward calibration method")
    for loop in (("caption_only",) if calibration.get("v5_scope") == "caption_only" else LOOPS):
        for component in COMPONENTS:
            stats = calibration.get("loops", {}).get(loop, {}).get(component, {})
            if (
                int(stats.get("count", 0)) < 1
                or not _finite(stats.get("mean"))
                or not _finite(stats.get("std"))
            ):
                raise ValueError(f"Invalid calibration statistics for {loop}.{component}")
            if float(stats["std"]) < 0.0:
                raise ValueError(f"Negative calibration std for {loop}.{component}")


def standardize(value: float, stats: dict[str, Any]) -> float:
    std = float(stats["std"])
    return 0.0 if std < 1.0e-12 else (float(value) - float(stats["mean"])) / std


def semantic_reward(
    raw: dict[str, float], *, loop_stats: dict[str, Any], reconstruction_weight: float,
    counterfactual_weight: float, anchor_penalty_weight: float,
    anchor_tolerance_z: float,
) -> tuple[float, dict[str, float]]:
    normalized = {
        component: standardize(raw[component], loop_stats[component])
        for component in COMPONENTS
    }
    counterfactual_std = float(loop_stats["counterfactual"]["std"])
    anchor_margin_z = (
        float(raw["anchor_margin"]) / counterfactual_std
        if counterfactual_std >= 1.0e-12 else 0.0
    )
    anchor_penalty = float(anchor_penalty_weight) * max(
        0.0, -anchor_margin_z - float(anchor_tolerance_z)
    )
    score = (
        float(reconstruction_weight) * normalized["reconstruction"]
        + float(counterfactual_weight) * normalized["counterfactual"]
        - anchor_penalty
    )
    return score, {
        **normalized,
        "anchor_margin_z": anchor_margin_z,
        "anchor_penalty": anchor_penalty,
    }


def score_calibration_examples(
    examples: list[dict[str, Any]], *, loop: str, calibration: dict[str, Any],
    reward_config: dict[str, Any], quality: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    validate_calibration(calibration)
    quality = quality or {}
    output = []
    for example in examples:
        valid, raw = candidate_components(loop, example["candidate"], quality)
        if not valid:
            continue
        score, normalized = semantic_reward(
            raw, loop_stats=calibration["loops"][loop],
            reconstruction_weight=float(reward_config["reconstruction_weight"]),
            counterfactual_weight=float(reward_config["counterfactual_weight"]),
            anchor_penalty_weight=float(reward_config["anchor_penalty_weight"]),
            anchor_tolerance_z=float(reward_config["anchor_tolerance_z"]),
        )
        output.append({
            "id": example["id"], "label": int(example["label"]), "score": score,
            "reward_components_raw": raw, "reward_components_normalized": normalized,
        })
    return output


def fit_sft_threshold(
    examples: list[dict[str, Any]], *, target_precision: float, min_recall: float,
) -> dict[str, Any]:
    positives = [item for item in examples if int(item["label"]) == 1 and _finite(item["score"])]
    negatives = [item for item in examples if int(item["label"]) == 0 and _finite(item["score"])]
    if not positives or not negatives:
        raise RuntimeError("SFT threshold calibration requires positive and negative examples")
    operating_points = []
    for threshold in sorted({float(item["score"]) for item in examples}):
        tp = sum(float(item["score"]) >= threshold for item in positives)
        fp = sum(float(item["score"]) >= threshold for item in negatives)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / len(positives)
        if precision >= float(target_precision) and recall >= float(min_recall):
            operating_points.append((recall, precision, -threshold, threshold, tp, fp))
    if not operating_points:
        return {
            "threshold": None,
            "status": "no_high_precision_operating_point",
            "target_precision": float(target_precision),
            "min_recall": float(min_recall),
            "positive_count": len(positives),
            "negative_count": len(negatives),
        }
    recall, precision, _negative_threshold, threshold, tp, fp = max(operating_points)
    return {
        "threshold": threshold,
        "status": "calibrated",
        "target_precision": float(target_precision),
        "min_recall": float(min_recall),
        "empirical_precision": precision,
        "empirical_recall": recall,
        "accepted_positives": tp,
        "accepted_negatives": fp,
        "positive_count": len(positives),
        "negative_count": len(negatives),
    }


def score_groups(
    groups: list[dict[str, Any]], *, loop: str, calibration: dict[str, Any],
    reward_config: dict[str, Any], sft_threshold: float | None,
    quality: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Keep V3 rewards/GRPO; sft_threshold is a diagnostic, never a V4 gate."""
    validate_calibration(calibration)
    quality = quality or {}
    loop_stats = calibration["loops"][loop]
    output: list[dict[str, Any]] = []
    for group in groups:
        candidates: list[dict[str, Any]] = []
        for source in group.get("candidates", []):
            candidate = reparse_synth_candidate(source) if loop == "audio_only" else deepcopy(source)
            candidate = candidate or deepcopy(source)
            valid, raw = candidate_components(loop, candidate, quality)
            normalized: dict[str, float] = {}
            reward = None
            if valid:
                reward, normalized = semantic_reward(
                    raw, loop_stats=loop_stats,
                    reconstruction_weight=float(reward_config["reconstruction_weight"]),
                    counterfactual_weight=float(reward_config["counterfactual_weight"]),
                    anchor_penalty_weight=float(reward_config["anchor_penalty_weight"]),
                    anchor_tolerance_z=float(reward_config["anchor_tolerance_z"]),
                )
            candidate.update({
                "structural_valid": valid,
                "semantic_valid": valid,
                "schema_progress": raw.get("schema_progress"),
                "reward_components_raw": raw,
                "reward_components_normalized": normalized,
                "semantic_reward": reward,
                "reward": reward,
                "sft_selected": False,
                "sft_confidence": reward,
                "sft_rejection_reason": "not_top1" if valid else "semantic_invalid",
                "sft_gate_would_select": False,
                "sft_gate_rejection_reason": "not_top1" if valid else "semantic_invalid",
                "skip_update": True,
                "grpo_mode": None,
            })
            candidates.append(candidate)

        semantic = [candidate for candidate in candidates if candidate["semantic_valid"]]
        active: list[dict[str, Any]] = []
        if len(semantic) >= 2:
            active = semantic
            for candidate in active:
                candidate["grpo_mode"] = "dual_semantic"
        elif loop == "audio_only":
            format_candidates = [
                candidate for candidate in candidates
                if bool(candidate.get("trajectory_valid"))
                and _finite(candidate.get("schema_progress"))
            ]
            if len(format_candidates) >= 2:
                active = format_candidates
                for candidate in active:
                    candidate["reward"] = float(candidate["schema_progress"])
                    candidate["grpo_mode"] = "schema_curriculum"

        if active:
            values = [float(item["reward"]) for item in active]
            mean = fmean(values)
            std = pstdev(values)
            for candidate in active:
                candidate["advantage"] = (
                    0.0 if std == 0.0 else (float(candidate["reward"]) - mean) / std
                )
                candidate["skip_update"] = False
        for candidate in candidates:
            if "advantage" not in candidate:
                candidate["advantage"] = 0.0

        top1_gap = None
        if semantic:
            ordered = sorted(
                semantic,
                key=lambda item: (float(item["semantic_reward"]), str(item["candidate_id"])),
                reverse=True,
            )
            winner = ordered[0]
            if len(ordered) > 1:
                top1_gap = float(ordered[0]["semantic_reward"]) - float(
                    ordered[1]["semantic_reward"]
                )
            if sft_threshold is None:
                winner["sft_gate_rejection_reason"] = "threshold_not_calibrated"
            elif float(winner["sft_confidence"]) >= float(sft_threshold):
                winner["sft_gate_would_select"] = True
                winner["sft_gate_rejection_reason"] = None
            else:
                winner["sft_gate_rejection_reason"] = "below_absolute_confidence_threshold"
            winner["sft_selected"] = True
            winner["sft_rejection_reason"] = None
        output.append({
            **group, "candidates": candidates,
            "cycle_sft_selection": CYCLE_SFT_SELECTION,
            "sft_gate_diagnostic_only": True,
            "sft_diagnostic_threshold": sft_threshold,
            "sft_top1_gap": top1_gap,
            "sft_top1_gap_is_hard_gate": False,
        })
    return output


def reward_summary(groups: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [candidate for group in groups for candidate in group.get("candidates", [])]
    return {
        "cycle_sft_selection": CYCLE_SFT_SELECTION,
        "sft_gate_diagnostic_only": True,
        "sft_eligible_groups": sum(
            any(item.get("semantic_valid") for item in group.get("candidates", []))
            for group in groups
        ),
        "sft_gate_would_select": sum(bool(item.get("sft_gate_would_select")) for item in candidates),
        "sft_selected_despite_gate": sum(
            bool(item.get("sft_selected")) and not bool(item.get("sft_gate_would_select"))
            for item in candidates
        ),
        **({
            "raw_schema_valid_candidates": sum(bool(item.get("raw_schema_valid")) for item in candidates),
            "semantic_input_valid_candidates": sum(bool(item.get("semantic_input_valid")) for item in candidates),
            "safely_repaired_candidates": sum(
                bool(item.get("semantic_input_valid")) and not bool(item.get("raw_schema_valid"))
                for item in candidates
            ),
        } if any("raw_text" in item for item in candidates) else {}),
        "semantic_nonzero_advantage_groups": sum(
            any(item.get("grpo_mode") == "dual_semantic" and item.get("advantage", 0.0) != 0.0
                for item in group.get("candidates", []))
            for group in groups
        ),
        "groups": len(groups),
        "candidates": len(candidates),
        "structurally_valid_candidates": sum(bool(item.get("structural_valid")) for item in candidates),
        "grpo_usable_groups": sum(
            sum(not item.get("skip_update", True) for item in group.get("candidates", [])) >= 2
            for group in groups
        ),
        "semantic_grpo_groups": sum(
            any(item.get("grpo_mode") == "dual_semantic" for item in group.get("candidates", []))
            for group in groups
        ),
        "schema_curriculum_groups": sum(
            any(item.get("grpo_mode") == "schema_curriculum" for item in group.get("candidates", []))
            for group in groups
        ),
        "sft_selected": sum(bool(item.get("sft_selected")) for item in candidates),
        "sft_rejected_groups": sum(
            bool(group.get("candidates"))
            and not any(item.get("sft_selected") for item in group.get("candidates", []))
            for group in groups
        ),
    }
