from __future__ import annotations

from copy import deepcopy

from dual_isl_train.counterfactuals import counterfactual_captions
from dual_isl_train.dual_space import project_synth_caption, synth_caption_json
from dual_isl_train.rewards import (
    fit_round0_calibration, fit_sft_threshold, score_groups, semantic_reward,
)


REWARD_CONFIG = {
    "reconstruction_weight": 0.5,
    "counterfactual_weight": 0.5,
    "anchor_penalty_weight": 0.25,
    "anchor_tolerance_z": 0.5,
}


def _audio_groups(caption):
    projected = project_synth_caption(caption)
    raw = synth_caption_json(projected)
    candidates = []
    for index in range(4):
        reconstruction = -0.8 + index * 0.1
        candidates.append({
            "candidate_id": f"audio::{index}", "caption": deepcopy(projected),
            "raw_text": raw, "raw_schema_valid": True, "normalized_schema_valid": True,
            "trajectory_valid": True, "asr_score": 0.70 + index * 0.05,
            "tts_target_logprob": reconstruction,
            "anchor_tts_target_logprob": reconstruction,
            "counterfactual_reconstruction": [{
                "reconstruction": reconstruction - 0.1 - index * 0.03,
            }],
            "anchor_counterfactual_reconstruction": [{
                "reconstruction": reconstruction - 0.1 - index * 0.03,
            }],
        })
    return [{"id": "audio", "candidates": candidates}]


def _caption_groups():
    candidates = []
    for index in range(4):
        reconstruction = -0.7 + index * 0.08
        candidates.append({
            "candidate_id": f"caption::{index}", "audio_path": f"{index}.wav",
            "trajectory_valid": True, "audio_health": 0.65 + index * 0.05,
            "asr_score": 0.7 + index * 0.04,
            "caption_target_logprob_macro": reconstruction,
            "anchor_caption_target_logprob_macro": reconstruction,
            "counterfactual_reconstruction": [{
                "reconstruction": reconstruction - 0.08 - index * 0.02,
            }],
            "anchor_counterfactual_reconstruction": [{
                "reconstruction": reconstruction - 0.08 - index * 0.02,
            }],
        })
    return [{"id": "caption", "candidates": candidates}]


def test_reward_keeps_reconstruction_counterfactual_and_diagnostic_gate(sample_caption):
    audio = _audio_groups(sample_caption)
    caption = _caption_groups()
    calibration = fit_round0_calibration(audio, caption)
    assert calibration["version"] == 2
    assert calibration["method"] == "round0_dual_counterfactual_zscore"
    scored = score_groups(
        audio, loop="audio_only", calibration=calibration,
        reward_config=REWARD_CONFIG, sft_threshold=-100.0,
    )
    candidates = scored[0]["candidates"]
    assert sum(item["sft_selected"] for item in candidates) == 1
    assert all(not item["skip_update"] for item in candidates)
    assert all("reconstruction" in item["reward_components_raw"] for item in candidates)
    assert all("counterfactual" in item["reward_components_raw"] for item in candidates)
    assert max(candidates, key=lambda item: item["semantic_reward"])["sft_selected"] is True

    high_threshold = score_groups(
        audio, loop="audio_only", calibration=calibration,
        reward_config=REWARD_CONFIG, sft_threshold=100.0,
    )[0]["candidates"]
    assert sum(item["sft_selected"] for item in high_threshold) == 1
    assert not any(item["sft_gate_would_select"] for item in high_threshold)
    assert [item["advantage"] for item in high_threshold] == [item["advantage"] for item in candidates]


def test_equal_semantic_rewards_produce_zero_advantages(sample_caption):
    groups = _audio_groups(sample_caption)
    for candidate in groups[0]["candidates"]:
        candidate["asr_score"] = 0.9
        candidate["tts_target_logprob"] = -0.5
        candidate["anchor_tts_target_logprob"] = -0.5
        candidate["counterfactual_reconstruction"] = [{"reconstruction": -0.7}]
        candidate["anchor_counterfactual_reconstruction"] = [{"reconstruction": -0.7}]
    calibration = {
        "version": 2, "method": "round0_dual_counterfactual_zscore", "fitted_round": 0,
        "frozen_across_rounds": True,
        "loops": {
            loop: {
                "reconstruction": {"count": 4, "mean": 0.0, "std": 1.0},
                "counterfactual": {"count": 4, "mean": 0.0, "std": 1.0},
            } for loop in ("audio_only", "caption_only")
        },
    }
    candidates = score_groups(
        groups, loop="audio_only", calibration=calibration,
        reward_config=REWARD_CONFIG, sft_threshold=-100.0,
    )[0]["candidates"]
    assert all(item["advantage"] == 0.0 for item in candidates)
    assert all(not item["skip_update"] for item in candidates)


def test_invalid_json_uses_schema_curriculum_not_semantic_reward(sample_caption):
    groups = _audio_groups(sample_caption)
    for index, candidate in enumerate(groups[0]["candidates"]):
        candidate["raw_text"] = "not json" if index == 0 else '{"partial": true}'
        candidate["raw_schema_valid"] = False
        candidate["normalized_schema_valid"] = False
        candidate["caption"] = None
    calibration = fit_round0_calibration(_audio_groups(sample_caption), _caption_groups())
    candidates = score_groups(
        groups, loop="audio_only", calibration=calibration,
        reward_config=REWARD_CONFIG, sft_threshold=-100.0,
    )[0]["candidates"]
    assert all(item["grpo_mode"] == "schema_curriculum" for item in candidates)
    assert all(item["semantic_reward"] is None for item in candidates)
    assert not any(item["sft_selected"] for item in candidates)
    assert candidates[0]["reward"] < candidates[1]["reward"]


def test_round0_calibration_uses_paired_fallback_for_empty_generated_loop(sample_caption):
    generated_audio = _audio_groups(sample_caption)
    for candidate in generated_audio[0]["candidates"]:
        candidate["raw_text"] = "not json"
        candidate["caption"] = None
        candidate["raw_schema_valid"] = False
        candidate["normalized_schema_valid"] = False

    paired_audio = _audio_groups(sample_caption)[0]["candidates"]
    calibration = fit_round0_calibration(
        generated_audio,
        _caption_groups(),
        fallback_candidates={"audio_only": paired_audio},
    )

    assert calibration["sources"]["audio_only"] == {
        "kind": "paired_counterfactual_calibration_fallback",
        "semantic_valid_candidates": 4,
    }
    assert calibration["sources"]["caption_only"] == {
        "kind": "round0_generated_candidates",
        "semantic_valid_candidates": 4,
    }
    assert calibration["loops"]["audio_only"]["reconstruction"]["count"] == 4


def test_sft_threshold_prefers_high_precision_with_minimum_recall():
    threshold = fit_sft_threshold(
        [
            {"label": 1, "score": 0.9}, {"label": 1, "score": 0.8},
            {"label": 1, "score": 0.2}, {"label": 0, "score": 0.4},
            {"label": 0, "score": 0.1}, {"label": 0, "score": -0.2},
        ],
        target_precision=0.9, min_recall=0.5,
    )
    assert threshold["status"] == "calibrated"
    assert threshold["threshold"] == 0.8
    assert threshold["empirical_precision"] == 1.0


def test_counterfactuals_keep_content_and_change_one_control_field(sample_caption):
    caption = project_synth_caption(sample_caption)
    values = counterfactual_captions(caption, key="fixed", max_count=3)
    assert len(values) == 3
    for item in values:
        assert item["caption"]["semantic_content"] == caption["semantic_content"]
        assert item["original_value"] != item["alternative_value"]


def test_frozen_anchor_only_penalizes_clear_contradiction():
    stats = {
        "reconstruction": {"count": 4, "mean": 0.0, "std": 1.0},
        "counterfactual": {"count": 4, "mean": 0.0, "std": 1.0},
    }
    supported, _ = semantic_reward(
        {"reconstruction": 1.0, "counterfactual": 1.0, "anchor_margin": 0.2},
        loop_stats=stats, reconstruction_weight=0.5, counterfactual_weight=0.5,
        anchor_penalty_weight=0.25, anchor_tolerance_z=0.5,
    )
    contradicted, components = semantic_reward(
        {"reconstruction": 1.0, "counterfactual": 1.0, "anchor_margin": -1.0},
        loop_stats=stats, reconstruction_weight=0.5, counterfactual_weight=0.5,
        anchor_penalty_weight=0.25, anchor_tolerance_z=0.5,
    )
    assert components["anchor_penalty"] > 0.0
    assert contradicted < supported
