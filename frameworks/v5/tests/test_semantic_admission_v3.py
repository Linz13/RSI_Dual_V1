from __future__ import annotations

import json
from copy import deepcopy

import pytest

from dual_isl_train.config import load_config
from dual_isl_train.dual_space import parse_synth_caption_text_detailed, reparse_synth_candidate, synth_validation_errors
from dual_isl_train.io import read_jsonl, write_jsonl
from dual_isl_train.orchestrator import DualISLOrchestrator
from dual_isl_train.render import synth_caption_example, synth_caption_prompt
from dual_isl_train.rewards import candidate_components, reward_summary, schema_progress, score_groups
from dual_isl_train.schema import set_path
from dual_isl_train.semantic_admission import admit_synth_text


def _caption():
    return synth_caption_example()["Target_JSON_Schema"]


@pytest.mark.parametrize("field,value,expected", [
    ("speaker_profile.gender", " Female ", "female"),
    ("paralinguistic.volume_level", "loud", "high"),
    ("paralinguistic.speaking_rate", "normal", "moderate"),
    ("paralinguistic.nonverbal_vocalization", "none", ["none"]),
    ("paralinguistic.nonverbal_vocalization", "laughing", ["laughter"]),
])
def test_safe_representation_repairs(field, value, expected):
    original = _caption()
    set_path(original, field, value)
    result = admit_synth_text(json.dumps(original))
    desired = deepcopy(original)
    set_path(desired, field, expected)
    assert result["semantic_input_valid"]
    assert result["semantic_caption"] == desired
    assert result["safe_normalization_rules"]


def test_drop_only_excluded_fields_and_keep_emphasized_words():
    caption = _caption()
    caption["semantic_content"].update(topic="example", intent="statement")
    caption["environment"] = {"location": "unknown"}
    caption["paralinguistic"]["emphasis"] = {"level": "emphasized", "emphasized_text": "example"}
    result = admit_synth_text(json.dumps({"Target_JSON_Schema": caption, "topic": "example"}))
    assert result["semantic_input_valid"]
    assert result["semantic_caption"]["paralinguistic"]["emphasis"]["emphasized_text"] == ["example"]
    assert "environment" not in result["semantic_caption"]
    assert "topic" not in result["semantic_caption"]["semantic_content"]


@pytest.mark.parametrize("field,value", [
    ("speaker_profile.age", "mature"),
    ("speaker_profile.accent", "southern"),
    ("paralinguistic.emphasis.level", "low emphasis"),
    ("paralinguistic.nonverbal_vocalization", []),
    ("paralinguistic.nonverbal_vocalization", ["none", "laughter"]),
    ("paralinguistic.nonverbal_vocalization", ["sigh", "sigh"]),
    ("paralinguistic.nonverbal_vocalization", [3]),
    ("paralinguistic.emotion_intensity", "high"),  # conflicts with neutral
    ("paralinguistic.emphasis.emphasized_text", ["example"]),  # conflicts with none
    ("paralinguistic.prosody", ""),
    ("semantic_content.transcript", "unknown"),
    ("semantic_content.transcript", "  "),
    ("speaker_profile", []),
])
def test_semantic_guesses_and_conflicts_are_rejected(field, value):
    caption = _caption()
    set_path(caption, field, value)
    result = admit_synth_text(json.dumps(caption))
    assert not result["semantic_input_valid"]
    assert result["semantic_caption"] is None
    assert result["semantic_rejection_reasons"]


def test_missing_field_is_not_filled_unknown_even_when_legacy_parser_can_fix_it():
    caption = _caption()
    del caption["speaker_profile"]["gender"]
    raw = json.dumps(caption)
    _, _, legacy = parse_synth_caption_text_detailed(raw)
    assert legacy["normalized_schema_valid"]
    item = reparse_synth_candidate({"raw_text": raw, "semantic_input_valid": True})
    assert not item["semantic_input_valid"]
    caption["speaker_profile"]["gender"] = "unknown"
    assert admit_synth_text(json.dumps(caption))["semantic_input_valid"]
    assert not reparse_synth_candidate({"caption": _caption(), "semantic_input_valid": True})["semantic_input_valid"]


@pytest.mark.parametrize("alter", [
    lambda raw: raw[:-1],
    lambda raw: raw.replace('"gender": "female"', '"gender": "male", "gender": "female"'),
    lambda raw: raw + raw,
    lambda raw: '[' + raw + ']',
    lambda raw: '{"broken":' + raw,
    lambda raw: '{"Target_JSON_Schema":' + raw + ', "alternative":' + raw + '}',
    lambda raw: raw.replace('"gender": "female"', '"gender": NaN'),
])
def test_malformed_or_ambiguous_json_is_not_salvaged(alter):
    assert not admit_synth_text(alter(json.dumps(_caption())))["semantic_input_valid"]


@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```", "Here is the result:\n{}"])
def test_single_complete_object_and_prompt_example(wrapper):
    raw = json.dumps(synth_caption_example())
    assert admit_synth_text(wrapper.format(raw))["semantic_input_valid"]
    assert not synth_validation_errors(synth_caption_example())
    prompt = synth_caption_prompt()
    assert json.dumps(synth_caption_example(), ensure_ascii=False, separators=(",", ":")) in prompt
    assert "STRUCTURE ONLY" in prompt


def test_repair_enters_semantic_reward_without_changing_trajectory_or_formula():
    caption = _caption()
    caption["paralinguistic"]["nonverbal_vocalization"] = "none"
    raw = json.dumps(caption)
    candidates = [{
        "candidate_id": str(i), "raw_text": raw, "trajectory_valid": True,
        "sampled_token_ids": [11, 12, i], "old_token_logprobs": [-0.2, -0.3, -0.4],
        "ref_token_logprobs": [-0.3, -0.4, -0.5], "prompt": "original prompt",
        "tts_target_logprob": -0.8 + i * 0.1, "anchor_tts_target_logprob": -0.7,
        "counterfactual_reconstruction": [{"reconstruction": -1.0}],
        "anchor_counterfactual_reconstruction": [{"reconstruction": -0.9}], "asr_score": 0.9,
    } for i in range(4)]
    original = deepcopy(candidates)
    calibration = {"version": 2, "method": "round0_dual_counterfactual_zscore", "fitted_round": 0,
                   "frozen_across_rounds": True, "loops": {
                       loop: {key: {"count": 4, "mean": 0.0, "std": 1.0}
                              for key in ("reconstruction", "counterfactual")}
                       for loop in ("audio_only", "caption_only")}}
    scored = score_groups([{"id": "group", "candidates": candidates}], loop="audio_only",
                          calibration=calibration, sft_threshold=100.0,
                          reward_config={"reconstruction_weight": 0.5, "counterfactual_weight": 0.5,
                                         "anchor_penalty_weight": 0.25, "anchor_tolerance_z": 0.5})
    assert candidates == original
    for before, after in zip(original, scored[0]["candidates"]):
        assert all(after[key] == before[key] for key in before)
        assert not after["raw_schema_valid"]
        assert after["grpo_mode"] == "dual_semantic"
        assert not after["sft_gate_would_select"]
        rec = before["tts_target_logprob"]
        assert after["semantic_reward"] == pytest.approx(0.5 * rec + 0.5 * (rec + 1.0))
    summary = reward_summary(scored)
    assert summary["sft_selected"] == summary["sft_selected_despite_gate"] == 1
    assert summary["safely_repaired_candidates"] == 4
    assert summary["semantic_grpo_groups"] == summary["semantic_nonzero_advantage_groups"] == 1
    # Admission alone cannot replace missing reverse-model scores or trajectory.
    for key, value in (("tts_target_logprob", None), ("trajectory_valid", False),
                       ("anchor_counterfactual_reconstruction", []), ("asr_score", float("nan"))):
        assert not candidate_components("audio_only", {**original[0], key: value}, {})[0]
    unsafe_raw = raw.replace('"prosody": "steady intonation"', '"prosody": ""')
    _, _, metadata = parse_synth_caption_text_detailed(unsafe_raw)
    assert schema_progress(reparse_synth_candidate({"raw_text": unsafe_raw})) == schema_progress(metadata)


def test_actual_collection_schedules_repaired_inputs_through_mock_round(tmp_path, monkeypatch):
    from dual_isl_train.workers import mock
    execute = mock.execute_mock
    originals = {}

    def repaired_rollout(**kwargs):
        execute(**kwargs)
        if kwargs["action"] != "rollout" or not kwargs["rows"] or "audio_path" not in kwargs["rows"][0]:
            return
        groups = list(read_jsonl(kwargs["output"]))
        for group in groups:
            for candidate in group["candidates"]:
                caption = json.loads(candidate["raw_text"])
                caption["Target_JSON_Schema"]["paralinguistic"]["nonverbal_vocalization"] = "none"
                candidate["raw_text"] = json.dumps(caption)
                originals[candidate["candidate_id"]] = deepcopy(candidate)
        write_jsonl(kwargs["output"], groups)

    monkeypatch.setattr(mock, "execute_mock", repaired_rollout)
    monkeypatch.setenv("DUALISL_RUN_DIR", str(tmp_path / "run"))
    config = load_config("configs/mock.yaml")
    config["training"]["rounds"] = 1
    config["tts"]["codec_cache_dir"] = str(tmp_path / "codecs")
    assert DualISLOrchestrator(config).train()["round"] == 0
    path = tmp_path / "run/round_000/training/round_000_caption_grpo.input.jsonl"
    groups = list(read_jsonl(path))
    for group in groups:
        assert group["prompt"] == synth_caption_prompt()
        for candidate in group["candidates"]:
            original = originals[candidate["candidate_id"]]
            assert candidate["semantic_input_valid"] and candidate["semantic_valid"]
            assert not candidate["raw_schema_valid"]
            assert candidate["grpo_mode"] == "dual_semantic"
            assert candidate["counterfactual_reconstruction"]
            for key in ("raw_text", "sampled_token_ids", "old_token_logprobs", "ref_token_logprobs"):
                assert candidate[key] == original[key]
    # Resume of the same committed V3 run remains valid.
    assert DualISLOrchestrator(config).train(resume_only=True)["round"] == 0
