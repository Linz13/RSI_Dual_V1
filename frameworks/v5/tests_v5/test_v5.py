from __future__ import annotations
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from dual_isl_train.partial_caption import admit
from dual_isl_train.attribute_reward import (reconstruction, reference_mask, score_audio_groups,
    EvaluationPending, JUDGE_FIELDS, EMPHASIS, EVENTS, emphasis_f1)
from dual_isl_train.dual_space import project_synth_caption
from dual_isl_train.workers.mock import mock_caption
from dual_isl_train.schema import set_path

ROOT = Path(__file__).resolve().parents[1]


def caption():
    return project_synth_caption(mock_caption(0))


def test_partial_admission_does_not_count_imputed_fields():
    data = {"semantic_content": {"transcript": "hello"}, "speaker_profile": {"gender": "Female", "age": "ancient"},
            "paralinguistic": {"emotion": "Happy"}, "extra": "value"}
    result = admit(json.dumps(data))
    assert result["semantic_input_valid"]
    assert len(result["valid_fields"]) == 3
    assert result["format_score"] == 0.5 + 0.5 * 3 / 16
    assert result["caption"]["speaker_profile"]["age"] == "unknown"
    assert result["caption"]["paralinguistic"]["emotion"] == "happy"


@pytest.mark.parametrize("text", ['{"speaker_profile":', '{"x":1,"x":2}', '[{"x":1}]', '{"x": NaN}'])
def test_ambiguous_or_broken_json_is_not_synthesized(text):
    result = admit(text)
    assert result["format_score"] == 0
    assert not result["semantic_input_valid"]


def test_legal_unknown_full_format_and_missing_text_admission():
    data = caption()
    data["semantic_content"]["transcript"] = "unknown"
    result = admit(json.dumps(data))
    assert result["format_score"] == 1
    assert not result["semantic_input_valid"]


def test_reference_mask_is_fixed_generated_unknown_cannot_shrink_denominator():
    ref = caption()
    pred = copy.deepcopy(ref)
    first = reconstruction(ref, pred, {f: 1 for f in JUDGE_FIELDS})
    pred["speaker_profile"]["gender"] = "unknown"
    second = reconstruction(ref, pred, {f: 1 for f in JUDGE_FIELDS})
    assert first["denominator"] == second["denominator"] == 15
    assert first["score"] == 1
    assert second["score"] == pytest.approx(14 / 15)
    ref["speaker_profile"]["gender"] = "unknown"
    assert "speaker_profile.gender" not in reference_mask(ref)


def test_transcript_not_a_reward_or_gate():
    ref, pred = caption(), caption()
    pred["semantic_content"]["transcript"] = "Completely different words"
    assert reconstruction(ref, pred, {f: 1 for f in JUDGE_FIELDS})["score"] == 1


def test_event_f1_absence_and_unknown():
    ref, pred = caption(), caption()
    set_path(ref, EVENTS, ["laughter", "cough"])
    set_path(pred, EVENTS, ["laughter"])
    assert reconstruction(ref, pred, {f: 1 for f in JUDGE_FIELDS})["fields"][EVENTS] == pytest.approx(2/3)
    set_path(ref, EVENTS, ["none"])
    set_path(pred, EVENTS, ["unknown"])
    assert reconstruction(ref, pred, {f: 1 for f in JUDGE_FIELDS})["fields"][EVENTS] == 0


@pytest.mark.parametrize("a,b,phrase_a,phrase_b,expected", [
    ("very good day", "very good day", ["very good"], ["good"], 2/3),
    ("今天真好", "今天真好", ["今天"], ["今"], 2/3),
    ("good day", "bad day", ["good"], ["bad"], 0),
    ("good day", "good day", [], ["missing"], 0)])
def test_emphasis_position_alignment(a,b,phrase_a,phrase_b,expected):
    ref, pred = caption(), caption()
    ref["semantic_content"]["transcript"] = a
    pred["semantic_content"]["transcript"] = b
    set_path(ref, EMPHASIS, phrase_a)
    set_path(pred, EMPHASIS, phrase_b)
    assert emphasis_f1(ref, pred) == pytest.approx(expected)


def test_judge_missing_is_failure_not_zero():
    with pytest.raises(EvaluationPending):
        reconstruction(caption(), caption(), {})


def candidate(cid, rec, fmt=1, valid=True):
    return {"candidate_id": cid, "trajectory_valid": True, "semantic_input_valid": valid, "format_score": fmt,
            "attribute_reconstruction": {"status": "complete" if valid else "unrenderable", "score": rec}}


def test_sft_uses_reconstruction_not_mixed_reward():
    # Mixed reward prefers b, but reconstruction prefers a.
    a, b = candidate("a", 0.8, 0.5), candidate("b", 0.79, 1)
    cs = score_audio_groups([{"candidates": [a,b]}])[0]["candidates"]
    assert cs[0]["sft_selected"] and not cs[1]["sft_selected"]
    assert cs[1]["advantage"] > cs[0]["advantage"]


def test_format_only_groups_and_flat_groups():
    a,b = candidate("a", None, .5, False), candidate("b", None, 0, False)
    cs = score_audio_groups([{"candidates": [a,b]}])[0]["candidates"]
    assert cs[0]["advantage"] == pytest.approx(1)
    assert not any(c["sft_selected"] for c in cs)
    flat = score_audio_groups([{"candidates": [a,copy.deepcopy(a)]}])[0]["candidates"]
    assert all(c["skip_update"] and c["advantage"] == 0 for c in flat)


def test_evaluator_failure_blocks_group():
    c = candidate("a", 1)
    c["attribute_reconstruction"] = {"status": "pending"}
    with pytest.raises(EvaluationPending):
        score_audio_groups([{"candidates": [c]}])


def test_core_grpo_functions_identical_to_v4():
    targets = {
        "scripts/midasheng_captioner_candidate.py": ["_native_candidates_backward", "_native_candidate_exact_backward"],
        "dual_isl_train/workers/qwen3_captioner.py": ["grpo_update", "_native_candidate_backward", "_native_candidates_backward"],
        "dual_isl_train/workers/qwen_voice_design.py": ["grpo_update", "_incremental_trajectory_impl", "incremental_candidate_logprobs", "_clipped_loss"],
    }
    for path, names in targets.items():
        def functions(root):
            tree = ast.parse((root/path).read_text())
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        old, new = functions(ROOT.with_name("DualISL_Train_RewardV4")), functions(ROOT)
        for name in names:
            assert new[name] == old[name], (path, name)


def test_cpu_end_to_end_resume_and_anchor_contract(tmp_path, monkeypatch):
    from dual_isl_train.config import load_config
    from dual_isl_train.orchestrator import DualISLOrchestrator
    from dual_isl_train.io import read_jsonl
    monkeypatch.setenv("DUALISL_RUN_DIR", str(tmp_path / "run"))
    cfg = load_config(ROOT / "configs/v5_mock.yaml")
    cfg["tts"]["codec_cache_dir"] = str(tmp_path / "codecs")
    orch = DualISLOrchestrator(cfg)
    result = orch.train()
    assert result["round"] == 1
    assert orch.train(resume_only=True) == result
    for rnd in range(2):
        rd = tmp_path / "run" / f"round_{rnd:03d}"
        summary = json.loads((rd / "summary.json").read_text())
        assert summary["caption_anchor_rows"] == summary["tts_anchor_rows"]
        assert summary["collections_complete_before_update"]
        assert not list(rd.rglob("*sftcal_tts*"))
        assert not list(rd.rglob("*audio_tts_reconstruction*"))
        for c in read_jsonl(rd / "training" / f"round_{rnd:03d}_caption_sft.input.jsonl"):
            assert c["target_schema"] == "synth_v1"
        for c in read_jsonl(rd / "training" / f"round_{rnd:03d}_tts_sft.input.jsonl"):
            assert c["audio_path"] == c["source_audio_path"]
    calibration = json.loads((tmp_path / "run/reward_calibration.json").read_text())
    assert set(calibration["loops"]) == {"caption_only"}


def test_demo_config_is_not_executed(tmp_path):
    from dual_isl_train.labeling import demo_credentials
    path = tmp_path / "demo.py"
    path.write_text('BASE_URL="https://example.test/v1"\nAPI_KEY="test-secret"\nraise RuntimeError("must not execute")\n')
    assert demo_credentials(path) == ("https://example.test/v1", "test-secret")
