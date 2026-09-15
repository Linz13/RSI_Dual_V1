from copy import deepcopy
import json

import pytest

from dual_isl_train.config import load_config, validate_config
from dual_isl_train.io import read_jsonl
from dual_isl_train.orchestrator import DualISLOrchestrator
from dual_isl_train.rewards import fit_round0_calibration, reward_summary, score_groups
from test_rewards import _audio_groups, _caption_groups, REWARD_CONFIG


@pytest.mark.parametrize("loop", ["audio_only", "caption_only"])
@pytest.mark.parametrize("threshold", [None, 100.0, -100.0])
def test_top1_selection_is_independent_of_diagnostic_threshold(sample_caption, loop, threshold):
    audio, caption = _audio_groups(sample_caption), _caption_groups()
    calibration = fit_round0_calibration(audio, caption)
    groups = audio if loop == "audio_only" else caption
    before = deepcopy(groups)
    scored = score_groups(groups, loop=loop, calibration=calibration,
                          reward_config=REWARD_CONFIG, sft_threshold=threshold)
    assert groups == before
    selected = [c for c in scored[0]["candidates"] if c["sft_selected"]]
    assert len(selected) == 1
    assert selected[0]["candidate_id"].endswith("::3")
    assert selected[0]["sft_rejection_reason"] is None
    assert selected[0]["sft_gate_would_select"] is (threshold == -100.0)
    assert scored[0]["cycle_sft_selection"] == "semantic_top1"
    summary = reward_summary(scored)
    assert summary["sft_selected"] == summary["sft_eligible_groups"] == 1


@pytest.mark.parametrize("loop", ["audio_only", "caption_only"])
@pytest.mark.parametrize("valid_count", [0, 1])
def test_one_valid_candidate_can_train_sft_but_not_semantic_grpo(sample_caption, loop, valid_count):
    audio, caption = _audio_groups(sample_caption), _caption_groups()
    calibration = fit_round0_calibration(audio, caption)
    groups = audio if loop == "audio_only" else caption
    # Invalid high-scoring candidates must never win, even without an absolute gate.
    for c in groups[0]["candidates"][valid_count:]:
        c["asr_score"] = float("nan")
    scored = score_groups(groups, loop=loop, calibration=calibration,
                          reward_config=REWARD_CONFIG, sft_threshold=None)
    cs = scored[0]["candidates"]
    assert sum(c["sft_selected"] for c in cs) == valid_count
    assert all(c["grpo_mode"] != "dual_semantic" for c in cs)
    assert all(not c["sft_selected"] for c in cs if not c["semantic_valid"])
    if valid_count:
        assert cs[0]["sft_selected"]
    if loop == "audio_only":
        assert all(c["grpo_mode"] == "schema_curriculum" for c in cs)
    else:
        assert all(c["skip_update"] for c in cs)


@pytest.mark.parametrize("loop", ["audio_only", "caption_only"])
def test_ties_keep_original_candidate_id_tiebreak(sample_caption, loop):
    audio, caption = _audio_groups(sample_caption), _caption_groups()
    calibration = fit_round0_calibration(audio, caption)
    groups = audio if loop == "audio_only" else caption
    original = groups[0]["candidates"][0]
    groups[0]["candidates"] = [{**deepcopy(original), "candidate_id": name} for name in ["b", "d", "a", "c"]]
    scored = score_groups(groups, loop=loop, calibration=calibration,
                          reward_config=REWARD_CONFIG, sft_threshold=None)
    assert [c["candidate_id"] for c in scored[0]["candidates"] if c["sft_selected"]] == ["d"]
    assert all(c["advantage"] == 0 for c in scored[0]["candidates"])


@pytest.mark.parametrize("field,value", [("cycle_sft_selection", "confidence_gate"), ("diagnostic_only", False)])
def test_v4_rejects_accidental_gate_reactivation(field, value):
    config = load_config("configs/mock.yaml")
    if field == "cycle_sft_selection":
        config["reward"][field] = value
    else:
        config["reward"]["sft_gate"][field] = value
    with pytest.raises(ValueError, match="RewardV4 requires"):
        validate_config(config)


@pytest.mark.parametrize("diagnostic_case", ["no_operating_point", "no_examples"])
def test_full_mock_keeps_both_sft_directions_when_diagnostic_unavailable(tmp_path, monkeypatch, diagnostic_case):
    monkeypatch.setenv("DUALISL_RUN_DIR", str(tmp_path / "run"))
    config = load_config("configs/mock.yaml")
    config["training"]["rounds"] = 1
    config["tts"]["codec_cache_dir"] = str(tmp_path / "codecs")
    if diagnostic_case == "no_operating_point":
        # Equal scores cannot separate balanced positive/negative examples at 90%.
        monkeypatch.setattr("dual_isl_train.orchestrator.score_calibration_examples", lambda *a, **k: [
            {"id": "positive", "label": 1, "score": 0.0},
            {"id": "negative", "label": 0, "score": 0.0},
        ])
        status = "no_high_precision_operating_point"
    else:
        monkeypatch.setattr("dual_isl_train.orchestrator.score_calibration_examples", lambda *a, **k: [])
        status = "insufficient_calibration_examples"
    result = DualISLOrchestrator(config).train()
    root = tmp_path / "run/round_000"
    assert result["round"] == 0 and (root / "commit.json").is_file()
    summary = json.loads((root / "summary.json").read_text())
    assert summary["sft_thresholds"]["diagnostic_only"]
    for loop in ("audio_only", "caption_only"):
        assert summary["sft_thresholds"]["loops"][loop]["threshold"] is None
        assert summary["sft_thresholds"]["loops"][loop]["status"] == status
        assert summary[loop]["sft_selected"] == 2
        assert summary[loop]["sft_gate_would_select"] == 0
    for model, source in (("caption", "caption_"), ("tts", "audio_")):
        rows = list(read_jsonl(root / f"training/round_000_{model}_sft.input.jsonl"))
        pseudo = [r for r in rows if not r["is_anchor"]]
        assert len(pseudo) == 2
        assert all(r["id"].startswith(source) for r in pseudo)
