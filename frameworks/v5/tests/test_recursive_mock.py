from __future__ import annotations

import json
from pathlib import Path

import pytest

from dual_isl_train.config import load_config
from dual_isl_train.dual_space import project_synth_caption
from dual_isl_train.io import read_jsonl
from dual_isl_train.orchestrator import DualISLOrchestrator


def _config(tmp_path: Path, monkeypatch, rounds: int = 3):
    monkeypatch.setenv("DUALISL_RUN_DIR", str(tmp_path / "run"))
    config = load_config("configs/mock.yaml")
    config["training"]["rounds"] = rounds
    config["tts"]["codec_cache_dir"] = str(tmp_path / "codecs")
    return config


def test_three_round_full_pool_recursive_dual_training(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch)
    root = Path(config["run"]["output_dir"])
    result = DualISLOrchestrator(config).train()
    assert result["round"] == 2
    audio_ids = ["audio_0", "audio_1"]
    caption_ids = ["caption_0", "caption_1"]
    previous = None
    for round_index in range(3):
        directory = root / f"round_{round_index:03d}"
        summary = json.loads((directory / "summary.json").read_text())
        assert summary["audio_pool_ids"] == audio_ids
        assert summary["caption_pool_ids"] == caption_ids
        assert summary["collections_complete_before_update"] is True
        assert summary["caption_anchor_rows"] == 4
        assert summary["tts_anchor_rows"] == 2
        assert summary["sft_thresholds"]["loops"]["audio_only"]["status"] == "calibrated"
        assert summary["sft_thresholds"]["loops"]["caption_only"]["status"] == "calibrated"
        assert summary["evaluation"] == "disabled; checkpoints are benchmark-ready"
        assert (directory / "commit.json").is_file()
        if previous is not None:
            assert summary["input_captioner"] == previous["output_captioner"]
            assert summary["input_tts"] == previous["output_tts"]
        previous = summary
        caption_sft = list(read_jsonl(directory / "training" / f"round_{round_index:03d}_caption_sft.input.jsonl"))
        tts_sft = list(read_jsonl(directory / "training" / f"round_{round_index:03d}_tts_sft.input.jsonl"))
        assert len(caption_sft) == 2 + 2 * 2
        source_anchors = [row for row in caption_sft if row["id"].startswith("anchor::source::")]
        assert len(source_anchors) == 2
        assert all(row["target_schema"] == "source_no_environment" for row in source_anchors)
        assert all("environment" not in row["caption"] for row in source_anchors)
        assert all("topic" in row["caption"]["semantic_content"] for row in source_anchors)
        assert all("intent" in row["caption"]["semantic_content"] for row in source_anchors)
        assert len(tts_sft) == 2 + 2
    calibration = json.loads((root / "reward_calibration.json").read_text())
    assert calibration["fitted_round"] == 0
    assert calibration["frozen_across_rounds"] is True
    assert calibration["version"] == 2
    assert (root / "reward_anchor.json").is_file()
    diagnostics = json.loads((root / "base_diagnostics.json").read_text())
    assert diagnostics["ok"] is True
    assert diagnostics["before_any_optimizer_step"] is True
    assert diagnostics["quality_thresholds_applied"] is True
    before = json.loads((root / "run_state.json").read_text())
    resumed = DualISLOrchestrator(config).train(resume_only=True)
    after = json.loads((root / "run_state.json").read_text())
    assert resumed == result
    assert after == before


def test_round0_schema_curriculum_bootstraps_from_paired_calibration(
    tmp_path, monkeypatch, sample_caption,
):
    config = _config(tmp_path, monkeypatch, rounds=1)
    root = Path(config["run"]["output_dir"])
    orchestrator = DualISLOrchestrator(config)

    repairable = project_synth_caption(sample_caption)
    repairable["paralinguistic"]["prosody"] = ""
    repairable["paralinguistic"]["pause"] = ""
    repairable["paralinguistic"]["nonverbal_vocalization"] = "none"
    repairable_raw = json.dumps({"Target_JSON_Schema": repairable})
    audio_groups = [{
        "id": "schema_only_audio",
        "source_id": "schema_only_audio",
        "audio_path": "schema_only.wav",
        "candidates": [
            {
                "candidate_id": f"schema_only_audio::{index}",
                "raw_text": repairable_raw if index >= 2 else "not json",
                "trajectory_valid": True,
            }
            for index in range(4)
        ],
    }]
    monkeypatch.setattr(
        orchestrator, "_collect_audio_only", lambda *_args, **_kwargs: audio_groups,
    )

    result = orchestrator.train()
    assert result["round"] == 0

    diagnostics = json.loads((root / "base_diagnostics.json").read_text())
    audio_diagnostics = diagnostics["branches"]["audio_only"]
    assert audio_diagnostics["ok"] is True
    assert audio_diagnostics["diagnostic_mode"] == "schema_curriculum"
    assert audio_diagnostics["reconstruction_valid_candidates"] == 0
    assert audio_diagnostics["schema_curriculum_usable_groups"] == 1

    calibration = json.loads((root / "reward_calibration.json").read_text())
    assert calibration["sources"]["audio_only"]["kind"] == (
        "paired_counterfactual_calibration_fallback"
    )
    assert calibration["sources"]["caption_only"]["kind"] == (
        "round0_generated_candidates"
    )

    caption_grpo = list(read_jsonl(
        root / "round_000/training/round_000_caption_grpo.input.jsonl"
    ))
    assert all(
        candidate["grpo_mode"] == "schema_curriculum"
        for group in caption_grpo
        for candidate in group["candidates"]
    )
    assert not any(
        candidate["sft_selected"]
        for group in caption_grpo
        for candidate in group["candidates"]
    )

    tts_sft = list(read_jsonl(
        root / "round_000/training/round_000_tts_sft.input.jsonl"
    ))
    assert tts_sft
    assert all(row["id"].startswith("anchor::") for row in tts_sft)
    assert (root / "round_000/commit.json").is_file()


def test_round_is_not_committed_when_reload_check_fails(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch, rounds=1)
    root = Path(config["run"]["output_dir"])
    orchestrator = DualISLOrchestrator(config)

    def fail_reload(*_args, **_kwargs):
        raise RuntimeError("injected reload failure")

    monkeypatch.setattr(orchestrator, "_reload_check", fail_reload)
    with pytest.raises(RuntimeError, match="injected reload failure"):
        orchestrator.train()
    assert not (root / "latest.json").exists()
    assert not (root / "round_000/commit.json").exists()
    assert not json.loads((root / "run_state.json").read_text())["current"]
    completed = DualISLOrchestrator(config).train(resume_only=True)
    assert completed["round"] == 0
    assert (root / "round_000/commit.json").is_file()


def test_committed_checkpoint_continuation_preserves_round_numbers_and_calibration(tmp_path, monkeypatch):
    source_config = _config(tmp_path / "source", monkeypatch, rounds=3)
    source_root = Path(source_config["run"]["output_dir"])
    source_result = DualISLOrchestrator(source_config).train()
    source_calibration = json.loads((source_root / "reward_calibration.json").read_text())

    continuation_root = tmp_path / "continuation" / "run"
    monkeypatch.setenv("DUALISL_RUN_DIR", str(continuation_root))
    continuation_config = load_config("configs/mock.yaml")
    continuation_config["training"]["rounds"] = 2
    continuation_config["training"]["round_offset"] = 3
    continuation_config["captioner"]["adapter_path"] = source_result["caption_checkpoint"]
    continuation_config["tts"]["adapter_path"] = source_result["tts_checkpoint"]
    continuation_config["tts"]["codec_cache_dir"] = str(tmp_path / "continuation" / "codecs")
    continuation_config["reward"]["calibration"]["initial_path"] = str(
        source_root / "reward_calibration.json"
    )
    continuation_config["reward"]["anchor"] = {
        "captioner_adapter_path": "",
        "tts_adapter_path": "",
    }

    result = DualISLOrchestrator(continuation_config).train()
    assert result["round"] == 4
    assert not (continuation_root / "round_000").exists()
    assert not (continuation_root / "base_diagnostics.json").exists()
    assert (continuation_root / "round_003/commit.json").is_file()
    assert (continuation_root / "round_004/commit.json").is_file()
    assert json.loads((continuation_root / "reward_calibration.json").read_text()) == source_calibration

    contract = json.loads((continuation_root / "continuation.json").read_text())
    assert contract["kind"] == "committed_dual_checkpoint_continuation"
    assert contract["round_offset"] == 3
    assert contract["round_count"] == 2
    assert contract["input_captioner"]["path"] == source_result["caption_checkpoint"]
    assert contract["input_tts"]["path"] == source_result["tts_checkpoint"]
    assert contract["frozen_reward_anchor"] == {
        "captioner_adapter": None,
        "tts_adapter": None,
    }
    assert json.loads((continuation_root / "reward_anchor.json").read_text()) == json.loads(
        (source_root / "reward_anchor.json").read_text()
    )

    first_commit = json.loads((continuation_root / "round_003/commit.json").read_text())
    assert first_commit["input_captioner"] == contract["input_captioner"]
    assert first_commit["input_tts"] == contract["input_tts"]
    before = json.loads((continuation_root / "run_state.json").read_text())
    assert DualISLOrchestrator(continuation_config).train(resume_only=True) == result
    assert json.loads((continuation_root / "run_state.json").read_text()) == before
