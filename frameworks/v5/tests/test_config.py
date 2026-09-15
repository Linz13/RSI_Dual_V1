from __future__ import annotations

from copy import deepcopy

import pytest

from dual_isl_train.config import load_config, validate_config


def test_reward_v4_config_has_counterfactual_reward_and_diagnostic_gate_without_warmstart():
    config = load_config("configs/mock.yaml")
    assert config["training"]["warmstart"] is False
    assert config["training"]["anchor_mode"] == "all_paired_once"
    assert config["training"]["group_size"] == 4
    assert config["evaluation"]["enabled"] is False
    assert "pseudo_label" not in config
    assert "filter" not in config
    assert config["reward"]["counterfactuals_per_candidate"] == 3
    assert config["reward"]["sft_gate"]["target_precision"] == 0.9
    assert config["reward"]["sft_gate"]["diagnostic_only"] is True
    assert config["reward"]["cycle_sft_selection"] == "semantic_top1"
    assert config["reward"]["audio_only"] == {
        "reconstruction_weight": 0.5, "counterfactual_weight": 0.5,
    }
    assert config["captioner"]["generation"]["rollout_batch_size"] == 4
    assert config["captioner"]["training"]["replay_batch_size"] == 4


def test_config_rejects_warmstart_and_non_single_epoch_grpo():
    config = load_config("configs/mock.yaml")
    invalid = deepcopy(config)
    invalid["training"]["warmstart"] = True
    with pytest.raises(ValueError, match="warmstart=false"):
        validate_config(invalid)
    invalid = deepcopy(config)
    invalid["tts"]["training"]["phases"]["grpo"]["epochs"] = 2
    with pytest.raises(ValueError, match="exactly 1"):
        validate_config(invalid)


def test_smoke_config_inherits_base_and_overrides_gpu_count():
    single = load_config("configs/gpu_smoke_1gpu.yaml")
    multi = load_config("configs/gpu_smoke_2gpu.yaml")
    assert single["training"]["rounds"] == 1
    assert single["data"]["max_records_per_role"] == 1
    assert single["captioner"]["generation"]["max_attempts_per_candidate"] == 1
    assert single["captioner"]["generation"]["rollout_batch_size"] == 1
    assert single["captioner"]["training"]["replay_batch_size"] == 1
    assert single["distributed"]["enabled"] is False
    assert multi["distributed"]["enabled"] is True
    assert multi["distributed"]["world_size"] == 2


def test_midasheng_candidate_configs_are_isolated(monkeypatch, tmp_path):
    run_dir = tmp_path / "midasheng-run"
    monkeypatch.setenv("DUALISL_RUN_DIR", str(run_dir))
    for path, world_size in (
        ("configs/gpu_smoke_1gpu_h100_midasheng_candidate.yaml", 1),
        ("configs/gpu_smoke_4gpu_h100_midasheng_candidate.yaml", 4),
        ("configs/train_4gpu_h100_midasheng_candidate.yaml", 4),
    ):
        config = load_config(path)
        assert config["run"]["output_dir"] == str(run_dir.resolve())
        assert config["captioner"]["worker_module"] == "scripts.midasheng_captioner_candidate"
        assert config["captioner"]["generation"]["rollout_batch_size"] == 1
        assert config["captioner"]["training"]["replay_batch_size"] == 1
        assert config["distributed"]["world_size"] == world_size
        assert config["tts"]["codec_cache_dir"] == str(
            (run_dir / "codec_cache/qwen3tts_12hz").resolve()
        )


def test_midasheng_reward_v2_configs_are_fresh_eight_gpu_runs(monkeypatch, tmp_path):
    run_dir = tmp_path / "midasheng-reward-v2"
    monkeypatch.setenv("DUALISL_RUN_DIR", str(run_dir))
    for path, rounds, limit in (
        ("configs/gpu_smoke_8gpu_h100_midasheng_reward_v2.yaml", 1, 8),
        ("configs/train_8gpu_h100_midasheng_reward_v2.yaml", 3, 0),
    ):
        config = load_config(path)
        assert config["run"]["output_dir"] == str(run_dir.resolve())
        assert config["distributed"]["enabled"] is True
        assert config["distributed"]["world_size"] == 8
        assert config["training"]["rounds"] == rounds
        assert config["training"]["round_offset"] == 0
        assert config["data"].get("max_records_per_role", 0) == limit
        assert config["captioner"]["worker_module"] == "scripts.midasheng_captioner_candidate"
        assert config["captioner"]["adapter_path"] == ""
        assert config["tts"]["adapter_path"] == ""
        assert config["reward"]["calibration"]["method"] == "round0_dual_counterfactual_zscore"
        assert config["tts"]["codec_cache_dir"] == str(
            (run_dir / "codec_cache/qwen3tts_12hz").resolve()
        )


def test_round_count_can_be_overridden_by_environment(monkeypatch):
    monkeypatch.setenv("DUALISL_ROUNDS", "5")
    config = load_config("configs/train_8gpu_h100_midasheng_reward_v2.yaml")
    assert config["training"]["rounds"] == 5


def test_round_count_environment_override_must_be_an_integer(monkeypatch):
    monkeypatch.setenv("DUALISL_ROUNDS", "three")
    with pytest.raises(ValueError, match="DUALISL_ROUNDS must be an integer"):
        load_config("configs/train_8gpu_h100_midasheng_reward_v2.yaml")


def test_legacy_midasheng_r2_continuation_configs_are_rejected(monkeypatch, tmp_path):
    run_dir = tmp_path / "midasheng-continuation"
    monkeypatch.setenv("DUALISL_RUN_DIR", str(run_dir))
    for path in (
        ("configs/gpu_smoke_8gpu_h100_midasheng_continuation_r2.yaml", 1),
        ("configs/train_8gpu_h100_midasheng_continuation_r2.yaml", 10),
    ):
        with pytest.raises(ValueError, match="round0_dual_counterfactual_zscore"):
            load_config(path[0])


def test_legacy_qwen_r2_continuation_config_is_rejected(monkeypatch, tmp_path):
    run_dir = tmp_path / "qwen-continuation"
    monkeypatch.setenv("DUALISL_RUN_DIR", str(run_dir))
    with pytest.raises(ValueError, match="round0_dual_counterfactual_zscore"):
        load_config("configs/train_8gpu_h100_qwen_continuation_r2.yaml")


def test_continuation_config_requires_complete_source_lineage():
    config = load_config("configs/mock.yaml")
    invalid = deepcopy(config)
    invalid["training"]["round_offset"] = -1
    with pytest.raises(ValueError, match="round_offset must be >= 0"):
        validate_config(invalid)

    invalid = deepcopy(config)
    invalid["training"]["round_offset"] = 3
    with pytest.raises(ValueError, match="captioner.adapter_path"):
        validate_config(invalid)

    invalid["captioner"]["adapter_path"] = "/checkpoint/caption"
    invalid["tts"]["adapter_path"] = "/checkpoint/tts"
    with pytest.raises(ValueError, match="reward.calibration.initial_path"):
        validate_config(invalid)

    invalid["reward"]["calibration"]["initial_path"] = "/calibration.json"
    with pytest.raises(ValueError, match="original frozen run anchor"):
        validate_config(invalid)

    invalid = deepcopy(config)
    invalid["reward"]["calibration"]["initial_path"] = "/calibration.json"
    with pytest.raises(ValueError, match="requires training.round_offset"):
        validate_config(invalid)


def test_midasheng_reward_v2_r9_continuation_configs_preserve_lineage(monkeypatch, tmp_path):
    run_dir = tmp_path / "midasheng-reward-v2-from-r9"
    monkeypatch.setenv("DUALISL_RUN_DIR", str(run_dir))
    for path, rounds, limit in (
        ("configs/gpu_smoke_8gpu_h100_midasheng_reward_v2_continuation_r9.yaml", 1, 8),
        ("configs/train_8gpu_h100_midasheng_reward_v2_continuation_r9.yaml", 10, 0),
    ):
        config = load_config(path)
        assert config["run"]["output_dir"] == str(run_dir.resolve())
        assert config["distributed"]["world_size"] == 8
        assert config["training"]["rounds"] == rounds
        assert config["training"]["round_offset"] == 10
        assert config["data"].get("max_records_per_role", 0) == limit
        assert config["captioner"]["adapter_path"].endswith(
            "/round_009/checkpoints/caption_final"
        )
        assert config["tts"]["adapter_path"].endswith(
            "/round_009/checkpoints/tts_final"
        )
        assert config["reward"]["calibration"]["initial_path"].endswith(
            "/dual_recursive_8gpu_h100_midasheng_reward_v2_20260901_run01/"
            "reward_calibration.json"
        )
        assert config["reward"]["anchor"] == {
            "captioner_adapter_path": "",
            "tts_adapter_path": "",
        }


@pytest.mark.parametrize("gpu_count", [4, 7, 8])
def test_h100_smoke_and_train_configs_use_all_ranks_with_safe_caption_batches(gpu_count):
    smoke = load_config(f"configs/gpu_smoke_{gpu_count}gpu_h100.yaml")
    training = load_config(f"configs/train_{gpu_count}gpu_h100.yaml")
    assert smoke["distributed"]["enabled"] is True
    assert smoke["distributed"]["world_size"] == gpu_count
    assert smoke["data"]["max_records_per_role"] == gpu_count
    assert smoke["training"]["rounds"] == 1
    assert training["distributed"]["enabled"] is True
    assert training["distributed"]["world_size"] == gpu_count
    assert training["training"]["rounds"] == 3
    assert training["data"]["duration"] == {"min": 1.2, "max": 30.0}
    assert training["critics"]["whisper_model"].endswith("/Caption/models/whisper/large-v3-turbo.pt")
    for config in (smoke, training):
        assert config["captioner"]["generation"]["rollout_batch_size"] == 1
        assert config["captioner"]["training"]["replay_batch_size"] == 1


def test_captioner_batch_sizes_are_bounded_and_require_neutral_sampling():
    config = load_config("configs/mock.yaml")
    invalid = deepcopy(config)
    invalid["captioner"]["generation"]["rollout_batch_size"] = 5
    with pytest.raises(ValueError, match="rollout_batch_size"):
        validate_config(invalid)
    invalid = deepcopy(config)
    invalid["captioner"]["training"]["replay_batch_size"] = 0
    with pytest.raises(ValueError, match="replay_batch_size"):
        validate_config(invalid)
    invalid = deepcopy(config)
    invalid["captioner"]["generation"]["temperature"] = 0.8
    with pytest.raises(ValueError, match="exact likelihood replay"):
        validate_config(invalid)
