from __future__ import annotations

import json

import pytest

from dual_isl_train.config import ENV_OVERRIDES, INTEGER_ENV_OVERRIDES, load_config
from scripts.reward_v4_run_guard import MARKER, check_run_directory, run_contract


@pytest.mark.parametrize("mode,stem,rounds,limit", [("train", "train", 10, None), ("smoke", "gpu_smoke", 1, 8)])
def test_v4_configs_start_from_base_with_fresh_calibration(monkeypatch, mode, stem, rounds, limit):
    for variable in (*ENV_OVERRIDES, *INTEGER_ENV_OVERRIDES):
        monkeypatch.delenv(variable, raising=False)
    config = load_config(f"configs/{stem}_8gpu_h100_midasheng_reward_v4.yaml")
    assert config["training"]["rounds"] == rounds
    assert config["data"].get("max_records_per_role") == limit
    assert config["captioner"]["lora"]["r"] == 4
    assert "reward_v4" in config["run"]["output_dir"]
    contract = run_contract(config, mode, rounds)
    assert contract["semantic_admission_version"] == "safe_semantic_input_v1"
    assert contract["cycle_sft_selection"] == "semantic_top1"
    assert contract["sft_gate_diagnostic_only"] is True
    assert contract["framework_version"] == "DualISL-Train-RewardV4-0.4"
    assert "scripts/midasheng_captioner_candidate.py" in contract["code_sha256"]
    config["reward"]["calibration"]["initial_path"] = "/old/calibration.json"
    with pytest.raises(ValueError, match="fresh"):
        run_contract(config, mode, rounds)


def test_run_guard_rejects_old_or_changed_runs_without_writing(tmp_path):
    project = tmp_path / "v4"
    old = tmp_path / "v2/runs/old"
    old.mkdir(parents=True)
    (old / "run_state.json").write_text("old-state")
    with pytest.raises(ValueError, match="dedicated directory"):
        check_run_directory(old, {}, project=project)
    assert list(p.name for p in old.iterdir()) == ["run_state.json"]
    root = project / "runs/new"
    assert check_run_directory(root, {}, project=project) == "train"
    assert not root.exists()
    root.mkdir(parents=True)
    (root / "run_state.json").write_text("{}")
    with pytest.raises(ValueError, match="matching V4 identity"):
        check_run_directory(root, {}, project=project)
    (root / MARKER).write_text(json.dumps({"version": "v4"}))
    assert check_run_directory(root, {"version": "v4"}, project=project) == "resume"
    with pytest.raises(ValueError, match="identity changed"):
        check_run_directory(root, {"version": "modified"}, project=project)
    with pytest.raises(ValueError, match="nested"):
        check_run_directory(root / "child", {}, project=project)


def test_run_guard_resolves_symlinks_before_checks(tmp_path):
    project = tmp_path / "v4"
    (project / "runs").mkdir(parents=True)
    old = tmp_path / "v2/run"
    old.mkdir(parents=True)
    (project / "runs/link").symlink_to(old, target_is_directory=True)
    with pytest.raises(ValueError, match="dedicated directory"):
        check_run_directory(project / "runs/link", {}, project=project)


def test_copied_v3_run_identity_is_not_a_v4_resume(tmp_path):
    project = tmp_path / "v4"
    root = project / "runs/copied_v3"
    root.mkdir(parents=True)
    (root / "reward_v3_run_identity.json").write_text("{}")
    (root / "run_state.json").write_text("{}")
    with pytest.raises(ValueError, match="matching V4 identity"):
        check_run_directory(root, {}, project=project)
    assert not (root / MARKER).exists()


def test_launcher_rejects_non_v4_sft_policy(monkeypatch):
    for variable in (*ENV_OVERRIDES, *INTEGER_ENV_OVERRIDES):
        monkeypatch.delenv(variable, raising=False)
    config = load_config("configs/train_8gpu_h100_midasheng_reward_v4.yaml")
    config["reward"]["sft_gate"]["diagnostic_only"] = False
    with pytest.raises(ValueError, match="diagnostic_only"):
        run_contract(config, "train", 10)
