from __future__ import annotations

import subprocess
from pathlib import Path

from dual_isl_train.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_new_captioner_configs_are_independent_base_reward_v2_runs():
    branches = {
        "midasheng": (
            "scripts.midasheng_0p6b_captioner_candidate",
            "MiDashengLM-0.6B-FP32",
            "midasheng-0p6b-captioner",
        ),
        "qwen2": (
            "scripts.qwen2_5_omni_3b_captioner_candidate",
            "Qwen2.5-Omni-3B",
            "qwen2_5-omni-3b-captioner",
        ),
    }
    config_paths = {
        "midasheng": (
            "configs/gpu_smoke_8gpu_h100_midasheng_0p6b_reward_v2.yaml",
            "configs/train_8gpu_h100_midasheng_0p6b_reward_v2.yaml",
        ),
        "qwen2": (
            "configs/gpu_smoke_8gpu_h100_qwen2_5_omni_3b_reward_v2.yaml",
            "configs/train_8gpu_h100_qwen2_5_omni_3b_reward_v2.yaml",
        ),
    }
    for branch, paths in config_paths.items():
        worker, model_name, env_name = branches[branch]
        for config_path in paths:
            config = load_config(ROOT / config_path)
            assert config["distributed"] == {
                **config["distributed"], "enabled": True, "world_size": 8, "backend": "nccl",
            }
            assert config["captioner"]["worker_module"] == worker
            assert config["captioner"]["model_path"].endswith(model_name)
            assert f"envs/{env_name}/bin/python" in config["captioner"]["python"]
            assert config["captioner"]["adapter_path"] == ""
            assert config["tts"]["adapter_path"] == ""
            assert config["captioner"]["generation"]["rollout_batch_size"] == 1
            assert config["captioner"]["training"]["replay_batch_size"] == 1
            assert config["training"]["round_offset"] == 0
            assert config["reward"]["calibration"]["method"] == "round0_dual_counterfactual_zscore"
            assert "qwen3tts_12hz" in config["tts"]["codec_cache_dir"]


def test_new_launchers_have_valid_shell_syntax_and_worker_contract_markers(tmp_path, monkeypatch):
    launchers = [
        "scripts/setup_new_captioner_envs.sh",
        "scripts/run_reward_v2_new_captioner_h100_single_node.sh",
        "scripts/run_midasheng_0p6b_reward_v2_h100_single_node.sh",
        "scripts/run_qwen2_5_omni_3b_reward_v2_h100_single_node.sh",
        "scripts/run_midasheng_reward_v2_continuation_r9_h100_single_node.sh",
    ]
    for relative in launchers:
        result = subprocess.run(["bash", "-n", str(ROOT / relative)], check=False)
        assert result.returncode == 0, relative
    worker = (ROOT / "scripts/qwen2_5_omni_3b_captioner_candidate.py").read_text(encoding="utf-8")
    assert "Qwen2_5OmniThinkerForConditionalGeneration" in worker
    assert "Qwen2_5OmniProcessor" in worker
    assert '"talker_loaded": False' in worker
    midasheng_worker = (ROOT / "scripts/midasheng_0p6b_captioner_candidate.py").read_text(encoding="utf-8")
    assert 'os.environ["HF_MODULES_CACHE"]' in midasheng_worker
    from scripts.midasheng_0p6b_captioner_candidate import _configure_rank_local_hf_modules_cache
    monkeypatch.setenv("LOCAL_RANK", "3")
    cache = _configure_rank_local_hf_modules_cache({"run": {"output_dir": str(tmp_path)}})
    assert cache == str(tmp_path / ".hf_modules_cache" / "rank_003")
    assert Path(cache).is_dir()
