from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from .io import expand_env, load_yaml
from .constants import CYCLE_SFT_SELECTION


ENV_OVERRIDES = {
    "DUALISL_RUN_DIR": ("run", "output_dir"),
    "DUALISL_PAIRED_PATH": ("data", "paired_path"),
    "DUALISL_AUDIO_ONLY_PATH": ("data", "audio_only_path"),
    "DUALISL_CAPTION_ONLY_PATH": ("data", "caption_only_path"),
    "DUALISL_CAPTION_PYTHON": ("captioner", "python"),
    "DUALISL_CAPTION_MODEL": ("captioner", "model_path"),
    "DUALISL_CAPTION_ADAPTER": ("captioner", "adapter_path"),
    "DUALISL_TTS_PYTHON": ("tts", "python"),
    "DUALISL_TTS_MODEL": ("tts", "model_path"),
    "DUALISL_TTS_TOKENIZER": ("tts", "tokenizer_path"),
    "DUALISL_TTS_ADAPTER": ("tts", "adapter_path"),
    "DUALISL_CRITIC_PYTHON": ("critics", "python"),
    "DUALISL_WHISPER_MODEL": ("critics", "whisper_model"),
}

INTEGER_ENV_OVERRIDES = {
    "DUALISL_ROUNDS": ("training", "rounds"),
}


def _resolve_paths(value: Any, base: Path, key: str = "") -> Any:
    if isinstance(value, dict):
        return {name: _resolve_paths(item, base, name) for name, item in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(item, base, key) for item in value]
    if isinstance(value, str) and value and (
        key.endswith("_path") or key.endswith("_dir") or key.endswith("_root")
        or key in {"python", "whisper_model"}
    ):
        path = Path(value)
        return str(path if path.is_absolute() else (base / path).resolve())
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_config_tree(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Circular config inheritance: {chain}")
    raw = expand_env(load_yaml(path))
    parent = raw.pop("extends", None)
    current = _resolve_paths(raw, path.parent)
    if not parent:
        return current
    parent_path = Path(str(parent))
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return _deep_merge(_load_config_tree(parent_path, (*stack, path)), current)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = _load_config_tree(config_path)
    for variable, target in ENV_OVERRIDES.items():
        value = os.environ.get(variable)
        if value and value.strip():
            config[target[0]][target[1]] = str(Path(value).expanduser().resolve())
    for variable, target in INTEGER_ENV_OVERRIDES.items():
        value = os.environ.get(variable)
        if value and value.strip():
            try:
                config[target[0]][target[1]] = int(value)
            except ValueError as exc:
                raise ValueError(f"{variable} must be an integer") from exc
    if config.get("version") == 5 and not config.get("labeling", {}).get("cache_dir"):
        config.setdefault("labeling", {})["cache_dir"] = str(Path(config["run"]["output_dir"]) / "label_cache")
    if config.get("version") == 5:
        config["tts"]["codec_cache_dir"] = str(Path(config["run"]["output_dir"]) / "codec_cache/qwen3tts_12hz")
    config["_config_path"] = str(config_path)
    validate_config(config)
    return config


def _worker_section(config: dict[str, Any], role: str) -> None:
    section = config[role]
    for key in ("python", "worker_module"):
        if not str(section.get(key, "")).strip():
            raise ValueError(f"{role}.{key} is required")


def validate_config(config: dict[str, Any]) -> None:
    if config.get("version") != 5:
        raise ValueError("V5 requires version: 5; use configs/v5_*.yaml")
    if int(config.get("training", {}).get("round_offset", 0)) != 0:
        raise ValueError("V5 starts new runs; resume within the same V5 run instead of cross-version continuation")
    audio_reward = config.get("reward", {}).get("audio_only", {})
    weights = [float(audio_reward.get(k, -1)) for k in ("reconstruction_weight", "format_weight")]
    if any(not 0 <= v <= 1 for v in weights) or abs(sum(weights)-1) > 1e-9:
        raise ValueError("V5 reconstruction/format weights must be nonnegative and sum to one")
    for k in ("synthesis_batch_size", "rollout_batch_size"):
        if int(config.get("tts", {}).get("generation", {}).get(k, 1)) < 1:
            raise ValueError("TTS batch size must be positive")
    required = {"run", "distributed", "data", "captioner", "tts", "critics", "training", "reward"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing config sections: {missing}")
    for role in ("captioner", "tts", "critics"):
        _worker_section(config, role)
    training = config["training"]
    if int(training.get("rounds", 0)) < 1:
        raise ValueError("training.rounds must be >= 1")
    round_offset = int(training.get("round_offset", 0))
    if round_offset < 0:
        raise ValueError("training.round_offset must be >= 0")
    if int(training.get("group_size", 0)) != 4:
        raise ValueError("training.group_size must be exactly 4")
    if bool(training.get("warmstart", False)):
        raise ValueError("The aligned main protocol requires training.warmstart=false")
    if str(training.get("anchor_mode", "")) != "all_paired_once":
        raise ValueError("training.anchor_mode must be all_paired_once")
    for role in ("captioner", "tts"):
        phases = config[role].get("training", {}).get("phases", {})
        if int(phases.get("grpo", {}).get("epochs", 0)) != 1:
            raise ValueError(f"{role}.training.phases.grpo.epochs must be exactly 1")
        if int(phases.get("cycle_sft", {}).get("epochs", 0)) != 1:
            raise ValueError(f"{role}.training.phases.cycle_sft.epochs must be exactly 1")
    if str(config["tts"].get("worker_module", "")).endswith("qwen_voice_design"):
        tts_grpo = config["tts"].get("training", {}).get("phases", {}).get("grpo", {})
        if str(tts_grpo.get("schedule", "")) != "length_bucketed":
            raise ValueError("tts.training.phases.grpo.schedule must be length_bucketed")
        if int(tts_grpo.get("heartbeat_frames", 0)) < 1:
            raise ValueError("tts.training.phases.grpo.heartbeat_frames must be >= 1")
    data = config["data"]
    for key in ("paired_path", "audio_only_path", "caption_only_path"):
        if not str(data.get(key, "")).strip():
            raise ValueError(f"data.{key} is required")
    if not bool(data.get("strict_disjoint", True)):
        raise ValueError("data.strict_disjoint must be true")
    if int(data.get("max_records_per_role", 0)) < 0:
        raise ValueError("data.max_records_per_role must be >= 0")
    if int(config["tts"].get("num_codebooks", 16)) != 16:
        raise ValueError("Qwen3-TTS must use exactly 16 codec codebooks")
    reward = config["reward"]
    for loop in ("caption_only",):
        loop_reward = reward.get(loop, {})
        reconstruction_weight = float(loop_reward.get("reconstruction_weight", -1.0))
        counterfactual_weight = float(loop_reward.get("counterfactual_weight", -1.0))
        if reconstruction_weight < 0.0 or counterfactual_weight < 0.0:
            raise ValueError(f"reward.{loop} weights must be non-negative")
        if abs(reconstruction_weight + counterfactual_weight - 1.0) > 1.0e-9:
            raise ValueError(f"reward.{loop} weights must sum to 1")
    if int(reward.get("counterfactuals_per_candidate", 0)) < 1:
        raise ValueError("reward.counterfactuals_per_candidate must be >= 1")
    if float(reward.get("anchor_penalty_weight", -1.0)) < 0.0:
        raise ValueError("reward.anchor_penalty_weight must be non-negative")
    if float(reward.get("anchor_tolerance_z", -1.0)) < 0.0:
        raise ValueError("reward.anchor_tolerance_z must be non-negative")
    gate = reward.get("sft_gate", {})
    if reward.get("cycle_sft_selection") != CYCLE_SFT_SELECTION:
        raise ValueError("RewardV4 requires reward.cycle_sft_selection=semantic_top1")
    if gate.get("diagnostic_only") is not True:
        raise ValueError("RewardV4 requires reward.sft_gate.diagnostic_only=true")
    if not 0.5 < float(gate.get("target_precision", 0.0)) <= 1.0:
        raise ValueError("reward.sft_gate.target_precision must be in (0.5,1]")
    if not 0.0 < float(gate.get("min_recall", 0.0)) <= 1.0:
        raise ValueError("reward.sft_gate.min_recall must be in (0,1]")
    for loop in ("audio_only", "caption_only"):
        for name, value in reward.get("validity", {}).get(loop, {}).items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"reward.validity.{loop}.{name} must be in [0,1]")
    if reward.get("calibration", {}).get("method") != "round0_dual_counterfactual_zscore":
        raise ValueError(
            "reward.calibration.method must be round0_dual_counterfactual_zscore"
        )
    if bool(config.get("evaluation", {}).get("enabled", False)):
        raise ValueError("Built-in benchmark evaluation is intentionally disabled in v0.1")
    initial_calibration = str(reward.get("calibration", {}).get("initial_path", "")).strip()
    reward_anchor = reward.get("anchor")
    if round_offset > 0:
        for role in ("captioner", "tts"):
            if not str(config[role].get("adapter_path", "")).strip():
                raise ValueError(
                    f"{role}.adapter_path is required when training.round_offset > 0"
                )
        if not initial_calibration:
            raise ValueError(
                "reward.calibration.initial_path is required when training.round_offset > 0"
            )
        required_anchor_keys = {"captioner_adapter_path", "tts_adapter_path"}
        if not isinstance(reward_anchor, dict) or not required_anchor_keys.issubset(reward_anchor):
            raise ValueError(
                "RewardV2 continuation requires reward.anchor.captioner_adapter_path "
                "and reward.anchor.tts_adapter_path so the original frozen run anchor is explicit"
            )
    elif initial_calibration:
        raise ValueError(
            "reward.calibration.initial_path requires training.round_offset > 0"
        )
    elif reward_anchor is not None:
        raise ValueError("reward.anchor is only valid when training.round_offset > 0")
    caption_generation = config["captioner"].get("generation", {})
    reuse = config["tts"].get("generation", {}).get("reference_replay_reuse", False)
    if config["tts"].get("generation", {}).get("rollout_schedule", "round_robin") not in ("round_robin", "previous_round_lpt"):
        raise ValueError("tts.generation.rollout_schedule must be round_robin or previous_round_lpt")
    if not isinstance(reuse, bool):
        raise ValueError("tts.generation.reference_replay_reuse must be a boolean")
    if int(caption_generation.get("max_attempts_per_candidate", 1)) != 1:
        raise ValueError("Captioner candidates must be sampled once without retry")
    group_size = int(training["group_size"])
    rollout_batch_size = int(caption_generation.get("rollout_batch_size", 1))
    replay_batch_size = int(config["captioner"].get("training", {}).get("replay_batch_size", 1))
    for name, value in (
        ("captioner.generation.rollout_batch_size", rollout_batch_size),
        ("captioner.training.replay_batch_size", replay_batch_size),
    ):
        if not 1 <= value <= group_size:
            raise ValueError(f"{name} must be in [1, training.group_size]")
    if rollout_batch_size > 1 or replay_batch_size > 1:
        sampling = (
            float(caption_generation.get("temperature", 1.0)),
            float(caption_generation.get("top_p", 1.0)),
            int(caption_generation.get("top_k", 0)),
        )
        if sampling != (1.0, 1.0, 0):
            raise ValueError(
                "Batched Captioner rollout/replay currently requires "
                "temperature=1.0, top_p=1.0 and top_k=0 for exact likelihood replay"
            )
    distributed = config["distributed"]
    if bool(distributed.get("enabled", False)):
        if int(distributed.get("world_size", 0)) < 2:
            raise ValueError("distributed.world_size must be >= 2 when enabled")
        if str(distributed.get("backend", "nccl")) != "nccl":
            raise ValueError("Real multi-GPU training requires NCCL")


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(config)
    value.pop("_config_path", None)
    return value
