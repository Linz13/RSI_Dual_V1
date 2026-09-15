"""Read-only preflight / locked reservation for the Qwen2.5-Omni-3B V4 four-GPU ten-round base launcher."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dual_isl_train.config import load_config, public_config, validate_config
from dual_isl_train.constants import FRAMEWORK_VERSION, CYCLE_SFT_SELECTION
from dual_isl_train.semantic_admission import ADMISSION_VERSION


MARKER = "reward_v4_run_identity.json"
PROJECT = Path(__file__).resolve().parents[1]


def run_contract(config: dict, mode: str, rounds: int) -> dict:
    if FRAMEWORK_VERSION != "DualISL-Train-RewardV4-0.4":
        raise ValueError("RewardV4 launcher loaded a different framework package")
    if mode != "train" or rounds != 10:
        raise ValueError("Unexpected launcher mode")
    distributed = config["distributed"]
    if (not distributed.get("enabled") or distributed["world_size"] != 4
            or distributed["backend"] != "nccl"):
        raise ValueError("RewardV4 Qwen2.5-Omni-3B requires 4-rank NCCL")
    if config["training"].get("round_offset", 0) != 0 or config["training"]["rounds"] != rounds:
        raise ValueError("RewardV4 base run requires round_offset=0 and the requested round count")
    if config["data"].get("max_records_per_role") is not None:
        raise ValueError("Qwen V4 ten rounds requires full data")
    for role, expected in (("paired", 39), ("audio_only", 184), ("caption_only", 196)):
        path = Path(config["data"][role + "_path"]).resolve()
        if path != (PROJECT.parent / "training_data_v2" / (role + ".jsonl")).resolve():
            raise ValueError("Unexpected dataset override")
        if sum(bool(line.strip()) for line in path.open()) != expected:
            raise ValueError("Unexpected source dataset count")
    if config["training"]["group_size"] != 4 or config["training"]["warmstart"]:
        raise ValueError("Expected four candidates and no warmstart")
    generation = config["tts"].get("generation", {})
    if generation.get("reference_replay_reuse") is not True or generation.get("rollout_schedule") != "previous_round_lpt":
        raise ValueError("Both validated TTS optimizations must be enabled")
    expected_python = PROJECT.parent.parent / "miniconda3/envs/qwen2_5-omni-3b-captioner/bin/python"
    if Path(config["captioner"]["python"]).resolve() != expected_python.resolve():
        raise ValueError("Unexpected Qwen Captioner interpreter")
    for role in ("captioner", "tts"):
        if config[role].get("adapter_path") or config["reward"].get("anchor", {}).get(f"{role}_adapter_path"):
            raise ValueError("RewardV4 base run must start with empty current and anchor adapters")
    if config["reward"]["calibration"].get("initial_path"):
        raise ValueError("RewardV4 must fit a fresh round-0 calibration")
    if config["reward"]["calibration"]["method"] != "round0_dual_counterfactual_zscore":
        raise ValueError("Unexpected calibration method")
    validate_config(config)
    if config["captioner"]["worker_module"] != "scripts.qwen2_5_omni_3b_captioner_candidate":
        raise ValueError("Configured Captioner is not Qwen2.5-Omni-3B")
    shared_models = PROJECT.parent / "models"
    for role, name in (("captioner", "Qwen2.5-Omni-3B"),
                       ("tts", "Qwen3-TTS-12Hz-1.7B-VoiceDesign")):
        if Path(config[role]["model_path"]).resolve() != (shared_models / name).resolve():
            raise ValueError(f"Unexpected {role} model override for this base launcher")
    files = sorted((PROJECT / "dual_isl_train").rglob("*.py"))
    files += sorted((PROJECT / "dual_isl_train").rglob("*.json"))
    files += [PROJECT / "scripts/qwen2_5_omni_3b_captioner_candidate.py",
              PROJECT / "scripts/qwen25_reward_v4_4gpu_run_guard.py",
              PROJECT / "scripts/run_qwen25_v4_10rounds_4gpu.sh",
              PROJECT / "scripts/verify_tts_reference_reuse.py"]
    return {
        "framework_version": FRAMEWORK_VERSION,
        "semantic_admission_version": ADMISSION_VERSION,
        "cycle_sft_selection": CYCLE_SFT_SELECTION,
        "sft_gate_diagnostic_only": True,
        "mode": mode, "rounds": rounds,
        "config": public_config(config),
        "code_sha256": {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in files},
    }


def check_run_directory(path: Path, contract: dict, *, project: Path = PROJECT) -> str:
    path = path.resolve()
    runs = (project / "runs").resolve()
    if path == runs or not path.is_relative_to(runs):
        raise ValueError(f"V4 output must be a dedicated directory under {runs}")
    for parent in path.parents:
        if parent == runs:
            break
        if (parent / "run_state.json").exists() or (parent / MARKER).exists():
            raise ValueError("Refusing a run nested inside another run")
    if not path.exists():
        return "train"
    if not path.is_dir():
        raise ValueError("Run path is not a directory")
    marker = path / MARKER
    if marker.exists():
        if json.loads(marker.read_text()) != contract:
            raise ValueError("Refusing reuse: V4 code/config/run identity changed; choose a new run directory")
    elif any(p.name != ".launcher.lock" for p in path.iterdir()):
        raise ValueError("Refusing non-empty directory without a matching V4 identity")
    if (path / "latest.json").exists() and json.loads((path / "latest.json").read_text()).get("round", -1) >= 9:
        raise ValueError("Ten rounds already complete; use a new run for a new experiment")
    if (path / "run_state.json").exists():
        if not marker.is_file():
            raise ValueError("Refusing resume without V4 identity")
        return "resume"
    if any(p.name not in {MARKER, ".launcher.lock"} for p in path.iterdir()):
        raise ValueError("Refusing partial run without run_state.json; choose a new run directory")
    return "train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("train",), required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--reserve", action="store_true", help="Only under the launcher's flock")
    args = parser.parse_args()
    config = load_config(args.config)
    contract = run_contract(config, args.mode, args.rounds)
    root = Path(config["run"]["output_dir"])
    action = check_run_directory(root, contract)
    if args.reserve and not (root / MARKER).exists():
        # The shell owns the directory lock. Exclusive create still prevents overwrite.
        with (root / MARKER).open("x") as handle:
            json.dump(contract, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print(action)


if __name__ == "__main__":
    main()
