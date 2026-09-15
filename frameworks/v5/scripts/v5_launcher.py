"""V5 entrypoint: CPU checks, real GPU smoke, training and immutable resume."""
from __future__ import annotations
import argparse
import atexit
import importlib
import json
import os
from pathlib import Path
import subprocess
import stat
import sys

from dual_isl_train.config import load_config, validate_config, ENV_OVERRIDES, INTEGER_ENV_OVERRIDES
from dual_isl_train.io import load_yaml

ROOT = Path(__file__).resolve().parents[1]


def share_run_outputs(root):
    """Also cover third-party checkpoint writers that create files with mode 0600."""
    if os.environ.get("DUALISL_SHARED_WRITABLE") != "1" or not root.is_dir() or root.is_symlink():
        return
    failed = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(directory) / name).is_symlink()]
        for path in [Path(directory), *(Path(directory) / name for name in files)]:
            try:
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode):
                    continue
                if stat.S_ISDIR(info.st_mode):
                    bits = 0o777
                elif stat.S_ISREG(info.st_mode):
                    bits = 0o666
                else:
                    continue
                mode = stat.S_IMODE(info.st_mode)
                if mode | bits != mode:
                    path.chmod(mode | bits)
            except OSError:
                failed += 1
    if failed:
        print(f"Shared permissions: {failed} paths could not be updated under {root}", file=sys.stderr)


def runtime_config(mode, run_dir):
    if mode == "resume":
        config_path = run_dir / "resolved_config.yaml"
        if not config_path.is_file():
            raise ValueError("No V5 resolved_config.yaml to resume")
        cfg = load_yaml(config_path)
        if Path(cfg["run"]["output_dir"]).resolve() != run_dir:
            raise ValueError("Resume must use the original run path")
        # Resume does not apply current shell overrides to the frozen training configuration.
        for name in ("DUALISL_ROUNDS", "DUALISL_PAIRED_PATH", "DUALISL_AUDIO_ONLY_PATH", "DUALISL_CAPTION_ONLY_PATH",
                     "DUALISL_CAPTION_ADAPTER", "DUALISL_TTS_ADAPTER"):
            os.environ.pop(name, None)
        validate_config(cfg)
        return cfg
    path = Path(os.environ.get("DUALISL_V5_CONFIG", ROOT / "configs" / (
        "v5_gpu_smoke.yaml" if mode == "gpu-smoke" else "v5_midasheng_4gpu.yaml")))
    cfg = load_config(path)
    cfg["run"]["output_dir"] = str(run_dir)
    cfg["labeling"]["cache_dir"] = str(run_dir / "label_cache")
    cfg["tts"]["codec_cache_dir"] = str(run_dir / "codec_cache/qwen3tts_12hz")
    if mode == "gpu-smoke":
        cfg["data"]["max_records_per_role"] = 2
        cfg["training"]["rounds"] = 1
    # A single node uses the explicitly selected visible GPUs. Never probe CUDA during check.
    ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")
    cfg["distributed"].update(enabled=len(ids) > 1, world_size=len(ids))
    if "DUALISL_INFERENCE_BATCH" in os.environ:
        size = int(os.environ["DUALISL_INFERENCE_BATCH"])
        cfg["captioner"]["generation"]["rollout_batch_size"] = size
        cfg["tts"]["generation"].update(rollout_batch_size=size, synthesis_batch_size=size)
    if "DUALISL_PAIRED_ANCHOR" in os.environ:
        value = os.environ["DUALISL_PAIRED_ANCHOR"]
        if value not in ("0", "1"):
            raise ValueError("DUALISL_PAIRED_ANCHOR must be 0 or 1")
        cfg["training"]["paired_anchor_enabled"] = value == "1"
    validate_config(cfg)
    return cfg


def check(cfg, dependencies=True):
    lab = cfg["labeling"]
    backend = json.loads(Path(lab["config_path"]).read_text())
    expert = backend["backends"]["experts"]
    paths = {r: cfg[r]["python"] for r in ("captioner", "tts", "critics")}
    paths.update(asr=lab["asr"]["python"], experts=expert["python"], rate=expert["rate_python"], voxlect=expert["voxlect_python"])
    required = {r + ".python": p for r, p in paths.items()}
    required.update({r + ".model_path": cfg[r]["model_path"] for r in ("captioner", "tts")})
    required.update(tts_tokenizer=cfg["tts"]["tokenizer_path"], whisper=cfg["critics"]["whisper_model"],
        asr_model=lab["asr"]["model_path"], emotion=expert["emotion_model_dir"], accent=expert["accent_model_dir"],
        accent_base=lab["accent_base_model_dir"], labeling_source=lab["source_root"], hf_home=expert["hf_home"])
    missing = {k: v for k, v in required.items() if not Path(v).exists()}
    if missing:
        raise ValueError("Missing runtime paths: " + json.dumps(missing))
    deps = {}
    if dependencies:
        for role, py in paths.items():
            p = subprocess.run([py, "-c", "import yaml,jsonschema,requests"], capture_output=True, text=True)
            deps[role] = p.returncode == 0
        if not all(deps.values()):
            raise ValueError("Missing small runtime dependencies in " + str([k for k, v in deps.items() if not v]) +
                             "; run bash scripts/setup_v5_runtime.sh")
    sys.path.insert(0, str(Path(lab["source_root"]) / "Experiment"))
    credentials = importlib.import_module("labeling2.run_with_local_api_key")
    # Validate the existing independent credential sources without printing their values.
    for model in ("gemini", "qwen35"):
        if not credentials.load_key(model):
            raise ValueError("Missing " + model + " key")
    from dual_isl_train.labeling import demo_credentials
    if not (os.getenv("V5_JUDGE_URL") and os.getenv("V5_JUDGE_KEY")):
        url, key = demo_credentials(lab["judge_demo_path"])
        if not url or not key:
            raise ValueError("Missing GPT judge credential")
    counts = {r: sum(bool(l.strip()) for l in Path(cfg["data"][r + "_path"]).open())
              for r in ("paired", "audio_only", "caption_only")}
    return {"cpu_preflight": "passed", "gpu_execution": "not_performed", "source_records": counts,
        "world_size": cfg["distributed"]["world_size"], "rounds": cfg["training"]["rounds"],
        "paired_anchor_enabled": cfg["training"]["paired_anchor_enabled"], "run_dir": cfg["run"]["output_dir"],
        "inference_batch": cfg["captioner"]["generation"]["rollout_batch_size"], "dependencies": deps}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("check", "gpu-smoke", "train", "resume", "verify"))
    p.add_argument("run_dir", type=Path)
    args = p.parse_args()
    root = args.run_dir.resolve()
    if args.mode in ("train", "gpu-smoke", "resume"):
        atexit.register(share_run_outputs, root)
    os.environ["DUALISL_RUN_DIR"] = str(root)
    if args.mode == "verify":
        from scripts.verify_v5 import verify
        report = verify(root)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(0 if report["ok"] else 1)
    cfg = runtime_config(args.mode, root)
    # Workers reload the resolved file; keep its values authoritative, including during resume.
    for name in (*ENV_OVERRIDES, *INTEGER_ENV_OVERRIDES):
        if name != "DUALISL_RUN_DIR":
            os.environ.pop(name, None)
    if args.mode in ("train", "gpu-smoke") and (root / "run_state.json").exists():
        raise ValueError("Run already exists; use resume for this path or choose a new run directory")
    print(json.dumps(check(cfg), ensure_ascii=False, indent=2), flush=True)
    if args.mode == "check":
        return
    visible = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(","))
    if visible != cfg["distributed"]["world_size"]:
        raise ValueError("Resume needs the original number of visible GPUs")
    from dual_isl_train.orchestrator import DualISLOrchestrator
    result = DualISLOrchestrator(cfg).train(resume_only=args.mode == "resume")
    print(json.dumps(result, indent=2), flush=True)
    from scripts.verify_v5 import verify
    report = verify(root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
