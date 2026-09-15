from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from labeling2.api_stage import finalize_api_stage
    from labeling2.manifest import load_manifest, sample_json, select_samples
    from labeling2.open_resolution import finalize_open_labels, resolve_open_labels
    from labeling2.pipeline import ensure_run_metadata, finalize, make_run_dirs, run_experts, run_general
except ModuleNotFoundError:  # package import: ``Experiment.labeling2.run_labeling``
    from .api_stage import finalize_api_stage
    from .manifest import load_manifest, sample_json, select_samples
    from .open_resolution import finalize_open_labels, resolve_open_labels
    from .pipeline import ensure_run_metadata, finalize, make_run_dirs, run_experts, run_general

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "default_config.json"
PRESET_MANIFESTS = {
    "test-labeling-smoke": HERE.parent / "data/test_labeling/v1/manifests/smoke_manifest.jsonl",
    "test-labeling-full": HERE.parent / "data/test_labeling/v1/manifests/labeling_manifest.jsonl",
}
PRESET_RUN_DIRS = {
    "test-labeling-smoke": HERE / "runs/test_labeling_v1/smoke",
    "test-labeling-full": HERE / "runs/test_labeling_v1/full",
}
MODEL_CHOICES = ("gemini", "qwen35", "qwen3_captioner", "kimi_audio", "step_audio_r1_1")
EXPERT_CHOICES = ("volume", "emotion", "accent_en", "accent_zh", "rate_en", "rate_zh")


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config must be a JSON object")
    # Paths that differ between servers can be supplied without editing the
    # checked-in JSON configuration. CUDA_VISIBLE_DEVICES remains the GPU
    # selector for each independent process.
    overrides = {
        "LABELING2_QWEN3_PYTHON": ("backends", "qwen3_captioner", "python"),
        "LABELING2_QWEN3_MODEL_DIR": ("backends", "qwen3_captioner", "model_dir"),
        "LABELING2_KIMI_PYTHON": ("backends", "kimi_audio", "python"),
        "LABELING2_KIMI_MODEL_DIR": ("backends", "kimi_audio", "model_dir"),
        "LABELING2_KIMI_TOKENIZER_DIR": ("backends", "kimi_audio", "glm4_tokenizer_path"),
        "LABELING2_STEP_PYTHON": ("backends", "step_audio_r1_1", "python"),
        "LABELING2_STEP_API_URL": ("backends", "step_audio_r1_1", "api_url"),
    }
    for env_name, keys in overrides.items():
        replacement = os.environ.get(env_name)
        if not replacement:
            continue
        target = value
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = replacement
    return value


def redact_config(value: Any) -> Any:
    """Keep run snapshots useful without copying literal credentials."""
    secret_values = {secret for secret in os.environ.values() if secret and len(secret) >= 16}

    def walk(item: Any, key: str = "") -> Any:
        lowered = key.casefold()
        if lowered in {"api_key", "access_token", "secret", "password"} or any(item == secret for secret in secret_values):
            return "***REDACTED***"
        if isinstance(item, dict):
            return {str(k): walk(v, str(k)) for k, v in item.items()}
        if isinstance(item, list):
            return [walk(v, key) for v in item]
        return item

    return walk(value)


def python_environment_check(python_path: str | None, modules: tuple[str, ...]) -> tuple[bool, str]:
    if not python_path or not Path(python_path).is_file():
        return False, "python_not_found"
    code = (
        "import importlib.util; missing=[m for m in "
        + repr(list(modules))
        + " if importlib.util.find_spec(m) is None]; print(','.join(missing))"
    )
    try:
        result = subprocess.run([python_path, "-c", code], capture_output=True, text=True, timeout=20, check=False)
    except Exception as exc:
        return False, type(exc).__name__
    if result.returncode != 0:
        return False, f"exit_{result.returncode}"
    missing = result.stdout.strip()
    return not missing, missing or "ok"


def resolve_scope(args: argparse.Namespace, *, require_run_dir: bool) -> tuple[Path, Path | None, str | None]:
    manifest_arg = getattr(args, "manifest", None)
    preset = getattr(args, "preset", None)
    if bool(manifest_arg) == bool(preset):
        raise ValueError("provide exactly one of --manifest or --preset")
    if preset:
        manifest = PRESET_MANIFESTS[preset].resolve()
        default_run_dir = PRESET_RUN_DIRS[preset].resolve()
    else:
        manifest = manifest_arg.expanduser().resolve()
        default_run_dir = None
    if require_run_dir:
        run_arg = getattr(args, "run_dir", None)
        run_dir = (run_arg.expanduser().resolve() if run_arg else default_run_dir)
        if run_dir is None:
            raise ValueError("--run-dir is required when --manifest is used")
    else:
        run_arg = getattr(args, "run_dir", None)
        run_dir = run_arg.expanduser().resolve() if run_arg else default_run_dir
    return manifest, run_dir, preset


def selected_backends(values: list[str] | None, choices: tuple[str, ...]) -> tuple[str, ...]:
    if not values:
        return choices
    unknown = sorted(set(values) - set(choices))
    if unknown:
        raise ValueError(f"unknown backend(s): {', '.join(unknown)}")
    return tuple(dict.fromkeys(values))


def preflight(
    manifest: Path,
    config: dict[str, Any],
    *,
    check_audio: bool = True,
    models: tuple[str, ...] | None = None,
    experts: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    samples = load_manifest(manifest, check_audio=check_audio)
    if models is None:
        models = MODEL_CHOICES
    if experts is None:
        experts = EXPERT_CHOICES
    checks: list[dict[str, Any]] = []
    for name in ("gemini", "qwen35"):
        if name not in models:
            continue
        item = config.get("backends", {}).get(name, {})
        if item.get("enabled", True):
            env_name = item.get("api_key_env", "")
            fallback_names = (env_name, "OPENAI_API_KEY", "SHELL_API_KEY") if name == "qwen35" else (env_name, "OPENAI_API_KEY")
            available = next((candidate for candidate in fallback_names if candidate and os.environ.get(candidate)), "")
            checks.append({"backend": name, "check": "api_key", "ok": bool(available), "detail": env_name})
    for name in ("qwen3_captioner", "kimi_audio", "step_audio_r1_1"):
        if name not in models:
            continue
        item = config.get("backends", {}).get(name, {})
        if not item and name == "step_audio_r1_1":
            item = config.get("backends", {}).get("step_audio", {})
        if item.get("enabled", True):
            python_path = item.get("python")
            checks.append({"backend": name, "check": "python", "ok": bool(python_path and Path(python_path).is_file()), "detail": python_path})
            modules = {
                "qwen3_captioner": ("torch", "transformers"),
                "kimi_audio": ("torch", "soundfile"),
                "step_audio_r1_1": ("requests",),
            }[name]
            module_ok, module_detail = python_environment_check(python_path, modules)
            checks.append({"backend": name, "check": "python_modules", "ok": module_ok, "detail": module_detail})
            if item.get("model_dir"):
                model_dir = Path(item["model_dir"])
                checks.append({"backend": name, "check": "model_dir", "ok": model_dir.is_dir(), "detail": str(model_dir)})
    expert_config = config.get("backends", {}).get("experts", {})
    expert_keys = set()
    if any(name in experts for name in ("volume", "emotion")):
        expert_keys.add("python")
    if any(name in experts for name in ("rate_en", "rate_zh")):
        expert_keys.add("rate_python")
    if "accent_zh" in experts:
        expert_keys.add("voxlect_python")
    for key in ("python", "rate_python", "voxlect_python"):
        if key not in expert_keys:
            continue
        value = expert_config.get(key)
        checks.append({"backend": f"experts.{key}", "check": "python", "ok": bool(value and Path(value).is_file()), "detail": value})
        modules = {
            "python": ("librosa", "funasr", "soundfile"),
            "rate_python": ("g2p", "brouhaha", "pyannote", "pypinyin", "whisper", "soundfile"),
            "voxlect_python": ("torch", "soundfile"),
        }[key]
        module_ok, module_detail = python_environment_check(value, modules)
        checks.append({"backend": f"experts.{key}", "check": "python_modules", "ok": module_ok, "detail": module_detail})
    if "emotion" in experts:
        value = expert_config.get("emotion_model_dir")
        checks.append({
            "backend": "experts.emotion",
            "check": "model_dir",
            "ok": bool(value and Path(value).is_dir()),
            "detail": value,
        })
    if "accent_en" in experts:
        value = expert_config.get("accent_model_dir")
        checks.append({
            "backend": "experts.accent_en",
            "check": "model_dir",
            "ok": bool(value and Path(value).is_dir()),
            "detail": value,
        })
    if "accent_zh" in experts:
        voxlect_source = HERE.parent / "acc_model_pool/Caption_Bench/third_party/voxlect"
        checks.append({"backend": "experts", "check": "voxlect_source", "ok": voxlect_source.is_dir(), "detail": str(voxlect_source)})
        hf_home = expert_config.get("hf_home")
        checks.append({
            "backend": "experts.accent_zh",
            "check": "hf_cache",
            "ok": bool(hf_home and (Path(hf_home) / "hub/models--tiantiaf--voxlect-mandarin-cantonese-dialect-whisper-small").is_dir()),
            "detail": hf_home,
        })
    step = config.get("backends", {}).get("step_audio_r1_1", {})
    if not step:
        step = config.get("backends", {}).get("step_audio", {})
    if "step_audio_r1_1" in models and step.get("enabled", True):
        url = str(step.get("api_url", ""))
        checks.append({"backend": "step_audio_r1_1", "check": "service_configured", "ok": bool(url), "detail": url})
        try:
            parsed = urllib.parse.urlparse(url)
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            with socket.create_connection((parsed.hostname, port), timeout=1.0):
                reachable = True
        except Exception:
            reachable = False
        checks.append({"backend": "step_audio_r1_1", "check": "service_reachable", "ok": reachable, "detail": url})
    summary = {"manifest": str(manifest.resolve()), "samples": len(samples), "checks": checks, "missing_or_invalid": [row for row in checks if not row["ok"]]}
    return summary


def finalize_samples(manifest: Path, run_dir: Path, start_index: int, max_samples: int):
    """Prefer the immutable run snapshot when finalizing an existing run."""
    snapshot = run_dir / "state" / "manifest_snapshot.jsonl"
    if start_index == 0 and max_samples == 0 and snapshot.is_file():
        return load_manifest(snapshot)
    return select_samples(load_manifest(manifest), start_index, max_samples)


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    manifest, run_dir, _preset = resolve_scope(args, require_run_dir=True)
    if args.stage == "general":
        models = selected_backends(args.models, MODEL_CHOICES)
        experts = tuple()
    elif args.stage == "experts":
        models = tuple()
        experts = selected_backends(args.experts, EXPERT_CHOICES)
    else:
        models = selected_backends(args.models, MODEL_CHOICES)
        experts = selected_backends(args.experts, EXPERT_CHOICES)
    all_samples = load_manifest(manifest, check_audio=True)
    samples = select_samples(all_samples, args.start_index, args.max_samples)
    assert run_dir is not None
    snapshot = run_dir / "state" / "manifest_snapshot.jsonl"
    existing_outputs = (
        list((run_dir / "raw_predictions").glob("*.jsonl"))
        + list((run_dir / "expert_predictions").glob("*.jsonl"))
        + list((run_dir / "final").glob("*.jsonl"))
    ) if run_dir.exists() else []
    if run_dir.exists() and not args.resume:
        if existing_outputs:
            raise RuntimeError(f"run directory already contains predictions; use --resume: {run_dir}")
    if args.resume and existing_outputs and not snapshot.is_file():
        raise RuntimeError(f"existing run has outputs but no manifest snapshot: {run_dir}")
    if args.resume and snapshot.is_file():
        existing = snapshot.read_text(encoding="utf-8")
        current = "".join(json.dumps(sample_json(s), ensure_ascii=False) + "\n" for s in samples)
        if existing != current:
            raise RuntimeError("manifest snapshot differs from the existing run; use a new --run-dir")
    make_run_dirs(run_dir)
    ensure_run_metadata(run_dir)
    state_config = run_dir / "state" / "config.json"
    state_config.write_text(json.dumps(redact_config(config), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "state" / "manifest_snapshot.jsonl").write_text("".join(json.dumps(sample_json(s), ensure_ascii=False) + "\n" for s in samples), encoding="utf-8")
    preflight_result = preflight(manifest, config, models=models, experts=experts)
    (run_dir / "state" / "preflight.json").write_text(json.dumps(preflight_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if preflight_result["missing_or_invalid"] and not args.allow_partial:
        raise RuntimeError(f"Preflight failed; use --allow-partial to continue: {preflight_result['missing_or_invalid']}")
    if args.stage in {"all", "general"}:
        run_general(samples, run_dir, config, resume=args.resume, models=models)
    if args.stage in {"all", "experts"}:
        run_experts(
            samples,
            run_dir,
            config,
            resume=args.resume,
            experts=experts,
            workers=args.expert_workers,
        )
    if args.stage in {"all", "finalize"}:
        print(json.dumps(finalize(samples, run_dir), ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable labeling2 audio annotation pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest", type=Path)
    common.add_argument("--preset", choices=tuple(PRESET_MANIFESTS))
    common.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    common.add_argument("--run-dir", type=Path)
    common.add_argument("--max-samples", type=int, default=0)
    common.add_argument("--start-index", type=int, default=0)
    common.add_argument("--resume", action="store_true")
    common.add_argument("--allow-partial", action="store_true")
    common.add_argument("--models", nargs="+", choices=MODEL_CHOICES)
    common.add_argument("--experts", nargs="+", choices=EXPERT_CHOICES)
    run_parser = sub.add_parser("run", parents=[common])
    run_parser.add_argument("--stage", choices=("all", "general", "experts", "finalize"), default="all")
    run_parser.add_argument(
        "--expert-workers",
        type=int,
        default=1,
        help="number of expert backends to run concurrently (default: 1)",
    )
    pre_parser = sub.add_parser("preflight")
    pre_parser.add_argument("--manifest", type=Path)
    pre_parser.add_argument("--preset", choices=tuple(PRESET_MANIFESTS))
    pre_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    pre_parser.add_argument("--no-audio-check", action="store_true")
    pre_parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES)
    pre_parser.add_argument("--experts", nargs="+", choices=EXPERT_CHOICES)
    fin_parser = sub.add_parser("finalize")
    fin_parser.add_argument("--manifest", type=Path)
    fin_parser.add_argument("--preset", choices=tuple(PRESET_MANIFESTS))
    fin_parser.add_argument("--run-dir", type=Path)
    fin_parser.add_argument("--max-samples", type=int, default=0)
    fin_parser.add_argument("--start-index", type=int, default=0)
    open_common = argparse.ArgumentParser(add_help=False)
    open_common.add_argument("--manifest", type=Path)
    open_common.add_argument("--preset", choices=tuple(PRESET_MANIFESTS))
    open_common.add_argument("--run-dir", type=Path)
    open_common.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    open_common.add_argument("--gpt-helper", type=Path)
    resolve_parser = sub.add_parser("resolve-open", parents=[open_common])
    resolve_parser.add_argument("--resume", action="store_true")
    resolve_parser.add_argument("--workers", type=int)
    resolve_parser.add_argument("--attempts", type=int)
    resolve_parser.add_argument("--timeout", type=int)
    sub.add_parser("finalize-open", parents=[open_common])
    api_stage_parser = sub.add_parser("finalize-api-stage")
    api_stage_parser.add_argument("--manifest", type=Path)
    api_stage_parser.add_argument("--preset", choices=tuple(PRESET_MANIFESTS))
    api_stage_parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.command == "preflight":
        manifest, _run_dir, _preset = resolve_scope(args, require_run_dir=False)
        result = preflight(manifest, load_config(args.config), check_audio=not args.no_audio_check,
                           models=selected_backends(args.models, MODEL_CHOICES),
                           experts=selected_backends(args.experts, EXPERT_CHOICES))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if not result["missing_or_invalid"] else 2)
    if args.command == "finalize":
        manifest, run_dir, _preset = resolve_scope(args, require_run_dir=True)
        assert run_dir is not None
        samples = finalize_samples(manifest, run_dir, args.start_index, args.max_samples)
        print(json.dumps(finalize(samples, run_dir), ensure_ascii=False, indent=2))
        return
    if args.command in {"resolve-open", "finalize-open"}:
        _manifest, run_dir, _preset = resolve_scope(args, require_run_dir=True)
        assert run_dir is not None
        config = load_config(args.config)
        open_config = config.get("open_resolution", {})
        helper_path = args.gpt_helper or Path(open_config.get(
            "helper_path",
            "/F00120250029/lixiang_share/Audio_caption_share/lzy/api/gpt_text.py",
        ))
        if args.command == "resolve-open":
            result = resolve_open_labels(
                run_dir,
                helper_path,
                resume=args.resume,
                workers=args.workers if args.workers is not None else int(open_config.get("workers", 4)),
                attempts=args.attempts if args.attempts is not None else int(open_config.get("attempts", 3)),
                timeout=args.timeout if args.timeout is not None else int(open_config.get("timeout_sec", 300)),
            )
        else:
            result = finalize_open_labels(run_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "finalize-api-stage":
        manifest, run_dir, _preset = resolve_scope(args, require_run_dir=True)
        assert run_dir is not None
        samples = finalize_samples(manifest, run_dir, 0, 0)
        print(json.dumps(finalize_api_stage(samples, run_dir), ensure_ascii=False, indent=2))
        return
    run(args)


if __name__ == "__main__":
    main()
