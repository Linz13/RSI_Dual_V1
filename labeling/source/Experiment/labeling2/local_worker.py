from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(os.environ.get("AUDIO_CAPTION_ROOT", Path(__file__).resolve().parents[2])).resolve()
CAPTION_ROOT = ROOT / "Experiment/acc_model_pool/Caption_Bench"
AIR_ROOT = CAPTION_ROOT.parent / "AIR-Bench"


def progress_line(model: str, completed: int, total: int, elapsed: float) -> str:
    """Return a dependency-free progress bar suitable for tmux and log files."""
    width = 30
    ratio = completed / total if total else 1.0
    filled = min(width, int(width * ratio))
    rate = completed / elapsed if elapsed > 0 else 0.0
    eta = (total - completed) / rate if rate > 0 else 0.0
    bar = "#" * filled + "-" * (width - filled)
    return (
        f"[{model}] [{bar}] {completed}/{total} ({ratio:6.2%}) "
        f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m"
    )


def load_backend(name: str) -> Any:
    path = AIR_ROOT / f"run_{name}.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    sys.path.insert(0, str(CAPTION_ROOT))
    sys.path.insert(1, str(AIR_ROOT))
    spec = importlib.util.spec_from_file_location(f"labeling2_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def backend_args(config: dict[str, Any], model: str) -> SimpleNamespace:
    item = config.get("backends", {}).get(model, {})
    if not item and model == "step_audio_r1_1":
        item = config.get("backends", {}).get("step_audio", {})
    run_config = config.get("run", {})
    return SimpleNamespace(
        device=item.get("device", "auto"), model_dir=Path(item["model_dir"]) if item.get("model_dir") else None,
        attn_backend=item.get("attn_backend", "sdpa"),
        max_new_tokens=int(item.get("max_new_tokens", run_config.get("max_new_tokens", 1024))),
        repetition_penalty=float(item.get("repetition_penalty", 1.0)),
        repetition_window_size=int(item.get("repetition_window_size", 64)),
        no_repeat_ngram_size=int(item.get("no_repeat_ngram_size", 0)),
        do_sample=bool(item.get("do_sample", False)),
        prompt_prefix="", api_url=item.get("api_url", "http://127.0.0.1:9999/v1/chat/completions"),
        step_model_name=item.get("step_model_name", "Step-Audio-R1.1"), system="",
        temperature=float(item.get("temperature", 0.7)),
        top_p=float(item.get("top_p", 0.9)),
        glm4_tokenizer_path=Path(item["glm4_tokenizer_path"]) if item.get("glm4_tokenizer_path") else None,
        allow_online_downloads=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("qwen3_captioner", "kimi_audio", "step_audio_r1_1"), required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    module_name = {"step_audio_r1_1": "step_audio_r1_1"}.get(args.model, args.model)
    if args.model == 'step_audio_r1_1' and config.get('backends', {}).get(args.model, {}).get('backend') == 'transformers':
        module_name = 'step_audio_r1_1_hf'
    model_args = backend_args(config, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line.strip()]
    total = len(tasks)
    progress_started = time.perf_counter()
    print(progress_line(args.model, 0, total, 0.0), flush=True)
    try:
        module = load_backend(module_name)
        # AIR-Bench backends use their own multiple-choice prompt helper. Patch
        # the module-local symbol so the validated model transport receives our
        # JSON task.
        module.build_prompt = lambda sample, prompt_prefix="": sample.prompt
        bundle = module.load_model(model_args)
    except Exception as exc:
        with args.output.open("a", encoding="utf-8") as target:
            for task in tasks:
                target.write(json.dumps({
                    "model_name": args.model, "sample_id": task["sample_id"],
                    "audio_path": task["audio_path"], "status": "error",
                    "response_text": "", "error": repr(exc),
                }, ensure_ascii=False) + "\n")
        print(f"[{args.model}] model load failed: {exc!r}", flush=True)
        return
    with args.output.open("a", encoding="utf-8") as target:
        for completed, task in enumerate(tasks, start=1):
            started = time.perf_counter()
            record = {"model_name": args.model, "sample_id": task["sample_id"], "audio_path": task["audio_path"], "status": "error", "response_text": "", "error": ""}
            try:
                sample = SimpleNamespace(sample_id=task["sample_id"], audio_path=task["audio_path"], prompt=task["prompt"])
                response = module.infer_one(bundle, sample, model_args)
                record.update({"status": "success", "response_text": str(response or "")})
            except Exception as exc:
                record["error"] = repr(exc)
            record["duration_sec"] = time.perf_counter() - started
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            target.flush()
            print(
                progress_line(
                    args.model,
                    completed,
                    total,
                    time.perf_counter() - progress_started,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
