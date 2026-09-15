from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .consensus import list_vote, scalar_vote
from .manifest import Sample, sample_json
from .parsing import extract_json, normalize_payload, validate_subset
from .prompts import build_prompt
from .target_schema import OPEN_FIELDS, empty_target, get_path, set_path, set_transcript, validate_target

ROOT = Path(os.environ.get("AUDIO_CAPTION_ROOT", Path(__file__).resolve().parents[2])).resolve()
CAPTION_ROOT = ROOT / "Experiment/acc_model_pool/Caption_Bench"
AIR_ROOT = CAPTION_ROOT.parent / "AIR-Bench"

MODEL_FIELDS: dict[str, list[str]] = {
    "gemini": [
        "semantic_content.language", "semantic_content.topic", "paralinguistic.pitch_level",
        "paralinguistic.prosody", "paralinguistic.pause", "environment.background_sound_events",
        "environment.acoustic_scene",
    ],
    "qwen35": [
        "semantic_content.intent", "speaker_profile.gender", "speaker_profile.age", "speaker_profile.timbre",
        "paralinguistic.emotion_intensity", "paralinguistic.emphasis.level", "paralinguistic.emphasis.emphasized_text",
        "paralinguistic.prosody", "paralinguistic.nonverbal_vocalization", "environment.background_sound_events",
        "environment.recording_quality", "environment.acoustic_scene",
    ],
    "qwen3_captioner": [
        "speaker_profile.gender", "speaker_profile.age", "paralinguistic.emotion_intensity",
        "paralinguistic.emphasis.level", "paralinguistic.emphasis.emphasized_text", "paralinguistic.prosody",
        "paralinguistic.nonverbal_vocalization", "environment.recording_quality",
    ],
    "kimi_audio": [
        "paralinguistic.emphasis.level", "paralinguistic.emphasis.emphasized_text",
        "paralinguistic.nonverbal_vocalization",
    ],
    "step_audio_r1_1": [
        "paralinguistic.nonverbal_vocalization", "environment.background_sound_events",
    ],
}

CONSENSUS_FIELDS = {
    "speaker_profile.gender": ("qwen35", "qwen3_captioner"),
    "speaker_profile.age": ("qwen35", "qwen3_captioner"),
    "paralinguistic.emotion_intensity": ("qwen35", "qwen3_captioner"),
    "paralinguistic.emphasis.level": ("qwen35", "qwen3_captioner", "kimi_audio"),
    "paralinguistic.nonverbal_vocalization": ("qwen35", "qwen3_captioner", "step_audio_r1_1", "kimi_audio"),
    "environment.recording_quality": ("qwen35", "qwen3_captioner"),
}

EXPERT_FIELDS = {
    "volume": "paralinguistic.volume_level", "emotion": "paralinguistic.emotion",
    "accent": "speaker_profile.accent", "rate": "paralinguistic.speaking_rate",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, row: dict[str, Any], *, sort_keys: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=sort_keys, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def latest_by_id(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("sample_id"):
            latest[str(row["sample_id"])] = row
    return latest


def make_run_dirs(run_dir: Path) -> None:
    for name in ("raw_predictions", "expert_predictions", "final", "state"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)


def ensure_run_metadata(run_dir: Path) -> dict[str, Any]:
    """Create/read immutable run metadata used by deterministic finalization."""
    path = run_dir / "state" / "run_metadata.json"
    if path.is_file():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    value = {"schema_version": "labeling2.v1", "created_utc": utc_now()}
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return value


@lru_cache(maxsize=None)
def api_module(name: str) -> Any:
    sys.path.insert(0, str(CAPTION_ROOT))
    path = CAPTION_ROOT / ("run_gemini_3_1_pro_preview.py" if name == "gemini" else "run_qwen35_omni_plus.py")
    spec = importlib.util.spec_from_file_location(f"labeling2_api_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def api_request(model: str, sample: Sample, prompt: str, config: dict[str, Any]) -> str:
    module = api_module(model)
    api = config.get("api", {})
    backend = config.get("backends", {}).get(model, {})
    if model == "gemini":
        args = SimpleNamespace(
            api_key=os.environ.get(backend.get("api_key_env", "GEMINI_API_KEY")),
            base_url=api.get("gemini_base_url"), api_model=api.get("gemini_model", "gemini-3.1-pro-preview"),
            timeout=int(api.get("timeout_sec", 300)), request_retries=max(1, int(api.get("retries", 3))),
            retry_backoff_sec=float(api.get("retry_backoff_sec", 3.0)),
            large_audio_threshold_mb=20.0,
        )
    else:
        args = SimpleNamespace(
            api_key=os.environ.get(backend.get("api_key_env", "QWEN_API_KEY")),
            base_url=api.get("qwen_base_url") or os.environ.get("QWEN_BASE_URL", ""),
            api_model=api.get("qwen_model", "qwen3.5-omni-plus"), timeout=int(api.get("timeout_sec", 300)),
            request_retries=max(1, int(api.get("retries", 3))),
            retry_backoff_sec=float(api.get("retry_backoff_sec", 3.0)),
        )
    bundle = module.load_model(args)
    return module.request_text(bundle, Path(sample.audio_path), prompt, args)


def backend_enabled(config: dict[str, Any], name: str) -> bool:
    backends = config.get("backends", {})
    item = backends.get(name)
    if item is None and name == "step_audio_r1_1":
        item = backends.get("step_audio")
    return bool((item or {}).get("enabled", True))


def run_api_model(samples: list[Sample], model: str, run_dir: Path, config: dict[str, Any], *, resume: bool) -> None:
    fields = MODEL_FIELDS[model]
    output = run_dir / "raw_predictions" / f"{model}.jsonl"
    latest = latest_by_id(output) if resume else {}
    attempts = max(1, int(config.get("run", {}).get("max_attempts", 2)))
    pending = []
    for sample in samples:
        old = latest.get(sample.sample_id)
        if old and old.get("status") == "success" and not old.get("field_errors"):
            continue
        pending.append((sample, old))

    if not pending:
        return

    # Import the backend once before worker threads start.  Besides avoiding
    # repeated module loading, this keeps importlib/sys.modules mutation out of
    # the concurrent request path.
    api_module(model)

    def process(sample: Sample, old: dict[str, Any] | None) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        accumulated: dict[str, Any] = (old.get("parsed") or {}) if old else {}
        errors: dict[str, str] = (old.get("field_errors") or {}) if old else {}
        # Only model-output validation failures belong in a repair prompt.
        # Transport failures have an empty response_text and must be retried
        # with the original clean prompt instead of exposing SSL/timeout
        # exception strings to the model.
        repair_errors: dict[str, str] = errors if old and old.get("response_text") else {}
        for attempt in range(1, attempts + 1):
            prompt = build_prompt(model, fields, repair_errors=repair_errors or None)
            started = time.perf_counter()
            record = {"timestamp_utc": utc_now(), "model_name": model, "sample_id": sample.sample_id, "audio_path": sample.audio_path, "attempt": attempt, "fields": fields, "prompt": prompt, "status": "error", "response_text": "", "parsed": {}, "field_errors": {}}
            try:
                response = api_request(model, sample, prompt, config)
                payload, parse_status = extract_json(response)
                record["response_text"] = response
                record["parse_status"] = parse_status
                if payload is None:
                    raise ValueError(parse_status)
                current_errors = validate_subset(payload, fields)
                current = normalize_payload(payload, fields)
                # A targeted repair is allowed to return only the failed fields.
                # Merge valid fields from every attempt so an otherwise valid
                # first response is never discarded by a partial retry.
                for field in fields:
                    if field not in current_errors:
                        value = get_path(current, field)
                        if value is not None:
                            set_path(accumulated, field, value)
                errors = {field: message for field, message in current_errors.items() if get_path(accumulated, field) is None}
                repair_errors = errors
                record["parsed"] = accumulated
                record["field_errors"] = errors
                record["status"] = "success" if not errors else "partial"
            except Exception as exc:
                record["error"] = repr(exc)
                errors = {field: repr(exc) for field in fields if get_path(accumulated, field) is None}
                repair_errors = {}
                record["parsed"] = accumulated
                record["field_errors"] = errors
                record["status"] = "partial" if accumulated else "error"
            record["duration_sec"] = time.perf_counter() - started
            records.append(record)
            if not errors:
                break

        return records

    workers = max(1, int(config.get("api", {}).get("workers", 1)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process, sample, old): sample.sample_id
            for sample, old in pending
        }
        for future in as_completed(futures):
            for record in future.result():
                # A single coordinator writes the append-only file, preventing
                # interleaved JSON when several requests finish together.
                append_jsonl(output, record)


def run_local_model(samples: list[Sample], model: str, run_dir: Path, config: dict[str, Any], *, resume: bool) -> None:
    fields = MODEL_FIELDS[model]
    output = run_dir / "raw_predictions" / f"{model}.jsonl"
    latest = latest_by_id(output) if resume else {}
    tasks = [sample for sample in samples if not (latest.get(sample.sample_id, {}).get("status") == "success" and not latest.get(sample.sample_id, {}).get("field_errors"))]
    if not tasks:
        return
    task_path = run_dir / "state" / f"{model}_tasks.jsonl"
    worker_path = run_dir / "state" / f"{model}_worker.jsonl"
    item = config.get("backends", {}).get(model, {})
    if not item and model == "step_audio_r1_1":
        item = config.get("backends", {}).get("step_audio", {})
    python_bin = item.get("python")
    if not python_bin or not Path(python_bin).is_file():
        for sample in tasks:
            append_jsonl(output, {"timestamp_utc": utc_now(), "model_name": model, "sample_id": sample.sample_id, "audio_path": sample.audio_path, "status": "error", "response_text": "", "parsed": {}, "field_errors": {field: "python_not_found" for field in fields}, "error": f"python_not_found:{python_bin}"})
        return
    # Keep a model-specific worker config so independently launched model
    # stages can safely share one run directory (for example, one tmux session
    # per GPU) without racing over state/config.json.
    worker_config_path = run_dir / "state" / f"{model}_config.json"
    worker_config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    command = [python_bin, str(Path(__file__).with_name("local_worker.py")), "--model", model, "--tasks", str(task_path), "--output", str(worker_path), "--config", str(worker_config_path)]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).parent), str(CAPTION_ROOT), str(AIR_ROOT), str(CAPTION_ROOT / "methods_low"), env.get("PYTHONPATH", "")])
    attempts = max(1, int(item.get("max_attempts", config.get("run", {}).get("max_attempts", 2))))
    accumulated = {sample.sample_id: (latest.get(sample.sample_id, {}).get("parsed") or {}) for sample in tasks}
    pending_errors = {sample.sample_id: (latest.get(sample.sample_id, {}).get("field_errors") or {}) for sample in tasks}
    pending = list(tasks)
    for attempt in range(1, attempts + 1):
        if not pending:
            break
        task_prompts = {
            sample.sample_id: build_prompt(
                model,
                fields,
                # Qwen3-Captioner is substantially more stable when a failed
                # sample is retried from the complete clean schema. Its repair
                # prompt tended to trigger dotted keys, placeholders, and
                # malformed field names.
                repair_errors=(
                    None
                    if model == "qwen3_captioner"
                    else pending_errors.get(sample.sample_id)
                    if (attempt > 1 or sample.sample_id in latest)
                    else None
                ),
            )
            for sample in pending
        }
        task_path.write_text("".join(json.dumps({
            "sample_id": sample.sample_id,
            "audio_path": sample.audio_path,
            "prompt": task_prompts[sample.sample_id],
        }, ensure_ascii=False) + "\n" for sample in pending), encoding="utf-8")
        worker_path.write_text("", encoding="utf-8")
        subprocess.run(command, check=False, env=env)
        worker_rows = {str(row.get("sample_id")): row for row in read_jsonl(worker_path)}
        next_pending: list[Sample] = []
        for sample in pending:
            row = worker_rows.get(sample.sample_id, {"status": "error", "error": "worker_no_result", "response_text": ""})
            fields_errors: dict[str, str] = {}
            current: dict[str, Any] = {}
            parse_status = "not_parsed"
            if row.get("status") == "success":
                payload, parse_status = extract_json(row.get("response_text", ""))
                if payload is not None:
                    fields_errors = validate_subset(payload, fields)
                    current = normalize_payload(payload, fields)
                else:
                    fields_errors = {field: parse_status for field in fields}
            else:
                fields_errors = {field: row.get("error", "backend_error") for field in fields}
            for field in fields:
                if field not in fields_errors:
                    value = get_path(current, field)
                    if value is not None:
                        set_path(accumulated[sample.sample_id], field, value)
            unresolved = {field: error for field, error in fields_errors.items() if get_path(accumulated[sample.sample_id], field) is None}
            pending_errors[sample.sample_id] = unresolved
            row.update({
                "timestamp_utc": utc_now(), "attempt": attempt, "fields": fields,
                "prompt": task_prompts[sample.sample_id],
                "parse_status": parse_status, "parsed": accumulated[sample.sample_id],
                "field_errors": unresolved,
                "status": "success" if not unresolved else "partial" if accumulated[sample.sample_id] else "error",
            })
            append_jsonl(output, row)
            if unresolved:
                next_pending.append(sample)
        pending = next_pending


def run_general(
    samples: list[Sample],
    run_dir: Path,
    config: dict[str, Any],
    *,
    resume: bool,
    models: tuple[str, ...] | None = None,
) -> None:
    if models is None:
        models = ("gemini", "qwen35", "qwen3_captioner", "kimi_audio", "step_audio_r1_1")
    for model in ("gemini", "qwen35"):
        if model not in models:
            continue
        if backend_enabled(config, model):
            run_api_model(samples, model, run_dir, config, resume=resume)
    for model in ("qwen3_captioner", "kimi_audio", "step_audio_r1_1"):
        if model not in models:
            continue
        if backend_enabled(config, model):
            run_local_model(samples, model, run_dir, config, resume=resume)


def latest_prediction(run_dir: Path, model: str, sample_id: str) -> dict[str, Any] | None:
    return latest_by_id(run_dir / "raw_predictions" / f"{model}.jsonl").get(sample_id)


def language_for(run_dir: Path, sample: Sample, gemini_rows: dict[str, dict[str, Any]] | None = None) -> str:
    row = (gemini_rows or {}).get(sample.sample_id) if gemini_rows is not None else latest_prediction(run_dir, "gemini", sample.sample_id)
    row = row or {}
    value = get_path(row.get("parsed", {}), "semantic_content.language")
    if isinstance(value, str) and value in {"English", "Chinese", "other", "unknown"}:
        return value
    hint = str(sample.language_hint or "").casefold()
    if hint in {"english", "en"}:
        return "English"
    if hint in {"chinese", "zh", "mandarin", "cantonese"}:
        return "Chinese"
    return "unknown"


def run_expert(
    samples: list[Sample],
    expert: str,
    run_dir: Path,
    config: dict[str, Any],
    *,
    resume: bool,
    gemini_rows: dict[str, dict[str, Any]] | None = None,
) -> None:
    output = run_dir / "expert_predictions" / f"{expert}.jsonl"
    latest = latest_by_id(output) if resume else {}
    selected: list[Sample] = []
    for sample in samples:
        old = latest.get(sample.sample_id, {})
        if old.get("status") == "success":
            continue
        # These failures describe the audio/transcript rather than a transient
        # backend problem. Preserve them for review instead of loading Whisper
        # again on every --resume invocation.
        old_error = str(old.get("error") or "")
        if old.get("status") == "error" and any(
            marker in old_error for marker in ("vad_no_speech", "no_chinese_phone_units", "empty_audio")
        ):
            continue
        language = language_for(run_dir, sample, gemini_rows)
        required_language = "English" if expert in {"accent_en", "rate_en"} else "Chinese" if expert in {"accent_zh", "rate_zh"} else None
        if required_language and language != required_language:
            skip_error = f"language={language}"
            if not (latest.get(sample.sample_id, {}).get("status") == "skipped" and latest.get(sample.sample_id, {}).get("error") == skip_error):
                append_jsonl(output, {"timestamp_utc": utc_now(), "expert": expert, "sample_id": sample.sample_id, "audio_path": sample.audio_path, "status": "skipped", "prediction": "", "error": skip_error})
            continue
        selected.append(sample)
    if not selected:
        return
    task_path = run_dir / "state" / f"{expert}_tasks.jsonl"
    worker_path = run_dir / "state" / f"{expert}_worker.jsonl"
    worker_path.write_text("", encoding="utf-8")
    def expert_task(sample: Sample) -> dict[str, Any]:
        value: dict[str, Any] = {"sample_id": sample.sample_id, "audio_path": sample.audio_path}
        if expert.startswith("rate"):
            value["transcript"] = sample.transcript
        return value
    task_path.write_text("".join(json.dumps(expert_task(s), ensure_ascii=False) + "\n" for s in selected), encoding="utf-8")
    experts = config.get("backends", {}).get("experts", {})
    python_bin = experts.get("rate_python" if expert.startswith("rate") else "voxlect_python" if expert == "accent_zh" else "python")
    if not python_bin or not Path(python_bin).is_file():
        for sample in selected:
            append_jsonl(output, {"timestamp_utc": utc_now(), "expert": expert, "sample_id": sample.sample_id, "audio_path": sample.audio_path, "status": "error", "prediction": "", "error": f"python_not_found:{python_bin}"})
        return
    command = [python_bin, str(Path(__file__).with_name("expert_worker.py")), "--expert", expert.replace("accent_en", "accent_en").replace("accent_zh", "accent_zh"), "--tasks", str(task_path), "--output", str(worker_path), "--device", experts.get("device", "cpu"), "--whisper-model", experts.get("whisper_model", "turbo")]
    if expert == "emotion" and experts.get("emotion_model_dir"):
        command.extend(["--emotion-model-dir", str(experts["emotion_model_dir"])])
        command.extend(["--emotion-model-revision", str(experts.get("emotion_model_revision", ""))])
    if expert == "accent_en" and experts.get("accent_model_dir"):
        command.extend(["--accent-model-dir", str(experts["accent_model_dir"])])
        command.extend(["--accent-model-revision", str(experts.get("accent_model_revision", ""))])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).parent), str(CAPTION_ROOT / "methods_low"), str(CAPTION_ROOT), env.get("PYTHONPATH", "")])
    # PyTorch/BLAS may otherwise size every independent expert process for the
    # whole host. Six workers on a 256-core machine were each creating hundreds
    # of threads, which made Whisper initialization dramatically slower.
    cpu_threads = str(max(1, int(experts.get("cpu_threads_per_worker", 8))))
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[variable] = cpu_threads
    if experts.get("hf_home"):
        env["HF_HOME"] = str(experts["hf_home"])
        env["HF_HUB_CACHE"] = str(Path(experts["hf_home"]) / "hub")
    completed = subprocess.run(command, check=False, env=env, cwd=str(CAPTION_ROOT / "methods_low"))
    for row in read_jsonl(worker_path):
        row["timestamp_utc"] = utc_now()
        append_jsonl(output, row)
    if completed.returncode:
        raise RuntimeError(f"expert worker {expert} exited with status {completed.returncode}")


def run_experts(
    samples: list[Sample],
    run_dir: Path,
    config: dict[str, Any],
    *,
    resume: bool,
    experts: tuple[str, ...] | None = None,
    workers: int = 1,
) -> None:
    if experts is None:
        experts = ("volume", "emotion", "accent_en", "accent_zh", "rate_en", "rate_zh")
    ordered = [
        expert for expert in ("volume", "emotion", "accent_en", "accent_zh", "rate_en", "rate_zh")
        if expert in experts
    ]
    if not ordered:
        return
    workers = max(1, min(int(workers), len(ordered)))
    # Route every expert from one immutable snapshot. Without this cache the
    # complete Gemini JSONL was reparsed once per sample and per expert.
    gemini_rows = latest_by_id(run_dir / "raw_predictions" / "gemini.jsonl")
    if workers == 1:
        for expert in ordered:
            run_expert(samples, expert, run_dir, config, resume=resume, gemini_rows=gemini_rows)
        return

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_expert,
                samples,
                expert,
                run_dir,
                config,
                resume=resume,
                gemini_rows=gemini_rows,
            ): expert
            for expert in ordered
        }
        for future in as_completed(futures):
            expert = futures[future]
            try:
                future.result()
                print(f"[experts] {expert} completed", flush=True)
            except Exception as exc:
                failures.append(f"{expert}: {exc!r}")
                print(f"[experts] {expert} failed: {exc!r}", file=sys.stderr, flush=True)
    if failures:
        raise RuntimeError("expert worker failure(s): " + "; ".join(failures))


def _model_value(run_dir: Path, model: str, sample_id: str, field: str) -> tuple[Any, dict[str, Any]]:
    row = latest_prediction(run_dir, model, sample_id) or {}
    parsed = row.get("parsed", {})
    value = get_path(parsed, field)
    if value is None or row.get("status") not in {"success", "partial"} or field in (row.get("field_errors") or {}):
        return None, {
            "model": model,
            "status": "missing",
            "error": row.get("error", ""),
            "raw_prediction_file": f"raw_predictions/{model}.jsonl",
            "attempt": row.get("attempt"),
        }
    return value, {
        "model": model,
        "status": "success",
        "field_errors": row.get("field_errors", {}),
        "raw_prediction_file": f"raw_predictions/{model}.jsonl",
        "attempt": row.get("attempt"),
        "parse_status": row.get("parse_status"),
    }


def _expert_value(run_dir: Path, expert: str, sample_id: str) -> tuple[Any, dict[str, Any]]:
    row = latest_by_id(run_dir / "expert_predictions" / f"{expert}.jsonl").get(sample_id, {})
    if row.get("status") == "success":
        return row.get("prediction", "unknown"), {"expert": expert, "status": "success", "evidence": row.get("evidence", {})}
    return "unknown", {"expert": expert, "status": row.get("status", "missing"), "error": row.get("error", "")}


def finalize(samples: list[Sample], run_dir: Path) -> dict[str, Any]:
    labels_path = run_dir / "final" / "labels.jsonl"
    provenance_path = run_dir / "final" / "provenance.jsonl"
    candidates_path = run_dir / "final" / "open_candidates.jsonl"
    review_path = run_dir / "final" / "review_queue.jsonl"
    for path in (labels_path, provenance_path, candidates_path, review_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    counts = Counter()
    field_total = Counter()
    field_resolved = Counter()
    consensus_total = 0
    consensus_passed = 0
    for sample in samples:
        target = empty_target()
        provenance: dict[str, Any] = {"schema_version": "labeling2.v1", "sample_id": sample.sample_id, "diarization": "not_run", "fields": {}}
        open_candidates: dict[str, Any] = {"schema_version": "labeling2.v1", "sample_id": sample.sample_id}
        issues: list[dict[str, Any]] = []
        # Single-source closed/open fields.
        single = {
            "semantic_content.language": "gemini", "semantic_content.topic": "gemini", "semantic_content.intent": "qwen35",
            "speaker_profile.timbre": "qwen35", "paralinguistic.pitch_level": "gemini", "paralinguistic.pause": "gemini",
        }
        for field, model in single.items():
            value, meta = _model_value(run_dir, model, sample.sample_id, field)
            if value is None:
                value = "unknown"
            set_path(target, field, value)
            provenance["fields"][field] = {**meta, "value": value}
            if value == "unknown":
                issues.append({"field": field, "reason": "missing_or_invalid_single_source"})
        for field, models in CONSENSUS_FIELDS.items():
            values, details = [], []
            for model in models:
                value, meta = _model_value(run_dir, model, sample.sample_id, field)
                values.append(value); details.append({**meta, "value": value})
            result = list_vote(values) if field in {"paralinguistic.nonverbal_vocalization"} else scalar_vote(values)
            set_path(target, field, result.value)
            provenance["fields"][field] = {"models": details, "value": result.value, "passed": result.passed, "required": result.required, "counts": result.counts}
            if not result.passed:
                issues.append({"field": field, "reason": "consensus_conflict", "counts": result.counts})
        for field, expert in (("paralinguistic.volume_level", "volume"), ("paralinguistic.emotion", "emotion")):
            value, meta = _expert_value(run_dir, expert, sample.sample_id)
            set_path(target, field, value)
            provenance["fields"][field] = meta | {"value": value}
            if value == "unknown":
                issues.append({"field": field, "reason": "expert_failed"})
        language = get_path(target, "semantic_content.language")
        accent_expert = "accent_en" if language == "English" else "accent_zh" if language == "Chinese" else None
        rate_expert = "rate_en" if language == "English" else "rate_zh" if language == "Chinese" else None
        for field, expert in (("speaker_profile.accent", accent_expert), ("paralinguistic.speaking_rate", rate_expert)):
            value, meta = _expert_value(run_dir, expert, sample.sample_id) if expert else ("unknown", {"status": "skipped_language"})
            set_path(target, field, value); provenance["fields"][field] = meta | {"value": value}
            if value == "unknown" and not (language == "other" and expert is None):
                issues.append({"field": field, "reason": "expert_failed_or_language_unknown"})
        # Open multi-model fields deliberately remain unresolved.
        for field, models, fallback in (
            ("paralinguistic.prosody", ("gemini", "qwen35", "qwen3_captioner"), "unknown"),
            ("environment.acoustic_scene", ("gemini", "qwen35"), "unknown"),
            ("environment.background_sound_events", ("gemini", "qwen35", "step_audio_r1_1"), ["unknown"]),
        ):
            candidates = []
            for model in models:
                value, meta = _model_value(run_dir, model, sample.sample_id, field)
                if value is None:
                    value = ["unknown"] if isinstance(fallback, list) else "unknown"
                candidates.append({"model": model, "value": value, **meta})
            set_path(target, field, fallback); open_candidates[field] = candidates; provenance["fields"][field] = {"status": "pending", "candidates": candidates, "value": fallback}
            issues.append({"field": field, "reason": "open_resolution_pending"})
        level = get_path(target, "paralinguistic.emphasis.level")
        level_meta = provenance["fields"].get("paralinguistic.emphasis.level", {})
        if level in {"none", "unknown"}:
            set_path(target, "paralinguistic.emphasis.emphasized_text", []); text_status = "resolved_empty"
        else:
            candidates = []
            for model in ("qwen35", "qwen3_captioner", "kimi_audio"):
                value, meta = _model_value(run_dir, model, sample.sample_id, "paralinguistic.emphasis.emphasized_text")
                if value is None:
                    value = []
                candidates.append({"model": model, "value": value, **meta})
            set_path(target, "paralinguistic.emphasis.emphasized_text", []); open_candidates["paralinguistic.emphasis.emphasized_text"] = candidates; text_status = "pending"
            issues.append({"field": "paralinguistic.emphasis.emphasized_text", "reason": "open_resolution_pending"})
        provenance["fields"]["paralinguistic.emphasis.emphasized_text"] = {"status": text_status, "value": [], "level": level, "level_source": level_meta}
        errors = validate_target(target)
        if errors:
            issues.append({"reason": "final_schema_invalid", "errors": errors})
        set_transcript(target, sample.transcript)
        label_row = {"schema_version": "labeling2.v1", "sample_id": sample.sample_id, "audio_path": sample.audio_path, "dataset": sample.dataset, "Target_JSON_Schema": target}
        for field, field_meta in provenance["fields"].items():
            field_total[field] += 1
            value = field_meta.get("value")
            resolved = field_meta.get("status") == "resolved_empty" or (
                field_meta.get("status") != "pending"
                and value not in (None, "unknown", ["unknown"])
            )
            if resolved:
                field_resolved[field] += 1
            if "passed" in field_meta:
                consensus_total += 1
                consensus_passed += int(bool(field_meta["passed"]))
        # Preserve schema field order, notably transcript immediately before intent.
        append_jsonl(labels_path, label_row, sort_keys=False)
        append_jsonl(provenance_path, provenance)
        if len(open_candidates) > 1:
            append_jsonl(candidates_path, open_candidates)
        if issues:
            counts.update(issue["reason"] for issue in issues)
            append_jsonl(review_path, {"schema_version": "labeling2.v1", "sample_id": sample.sample_id, "issues": issues})
    review_count = sum(1 for _ in read_jsonl(review_path))
    coverage = {
        field: {
            "resolved": field_resolved[field],
            "total": field_total[field],
            "rate": field_resolved[field] / field_total[field] if field_total[field] else 0.0,
        }
        for field in sorted(field_total)
    }
    summary = {
        "schema_version": "labeling2.v1",
        "run_started_utc": (json.loads((run_dir / "state" / "run_metadata.json").read_text(encoding="utf-8"))
                            if (run_dir / "state" / "run_metadata.json").is_file() else {}).get("created_utc", ""),
        "samples": len(samples),
        "review_samples": review_count,
        "pending_review_count": review_count,
        "coverage": coverage,
        "consensus_pass_rate": {
            "passed": consensus_passed,
            "total": consensus_total,
            "rate": consensus_passed / consensus_total if consensus_total else 0.0,
        },
        "issue_counts": dict(counts),
        "error_types": dict(counts),
        "labels": str(labels_path),
        "provenance": str(provenance_path),
        "open_candidates": str(candidates_path),
        "review_queue": str(review_path),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
