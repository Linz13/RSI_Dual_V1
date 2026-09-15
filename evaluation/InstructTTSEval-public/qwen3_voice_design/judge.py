#!/usr/bin/env python3
"""Restartable official-prompt Gemini judge with a strictly offline dry-run mode."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import re
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import json_repair
from tqdm import tqdm

from common import (
    JsonlCheckpointWriter,
    atomic_write_json,
    load_latest_records,
    record_key,
    sha256_file,
    sha256_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
SHARED_BENCHMARK_ROOT = BENCHMARK_ROOT.parent
DEFAULT_PROMPT = BENCHMARK_ROOT / "eval/eval_prompt.txt"
LEGACY_CONFIG = (
    SHARED_BENCHMARK_ROOT
    / "EmergentTTS-Eval-public/qwen3_voice_design/local_judger_config.py"
)
PLACEHOLDER = "<此处插入待评测的语音风格描述>"
NON_RETRYABLE_STATUS = re.compile(r"\b(?:400|401|403)\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--model", default="models/gemini-2.5-pro")
    parser.add_argument("--backend", choices=("auto", "files", "inline"), default="auto")
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("INSTRUCTTTS_GEMINI_WORKERS", "128")),
    )
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-paid", action="store_true")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Return success after all retries are exhausted even when individual "
            "items failed. Failed records remain in the JSONL checkpoint."
        ),
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry items whose latest checkpoint record is already failed.",
    )
    parser.add_argument("--check-config", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_legacy_config(path: Path) -> tuple[str | None, str | None]:
    if not path.is_file():
        return None, None
    spec = importlib.util.spec_from_file_location("instructtts_legacy_judger_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load local judger config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (
        getattr(module, "DEFAULT_JUDGER_API_KEY", None),
        getattr(module, "DEFAULT_JUDGER_BASE_URL", None),
    )


def configure_judger() -> tuple[str, str | None, str]:
    api_key = os.environ.get("JUDGER_API_KEY") or os.environ.get("GENAI_API_KEY")
    base_url = os.environ.get("JUDGER_BASE_URL")
    source = "environment"
    if not api_key:
        legacy_key, legacy_base = _load_legacy_config(LEGACY_CONFIG)
        api_key = legacy_key
        if base_url is None:
            base_url = legacy_base
        source = str(LEGACY_CONFIG)
    if not isinstance(api_key, str) or not api_key.strip():
        raise RuntimeError(
            "No Gemini key found in JUDGER_API_KEY, GENAI_API_KEY, or the existing "
            f"local config at {LEGACY_CONFIG}"
        )
    if base_url is not None:
        if not isinstance(base_url, str):
            raise TypeError("Gemini base URL must be a string")
        base_url = base_url.strip().rstrip("/") or None
    return api_key.strip(), base_url, source


def create_client(api_key: str, base_url: str | None, timeout_seconds: int):
    # Keep offline checks and --dry-run independent of Google SDK binary
    # dependencies; import the client only for config checks or paid requests.
    from google import genai
    from google.genai import types

    options = None
    if base_url:
        options = types.HttpOptions(
            base_url=base_url, api_version="", timeout=timeout_seconds * 1000
        )
    return genai.Client(api_key=api_key, http_options=options)


def resolve_backend(requested: str, base_url: str | None) -> str:
    if requested == "auto":
        return "inline" if base_url else "files"
    if requested == "files" and base_url:
        raise ValueError(
            "The Google Files backend is not assumed to exist on a custom base URL; "
            "use --backend inline"
        )
    return requested


def load_prompt(path: Path) -> str:
    template = path.read_text(encoding="utf-8")
    count = template.count(PLACEHOLDER)
    if count != 1:
        raise ValueError(f"Expected exactly one prompt placeholder, found {count}: {path}")
    return template


def build_prompt(template: str, instruction: str) -> str:
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Instruction is empty")
    prompt = template.replace(PLACEHOLDER, instruction)
    if PLACEHOLDER in prompt:
        raise RuntimeError("Prompt placeholder was not fully replaced")
    return prompt


def extract_result(text: str) -> dict[str, Any]:
    parsed = json_repair.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Gemini response is not a JSON object")
    consistency = parsed.get("一致性")
    if not isinstance(consistency, bool):
        raise ValueError("Gemini response does not contain boolean field '一致性'")
    return parsed


def result_exit_code(
    success_count: int, expected_count: int, allow_incomplete: bool
) -> int:
    if success_count == expected_count:
        return 0
    return 0 if allow_incomplete else 1


def checkpoint_progress(
    jobs: list[dict[str, Any]],
    latest: dict[tuple[str, str, str], dict[str, Any]],
    dry_run: bool,
    allow_incomplete: bool,
    retry_failed: bool,
) -> tuple[
    list[dict[str, Any]],
    set[tuple[str, str, str]],
    set[tuple[str, str, str]],
]:
    completed = {
        key
        for key, row in latest.items()
        if row.get("status") == "success" and bool(row.get("dry_run")) == dry_run
    }
    recorded_failed = {
        key
        for key, row in latest.items()
        if row.get("status") == "failed" and bool(row.get("dry_run")) == dry_run
    }
    resolved = set(completed)
    if allow_incomplete and not retry_failed:
        resolved.update(recorded_failed)
    pending = [job for job in jobs if record_key(job) not in resolved]
    return pending, completed, recorded_failed


def usage_to_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json", exclude_none=True)
    fields = (
        "prompt_token_count",
        "candidates_token_count",
        "total_token_count",
        "thoughts_token_count",
        "cached_content_token_count",
    )
    return {name: getattr(usage, name) for name in fields if hasattr(usage, name)}


def generation_jobs(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise RuntimeError(f"Generation manifest is incomplete: {path}")
    jobs = [
        item
        for item in manifest.get("items", [])
        if item.get("status") in {"generated", "existing"}
    ]
    if len(jobs) != manifest.get("expected_audios"):
        raise RuntimeError(
            f"Generation manifest has {len(jobs)} valid items; "
            f"expected {manifest.get('expected_audios')}"
        )
    keys = [record_key(job) for job in jobs]
    if len(keys) != len(set(keys)):
        raise ValueError("Generation manifest contains duplicate language/id/task keys")
    for job in jobs:
        audio = Path(job["audio_path"])
        if not audio.is_file() or audio.stat().st_size <= 44:
            raise FileNotFoundError(audio)
    return manifest, jobs


def ensure_metadata(path: Path, payload: dict[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"Judge identity mismatch in existing output: {path}")
        return
    atomic_write_json(path, payload)


def call_gemini(
    client,
    backend: str,
    model: str,
    prompt: str,
    audio_path: Path,
) -> tuple[str, Any]:
    from google.genai import types

    safety = [
        types.SafetySetting(category=name, threshold="BLOCK_NONE")
        for name in (
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
        )
    ]
    config = types.GenerateContentConfig(
        response_mime_type="text/plain", temperature=0, safety_settings=safety
    )
    uploaded = None
    try:
        if backend == "files":
            uploaded = client.files.upload(file=str(audio_path))
            contents = [prompt, uploaded]
        else:
            part = types.Part.from_bytes(data=audio_path.read_bytes(), mime_type="audio/wav")
            contents = [prompt, part]
        response = client.models.generate_content(
            model=model, contents=contents, config=config
        )
        if not isinstance(response.text, str) or not response.text.strip():
            raise ValueError("Gemini returned an empty response")
        return response.text.strip(), response.usage_metadata
    finally:
        if uploaded is not None:
            try:
                client.files.delete(name=uploaded.name)
            except Exception as exc:
                print(f"[WARN] Failed to delete uploaded file {uploaded.name}: {exc}")


def main() -> int:
    args = parse_args()
    if args.workers <= 0 or args.attempts <= 0 or args.timeout <= 0:
        raise ValueError("workers, attempts, and timeout must be positive")
    if args.retry_delay < 0:
        raise ValueError("retry delay cannot be negative")
    if args.check_config:
        api_key, base_url, source = configure_judger()
        client = create_client(api_key, base_url, args.timeout)
        client.close()
        backend = resolve_backend(args.backend, base_url)
        print(f"[CHECK] Gemini config source={source}")
        print(f"[CHECK] endpoint={base_url or 'Google default'} backend={backend}")
        print("[CHECK] Client initialized and closed without sending a request.")
        return 0
    if args.generation_manifest is None or args.output_dir is None:
        raise ValueError("--generation-manifest and --output-dir are required")
    if not args.dry_run and not args.confirm_paid:
        raise RuntimeError("Real Gemini calls require --confirm-paid")

    generation_path = args.generation_manifest.resolve()
    output_dir = args.output_dir.resolve()
    prompt_path = args.prompt_file.resolve()
    generation, jobs = generation_jobs(generation_path)
    template = load_prompt(prompt_path)

    api_key = None
    base_url = None
    source = None
    backend = "offline-inline"
    if not args.dry_run:
        api_key, base_url, source = configure_judger()
        backend = resolve_backend(args.backend, base_url)
    metadata = {
        "benchmark": "InstructTTSEval",
        "generation_manifest": str(generation_path),
        "generation_identity_sha256": generation["generation_identity_sha256"],
        "prompt_file": str(prompt_path),
        "prompt_sha256": sha256_file(prompt_path),
        "judge_model": args.model,
        "backend": backend,
        "endpoint": base_url,
        "temperature": 0,
        "google_genai_version": importlib.metadata.version("google-genai"),
        "dry_run": args.dry_run,
        "expected_items": len(jobs),
    }
    ensure_metadata(output_dir / "judge_metadata.json", metadata)
    result_path = output_dir / "judge_results.jsonl"
    latest = load_latest_records(result_path)
    pending, completed, recorded_failed = checkpoint_progress(
        jobs,
        latest,
        args.dry_run,
        args.allow_incomplete,
        args.retry_failed,
    )
    print(
        f"[JUDGE] expected={len(jobs)} complete={len(completed)} pending={len(pending)} "
        f"recorded_failed={len(recorded_failed)} backend={backend} "
        f"dry_run={args.dry_run}"
    )
    if not pending:
        if len(completed) == len(jobs):
            print("[DONE] Judge checkpoint is already complete.")
        else:
            print(
                f"[WARN] Keeping {len(recorded_failed)} recorded failure(s); "
                "the score will use successful judge results only."
            )
        return result_exit_code(len(completed), len(jobs), args.allow_incomplete)

    thread_state = threading.local()

    def get_client():
        if args.dry_run:
            raise AssertionError("Dry-run attempted to construct a Gemini client")
        client = getattr(thread_state, "client", None)
        if client is None:
            client = create_client(api_key, base_url, args.timeout)
            thread_state.client = client
        return client

    def judge_one(job: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        audio_path = Path(job["audio_path"])
        prompt = build_prompt(template, job["instruction"])
        base = {
            "timestamp_utc": utc_now(),
            "language": job["language"],
            "id": job["id"],
            "task": job["task"],
            "audio_path": str(audio_path),
            "audio_sha256": sha256_file(audio_path),
            "prompt_sha256": sha256_json(prompt),
            "judge_model": args.model,
            "backend": backend,
            "dry_run": args.dry_run,
        }
        if args.dry_run:
            # Exercise the same inline-audio bytes/mime structure without
            # importing or constructing any Google SDK object.
            inline_part = {
                "inline_data": {
                    "data": audio_path.read_bytes(),
                    "mime_type": "audio/wav",
                }
            }
            if not inline_part["inline_data"]["data"]:
                raise ValueError(f"Dry-run audio is empty: {audio_path}")
            raw = '{"一致性": true, "dry_run": true}'
            parsed = extract_result(raw)
            return {
                **base,
                "status": "success",
                "gemini_score": parsed["一致性"],
                "parsed_response": parsed,
                "raw_response": raw,
                "usage": None,
                "attempt": 0,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }

        last_error: Exception | None = None
        last_traceback = ""
        for attempt in range(1, args.attempts + 1):
            try:
                raw, usage = call_gemini(
                    get_client(), backend, args.model, prompt, audio_path
                )
                parsed = extract_result(raw)
                return {
                    **base,
                    "status": "success",
                    "gemini_score": parsed["一致性"],
                    "parsed_response": parsed,
                    "raw_response": raw,
                    "usage": usage_to_dict(usage),
                    "attempt": attempt,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
            except Exception as exc:
                last_error = exc
                last_traceback = traceback.format_exc()
                if attempt >= args.attempts or NON_RETRYABLE_STATUS.search(str(exc)):
                    break
                time.sleep(args.retry_delay)
        assert last_error is not None
        return {
            **base,
            "status": "failed",
            "error_type": type(last_error).__name__,
            "error": str(last_error),
            "traceback": last_traceback,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }

    with JsonlCheckpointWriter(result_path) as writer:
        with ThreadPoolExecutor(max_workers=min(args.workers, len(pending))) as executor:
            futures = [executor.submit(judge_one, job) for job in pending]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Judge"):
                writer.write(future.result())

    final = load_latest_records(result_path)
    success_count = sum(
        row.get("status") == "success" and bool(row.get("dry_run")) == args.dry_run
        for row in final.values()
    )
    print(f"[DONE] success={success_count}/{len(jobs)} results={result_path}")
    if success_count != len(jobs) and args.allow_incomplete:
        print(
            f"[WARN] Continuing with {len(jobs) - success_count} failed item(s); "
            "the score will use successful judge results only."
        )
    return result_exit_code(success_count, len(jobs), args.allow_incomplete)


if __name__ == "__main__":
    raise SystemExit(main())
