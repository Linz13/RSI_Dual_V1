#!/usr/bin/env python3
"""Generate pinned InstructTTSEval audio with Qwen3-TTS VoiceDesign."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel
from tqdm.auto import tqdm

from common import (
    DATASET_REVISION,
    LANGUAGES,
    OFFICIAL_COMMIT,
    QWEN_LANGUAGE,
    TASKS,
    atomic_write_json,
    audio_relative_path,
    load_samples,
    selected_samples,
    sha256_file,
    sha256_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
SHARED_BENCHMARK_ROOT = BENCHMARK_ROOT.parent
CAPTION_ROOT = SHARED_BENCHMARK_ROOT.parent
sys.path.insert(0, str(SHARED_BENCHMARK_ROOT))
from model_adapter_utils import describe_adapter, load_peft_adapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=Path,
        default=CAPTION_ROOT / "models/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    )
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument(
        "--manifest-dir", type=Path, default=BENCHMARK_ROOT / "data/manifests"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--languages", nargs="+", choices=LANGUAGES, default=list(LANGUAGES)
    )
    parser.add_argument("--num-samples-per-language", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--attn", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--gpu-memory-gib", type=float)
    parser.add_argument("--seed-per-batch", action="store_true")
    return parser.parse_args()


def valid_wav(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 44:
        return False
    try:
        info = sf.info(path)
    except Exception:
        return False
    return info.frames > 0 and info.channels == 1 and info.samplerate > 0


def base_model_descriptor(model_path: Path) -> dict[str, Any]:
    required = ("config.json", "model.safetensors")
    files = []
    for name in required:
        path = model_path / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(
            {"name": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    provenance = model_path / ".download_provenance.json"
    if provenance.is_file():
        files.append(
            {
                "name": provenance.name,
                "bytes": provenance.stat().st_size,
                "sha256": sha256_file(provenance),
            }
        )
    return {"path": str(model_path), "files": files}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str) -> tuple[str, torch.dtype]:
    if not device.startswith("cuda"):
        return "cpu", torch.float32
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    major, _ = torch.cuda.get_device_capability(torch.device(device))
    return device, torch.bfloat16 if major >= 8 else torch.float16


def make_jobs(
    samples: list[dict[str, Any]], output_dir: Path, tasks=TASKS
) -> list[dict[str, Any]]:
    jobs = []
    for sample in samples:
        for task in tasks:
            relative = audio_relative_path(sample["language"], task, sample["id"])
            path = (output_dir / relative).resolve()
            try:
                path.relative_to(output_dir)
            except ValueError as exc:
                raise ValueError(f"Audio path escapes output directory: {path}") from exc
            jobs.append(
                {
                    "id": sample["id"],
                    "language": sample["language"],
                    "task": task,
                    "text": sample["text"],
                    "instruction": sample[task],
                    "audio_path": str(path),
                    "audio_relative_path": str(relative),
                }
            )
    return jobs


def shard_samples(samples, num_shards, shard_index):
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Invalid shard count/index")
    return samples[shard_index::num_shards]


def batch_seed(seed, jobs):
    # Independent of shard/GPU and of how many previous items were resumed.
    material = {"seed": seed, "keys": [(j["language"], j["id"], j["task"]) for j in jobs]}
    return int(sha256_json(material)[:8], 16)


def pending_batches(jobs, pending, batch_size, fixed):
    # Keep batch membership stable even if a crash left some WAVs on disk
    # before the batch manifest was committed. Completed WAVs are preserved.
    ordered = sorted(jobs if fixed else pending, key=lambda job: len(job["text"]))
    pending_ids = {job["audio_path"] for job in pending}
    batches = []
    for offset in range(0, len(ordered), batch_size):
        batch = ordered[offset:offset + batch_size]
        if any(job["audio_path"] in pending_ids for job in batch):
            batches.append(batch)
    return batches


def ensure_generation_identity(manifest_path: Path, identity_sha256: str) -> None:
    if not manifest_path.is_file():
        return
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    if previous.get("generation_identity_sha256") != identity_sha256:
        raise RuntimeError(
            "Refusing to reuse output directory with a different generation identity: "
            f"{manifest_path}"
        )


def generate_batch(
    model: Qwen3TTSModel, jobs: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    started = time.monotonic()
    keys = [f"{job['id']}:{job['task']}" for job in jobs]
    if getattr(args, "seed_per_batch", False):
        set_seed(batch_seed(args.seed, jobs))
    try:
        wavs, sample_rate = model.generate_voice_design(
            text=[job["text"] for job in jobs],
            instruct=[job["instruction"] for job in jobs],
            language=[QWEN_LANGUAGE[job["language"]] for job in jobs],
            non_streaming_mode=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        if len(wavs) != len(jobs):
            raise RuntimeError(f"Model returned {len(wavs)} WAVs for {len(jobs)} jobs")
        elapsed = round(time.monotonic() - started, 3)
        results = []
        for job, wav in zip(jobs, wavs):
            if len(wav) == 0:
                raise RuntimeError(f"Empty audio for {job['id']}:{job['task']}")
            path = Path(job["audio_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            preserve = getattr(args, "seed_per_batch", False) and valid_wav(path)
            if not preserve:
                sf.write(path, wav, sample_rate, subtype="PCM_16")
            if not valid_wav(path):
                raise RuntimeError(f"Generated invalid WAV: {path}")
            info = sf.info(path)
            results.append(
                {
                    **job,
                    "status": "existing" if preserve else "generated",
                    "sample_rate": info.samplerate,
                    "frames": info.frames,
                    "channels": info.channels,
                    "audio_bytes": path.stat().st_size,
                    "audio_sha256": sha256_file(path),
                    "batch_size": len(jobs),
                    "batch_elapsed_seconds": elapsed,
                }
            )
        print(f"[BATCH][OK] {keys} elapsed={elapsed}s", flush=True)
        return results
    except Exception as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if isinstance(exc, torch.cuda.OutOfMemoryError) and len(jobs) == 1:
            # Stop promptly when sharing a GPU; do not retry every remaining item.
            raise
        if len(jobs) > 1:
            midpoint = len(jobs) // 2
            print(f"[BATCH][SPLIT] {keys}: {type(exc).__name__}: {exc}", flush=True)
            return generate_batch(model, jobs[:midpoint], args) + generate_batch(
                model, jobs[midpoint:], args
            )
        job = jobs[0]
        return [
            {
                **job,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        ]


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if len(set(args.tasks)) != len(args.tasks):
        raise ValueError("--tasks contains duplicates")
    if args.gpu_memory_gib is not None and args.gpu_memory_gib <= 0:
        raise ValueError("--gpu-memory-gib must be positive")
    output_dir = args.output_dir.resolve()
    model_path = args.model_path.resolve()
    manifest_dir = args.manifest_dir.resolve()
    samples = selected_samples(
        load_samples(manifest_dir, args.languages), args.num_samples_per_language
    )
    samples = shard_samples(samples, args.num_shards, args.shard_index)
    if not samples:
        raise ValueError("Selection produced no samples")
    adapter = describe_adapter(args.adapter_dir, model_path)
    base_model = base_model_descriptor(model_path)
    data_files = {
        language: {
            "path": str((manifest_dir / f"{language}.jsonl").resolve()),
            "sha256": sha256_file(manifest_dir / f"{language}.jsonl"),
        }
        for language in args.languages
    }
    identity = {
        "benchmark": "InstructTTSEval",
        "official_commit": OFFICIAL_COMMIT,
        "dataset_revision": DATASET_REVISION,
        "data_files": data_files,
        "model": base_model,
        "adapter": adapter,
        "languages": list(args.languages),
        "selected_ids": [sample["id"] for sample in samples],
        "tasks": list(args.tasks),
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "attention": args.attn,
        "software": {
            package: importlib.metadata.version(package)
            for package in ("qwen-tts", "torch", "peft", "soundfile")
        },
    }
    # Preserve identity compatibility for existing unsharded benchmark runs.
    if args.num_shards != 1:
        identity["shard"] = {"count": args.num_shards, "index": args.shard_index}
    if args.seed_per_batch:
        identity["seed_strategy"] = "sha256_seed_and_batch_keys_v1"
    identity_sha256 = sha256_json(identity)
    manifest_path = output_dir / "generation_manifest.json"
    ensure_generation_identity(manifest_path, identity_sha256)

    jobs = make_jobs(samples, output_dir, args.tasks)
    existing = []
    pending = []
    for job in jobs:
        path = Path(job["audio_path"])
        if valid_wav(path):
            info = sf.info(path)
            existing.append(
                {
                    **job,
                    "status": "existing",
                    "sample_rate": info.samplerate,
                    "frames": info.frames,
                    "channels": info.channels,
                    "audio_bytes": path.stat().st_size,
                    "audio_sha256": sha256_file(path),
                }
            )
            print(f"[SKIP] {job['id']}:{job['task']}", flush=True)
        else:
            pending.append(job)

    manifest = {
        "benchmark": "InstructTTSEval",
        "official_commit": OFFICIAL_COMMIT,
        "dataset_revision": DATASET_REVISION,
        "generation_identity": identity,
        "generation_identity_sha256": identity_sha256,
        "expected_audios": len(jobs),
        "items": existing,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(manifest_path, manifest)
    if pending:
        set_seed(args.seed)
        device, dtype = resolve_device(args.device)
        if args.gpu_memory_gib is not None and device.startswith("cuda"):
            total = torch.cuda.get_device_properties(torch.device(device)).total_memory
            fraction = args.gpu_memory_gib * 1024**3 / total
            if fraction > 1:
                raise ValueError("GPU memory budget exceeds device capacity")
            torch.cuda.set_per_process_memory_fraction(fraction, torch.device(device))
            print(f"[MEMORY] PyTorch allocator budget={args.gpu_memory_gib:g} GiB", flush=True)
        print(f"[LOAD] model={model_path} device={device} dtype={dtype}", flush=True)
        model = Qwen3TTSModel.from_pretrained(
            str(model_path),
            device_map=device,
            dtype=dtype,
            attn_implementation=args.attn,
        )
        if adapter is not None:
            model.model, runtime_adapter = load_peft_adapter(model.model, adapter)
            manifest["runtime_adapter"] = runtime_adapter
            print(f"[LOAD] adapter={adapter['path']}", flush=True)
        batches = pending_batches(jobs, pending, args.batch_size, args.seed_per_batch)
        pending_paths = {job["audio_path"] for job in pending}
        progress = tqdm(total=len(pending), desc="Qwen3-TTS", unit="audio")
        for batch in batches:
            results = [r for r in generate_batch(model, batch, args)
                       if r["audio_path"] in pending_paths]
            manifest["items"].extend(results)
            manifest["items"].sort(
                key=lambda item: (item["language"], item["id"], item["task"])
            )
            atomic_write_json(manifest_path, manifest)
            progress.update(len(results))
        progress.close()

    successes = sum(
        item["status"] in {"generated", "existing"} for item in manifest["items"]
    )
    failures = sum(item["status"] == "failed" for item in manifest["items"])
    manifest["complete"] = successes == len(jobs) and failures == 0
    atomic_write_json(manifest_path, manifest)
    print(f"[DONE] valid={successes}/{len(jobs)} failures={failures}")
    print(f"[DONE] manifest={manifest_path}")
    return 0 if manifest["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
