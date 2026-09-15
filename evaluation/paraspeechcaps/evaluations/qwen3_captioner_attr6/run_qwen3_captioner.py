#!/usr/bin/env python3
"""Run Qwen3 or MiDasheng Captioner Attr6 inference on one/two GPUs."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BENCHMARK_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BENCHMARK_ROOT))
from model_adapter_utils import describe_adapter, load_peft_adapter, sha256_json  # noqa: E402

from common import (
    DEFAULT_MIDASHENG_MODEL_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_RUN_ROOT,
    MANIFEST_PATH,
    MIDASHENG_ENV_PYTHON,
    QWEN_ENV_PYTHON,
    append_jsonl,
    build_prompt,
    duration_balanced_partition,
    latest_records,
    load_schema,
    parse_prediction_with_violations,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_jsonl,
)


EVAL_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_ROOT / "outputs"


def parse_args(
    *,
    default_backend: str = "qwen3",
    default_model_dir: Path = DEFAULT_MODEL_DIR,
    default_python: Path = QWEN_ENV_PYTHON,
    default_attn_backend: str = "flash_attention_2",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backend", choices=("qwen3", "midasheng"), default=default_backend)
    parser.add_argument("--model-dir", type=Path, default=default_model_dir)
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=None,
        help="Optional PEFT adapter; the complete base model still comes from --model-dir.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=default_python,
        help="Python executable used for per-GPU workers (or set CAPTIONER_PYTHON).",
    )
    parser.add_argument("--gpu-list", default="auto", help="auto or comma-separated physical GPU IDs")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all selected samples")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="run one sample from each source")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--attn-backend", default=default_attn_backend,
        choices=["flash_attention_2", "sdpa"]
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--sample-ids", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def evaluation_identity(args: argparse.Namespace, manifest_path: Path) -> tuple[str, dict[str, Any]]:
    model_dir = args.model_dir.expanduser().resolve()
    adapter = describe_adapter(args.adapter_dir, model_dir)
    payload = {
        "protocol": load_schema()["protocol"],
        "backend": args.backend,
        "model_dir": str(model_dir),
        "adapter": adapter,
        "manifest_sha256": sha256_file(manifest_path),
        "max_new_tokens": args.max_new_tokens,
        "attn_backend": args.attn_backend,
        "do_sample": False,
    }
    return sha256_json(payload), payload


def discover_gpu_ids() -> list[str]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def resolve_gpu_ids(value: str) -> list[str]:
    available = discover_gpu_ids()
    if value == "auto":
        selected = available[:2]
    else:
        selected = [item.strip() for item in value.split(",") if item.strip()]
        unavailable = sorted(set(selected) - set(available))
        if unavailable:
            raise ValueError(f"requested GPU IDs are not visible: {unavailable}; visible={available}")
        selected = selected[:2]
    if not selected:
        raise RuntimeError("Qwen3-Captioner Attr6 requires at least one visible CUDA GPU")
    return selected


def patch_initializer_range(config: Any) -> None:
    """Patch fields absent from this local Qwen3-Omni checkpoint config."""
    if not hasattr(config, "initializer_range"):
        config.initializer_range = 0.02
    for name in ("thinker_config", "talker_config", "code2wav_config"):
        child = getattr(config, name, None)
        if child is not None and not hasattr(child, "initializer_range"):
            child.initializer_range = 0.02


def load_qwen3_captioner(args: argparse.Namespace) -> dict[str, Any]:
    """Load Qwen3-Omni and attach the training adapter to its thinker."""
    import torch
    from qwen_omni_utils import process_mm_info
    from transformers import (
        AutoConfig,
        Qwen3OmniMoeForConditionalGeneration,
        Qwen3OmniMoeProcessor,
    )

    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Captioner model directory not found: {model_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-Captioner inference requires a visible CUDA GPU")
    config = AutoConfig.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=True
    )
    patch_initializer_range(config)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_dir,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation=args.attn_backend,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
    )
    adapter = describe_adapter(args.adapter_dir, model_dir)
    if adapter is not None:
        model.thinker, adapter = load_peft_adapter(model.thinker, adapter)
    model.eval()
    processor = Qwen3OmniMoeProcessor.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=True
    )
    return {
        "model": model,
        "processor": processor,
        "process_mm_info": process_mm_info,
        "torch": torch,
        "backend": "qwen3",
        "adapter": adapter,
    }


def load_midasheng_captioner(args: argparse.Namespace) -> dict[str, Any]:
    """Load MiDasheng with its native audio processor and optional adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Captioner model directory not found: {model_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("MiDasheng inference requires a visible CUDA GPU")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation=args.attn_backend,
        low_cpu_mem_usage=True,
    )
    adapter = describe_adapter(args.adapter_dir, model_dir)
    if adapter is not None:
        model, adapter = load_peft_adapter(model, adapter)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True,
    )
    configured_eos = model.generation_config.eos_token_id
    eos_token_ids = (
        [int(value) for value in configured_eos]
        if isinstance(configured_eos, (list, tuple)) else [int(configured_eos)]
    )
    return {
        "model": model,
        "processor": processor,
        "tokenizer": tokenizer,
        "torch": torch,
        "backend": "midasheng",
        "adapter": adapter,
        "eos_token_ids": eos_token_ids,
        "pad_token_id": int(
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None else eos_token_ids[0]
        ),
    }


def load_captioner(args: argparse.Namespace) -> dict[str, Any]:
    if args.backend == "qwen25":
        from qwen25_caption_backend import load_bundle
        return load_bundle(args)
    if args.backend == "qwen3":
        return load_qwen3_captioner(args)
    if args.backend == "midasheng":
        return load_midasheng_captioner(args)
    raise ValueError(f"Unsupported Captioner backend: {args.backend}")


def infer_qwen3(
    bundle: dict[str, Any],
    audio_path: str,
    prompt: str,
    args: argparse.Namespace,
) -> str:
    """Run deterministic single-audio, text-only Qwen3-Omni generation."""
    model = bundle["model"]
    processor = bundle["processor"]
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    rendered = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, _, _ = bundle["process_mm_info"](
        conversation, use_audio_in_video=False
    )
    inputs = processor(
        text=rendered,
        audio=audios,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    prepared: dict[str, Any] = {}
    for key, value in inputs.items():
        prepared[key] = value.to(device) if hasattr(value, "to") else value
    for key in ("input_features", "pixel_values", "pixel_values_videos"):
        if key in prepared:
            prepared[key] = prepared[key].to(dtype=dtype)
    with bundle["torch"].inference_mode():
        text_ids, _ = model.generate(
            **prepared,
            return_audio=False,
            thinker_max_new_tokens=args.max_new_tokens,
            thinker_do_sample=False,
            thinker_return_dict_in_generate=True,
        )
    sequences = text_ids.sequences if hasattr(text_ids, "sequences") else text_ids
    generated = sequences[:, prepared["input_ids"].shape[1] :]
    return processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def infer_midasheng(
    bundle: dict[str, Any],
    audio_path: str,
    prompt: str,
    args: argparse.Namespace,
) -> str:
    model = bundle["model"]
    processor = bundle["processor"]
    tokenizer = bundle["tokenizer"]
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "audio", "path": audio_path},
        ],
    }]
    values = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        add_special_tokens=True,
        return_dict=True,
    )
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    prepared: dict[str, Any] = {}
    for key, value in values.items():
        if hasattr(value, "to"):
            value = value.to(device)
            if getattr(value, "is_floating_point", lambda: False)():
                value = value.to(dtype=dtype)
        prepared[key] = value
    with bundle["torch"].inference_mode():
        output = model.generate(
            **prepared,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=bundle["eos_token_ids"],
            pad_token_id=bundle["pad_token_id"],
            return_dict_in_generate=True,
        )
    sequence = output.sequences[0].detach().long()
    prompt_ids = prepared["input_ids"][0].detach().long()
    prefix_length = int(prompt_ids.numel())
    if sequence.numel() >= prefix_length and bundle["torch"].equal(
        sequence[:prefix_length], prompt_ids
    ):
        sequence = sequence[prefix_length:]
    return tokenizer.decode(
        sequence.cpu().tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def infer_one(
    bundle: dict[str, Any],
    audio_path: str,
    prompt: str,
    args: argparse.Namespace,
) -> str:
    if bundle["backend"] == "qwen25":
        from qwen25_caption_backend import infer_one as infer_qwen25
        return infer_qwen25(bundle, audio_path, prompt, args)
    if bundle["backend"] == "qwen3":
        return infer_qwen3(bundle, audio_path, prompt, args)
    if bundle["backend"] == "midasheng":
        return infer_midasheng(bundle, audio_path, prompt, args)
    raise ValueError(f"Unsupported Captioner backend: {bundle['backend']}")


def run_worker(args: argparse.Namespace) -> int:
    if args.sample_ids is None or args.worker_output is None:
        raise ValueError("worker mode requires --sample-ids and --worker-output")
    sample_ids = set(json.loads(args.sample_ids.read_text(encoding="utf-8")))
    samples = [row for row in read_jsonl(args.manifest) if row["sample_id"] in sample_ids]
    existing = latest_records([args.worker_output]) if args.resume else {}
    identity_sha256, identity = evaluation_identity(args, args.manifest.resolve())
    mismatched = [
        sample_id for sample_id, row in existing.items()
        if row.get("evaluation_identity_sha256") != identity_sha256
    ]
    if mismatched:
        raise RuntimeError(
            "Refusing --resume because existing Captioner records belong to a different "
            f"model/protocol identity; examples={mismatched[:3]}"
        )
    pending = [row for row in samples if existing.get(row["sample_id"], {}).get("status") != "success"]
    print(
        f"worker_cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', '')} "
        f"assigned={len(samples)} pending={len(pending)} output={args.worker_output}",
        flush=True,
    )
    if not pending:
        return 0
    schema = load_schema()
    prompt = build_prompt(schema)
    load_started = time.perf_counter()
    bundle = load_captioner(args)
    print(f"model_loaded_seconds={time.perf_counter() - load_started:.2f}", flush=True)
    if bundle["adapter"] is not None:
        print(
            "adapter_loaded="
            + json.dumps(bundle["adapter"], ensure_ascii=False, sort_keys=True),
            flush=True,
        )

    for ordinal, sample in enumerate(pending, 1):
        started = time.perf_counter()
        raw = ""
        parsed: dict[str, Any] | None = None
        schema_violations: list[str] = []
        status, error = "error", ""
        try:
            raw = str(infer_one(bundle, sample["audio_path"], prompt, args) or "")
            parsed, schema_violations = parse_prediction_with_violations(raw, schema)
            status = "success"
        except ValueError as exc:
            status, error = "invalid", repr(exc)
        except Exception as exc:
            status, error = "error", repr(exc)
        elapsed = time.perf_counter() - started
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": schema["protocol"],
            "model_name": f"{args.backend}_captioner",
            "backend": args.backend,
            "model_dir": str(args.model_dir.expanduser().resolve()),
            "adapter": bundle["adapter"],
            "evaluation_identity_sha256": identity_sha256,
            "evaluation_identity": identity,
            "sample_id": sample["sample_id"],
            "benchmark_indices": sample["benchmark_indices"],
            "audio_path": sample["audio_path"],
            "source": sample["source"],
            "response_text": raw,
            "prediction": parsed,
            "schema_violations": schema_violations,
            "status": status,
            "error": error,
            "elapsed_seconds": round(elapsed, 6),
            "prompt_sha256": sha256_text(prompt),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
        }
        append_jsonl(args.worker_output, record)
        print(
            f"[{ordinal}/{len(pending)}] {sample['sample_id']} source={sample['source']} "
            f"status={status} seconds={elapsed:.2f}",
            flush=True,
        )
        if error:
            print(error, flush=True)
    return 0


def select_samples(args: argparse.Namespace, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = samples[args.start_index :]
    if args.smoke:
        by_source: dict[str, dict[str, Any]] = {}
        for sample in selected:
            by_source.setdefault(sample["source"], sample)
        selected = [by_source[source] for source in sorted(by_source)]
    elif args.max_samples:
        selected = selected[: args.max_samples]
    return selected


def merge_outputs(output_dir: Path, manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paths = list((output_dir / "shards").glob("*.jsonl"))
    latest = latest_records(paths)
    rows = [latest[sample["sample_id"]] for sample in manifest if sample["sample_id"] in latest]
    write_jsonl(output_dir / "predictions.jsonl", rows)
    return rows


def launch(args: argparse.Namespace) -> int:
    if args.max_samples < 0 or args.start_index < 0:
        raise ValueError("--max-samples and --start-index must be non-negative")
    manifest_path = args.manifest.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    python_executable = args.python.expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Captioner model directory not found: {model_dir}")
    if not python_executable.is_file() or not os.access(python_executable, os.X_OK):
        raise FileNotFoundError(f"Captioner Python executable not found: {python_executable}")
    samples = read_jsonl(manifest_path)
    if not samples:
        raise FileNotFoundError(f"manifest is empty or missing: {manifest_path}")
    selected = select_samples(args, samples)
    output_dir = args.output_dir.expanduser().resolve()
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    existing = latest_records(list(shard_dir.glob("*.jsonl"))) if args.resume else {}
    identity_sha256, identity = evaluation_identity(args, manifest_path)
    mismatched = [
        sample_id for sample_id, row in existing.items()
        if row.get("evaluation_identity_sha256") != identity_sha256
    ]
    if mismatched:
        raise RuntimeError(
            "Refusing --resume because existing Captioner shards belong to a different "
            f"model/protocol identity; examples={mismatched[:3]}"
        )
    pending = [row for row in selected if existing.get(row["sample_id"], {}).get("status") != "success"]
    gpu_ids = resolve_gpu_ids(args.gpu_list)
    worker_count = min(len(gpu_ids), max(1, len(pending)))
    mode = "dual_gpu_data_parallel" if worker_count >= 2 else "single_gpu_fallback"
    print(
        f"manifest={len(samples)} selected={len(selected)} pending={len(pending)} "
        f"visible_gpus={discover_gpu_ids()} selected_gpus={gpu_ids[:worker_count]} mode={mode}",
        flush=True,
    )
    started = time.perf_counter()
    if pending:
        shards = duration_balanced_partition(pending, worker_count)
        processes: list[subprocess.Popen[Any]] = []
        for worker_index, (gpu_id, shard) in enumerate(zip(gpu_ids, shards)):
            id_path = shard_dir / f"worker_{worker_index}_sample_ids.json"
            id_path.write_text(
                json.dumps([row["sample_id"] for row in shard], indent=2) + "\n", encoding="utf-8"
            )
            output_path = shard_dir / f"worker_{worker_index}.jsonl"
            command = [
                str(python_executable), str(Path(__file__).resolve()),
                "--worker", "--manifest", str(manifest_path),
                "--sample-ids", str(id_path), "--worker-output", str(output_path),
                "--backend", args.backend,
                "--model-dir", str(model_dir),
                "--max-new-tokens", str(args.max_new_tokens),
                "--attn-backend", args.attn_backend,
            ]
            if args.adapter_dir is not None:
                command.extend(["--adapter-dir", str(args.adapter_dir.expanduser().resolve())])
            if args.resume:
                command.append("--resume")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu_id
            print(
                f"launch worker={worker_index} physical_gpu={gpu_id} samples={len(shard)} "
                f"duration={sum(float(row['duration_seconds']) for row in shard):.1f}s",
                flush=True,
            )
            processes.append(subprocess.Popen(command, env=environment))
        return_codes = [process.wait() for process in processes]
        if any(code != 0 for code in return_codes):
            raise RuntimeError(f"one or more inference workers failed: {return_codes}")
    merged = merge_outputs(output_dir, samples)
    selected_ids = {row["sample_id"] for row in selected}
    selected_records = [row for row in merged if row["sample_id"] in selected_ids]
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": load_schema()["protocol"],
        "mode": mode,
        "visible_gpu_ids": discover_gpu_ids(),
        "selected_gpu_ids": gpu_ids[:worker_count],
        "worker_count": worker_count,
        "manifest_sha256": sha256_file(manifest_path),
        "backend": args.backend,
        "model_dir": str(args.model_dir.expanduser().resolve()),
        "adapter": identity["adapter"],
        "evaluation_identity_sha256": identity_sha256,
        "evaluation_identity": identity,
        "python": sys.version,
        "platform": platform.platform(),
        "selected_samples": len(selected),
        "merged_selected_records": len(selected_records),
        "successes": sum(row.get("status") == "success" for row in selected_records),
        "wall_seconds": round(time.perf_counter() - started, 6),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    return 0


def main(
    *,
    default_backend: str = "qwen3",
    default_model_dir: Path = DEFAULT_MODEL_DIR,
    default_python: Path = QWEN_ENV_PYTHON,
    default_attn_backend: str = "flash_attention_2",
) -> None:
    args = parse_args(
        default_backend=default_backend,
        default_model_dir=default_model_dir,
        default_python=default_python,
        default_attn_backend=default_attn_backend,
    )
    raise SystemExit(run_worker(args) if args.worker else launch(args))


if __name__ == "__main__":
    main()
