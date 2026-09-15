#!/usr/bin/env python3
"""Run deterministic Qwen3-Omni or MiDasheng inference on StyleCap MCQs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = ROOT.parent
CAPTION_ROOT = BENCHMARK_ROOT.parent
sys.path.insert(0, str(BENCHMARK_ROOT))

from model_adapter_utils import describe_adapter, load_peft_adapter, sha256_json  # noqa: E402

from common import BENCHMARK_JSONL, read_jsonl, write_json, write_jsonl  # noqa: E402
from evaluate import normalize_prediction  # noqa: E402


DEFAULT_MODEL_DIR = CAPTION_ROOT / "models" / "MiDashengLM-7B-1021-BF16"
DEFAULT_ADAPTER_DIR = (
    CAPTION_ROOT
    / "DualISL_Train"
    / "runs"
    / "dual_recursive_4gpu_h100_midasheng_20260830_run01"
    / "round_002"
    / "checkpoints"
    / "caption_final"
)
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "midasheng4_round2_full_20260831_run01"
PROMPT_VERSION = "stylecap-mcq-exact-letter-v1"


def optional_path(value: str) -> Path | None:
    if value.strip().casefold() in {"", "none", "base"}:
        return None
    return Path(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK_JSONL)
    parser.add_argument("--backend", choices=("midasheng", "qwen3", "qwen25"), default="midasheng")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--adapter-dir",
        type=optional_path,
        default=DEFAULT_ADAPTER_DIR,
        help="PEFT adapter directory. Pass 'none' or 'base' to evaluate the base model.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Questions per model.generate call. Failed batches are split recursively.",
    )
    parser.add_argument("--max-questions", type=int, default=0, help="0 means all questions")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--attn-backend", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_prompt(question: dict[str, Any]) -> str:
    choices = "\n".join(
        f"{letter}: {label}" for letter, label in question["choices"].items()
    )
    return (
        "You are taking an audio multiple-choice test. Listen carefully to the "
        "audio and answer the question. Output exactly one option letter and "
        "nothing else.\n\n"
        f"Question: {question['question']}\n"
        f"Options:\n{choices}\n\n"
        "Answer:"
    )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def latest_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    return {row["question_id"]: row for row in read_jsonl(path)}


def load_midasheng(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("MiDasheng evaluation requires a visible CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"BF16 is not supported by {torch.cuda.get_device_name(0)}")

    model_dir = args.model_dir.expanduser().resolve()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation=args.attn_backend,
        low_cpu_mem_usage=True,
    )
    descriptor = describe_adapter(args.adapter_dir, model_dir)
    adapter = None
    if descriptor is not None:
        model, adapter = load_peft_adapter(model, descriptor)
    model.requires_grad_(False)
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    processor = AutoProcessor.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    )
    configured_eos = model.generation_config.eos_token_id
    eos_token_ids = (
        [int(value) for value in configured_eos]
        if isinstance(configured_eos, (list, tuple))
        else [int(configured_eos)]
    )
    pad_token_id = int(
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else eos_token_ids[0]
    )
    return {
        "backend": "midasheng",
        "model": model,
        "processor": processor,
        "tokenizer": tokenizer,
        "torch": torch,
        "adapter": adapter,
        "eos_token_ids": eos_token_ids,
        "pad_token_id": pad_token_id,
    }


def patch_initializer_range(config: Any) -> None:
    """Patch fields absent from the local Qwen3-Omni checkpoint config."""
    if not hasattr(config, "initializer_range"):
        config.initializer_range = 0.02
    for name in ("thinker_config", "talker_config", "code2wav_config"):
        child = getattr(config, name, None)
        if child is not None and not hasattr(child, "initializer_range"):
            child.initializer_range = 0.02


def load_qwen3(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from qwen_omni_utils import process_mm_info
    from transformers import (
        AutoConfig,
        Qwen3OmniMoeForConditionalGeneration,
        Qwen3OmniMoeProcessor,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-Omni evaluation requires a visible CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"BF16 is not supported by {torch.cuda.get_device_name(0)}")

    model_dir = args.model_dir.expanduser().resolve()
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
    descriptor = describe_adapter(args.adapter_dir, model_dir)
    adapter = None
    if descriptor is not None:
        model.thinker, adapter = load_peft_adapter(model.thinker, descriptor)
    model.requires_grad_(False)
    model.eval()
    processor = Qwen3OmniMoeProcessor.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=True
    )
    return {
        "backend": "qwen3",
        "model": model,
        "processor": processor,
        "process_mm_info": process_mm_info,
        "torch": torch,
        "adapter": adapter,
    }


def load_model(args: argparse.Namespace) -> dict[str, Any]:
    if args.backend == "qwen25":
        from qwen25_caption_backend import load_bundle
        return load_bundle(args)
    if args.backend == "midasheng":
        return load_midasheng(args)
    if args.backend == "qwen3":
        return load_qwen3(args)
    raise ValueError(f"unsupported Captioner backend: {args.backend}")


def infer_midasheng_batch(
    bundle: dict[str, Any],
    audio_paths: list[str],
    prompts: list[str],
    args: argparse.Namespace,
) -> list[str]:
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "path": audio_path},
                ],
            }
        ]
        for audio_path, prompt in zip(audio_paths, prompts, strict=True)
    ]
    values = bundle["processor"].apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        add_special_tokens=True,
        return_dict=True,
        padding=True,
    )
    device = next(bundle["model"].parameters()).device
    dtype = next(bundle["model"].parameters()).dtype
    prepared: dict[str, Any] = {}
    for key, value in values.items():
        if hasattr(value, "to"):
            value = value.to(device)
            if getattr(value, "is_floating_point", lambda: False)():
                value = value.to(dtype=dtype)
        prepared[key] = value

    with bundle["torch"].inference_mode():
        output = bundle["model"].generate(
            **prepared,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=bundle["eos_token_ids"],
            pad_token_id=bundle["pad_token_id"],
            return_dict_in_generate=True,
        )
    sequences = output.sequences.detach().long()
    prompt_ids = prepared["input_ids"].detach().long()
    if (
        sequences.shape[1] >= prompt_ids.shape[1]
        and bundle["torch"].equal(sequences[:, : prompt_ids.shape[1]], prompt_ids)
    ):
        sequences = sequences[:, prompt_ids.shape[1] :]
    return [
        value.strip()
        for value in bundle["tokenizer"].batch_decode(
            sequences.cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    ]


def infer_midasheng(
    bundle: dict[str, Any], audio_path: str, prompt: str, args: argparse.Namespace
) -> str:
    return infer_midasheng_batch(bundle, [audio_path], [prompt], args)[0]


def infer_qwen3_batch(
    bundle: dict[str, Any],
    audio_paths: list[str],
    prompts: list[str],
    args: argparse.Namespace,
) -> list[str]:
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        for audio_path, prompt in zip(audio_paths, prompts, strict=True)
    ]
    rendered = bundle["processor"].apply_chat_template(
        conversations, add_generation_prompt=True, tokenize=False
    )
    audios, _, _ = bundle["process_mm_info"](
        conversations, use_audio_in_video=False
    )
    inputs = bundle["processor"](
        text=rendered,
        audio=audios,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    model = bundle["model"]
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    prepared = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
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
    return [
        value.strip()
        for value in bundle["processor"].batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    ]


def infer_qwen3(
    bundle: dict[str, Any], audio_path: str, prompt: str, args: argparse.Namespace
) -> str:
    return infer_qwen3_batch(bundle, [audio_path], [prompt], args)[0]


def infer(
    bundle: dict[str, Any], audio_path: str, prompt: str, args: argparse.Namespace
) -> str:
    if bundle["backend"] == "qwen25":
        from qwen25_caption_backend import infer_one
        return infer_one(bundle, audio_path, prompt, args)
    if bundle["backend"] == "midasheng":
        return infer_midasheng(bundle, audio_path, prompt, args)
    if bundle["backend"] == "qwen3":
        return infer_qwen3(bundle, audio_path, prompt, args)
    raise ValueError(f"unsupported Captioner backend: {bundle['backend']}")


def infer_batch(
    bundle: dict[str, Any], questions: list[dict[str, Any]], args: argparse.Namespace
) -> list[str]:
    audio_paths = [str(question["audio_path"]) for question in questions]
    prompts = [build_prompt(question) for question in questions]
    if bundle["backend"] == "qwen25":
        return bundle["captioner"].generate_batch(audio_paths, prompts, args.max_new_tokens)
    if bundle["backend"] == "midasheng":
        return infer_midasheng_batch(bundle, audio_paths, prompts, args)
    if bundle["backend"] == "qwen3":
        return infer_qwen3_batch(bundle, audio_paths, prompts, args)
    raise ValueError(f"unsupported Captioner backend: {bundle['backend']}")


def infer_batch_resilient(
    bundle: dict[str, Any], questions: list[dict[str, Any]], args: argparse.Namespace
) -> list[tuple[str, str]]:
    """Infer a batch, recursively isolating OOMs or sample-specific failures."""
    try:
        responses = infer_batch(bundle, questions, args)
        if len(responses) != len(questions):
            raise RuntimeError(
                f"batch output count mismatch: {len(responses)} != {len(questions)}"
            )
        return [(response, "") for response in responses]
    except Exception as exc:
        if bundle["torch"].cuda.is_available():
            bundle["torch"].cuda.empty_cache()
        if len(questions) == 1:
            return [("", repr(exc))]
        midpoint = len(questions) // 2
        print(
            f"[BATCH][SPLIT] size={len(questions)} error={type(exc).__name__}: {exc}",
            flush=True,
        )
        return infer_batch_resilient(
            bundle, questions[:midpoint], args
        ) + infer_batch_resilient(
            bundle, questions[midpoint:], args
        )


def validate_benchmark(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("benchmark is empty")
    ids = [str(row.get("question_id", "")) for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("benchmark question_id values must be non-empty and unique")
    for row in rows:
        audio_path = Path(row["audio_path"])
        if not audio_path.is_file() or audio_path.stat().st_size <= 0:
            raise FileNotFoundError(f"missing or empty benchmark audio: {audio_path}")


def main() -> int:
    args = parse_args()
    if (
        args.max_new_tokens < 1
        or args.batch_size < 1
        or args.max_questions < 0
        or args.start_index < 0
    ):
        raise ValueError("token/question counts and start index must be non-negative")
    if args.log_every < 1:
        raise ValueError("--log-every must be positive")
    benchmark_path = args.benchmark.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not benchmark_path.is_file():
        raise FileNotFoundError(f"benchmark not found: {benchmark_path}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model not found: {model_dir}")
    if args.adapter_dir is not None:
        args.adapter_dir = args.adapter_dir.expanduser().resolve()

    rows = read_jsonl(benchmark_path)
    validate_benchmark(rows)
    selected = rows[args.start_index :]
    if args.max_questions:
        selected = selected[: args.max_questions]

    descriptor = describe_adapter(args.adapter_dir, model_dir)
    identity = {
        "protocol": "stylecap-promptspeech-speaker-open-mcq-v1",
        "prompt_version": PROMPT_VERSION,
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": sha256_file(benchmark_path),
        "model_dir": str(model_dir),
        "adapter": descriptor,
        "max_new_tokens": args.max_new_tokens,
        "attn_backend": args.attn_backend,
        "do_sample": False,
    }
    # Preserve the identity of the already-completed MiDasheng r2 run while
    # making Qwen executions explicitly distinguishable in new output roots.
    if args.backend != "midasheng":
        identity["backend"] = args.backend
    if args.batch_size != 1:
        identity["batch_size"] = args.batch_size
    identity_sha256 = sha256_json(identity)
    generations_path = output_dir / "generations.jsonl"
    predictions_path = output_dir / "predictions.jsonl"
    metadata_path = output_dir / "run_metadata.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = latest_records(generations_path) if args.resume else {}
    mismatched = [
        question_id
        for question_id, record in existing.items()
        if record.get("evaluation_identity_sha256") != identity_sha256
    ]
    if mismatched:
        raise RuntimeError(
            "refusing --resume because existing records use a different evaluation "
            f"identity; examples={mismatched[:3]}"
        )
    completed_statuses = {"success", "invalid"}
    pending = [
        row
        for row in selected
        if existing.get(row["question_id"], {}).get("status") not in completed_statuses
    ]
    print(
        f"benchmark={len(rows)} selected={len(selected)} completed={len(selected) - len(pending)} "
        f"pending={len(pending)} output={output_dir}",
        flush=True,
    )

    started = time.perf_counter()
    bundle: dict[str, Any] | None = None
    if pending:
        load_started = time.perf_counter()
        bundle = load_model(args)
        print(
            f"model_loaded_seconds={time.perf_counter() - load_started:.2f} "
            f"gpu={bundle['torch'].cuda.get_device_name(0)}",
            flush=True,
        )

    error_count = 0
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset : offset + args.batch_size]
        batch_started = time.perf_counter()
        assert bundle is not None
        batch_results = infer_batch_resilient(bundle, batch, args)
        elapsed = time.perf_counter() - batch_started
        for batch_index, (question, result) in enumerate(
            zip(batch, batch_results, strict=True)
        ):
            ordinal = offset + batch_index + 1
            raw_response, error = result
            prediction = ""
            status = "error"
            if error:
                error_count += 1
            else:
                normalized = normalize_prediction(raw_response, question)
                prediction = normalized if normalized is not None else raw_response
                status = "success" if normalized is not None else "invalid"
            record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "question_id": question["question_id"],
                "audio_id": question["audio_id"],
                "task": question["task"],
                "backend": args.backend,
                "audio_path": question["audio_path"],
                "prediction": prediction,
                "response_text": raw_response,
                "status": status,
                "error": error,
                "batch_size": len(batch),
                "batch_elapsed_seconds": round(elapsed, 6),
                "evaluation_identity_sha256": identity_sha256,
            }
            append_jsonl(generations_path, record)
            if ordinal == 1 or ordinal % args.log_every == 0 or ordinal == len(pending):
                print(
                    f"[{ordinal}/{len(pending)}] {question['question_id']} status={status} "
                    f"response={raw_response!r} batch={len(batch)} "
                    f"batch_seconds={elapsed:.2f}",
                    flush=True,
                )
            if error:
                print(f"  error={error}", flush=True)

    latest = latest_records(generations_path)
    selected_predictions = [
        {
            "question_id": row["question_id"],
            "prediction": latest[row["question_id"]]["prediction"],
        }
        for row in selected
        if latest.get(row["question_id"], {}).get("status") in completed_statuses
    ]
    write_jsonl(predictions_path, selected_predictions)
    statuses: dict[str, int] = {}
    for row in selected:
        status = latest.get(row["question_id"], {}).get("status", "missing")
        statuses[status] = statuses.get(status, 0) + 1
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "identity_sha256": identity_sha256,
        "identity": identity,
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "selected_questions": len(selected),
        "prediction_rows": len(selected_predictions),
        "statuses": statuses,
        "wall_seconds_this_invocation": round(time.perf_counter() - started, 6),
    }
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    if error_count or len(selected_predictions) != len(selected):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
