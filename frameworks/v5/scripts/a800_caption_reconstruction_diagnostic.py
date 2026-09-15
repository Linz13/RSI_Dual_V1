#!/usr/bin/env python
"""Diagnostic-only P_syn reconstruction after order-only dictionary rebuilding.

The production worker currently assumes JSON keys are serialized in its field
iteration order. This harness preserves every value but rebuilds insertion order
to demonstrate whether the underlying GPU reconstruction forward is finite.
It does not replace or modify the failing canonical smoke stage.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from dual_isl_train.config import load_config
from dual_isl_train.io import atomic_json, read_jsonl
from dual_isl_train.workers.qwen3_captioner import Qwen3CaptionerWorker


def ordered_caption(value):
    semantic = value["semantic_content"]
    speaker = value["speaker_profile"]
    para = value["paralinguistic"]
    emphasis = para["emphasis"]
    return {
        "semantic_content": {
            "language": semantic["language"],
            "transcript": semantic["transcript"],
        },
        "speaker_profile": {
            "gender": speaker["gender"],
            "age": speaker["age"],
            "timbre": speaker["timbre"],
            "accent": speaker["accent"],
        },
        "paralinguistic": {
            "speaking_rate": para["speaking_rate"],
            "pitch_level": para["pitch_level"],
            "volume_level": para["volume_level"],
            "emotion": para["emotion"],
            "emotion_intensity": para["emotion_intensity"],
            "emphasis": {
                "level": emphasis["level"],
                "emphasized_text": emphasis["emphasized_text"],
            },
            "prosody": para["prosody"],
            "pause": para["pause"],
            "nonverbal_vocalization": para["nonverbal_vocalization"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)

    import torch

    config = load_config(args.config)
    config["distributed"]["enabled"] = False
    config["distributed"]["world_size"] = 1
    rows = list(read_jsonl(args.input))
    load_started = time.perf_counter()
    worker = Qwen3CaptionerWorker(config)
    torch.cuda.synchronize()
    result = {
        "status": "running",
        "diagnostic_only": True,
        "canonical_failure_not_fixed": True,
        "input": str(Path(args.input).resolve()),
        "model_load_seconds": time.perf_counter() - load_started,
        "model_allocated_bytes": int(torch.cuda.memory_allocated()),
        "rows": [],
    }
    atomic_json(output_dir / "results.json", result)
    for row in rows:
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        macro, per_field, token_count = worker.synth_value_macro_logprob(
            row["audio_path"], row["prompt"], ordered_caption(row["target_caption"]),
            reference=True,
        )
        torch.cuda.synchronize()
        result["rows"].append({
            "candidate_id": row["candidate_id"],
            "caption_target_logprob_macro": macro,
            "per_field": per_field,
            "value_token_count": token_count,
            "finite": math.isfinite(macro) and all(math.isfinite(value) for value in per_field.values()),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        })
        atomic_json(output_dir / "results.json", result)
    result["finite_count"] = sum(row["finite"] for row in result["rows"])
    result["status"] = "complete" if result["finite_count"] == len(rows) else "complete_with_failures"
    atomic_json(output_dir / "results.json", result)


if __name__ == "__main__":
    main()
