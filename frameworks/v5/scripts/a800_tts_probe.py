#!/usr/bin/env python
"""Validation-only single-A800 Qwen3-TTS trajectory/update/reload probe."""

from __future__ import annotations

import argparse
import gc
import math
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from dual_isl_train.adapters import audit_adapter_pair, load_reference_adapter_exact
from dual_isl_train.config import load_config
from dual_isl_train.constants import SOURCE_DOMAIN_TARGET
from dual_isl_train.dual_space import project_synth_caption
from dual_isl_train.io import atomic_json, read_jsonl
from dual_isl_train.render import render_qwen_request
from dual_isl_train.trajectory import tts_trajectory_errors
from dual_isl_train.workers.qwen_voice_design import QwenVoiceDesignWorker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--paired-manifest", required=True)
    parser.add_argument("--caption-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs=4, type=int, default=[21001, 21002, 21003, 21004])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing probe directory: {output_dir}")
    output_dir.mkdir(parents=True)
    result_path = output_dir / "results.json"
    log_path = output_dir / "probe.log"
    results: dict[str, Any] = {
        "status": "running", "stages": {}, "errors": [], "seeds": args.seeds,
    }

    def persist() -> None:
        atomic_json(result_path, results)

    def log(message: str) -> None:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    config = load_config(args.config)
    config["distributed"]["enabled"] = False
    config["distributed"]["world_size"] = 1
    config["tts"]["codec_cache_dir"] = str(output_dir / "codec_cache")
    paired = list(read_jsonl(args.paired_manifest))[0]
    caption_only = list(read_jsonl(args.caption_manifest))[0]
    source_caption = project_synth_caption(caption_only["caption"])
    request = render_qwen_request(source_caption)
    results["inputs"] = {
        "paired_id": paired["id"], "paired_audio_path": paired["audio_path"],
        "caption_only_id": caption_only["id"], "source_caption": source_caption,
    }
    persist()

    import torch

    def stage(name: str, function: Callable[[], Any]) -> Any:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        free_before, total = torch.cuda.mem_get_info()
        baseline = torch.cuda.memory_allocated()
        started = time.perf_counter()
        try:
            value = function()
            torch.cuda.synchronize()
        except Exception as exc:
            error = {
                "stage": name, "type": type(exc).__name__, "message": str(exc),
                "traceback": traceback.format_exc(),
                "cuda_oom": isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower(),
            }
            results["errors"].append(error)
            results["status"] = "failed"
            persist()
            log(f"FAILED {name}: {type(exc).__name__}: {exc}")
            raise
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated()
        free_after, _ = torch.cuda.mem_get_info()
        results["stages"][name] = {
            "elapsed_seconds": elapsed,
            "baseline_allocated_bytes": int(baseline),
            "peak_allocated_bytes": int(peak),
            "peak_increment_bytes": int(max(0, peak - baseline)),
            "free_before_bytes": int(free_before), "free_after_bytes": int(free_after),
            "total_bytes": int(total),
        }
        persist()
        log(
            f"{name}: {elapsed:.3f}s, peak={peak / 2**30:.3f} GiB, "
            f"increment={max(0, peak - baseline) / 2**30:.3f} GiB"
        )
        return value

    log("Loading production Qwen3-TTS VoiceDesign worker")
    started = time.perf_counter()
    worker = QwenVoiceDesignWorker(config)
    torch.cuda.synchronize()
    results["model_load"] = {
        "elapsed_seconds": time.perf_counter() - started,
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "trainable_parameters": sum(value.numel() for value in worker.trainable_parameters),
        "model_type": str(worker.core.tts_model_type),
        "num_codebooks": int(worker.core.config.talker_config.num_code_groups),
    }
    initial_checkpoint = output_dir / "tts_initial_checkpoint"
    worker._save(str(initial_checkpoint), {"update": "validation_initial", "steps": 0})
    reference_audit = load_reference_adapter_exact(worker.policy, initial_checkpoint, "reference")
    worker.has_reference_adapter = True
    worker.policy.set_adapter("default")
    pair_audit = audit_adapter_pair(worker.policy)
    if not pair_audit["exact"]:
        raise RuntimeError(f"Initial TTS policy/reference mismatch: {pair_audit}")
    results["initial_checkpoint"] = str(initial_checkpoint)
    results["adapter_audits"] = {"reference": reference_audit, "policy_reference": pair_audit}
    persist()
    log(f"Model ready; allocated={torch.cuda.memory_allocated() / 2**30:.3f} GiB")

    codec_rows = [{"id": paired["id"], "audio_path": paired["audio_path"]}]
    first_codecs = stage("codec_encode", lambda: worker.prepare_codecs(codec_rows))
    second_codecs = stage("codec_cache_reuse", lambda: worker.prepare_codecs(codec_rows))
    codec_path = first_codecs[0]["codec_path"]
    import json
    codec_payload = json.loads(Path(codec_path).read_text(encoding="utf-8"))
    results["codec"] = {
        "first_cache_reused": first_codecs[0]["cache_reused"],
        "second_cache_reused": second_codecs[0]["cache_reused"],
        "codec_path": codec_path,
        "shape": codec_payload["shape"],
        "shape_valid_T16": len(codec_payload["shape"]) == 2 and codec_payload["shape"][1] == 16,
    }
    persist()

    rollout_rows = [{
        "id": caption_only["id"], "source_id": caption_only["id"],
        "source_caption": source_caption, "request": request, "group_size": 4,
        "candidate_seeds": args.seeds,
    }]
    rollout_path = output_dir / "tts_rollout.jsonl"
    rollout = stage("rollout_4_candidates", lambda: worker.rollout(rollout_rows, str(rollout_path)))
    group = rollout[0]
    candidates = group["candidates"]
    results["rollout"] = {
        "candidate_count": len(candidates),
        "candidate_ids": [value["candidate_id"] for value in candidates],
        "candidate_seeds": [value["generation_seed"] for value in candidates],
        "codec_shapes": [[len(value["codec_codes"]), len(value["codec_codes"][0])] for value in candidates],
        "trajectory_valid_count": sum(bool(value["trajectory_valid"]) for value in candidates),
        "trajectory_errors": [tts_trajectory_errors(value) for value in candidates],
        "behavior_main_all_finite": all(
            all(math.isfinite(float(item)) for item in value["old_main_logprobs"])
            and all(math.isfinite(float(item)) for item in value["ref_main_logprobs"])
            for value in candidates
        ),
        "behavior_sub_all_finite": all(
            all(math.isfinite(float(item)) for frame in value["old_sub_logprobs"] for item in frame)
            and all(math.isfinite(float(item)) for frame in value["ref_sub_logprobs"] for item in frame)
            for value in candidates
        ),
        "main_behavior_replay_max_abs_error": max(
            value["main_behavior_replay_max_abs_error"] for value in candidates
        ),
        "subtalker_behavior_replay_max_abs_error": max(
            value["subtalker_behavior_replay_max_abs_error"] for value in candidates
        ),
        "candidates": candidates,
    }
    persist()

    reconstruction_rows = [{
        "candidate_id": f"paired::{paired['id']}",
        "request": render_qwen_request(project_synth_caption(paired["caption"])),
        "codec_path": codec_path,
    }]
    reconstruction = stage(
        "reconstruction_score", lambda: worker.score_reconstruction(reconstruction_rows),
    )
    results["reconstruction"] = {
        **reconstruction[0],
        "finite": all(
            math.isfinite(float(reconstruction[0][key]))
            for key in ("tts_target_logprob", "tts_main_logprob", "tts_sub_logprob")
        ),
    }
    persist()

    advantages = [-1.3416407865, -0.4472135955, 0.4472135955, 1.3416407865]
    for candidate, advantage in zip(candidates, advantages, strict=True):
        candidate["advantage"] = advantage
        candidate["skip_update"] = False
    grpo_checkpoint = output_dir / "tts_after_grpo"
    grpo_output = stage(
        "grpo_optimizer_step",
        lambda: worker.grpo_update([group], str(grpo_checkpoint), "grpo"),
    )
    results["grpo"] = grpo_output[0]
    persist()

    sft_rows = [{
        "id": f"anchor::{paired['id']}", "audio_path": paired["audio_path"],
        "source_audio_path": paired["audio_path"], "codec_path": codec_path,
        "request": render_qwen_request(project_synth_caption(paired["caption"])),
        "target_origin": SOURCE_DOMAIN_TARGET, "is_anchor": True,
    }]
    final_checkpoint = output_dir / "tts_final"
    sft_output = stage(
        "sft_optimizer_step",
        lambda: worker.sft_update(sft_rows, str(final_checkpoint), "cycle_sft"),
    )
    results["sft"] = sft_output[0]
    persist()

    del worker
    gc.collect()
    torch.cuda.empty_cache()

    def reload_final() -> dict[str, Any]:
        reloaded = QwenVoiceDesignWorker(config, str(final_checkpoint))
        value = {
            "model_type": str(reloaded.core.tts_model_type),
            "num_codebooks": int(reloaded.core.config.talker_config.num_code_groups),
            "trainable_parameters": sum(item.numel() for item in reloaded.trainable_parameters),
            "adapter_audits": reloaded.adapter_audits,
            "parameter_signature": reloaded.parameter_signature(),
        }
        del reloaded
        return value

    results["reload"] = stage("final_checkpoint_reload", reload_final)
    results["contracts"] = {
        "codec_cache_ok": not results["codec"]["first_cache_reused"]
        and results["codec"]["second_cache_reused"],
        "four_candidates": results["rollout"]["candidate_count"] == 4,
        "all_codec_T16": all(shape[1] == 16 and shape[0] > 0 for shape in results["rollout"]["codec_shapes"]),
        "all_trajectories_valid": results["rollout"]["trajectory_valid_count"] == 4,
        "main_replay_within_5e4": results["rollout"]["main_behavior_replay_max_abs_error"] <= 5e-4,
        "sub_replay_within_5e4": results["rollout"]["subtalker_behavior_replay_max_abs_error"] <= 5e-4,
        "reconstruction_finite": results["reconstruction"]["finite"],
        "grpo_parameter_changed": bool(results["grpo"]["parameter_delta"]["ok"]),
        "sft_parameter_changed": bool(results["sft"]["parameter_delta"]["ok"]),
        "final_checkpoint_reload": bool(results["reload"]["adapter_audits"]["policy"]["exact"]),
    }
    results["status"] = "complete" if all(results["contracts"].values()) else "complete_with_failures"
    persist()
    log(f"Probe finished with status={results['status']}")


if __name__ == "__main__":
    main()
