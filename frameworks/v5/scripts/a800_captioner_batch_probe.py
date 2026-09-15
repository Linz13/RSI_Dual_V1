#!/usr/bin/env python
"""Controlled single-A800 Captioner batch/replay/GRPO equivalence probe.

This is a validation-only harness. It calls the production worker methods
without changing model, reward, candidate count, or replay contracts.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from dual_isl_train.adapters import audit_adapter_pair, load_reference_adapter_exact
from dual_isl_train.config import load_config
from dual_isl_train.constants import TRAJECTORY_VERSION
from dual_isl_train.io import atomic_json, read_jsonl
from dual_isl_train.render import synth_caption_prompt
from dual_isl_train.trajectory import caption_trajectory_errors, mark_trajectory
from dual_isl_train.workers.qwen3_captioner import Qwen3CaptionerWorker, _batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--audio-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs=4, type=int, default=[11001, 11002, 11003, 11004])
    return parser.parse_args()


def finite(values: list[float]) -> bool:
    return bool(values) and all(math.isfinite(float(value)) for value in values)


def tensor_lists(values: list[Any]) -> list[list[float]]:
    return [value.detach().float().cpu().tolist() for value in values]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing probe directory: {output_dir}")
    output_dir.mkdir(parents=True)
    result_path = output_dir / "results.json"
    log_path = output_dir / "probe.log"
    results: dict[str, Any] = {
        "status": "running",
        "config": str(Path(args.config).resolve()),
        "audio_manifest": str(Path(args.audio_manifest).resolve()),
        "seeds": args.seeds,
        "batches": {},
        "fixed_replay": {},
        "optimizer_steps": {},
        "errors": [],
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
    config["captioner"]["generation"].update({
        "temperature": 1.0, "top_p": 1.0, "top_k": 0,
    })
    rows = list(read_jsonl(args.audio_manifest))
    if not rows:
        raise ValueError("Audio manifest is empty")
    row = rows[0]
    audio_path = str(row["audio_path"])
    prompt = synth_caption_prompt()
    results["sample"] = {"id": str(row["id"]), "audio_path": audio_path, "prompt": prompt}
    persist()

    import torch

    def stage(name: str, function: Callable[[], Any]) -> tuple[Any | None, dict[str, Any]]:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        free_before, total = torch.cuda.mem_get_info()
        baseline = torch.cuda.memory_allocated()
        started = time.perf_counter()
        value = None
        error = None
        try:
            value = function()
            torch.cuda.synchronize()
        except Exception as exc:  # keep lower batch results if a larger batch OOMs
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "cuda_oom": isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower(),
            }
            results["errors"].append({"stage": name, **error})
            log(f"FAILED {name}: {type(exc).__name__}: {exc}")
            gc.collect()
            torch.cuda.empty_cache()
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated()
        free_after, _ = torch.cuda.mem_get_info()
        metrics = {
            "elapsed_seconds": elapsed,
            "baseline_allocated_bytes": int(baseline),
            "peak_allocated_bytes": int(peak),
            "peak_increment_bytes": int(max(0, peak - baseline)),
            "free_before_bytes": int(free_before),
            "free_after_bytes": int(free_after),
            "total_bytes": int(total),
            "error": error,
        }
        log(
            f"{name}: {elapsed:.3f}s, peak={peak / 2**30:.3f} GiB, "
            f"increment={max(0, peak - baseline) / 2**30:.3f} GiB"
        )
        persist()
        return value, metrics

    log("Loading production Qwen3 Captioner worker")
    load_started = time.perf_counter()
    worker = Qwen3CaptionerWorker(config)
    torch.cuda.synchronize()
    results["model_load"] = {
        "elapsed_seconds": time.perf_counter() - load_started,
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "preflight": worker.preflight()[0],
    }
    initial_checkpoint = output_dir / "captioner_initial_checkpoint"
    worker._save(str(initial_checkpoint), {"update": "validation_initial", "steps": 0})
    results["initial_checkpoint"] = str(initial_checkpoint)
    reference_audit = load_reference_adapter_exact(worker.model, initial_checkpoint, "reference")
    worker.has_reference_adapter = True
    worker.model.set_adapter("default")
    worker._audit_adapters()
    pair_audit = audit_adapter_pair(worker.model)
    if not pair_audit["exact"]:
        raise RuntimeError(f"Initial policy/reference mismatch: {pair_audit}")
    results["adapter_audits"] = {"reference": reference_audit, "policy_reference": pair_audit}
    initial_parameters = {
        name: parameter.detach().cpu().clone()
        for name, parameter in worker._current_adapter_named_parameters()
    }
    results["trainable_parameter_tensors"] = len(initial_parameters)
    results["trainable_parameters"] = sum(value.numel() for value in initial_parameters.values())
    persist()
    log(
        f"Model ready; allocated={torch.cuda.memory_allocated() / 2**30:.3f} GiB, "
        f"trainable={results['trainable_parameters']}"
    )

    rollout_values: dict[int, dict[str, Any]] = {}
    for batch_size in (1, 2, 4):
        config["captioner"]["generation"]["rollout_batch_size"] = batch_size
        config["captioner"]["training"]["replay_batch_size"] = batch_size
        worker.cfg["generation"]["rollout_batch_size"] = batch_size
        worker.cfg["training"]["replay_batch_size"] = batch_size
        batch_record: dict[str, Any] = {"batch_size": batch_size}
        results["batches"][str(batch_size)] = batch_record

        def do_rollout() -> list[dict[str, Any]]:
            generated: list[dict[str, Any]] = []
            for seed_batch in _batches(args.seeds, batch_size):
                generated.extend(worker.generate_batch(audio_path, prompt, seed_batch))
            return generated

        samples, batch_record["rollout"] = stage(f"batch{batch_size}.rollout", do_rollout)
        if samples is None:
            batch_record["status"] = "rollout_failed"
            persist()
            continue
        batch_record["candidate_count"] = len(samples)
        if len(samples) != 4:
            raise RuntimeError(f"Batch {batch_size} retained {len(samples)} candidates instead of 4")
        trajectories = [sample["sampled_token_ids"] for sample in samples]

        def do_policy_replay() -> list[Any]:
            replayed: list[Any] = []
            for values in _batches(trajectories, batch_size):
                replayed.extend(worker.sampled_token_logprobs_batch(
                    audio_path, prompt, values, reference=False,
                ))
            return replayed

        policy, batch_record["policy_replay"] = stage(f"batch{batch_size}.policy_replay", do_policy_replay)

        def do_reference_replay() -> list[Any]:
            replayed: list[Any] = []
            for values in _batches(trajectories, batch_size):
                replayed.extend(worker.sampled_token_logprobs_batch(
                    audio_path, prompt, values, reference=True,
                ))
            return replayed

        reference, batch_record["reference_replay"] = stage(
            f"batch{batch_size}.reference_replay", do_reference_replay,
        )
        if policy is None or reference is None:
            batch_record["status"] = "replay_failed"
            persist()
            continue
        policy_lists = tensor_lists(policy)
        reference_lists = tensor_lists(reference)
        candidates = []
        for index, (sample, policy_values, reference_values) in enumerate(zip(
            samples, policy_lists, reference_lists, strict=True,
        )):
            error = max(abs(a - b) for a, b in zip(sample["old_token_logprobs"], policy_values, strict=True))
            candidate = {
                "candidate_id": f"{row['id']}::{index}",
                "generation_seed": args.seeds[index],
                "raw_text": sample["raw_text"],
                "sampled_token_ids": sample["sampled_token_ids"],
                "old_token_logprobs": sample["old_token_logprobs"],
                "ref_token_logprobs": reference_values,
                "behavior_replay_max_abs_error": error,
                "behavior_logprob_mode": "generation_scores",
                "trajectory_version": TRAJECTORY_VERSION,
                **{key: sample[key] for key in (
                    "finish_reason", "terminal_token_id", "terminated_by_eos",
                    "post_eos_token_count", "generated_token_count", "eos_token_id",
                )},
            }
            candidates.append(mark_trajectory(candidate, "caption"))
        max_length = max(len(value) for value in trajectories)
        batch_record.update({
            "status": "ok",
            "candidate_ids": [value["candidate_id"] for value in candidates],
            "candidate_seeds": [value["generation_seed"] for value in candidates],
            "generated_token_counts": [value["generated_token_count"] for value in candidates],
            "terminated_by_eos": [value["terminated_by_eos"] for value in candidates],
            "finish_reasons": [value["finish_reason"] for value in candidates],
            "padding_alignment_valid": all(
                len(value["sampled_token_ids"]) == max_length or value["terminated_by_eos"]
                for value in candidates
            ),
            "policy_replay_all_finite": all(finite(value) for value in policy_lists),
            "reference_replay_all_finite": all(finite(value) for value in reference_lists),
            "behavior_replay_max_abs_error": max(
                value["behavior_replay_max_abs_error"] for value in candidates
            ),
            "trajectory_valid_count": sum(bool(value["trajectory_valid"]) for value in candidates),
            "trajectory_errors": [caption_trajectory_errors(value) for value in candidates],
            "candidates": candidates,
        })
        rollout_values[batch_size] = {
            "trajectories": trajectories,
            "policy": policy_lists,
            "reference": reference_lists,
        }
        persist()

    if 1 not in rollout_values:
        raise RuntimeError("Batch-1 rollout/replay did not produce the fixed trajectory set")
    fixed = rollout_values[1]
    results["fixed_trajectory_source_batch"] = 1
    for batch_size in (1, 2, 4):
        fixed_record: dict[str, Any] = {}
        results["fixed_replay"][str(batch_size)] = fixed_record

        def replay_fixed_policy() -> list[Any]:
            replayed: list[Any] = []
            for values in _batches(fixed["trajectories"], batch_size):
                replayed.extend(worker.sampled_token_logprobs_batch(
                    audio_path, prompt, values, reference=False,
                ))
            return replayed

        policy, fixed_record["policy_replay"] = stage(
            f"fixed.batch{batch_size}.policy_replay", replay_fixed_policy,
        )

        def replay_fixed_reference() -> list[Any]:
            replayed: list[Any] = []
            for values in _batches(fixed["trajectories"], batch_size):
                replayed.extend(worker.sampled_token_logprobs_batch(
                    audio_path, prompt, values, reference=True,
                ))
            return replayed

        reference, fixed_record["reference_replay"] = stage(
            f"fixed.batch{batch_size}.reference_replay", replay_fixed_reference,
        )
        if policy is None or reference is None:
            fixed_record["status"] = "failed"
            persist()
            continue
        policy_lists = tensor_lists(policy)
        reference_lists = tensor_lists(reference)
        fixed_record.update({
            "status": "ok",
            "policy_all_finite": all(finite(value) for value in policy_lists),
            "reference_all_finite": all(finite(value) for value in reference_lists),
            "policy_max_abs_diff_vs_batch1": max(
                abs(a - b)
                for row_a, row_b in zip(policy_lists, fixed["policy"], strict=True)
                for a, b in zip(row_a, row_b, strict=True)
            ),
            "reference_max_abs_diff_vs_batch1": max(
                abs(a - b)
                for row_a, row_b in zip(reference_lists, fixed["reference"], strict=True)
                for a, b in zip(row_a, row_b, strict=True)
            ),
        })
        persist()

    advantages = [-1.3416407865, -0.4472135955, 0.4472135955, 1.3416407865]
    probe_candidates = [
        {
            "candidate_id": f"{row['id']}::{index}",
            "sampled_token_ids": fixed["trajectories"][index],
            "old_token_logprobs": fixed["policy"][index],
            "ref_token_logprobs": fixed["reference"][index],
            "advantage": advantages[index],
            "skip_update": False,
        }
        for index in range(4)
    ]
    group = {"id": str(row["id"]), "audio_path": audio_path, "prompt": prompt, "candidates": probe_candidates}
    baseline_gradients: dict[str, Any] | None = None
    baseline_updates: dict[str, Any] | None = None

    for batch_size in (1, 2, 4):
        worker.cfg["training"]["replay_batch_size"] = batch_size
        with torch.no_grad():
            for name, parameter in worker._current_adapter_named_parameters():
                parameter.copy_(initial_parameters[name].to(device=parameter.device, dtype=parameter.dtype))
        worker.model.set_adapter("default")
        worker.model.eval()
        worker.model.zero_grad(set_to_none=True)
        training = worker._training_config("grpo")
        optimizer = worker._optimizer(training)

        def do_current_step() -> dict[str, Any]:
            optimizer.zero_grad(set_to_none=True)
            candidate_results = []
            for values in _batches(probe_candidates, batch_size):
                candidate_results.extend(worker._native_candidates_backward(
                    group, values,
                    clip=float(training.get("clip_range", 0.2)),
                    kl_beta=float(training.get("kl_beta", 0.02)),
                    candidate_count=4,
                ))
            named = worker._current_adapter_named_parameters()
            gradient_snapshots = {
                name: parameter.grad.detach().cpu().clone()
                for name, parameter in named if parameter.grad is not None
            }
            gradient_family_nonzero = {"audio": 0, "text": 0}
            all_finite = True
            squared_norm = 0.0
            max_abs = 0.0
            for name, gradient in gradient_snapshots.items():
                all_finite = all_finite and bool(torch.isfinite(gradient).all())
                nonzero = bool(gradient.abs().max() > 0)
                if nonzero:
                    gradient_family_nonzero["audio" if "audio" in name.lower() else "text"] += 1
                squared_norm += float(gradient.double().square().sum())
                max_abs = max(max_abs, float(gradient.abs().max()))
            reference_gradients = [
                name for name, parameter in worker.model.named_parameters()
                if ".reference." in name and parameter.grad is not None and bool(parameter.grad.abs().max() > 0)
            ]
            clipped_norm = float(torch.nn.utils.clip_grad_norm_(
                worker.trainable_parameters, float(training.get("grad_clip", 1.0)),
            ).detach().cpu())
            optimizer.step()
            updated = {
                name: parameter.detach().cpu().clone()
                for name, parameter in worker._current_adapter_named_parameters()
            }
            update_max_abs = 0.0
            update_squared_norm = 0.0
            changed_tensors = 0
            for name, value in updated.items():
                delta = value.double() - initial_parameters[name].double()
                local_max = float(delta.abs().max()) if delta.numel() else 0.0
                update_max_abs = max(update_max_abs, local_max)
                update_squared_norm += float(delta.square().sum())
                changed_tensors += int(local_max > 0)
            return {
                "candidate_results": candidate_results,
                "loss": sum(value[0] for value in candidate_results) / len(candidate_results),
                "current_behavior_replay_max_abs_error": max(value[1] for value in candidate_results),
                "mean_kl": sum(value[2] for value in candidate_results) / len(candidate_results),
                "gradient_all_finite": all_finite,
                "gradient_nonzero_tensors": sum(
                    bool(value.abs().max() > 0) for value in gradient_snapshots.values()
                ),
                "gradient_family_nonzero": gradient_family_nonzero,
                "gradient_l2_norm": math.sqrt(squared_norm),
                "gradient_max_abs": max_abs,
                "clipped_grad_norm": clipped_norm,
                "reference_gradient_tensors": len(reference_gradients),
                "reference_gradient_names": reference_gradients,
                "parameter_changed_tensors": changed_tensors,
                "parameter_update_l2_norm": math.sqrt(update_squared_norm),
                "parameter_update_max_abs": update_max_abs,
                "_gradient_snapshots": gradient_snapshots,
                "_updated": updated,
            }

        update, metrics = stage(f"batch{batch_size}.current_replay_optimizer_step", do_current_step)
        record = {"current_replay": metrics}
        results["optimizer_steps"][str(batch_size)] = record
        if update is None:
            record["status"] = "failed"
            persist()
            continue
        gradients = update.pop("_gradient_snapshots")
        updated = update.pop("_updated")
        record.update({"status": "ok", **update})
        if baseline_gradients is None:
            baseline_gradients = gradients
            baseline_updates = updated
            record["gradient_max_abs_diff_vs_batch1"] = 0.0
            record["parameter_max_abs_diff_vs_batch1"] = 0.0
        else:
            record["gradient_max_abs_diff_vs_batch1"] = max(
                float((gradients[name].float() - baseline_gradients[name].float()).abs().max())
                for name in baseline_gradients
            )
            record["parameter_max_abs_diff_vs_batch1"] = max(
                float((updated[name].float() - baseline_updates[name].float()).abs().max())
                for name in baseline_updates
            )
        persist()
        del gradients, updated
        gc.collect()

    results["rollout_token_identity_vs_batch1"] = {
        str(batch_size): [
            trajectories == baseline
            for trajectories, baseline in zip(
                rollout_values[batch_size]["trajectories"], fixed["trajectories"], strict=True,
            )
        ]
        for batch_size in rollout_values
    }
    results["contracts"] = {
        "group_size_preserved": all(
            value.get("candidate_count") == 4 for value in results["batches"].values()
            if value.get("status") == "ok"
        ),
        "behavior_replay_within_5e4": all(
            value.get("behavior_replay_max_abs_error", float("inf")) <= 5e-4
            for value in results["batches"].values() if value.get("status") == "ok"
        ),
        "all_successful_trajectories_valid": all(
            value.get("trajectory_valid_count") == 4 for value in results["batches"].values()
            if value.get("status") == "ok"
        ),
        "optimizer_reference_has_no_gradient": all(
            value.get("reference_gradient_tensors") == 0
            for value in results["optimizer_steps"].values() if value.get("status") == "ok"
        ),
        "optimizer_audio_text_gradients_nonzero": all(
            value.get("gradient_family_nonzero", {}).get("audio", 0) > 0
            and value.get("gradient_family_nonzero", {}).get("text", 0) > 0
            for value in results["optimizer_steps"].values() if value.get("status") == "ok"
        ),
    }
    results["status"] = "complete" if not results["errors"] else "complete_with_failures"
    persist()
    log(f"Probe finished with status={results['status']}")


if __name__ == "__main__":
    main()
