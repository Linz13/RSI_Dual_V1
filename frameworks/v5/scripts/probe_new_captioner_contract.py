#!/usr/bin/env python3
"""Run the bounded single-H100 Captioner training contract for a new branch.

This is deliberately not an 8-rank launcher.  It exercises one real paired
audio, four sampled candidates, exact behavior/policy/reference replay, the
configured LoRA audit, one GRPO step, one SFT step, non-zero audio/text
gradients, and adapter reload.  ``--max-new-tokens`` keeps this preparation
probe bounded; production configs remain at 384 tokens.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

from dual_isl_train.config import load_config
from dual_isl_train.constants import SOURCE_DOMAIN_TARGET
from dual_isl_train.data import load_records
from dual_isl_train.distributed import DistributedContext
from dual_isl_train.render import caption_prompt


def _worker_for(kind: str):
    if kind == "midasheng_0p6b":
        from scripts.midasheng_0p6b_captioner_candidate import MiDasheng06CaptionerWorker
        return MiDasheng06CaptionerWorker
    if kind == "qwen2_5_omni_3b":
        from scripts.qwen2_5_omni_3b_captioner_candidate import Qwen25Omni3BCaptionerWorker
        return Qwen25Omni3BCaptionerWorker
    raise ValueError(f"Unsupported --kind: {kind}")


def _max_abs_difference(left: dict[str, object], right: dict[str, object]) -> float:
    import torch

    names = sorted(set(left) & set(right))
    if not names:
        raise RuntimeError("No common adapter tensors found during reload audit")
    return max(float((left[name].float() - right[name].float()).abs().max()) for name in names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["midasheng_0p6b", "qwen2_5_omni_3b"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()
    if args.max_new_tokens < 2:
        raise SystemExit("--max-new-tokens must be >= 2")

    config = load_config(args.config)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Probe output directory must be new and empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(config["data"]["paired_path"])
    if not records:
        raise RuntimeError("No paired records found for Captioner probe")
    row = dict(records[0])
    row["prompt"] = caption_prompt()
    row["group_size"] = 4
    row["candidate_seeds"] = [101, 202, 303, 404]
    row["caption_schema"] = "full"
    row["target_origin"] = SOURCE_DOMAIN_TARGET
    row["target_schema"] = "source_no_environment"
    Worker = _worker_for(args.kind)
    distributed = DistributedContext()
    worker = None
    report: dict[str, object] = {
        "status": "started", "kind": args.kind, "config": str(Path(args.config).resolve()),
        "model_path": config["captioner"]["model_path"], "audio_path": row["audio_path"],
        "requested_candidates": 4, "probe_max_new_tokens": args.max_new_tokens,
    }
    try:
        worker = Worker(config, distributed=distributed)
        worker.cfg["generation"]["max_new_tokens"] = args.max_new_tokens
        preflight = worker.preflight()[0]
        report["preflight"] = preflight
        trainable = [name for name, parameter in worker.model.named_parameters() if parameter.requires_grad]
        report["trainable_parameter_tensors"] = len(trainable)
        report["trainable_audio_tensors"] = sum("audio" in name.lower() for name in trainable)
        report["trainable_text_tensors"] = sum("audio" not in name.lower() for name in trainable)
        if not report["trainable_audio_tensors"] or not report["trainable_text_tensors"]:
            raise RuntimeError("LoRA audit did not find both audio and text trainable tensors")

        rollout = worker.rollout([row])
        if len(rollout) != 1 or len(rollout[0].get("candidates", [])) != 4:
            raise RuntimeError("Captioner probe did not produce exactly four candidates")
        group = rollout[0]
        report["candidate_parse"] = {
            "count": len(group["candidates"]),
            "parse_errors": [len(item.get("parse_errors", [])) for item in group["candidates"]],
            "trajectory_valid": [bool(item.get("trajectory_valid")) for item in group["candidates"]],
        }
        report["behavior_replay_max_abs_error"] = max(
            float(item["behavior_replay_max_abs_error"]) for item in group["candidates"]
        )
        report["policy_reference_max_abs_error"] = max(
            float(item["policy_reference_max_abs_error"]) for item in group["candidates"]
        )
        if report["behavior_replay_max_abs_error"] > 5e-4:
            raise RuntimeError(f"Behavior replay mismatch: {report['behavior_replay_max_abs_error']}")
        # The candidates are intentionally assigned a non-flat synthetic probe
        # reward.  This tests the native GRPO optimizer contract independently
        # of the full RewardV2 critic/TTS loop.
        for index, candidate in enumerate(group["candidates"]):
            candidate["skip_update"] = False
            candidate["advantage"] = [-1.5, -0.5, 0.5, 1.5][index]
            candidate["reward"] = float(candidate["advantage"])
        grpo_checkpoint = output_dir / "grpo_adapter"
        grpo_result = worker.grpo_update([group], str(grpo_checkpoint), phase="grpo")
        report["grpo"] = grpo_result[0] if grpo_result else {"status": "missing"}
        if not grpo_result or grpo_result[0].get("parameter_before") == grpo_result[0].get("parameter_after"):
            raise RuntimeError("GRPO probe did not change trainable adapter parameters")

        sft_checkpoint = output_dir / "sft_adapter"
        sft_result = worker.sft_update([row], str(sft_checkpoint), phase="cycle_sft")
        report["sft"] = sft_result[0] if sft_result else {"status": "missing"}
        if not sft_result or sft_result[0].get("parameter_before") == sft_result[0].get("parameter_after"):
            raise RuntimeError("SFT probe did not change trainable adapter parameters")
        gradient_families = sft_result[0].get("gradient_family_nonzero", {})
        report["gradient_family_nonzero"] = gradient_families
        if int(gradient_families.get("audio", 0)) < 1 or int(gradient_families.get("text", 0)) < 1:
            raise RuntimeError(f"Audio/text gradient contract failed: {gradient_families}")

        saved_tensors = {
            name: parameter.detach().cpu().clone()
            for name, parameter in worker.model.named_parameters()
            if parameter.requires_grad and ".default." in name
        }
        worker.model = None
        gc.collect()
        worker.torch.cuda.empty_cache()
        worker = Worker(config, checkpoint=str(sft_checkpoint), distributed=distributed)
        reloaded_tensors = {
            name: parameter.detach().cpu().clone()
            for name, parameter in worker.model.named_parameters()
            if parameter.requires_grad and ".default." in name
        }
        reload_error = _max_abs_difference(saved_tensors, reloaded_tensors)
        report["adapter_reload_max_abs_error"] = reload_error
        if reload_error > 0.0:
            raise RuntimeError(f"Adapter reload mismatch: {reload_error}")
        report["status"] = "ok"
        (output_dir / "contract_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8",
        )
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    finally:
        if worker is not None:
            worker.model = None
            gc.collect()
            try:
                worker.torch.cuda.empty_cache()
            except Exception:
                pass


if __name__ == "__main__":
    main()
