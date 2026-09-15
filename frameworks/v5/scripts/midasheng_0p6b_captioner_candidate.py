#!/usr/bin/env python3
"""MiDasheng-0.6B RewardV2 Captioner entry point.

The replay/GRPO/SFT implementation remains the validated MiDasheng worker.
This thin branch-only subclass provides the 0.6B audio-frontend boundary.
The downloaded snapshot is kept in FP32: casting this particular FP32
checkpoint to BF16 produces non-finite decoder logits on the real audio path.
The MiDasheng-7B worker and its module hash are not changed.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

from dual_isl_train.distributed import DistributedContext, run_sharded_inference
from dual_isl_train.workers.common import load_job, worker_parser, write_output
from dual_isl_train.workers.qwen3_captioner import _close_captioner_worker
from scripts.midasheng_captioner_candidate import MiDashengCaptionerWorker


def _configure_rank_local_hf_modules_cache(config: dict[str, Any]) -> str:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    modules_cache = (
        Path(config["run"]["output_dir"]).resolve()
        / ".hf_modules_cache"
        / f"rank_{local_rank:03d}"
    )
    modules_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_MODULES_CACHE"] = str(modules_cache)
    return str(modules_cache)


class MiDasheng06CaptionerWorker(MiDashengCaptionerWorker):
    """MiDashengLM-0.6B with a branch-local audio-frontend boundary."""

    def __init__(
        self,
        config: dict[str, Any],
        checkpoint: str = "",
        target_checkpoint: str = "",
        distributed: DistributedContext | None = None,
    ):
        # AutoModel trust_remote_code copies the local modeling files into the
        # Transformers modules cache.  Eight ranks writing one cache can expose
        # a partially initialized module to another rank, so isolate it by run
        # and LOCAL_RANK before Transformers is imported by the parent worker.
        self.hf_modules_cache = _configure_rank_local_hf_modules_cache(config)
        super().__init__(config, checkpoint, target_checkpoint, distributed)
        base_model = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        original = base_model._forward_audio_encoder
        dtype = next(base_model.parameters()).dtype

        def autocast_audio(audios: Any, audio_length: Any):
            if dtype in (self.torch.float16, self.torch.bfloat16):
                with self.torch.autocast(device_type="cuda", dtype=dtype):
                    return original(audios, audio_length)
            with contextlib.nullcontext():
                return original(audios, audio_length)

        base_model._forward_audio_encoder = autocast_audio
        self.audio_frontend_autocast_dtype = str(dtype)

    def preflight(self) -> list[dict[str, Any]]:
        result = super().preflight()[0]
        result["backend"] = "MiDashengLMModel-0.6B"
        result["audio_frontend_autocast_dtype"] = self.audio_frontend_autocast_dtype
        result["hf_modules_cache"] = self.hf_modules_cache
        result["model_precision_source"] = "FP32 ModelScope snapshot kept as FP32 (BF16 logits probe was non-finite)"
        return [result]


def main() -> None:
    parser = worker_parser(
        "MiDasheng-0.6B structured caption worker",
        ["rollout", "score", "grpo-update", "sft-update", "gradient-audit", "preflight"],
    )
    args = parser.parse_args()
    config, rows = load_job(args)
    distributed = DistributedContext.initialize(config)
    worker: MiDasheng06CaptionerWorker | None = None
    try:
        worker = MiDasheng06CaptionerWorker(
            config, args.checkpoint_in, args.target_checkpoint, distributed,
        )
        extra_metrics: dict[str, Any] = {}
        if args.action == "rollout":
            if distributed.enabled:
                output, extra_metrics = run_sharded_inference(rows, args.output, distributed, worker.rollout)
            else:
                output = worker.rollout(rows)
        elif args.action == "score":
            if distributed.enabled:
                output, extra_metrics = run_sharded_inference(rows, args.output, distributed, worker.score)
            else:
                output = worker.score(rows)
        elif args.action == "grpo-update":
            output = worker.grpo_update(rows, args.checkpoint_out, args.training_phase)
            extra_metrics = worker.output_metrics
        elif args.action == "sft-update":
            output = worker.sft_update(rows, args.checkpoint_out, args.training_phase)
            extra_metrics = worker.output_metrics
        elif args.action == "gradient-audit":
            output = worker.gradient_audit(rows)
        else:
            output = worker.preflight()
        extra_metrics["captioner_batching"] = {
            "rollout_batch_size": int(worker.cfg.get("generation", {}).get("rollout_batch_size", 1)),
            "replay_batch_size": int(worker.cfg.get("training", {}).get("replay_batch_size", 1)),
            "group_size_preserved": 4,
            "completion_only_sequences_supported": True,
            "all_policy_reference_current_replays_preserved": True,
        }
        extra_metrics["adapter_audits"] = worker.adapter_audits
        if not distributed.enabled or (distributed.is_main and args.action not in {"rollout", "score"}):
            write_output(args.output, output, extra_metrics)
        elif distributed.is_main:
            from dual_isl_train.io import atomic_json
            atomic_json(str(args.output) + ".metrics.json", extra_metrics)
    finally:
        _close_captioner_worker(worker, distributed)


if __name__ == "__main__":
    main()
