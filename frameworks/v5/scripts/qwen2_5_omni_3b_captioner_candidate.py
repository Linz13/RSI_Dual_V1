#!/usr/bin/env python3
"""Qwen2.5-Omni-3B thinker-only RewardV2 Captioner worker.

The implementation inherits the already validated Qwen Captioner replay,
GRPO, SFT, parsing, trajectory and adapter-audit logic.  Only model loading,
processor construction and the exact LoRA module expression are specialized
here.  The full Qwen2.5-Omni container/talker is never instantiated: the
official Transformers thinker-only class is loaded directly.
"""

from __future__ import annotations

from typing import Any

from dual_isl_train.distributed import DistributedContext, run_sharded_inference
from dual_isl_train.workers.common import load_job, worker_parser, write_output
from dual_isl_train.workers.qwen3_captioner import (
    Qwen3CaptionerWorker,
    _close_captioner_worker,
)


class Qwen25Omni3BCaptionerWorker(Qwen3CaptionerWorker):
    """Qwen2.5-Omni text thinker with audio tower adapters only."""

    def __init__(
        self,
        config: dict[str, Any],
        checkpoint: str = "",
        target_checkpoint: str = "",
        distributed: DistributedContext | None = None,
    ):
        import torch
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
        from qwen_omni_utils import process_mm_info

        self.torch = torch
        self.cfg = config["captioner"]
        self.seed = int(config["run"].get("seed", 42))
        self.parent_checkpoint = checkpoint or None
        self.distributed = distributed or DistributedContext()
        self.device = torch.device(self.distributed.device or str(self.cfg.get("device", "cuda:0")))
        if self.device.type != "cuda":
            raise ValueError("Qwen2.5-Omni Captioner smoke training requires a CUDA device")
        self.ddp_model = None
        self.output_metrics: dict[str, Any] = {}
        self._process_mm_info = process_mm_info

        dtype_name = str(self.cfg.get("dtype", "bfloat16")).lower()
        dtype = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(dtype_name)
        if dtype is None:
            raise ValueError(f"Unsupported Qwen2.5-Omni Captioner dtype: {dtype_name}")

        model_path = str(self.cfg["model_path"])
        # Qwen2.5-Omni publishes this class as the supported text-only path.
        # Loading it directly also prevents any talker/voice decoder parameters
        # from entering the optimizer or the adapter audit.
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            device_map={"": str(self.device)},
            attn_implementation=self.cfg.get("attn_implementation", "flash_attention_2"),
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True,
        )
        self.tokenizer = self.processor.tokenizer
        eos_token = str(self.cfg.get("generation", {}).get("eos_token", "<|im_end|>"))
        self.eos_token_id = int(self.tokenizer.convert_tokens_to_ids(eos_token))
        if self.eos_token_id < 0 or self.tokenizer.eos_token_id != self.eos_token_id:
            raise RuntimeError(
                f"Qwen2.5 EOS mismatch: {eos_token}={self.eos_token_id}, "
                f"tokenizer.eos_token_id={self.tokenizer.eos_token_id}"
            )
        self.pad_token_id = int(self.tokenizer.pad_token_id)
        if self.pad_token_id < 0:
            raise RuntimeError("Qwen2.5 tokenizer has no valid pad_token_id")
        self.has_reference_adapter = False
        self.adapter_audits: dict[str, Any] = {}
        self._attach_or_load_lora(checkpoint, target_checkpoint)
        for peft_config in getattr(self.model, "peft_config", {}).values():
            peft_config.base_model_name_or_path = model_path
        if bool(self.cfg.get("gradient_checkpointing", False)) and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
        self.model.eval()
        self.freeze_batchnorm()
        self._audit_adapters()
        self._broadcast_current_adapter_parameters()

    def _lora_config(self):
        from peft import LoraConfig

        lora = self.cfg.get("lora", {})
        # Peft treats a string target_modules value as a full regex.  Anchors
        # make this an exact audit: no vision, talker, or text MLP modules can
        # be selected accidentally by a broad suffix match.
        targets = lora.get("target_modules") or (
            r"^model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$|"
            r"^audio_tower\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|out_proj)$"
        )
        return LoraConfig(
            r=int(lora.get("r", 8)),
            lora_alpha=int(lora.get("alpha", 16)),
            lora_dropout=float(lora.get("dropout", 0.0)),
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        )

    @staticmethod
    def _messages(audio_path: str, prompt: str, completion: str | None = None) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt},
            ],
        }]
        if completion is not None:
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": completion}],
            })
        return messages

    def _inputs(self, audio_path: str, prompt: str, completion: str | None = None) -> dict[str, Any]:
        conversation = self._messages(audio_path, prompt, completion)
        rendered = self.processor.apply_chat_template(
            conversation, add_generation_prompt=completion is None, tokenize=False,
        )
        audios, _, _ = self._process_mm_info(conversation, use_audio_in_video=False)
        values = self.processor(
            text=rendered,
            audio=audios,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        built: dict[str, Any] = {}
        for key, value in values.items():
            built[key] = value.to(self.device) if hasattr(value, "to") else value
        model_dtype = next(self.model.parameters()).dtype
        for key in ("input_features", "pixel_values", "pixel_values_videos"):
            if key in built:
                built[key] = built[key].to(dtype=model_dtype)
        return built

    def preflight(self) -> list[dict[str, Any]]:
        result = super().preflight()[0]
        result.update({
            "backend": "Qwen2_5OmniThinkerForConditionalGeneration",
            "model_mode": "eval" if not self.model.training else "train",
            "thinker_only": True,
            "talker_loaded": False,
            "vision_adapters_allowed": False,
            "audio_lora_target_expression": str(self.cfg.get("lora", {}).get("target_modules", "")),
        })
        return [result]


def main() -> None:
    parser = worker_parser(
        "Qwen2.5-Omni-3B thinker-only structured caption worker",
        ["rollout", "score", "grpo-update", "sft-update", "gradient-audit", "preflight"],
    )
    args = parser.parse_args()
    config, rows = load_job(args)
    distributed = DistributedContext.initialize(config)
    worker: Qwen25Omni3BCaptionerWorker | None = None
    try:
        worker = Qwen25Omni3BCaptionerWorker(
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
            "thinker_only": True,
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
