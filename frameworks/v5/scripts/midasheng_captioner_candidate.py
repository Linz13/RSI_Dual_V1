#!/usr/bin/env python3
"""Isolated production-candidate MiDasheng Captioner worker.

This module deliberately lives outside ``dual_isl_train/`` while the active
Qwen run may still need to resume.  StageManager hashes the package, so keeping
the candidate here prevents MiDasheng development from changing that run's
implementation hash.  Promote this file into the package only after the Qwen
run no longer needs the old hash.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from dual_isl_train.distributed import DistributedContext, run_sharded_inference
from dual_isl_train.workers.common import load_job, worker_parser, write_output
from dual_isl_train.workers.qwen3_captioner import (
    Qwen3CaptionerWorker,
    _close_captioner_worker,
)


class MiDashengCaptionerWorker(Qwen3CaptionerWorker):
    """MiDashengLM worker with exact completion-only sampled-ID replay."""

    def __init__(
        self,
        config: dict[str, Any],
        checkpoint: str = "",
        target_checkpoint: str = "",
        distributed: DistributedContext | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        self.torch = torch
        self.cfg = config["captioner"]
        self.seed = int(config["run"].get("seed", 42))
        self.parent_checkpoint = checkpoint or None
        self.distributed = distributed or DistributedContext()
        self.device = torch.device(self.distributed.device or str(self.cfg.get("device", "cuda:0")))
        if self.device.type != "cuda":
            raise ValueError("MiDasheng training requires a CUDA device")
        self.ddp_model = None
        self.output_metrics: dict[str, Any] = {}

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
            raise ValueError(f"Unsupported MiDasheng dtype: {dtype_name}")

        model_path = str(self.cfg["model_path"])
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            dtype=dtype,
            device_map={"": str(self.device)},
            attn_implementation=self.cfg.get("attn_implementation", "sdpa"),
            low_cpu_mem_usage=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True,
        )
        configured_eos = self.model.generation_config.eos_token_id
        self.eos_token_ids = (
            [int(value) for value in configured_eos]
            if isinstance(configured_eos, (list, tuple)) else [int(configured_eos)]
        )
        im_end = int(self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        self.eos_token_id = im_end if im_end in self.eos_token_ids else self.eos_token_ids[-1]
        self.pad_token_id = int(
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None else self.eos_token_ids[0]
        )
        self.has_reference_adapter = False
        self.adapter_audits: dict[str, Any] = {}
        self._attach_or_load_lora(checkpoint, target_checkpoint)
        for peft_config in getattr(self.model, "peft_config", {}).values():
            peft_config.base_model_name_or_path = model_path
        if bool(self.cfg.get("gradient_checkpointing", False)) and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        self.model.eval()
        self.freeze_batchnorm()
        self._audit_adapters()
        self._broadcast_current_adapter_parameters()

    def _lora_config(self):
        from peft import LoraConfig

        lora = self.cfg.get("lora", {})
        targets = lora.get("target_modules") or "|".join([
            r"^decoder\.model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$",
            r"^decoder\.model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)$",
            r"^audio_projector\.net\.(0|2)$",
        ])
        return LoraConfig(
            r=int(lora.get("r", 4)),
            lora_alpha=int(lora.get("alpha", 8)),
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
                {"type": "text", "text": prompt},
                {"type": "audio", "path": audio_path},
            ],
        }]
        if completion is not None:
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": completion}],
            })
        return messages

    def _inputs(self, audio_path: str, prompt: str, completion: str | None = None) -> dict[str, Any]:
        values = self.processor.apply_chat_template(
            self._messages(audio_path, prompt, completion),
            tokenize=True,
            add_generation_prompt=completion is None,
            add_special_tokens=True,
            return_dict=True,
        )
        dtype = next(self.model.parameters()).dtype
        built: dict[str, Any] = {}
        for key, value in values.items():
            if hasattr(value, "to"):
                value = value.to(self.device)
                if getattr(value, "is_floating_point", lambda: False)():
                    value = value.to(dtype=dtype)
            built[key] = value
        return built

    @staticmethod
    def _completion_from_sequence(torch_module: Any, sequence: Any, prompt_ids: Any, expected: list[int]):
        sequence = sequence.detach().long()
        target = torch_module.tensor(expected, device=sequence.device, dtype=sequence.dtype)
        if torch_module.equal(sequence, target):
            return sequence, "completion_only"
        prefix = int(prompt_ids.numel())
        if sequence.numel() >= prefix and torch_module.equal(sequence[:prefix], prompt_ids):
            completion = sequence[prefix:prefix + target.numel()]
            if torch_module.equal(completion, target):
                return completion, "prompt_plus_completion"
        raise RuntimeError("MiDasheng replay did not preserve sampled token IDs")

    def _warped_logprobs(self, input_ids: Any, scores: Any, temperature: float):
        warped = self._sampling_warpers(temperature)(input_ids, scores.float())
        return self.torch.log_softmax(warped, dim=-1)

    def generate_batch(self, audio_path, prompt, seeds, *, temperature=None):
        from dual_isl_train.workers.midasheng_batch import generate_batch
        fallback = getattr(self, "_v5_rollout_serial_reason", None)
        if fallback:
            rows = self._serial_generate_batch(audio_path, prompt, seeds, temperature=temperature)
            for row in rows:
                row.update(generation_batch_size=1, generation_batch_fallback=fallback)
            return rows
        try:
            return generate_batch(self, audio_path, prompt, seeds, temperature)
        except self.torch.cuda.OutOfMemoryError:
            if not self.cfg.get("generation", {}).get("batch_fallback", True):
                raise
        # Leave the exception scope so the failed batch's traceback releases GPU tensors.
        import gc
        gc.collect()
        self.torch.cuda.empty_cache()
        rows = self._serial_generate_batch(audio_path, prompt, seeds, temperature=temperature)
        for row in rows:
            row.update(generation_batch_size=1, generation_batch_fallback="cuda_oom")
        return rows

    def rollout(self, rows):
        output = []
        for row in rows:
            group = super().rollout([row])[0]
            rejected = [c for c in group["candidates"]
                        if c.get("generation_batch_size", 1) > 1 and not c["trajectory_valid"]]
            if rejected:
                # Batched bf16 decoding can differ from the unchanged single-sequence
                # GRPO replay. Resample serially; never replace behavior probabilities
                # with replay values or relax the trajectory-consistency threshold.
                if not self.cfg.get("generation", {}).get("batch_fallback", True):
                    raise RuntimeError("MiDasheng batched rollout failed trajectory validation")
                self._v5_rollout_serial_reason = "trajectory_mismatch"
                errors = [c["behavior_replay_max_abs_error"] for c in rejected]
                print(f"MIDASHENG_BATCH_FALLBACK id={row['id']} reason=trajectory_mismatch "
                      f"max_replay_error={max(errors)}", flush=True)
                group = super().rollout([row])[0]
                group["batch_fallback"] = self._v5_rollout_serial_reason
                group["rejected_batch_replay_max_abs_error"] = max(errors)
                for candidate in group["candidates"]:
                    candidate["generation_attempts"] = 2
            output.append(group)
        return output

    def _serial_generate_batch(
        self,
        audio_path: str,
        prompt: str,
        seeds: list[int],
        *,
        temperature: float | None = None,
    ) -> list[dict[str, Any]]:
        from transformers import LogitsProcessorList

        generation = self.cfg.get("generation", {})
        temperature = float(generation.get("temperature", 1.0) if temperature is None else temperature)
        results = []
        for seed in seeds:
            values = self._inputs(audio_path, prompt)
            generator = self.torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
            captured_ids: list[int] = []
            captured_logps: list[float] = []

            class SampleAndForce:
                def __call__(processor_self, input_ids, scores):
                    logps = self._warped_logprobs(input_ids, scores, temperature)
                    if temperature > 0:
                        target = self.torch.multinomial(
                            logps.exp(), num_samples=1, generator=generator,
                        )[0, 0]
                    else:
                        target = logps.argmax(dim=-1)[0]
                    captured_ids.append(int(target.detach().cpu()))
                    captured_logps.append(float(logps[0, target].detach().cpu()))
                    forced = self.torch.full_like(scores, -self.torch.inf)
                    forced[0, target] = 0
                    return forced

            self.model.set_adapter("default")
            self.model.eval()
            self.freeze_batchnorm()
            with self.torch.no_grad():
                output = self.model.generate(
                    **values,
                    max_new_tokens=int(generation.get("max_new_tokens", 384)),
                    do_sample=False,
                    use_cache=True,
                    logits_processor=LogitsProcessorList([SampleAndForce()]),
                    eos_token_id=self.eos_token_ids,
                    pad_token_id=self.pad_token_id,
                    return_dict_in_generate=True,
                )
            completion, sequence_mode = self._completion_from_sequence(
                self.torch, output.sequences[0], values["input_ids"][0], captured_ids,
            )
            token_ids = completion.cpu().tolist()
            eos_positions = [index for index, token_id in enumerate(token_ids) if token_id in self.eos_token_ids]
            token_count = eos_positions[0] + 1 if eos_positions else len(token_ids)
            token_ids = token_ids[:token_count]
            token_logps = captured_logps[:token_count]
            terminated = bool(eos_positions and eos_positions[0] == token_count - 1)
            results.append({
                "raw_text": self.tokenizer.decode(
                    token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
                ).strip(),
                "sampled_token_ids": token_ids,
                "old_token_logprobs": token_logps,
                "finish_reason": "eos" if terminated else "length",
                "terminal_token_id": token_ids[-1] if token_ids else None,
                "terminated_by_eos": terminated,
                "post_eos_token_count": 0,
                "generated_token_count": len(token_ids),
                "eos_token_id": self.eos_token_id,
                "sequence_mode": sequence_mode,
            })
        return results

    def sampled_token_logprobs_batch(
        self,
        audio_path: str,
        prompt: str,
        sampled_token_ids_batch: list[list[int]],
        *,
        reference: bool = False,
        grad: bool = False,
    ) -> list[Any]:
        if grad:
            raise RuntimeError("MiDasheng gradient replay must use the bounded native GRPO path")
        from transformers import LogitsProcessorList

        generation = self.cfg.get("generation", {})
        temperature = float(generation.get("temperature", 1.0))
        results = []
        for sampled_ids in sampled_token_ids_batch:
            if not sampled_ids:
                results.append(self.torch.empty(0, device=self.device, dtype=self.torch.float32))
                continue
            values = self._inputs(audio_path, prompt)
            targets = self.torch.tensor(sampled_ids, device=self.device, dtype=self.torch.long)
            captured: list[Any] = []

            class CaptureAndForce:
                def __init__(processor_self):
                    processor_self.index = 0

                def __call__(processor_self, input_ids, scores):
                    index = processor_self.index
                    if index >= targets.numel():
                        raise RuntimeError("MiDasheng replay exceeded sampled tokens")
                    logps = self._warped_logprobs(input_ids, scores, temperature)
                    target = targets[index]
                    captured.append(logps[0, target].detach())
                    forced = self.torch.full_like(scores, -self.torch.inf)
                    forced[0, target] = 0
                    processor_self.index += 1
                    return forced

            capture = CaptureAndForce()
            self.model.eval()
            with self.torch.no_grad(), self.adapter_context(reference):
                output = self.model.generate(
                    **values,
                    max_new_tokens=len(sampled_ids),
                    do_sample=False,
                    use_cache=True,
                    logits_processor=LogitsProcessorList([capture]),
                    eos_token_id=self.eos_token_ids,
                    pad_token_id=self.pad_token_id,
                    return_dict_in_generate=True,
                )
            completion, _ = self._completion_from_sequence(
                self.torch, output.sequences[0], values["input_ids"][0], sampled_ids,
            )
            if capture.index != targets.numel() or completion.numel() != targets.numel():
                raise RuntimeError("MiDasheng replay token count mismatch")
            results.append(self.torch.stack(captured).float())
        return results

    @staticmethod
    def _detach_cache(cache: Any) -> None:
        layers = getattr(cache, "layers", None)
        if layers is not None:
            for layer in layers:
                if getattr(layer, "keys", None) is not None:
                    layer.keys = layer.keys.detach()
                if getattr(layer, "values", None) is not None:
                    layer.values = layer.values.detach()
            return
        if isinstance(cache, (tuple, list)):
            for layer in cache:
                if isinstance(layer, (tuple, list)):
                    for tensor in layer:
                        if hasattr(tensor, "detach_"):
                            tensor.detach_()

    def _native_candidates_backward(
        self,
        group: dict[str, Any],
        candidates: list[dict[str, Any]],
        *,
        clip: float,
        kl_beta: float,
        candidate_count: int,
    ) -> list[tuple[float, float, float]]:
        return [
            self._native_candidate_exact_backward(
                group, candidate, clip=clip, kl_beta=kl_beta, candidate_count=candidate_count,
            )
            for candidate in candidates
        ]

    def _native_candidate_exact_backward(
        self,
        group: dict[str, Any],
        candidate: dict[str, Any],
        *,
        clip: float,
        kl_beta: float,
        candidate_count: int,
    ) -> tuple[float, float, float]:
        from transformers import LogitsProcessorList

        targets = self.torch.tensor(candidate["sampled_token_ids"], device=self.device, dtype=self.torch.long)
        old = self.torch.tensor(candidate["old_token_logprobs"], device=self.device, dtype=self.torch.float32)
        reference = self.torch.tensor(candidate["ref_token_logprobs"], device=self.device, dtype=self.torch.float32)
        if targets.numel() != old.numel() or targets.numel() != reference.numel() or not targets.numel():
            raise RuntimeError("MiDasheng GRPO policy/behavior/reference token counts differ")

        values = self._inputs(group["audio_path"], group["prompt"])
        generation_model = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        prompt_ids = values.pop("input_ids")
        inputs_embeds = generation_model._prepare_inputs_embeds(
            input_ids=prompt_ids,
            input_values=values.pop("input_values", None),
            inputs_embeds=values.pop("inputs_embeds", None),
            audio_length=values.pop("audio_length", None),
        )
        decoder = generation_model.decoder
        temperature = float(self.cfg.get("generation", {}).get("temperature", 1.0))
        chunk_size = int(self.cfg.get("training", {}).get("grpo_backward_chunk_tokens", 16))
        pending: list[Any] = []
        losses: list[Any] = []
        errors: list[Any] = []
        kls: list[Any] = []

        class ForceBackward:
            def __init__(processor_self):
                processor_self.index = 0

            def __call__(processor_self, input_ids, scores):
                index = processor_self.index
                if index >= targets.numel():
                    raise RuntimeError("MiDasheng GRPO decoder replay exceeded sampled tokens")
                logps = self._warped_logprobs(input_ids, scores, temperature)
                target = targets[index]
                current = logps[0, target]
                ratio = self.torch.exp(self.torch.clamp(current - old[index], -10, 10))
                advantage = self.torch.tensor(float(candidate["advantage"]), device=self.device)
                policy = -self.torch.minimum(
                    ratio * advantage,
                    self.torch.clamp(ratio, 1 - clip, 1 + clip) * advantage,
                )
                log_ratio = reference[index] - current
                kl = self.torch.exp(self.torch.clamp(log_ratio, -10, 10)) - log_ratio - 1
                loss = policy + kl_beta * kl
                pending.append(loss / (candidate_count * targets.numel()))
                losses.append(loss.detach())
                errors.append((current.detach() - old[index]).abs())
                kls.append(kl.detach())
                forced = self.torch.full_like(scores, -self.torch.inf)
                forced[0, target] = 0
                processor_self.index += 1
                if processor_self.index % chunk_size == 0 or processor_self.index == targets.numel():
                    self.torch.stack(pending).sum().backward()
                    pending.clear()
                return forced

        original_update = decoder._update_model_kwargs_for_generation
        had_instance_update = "_update_model_kwargs_for_generation" in decoder.__dict__
        previous_instance_update = decoder.__dict__.get("_update_model_kwargs_for_generation")
        generation_index = 0

        def detached_update(outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1):
            nonlocal generation_index
            updated = original_update(outputs, model_kwargs, is_encoder_decoder, num_new_tokens)
            if (generation_index + 1) % chunk_size == 0:
                self._detach_cache(updated.get("past_key_values"))
            generation_index += 1
            return updated

        decoder._update_model_kwargs_for_generation = detached_update
        force = ForceBackward()
        try:
            undecorated = getattr(decoder.generate, "__wrapped__", None)
            if undecorated is None:
                raise RuntimeError("MiDasheng decoder generate lacks its no-grad wrapped implementation")
            output = undecorated(
                decoder,
                inputs_embeds=inputs_embeds,
                generation_config=generation_model.generation_config,
                **values,
                max_new_tokens=int(targets.numel()),
                do_sample=False,
                use_cache=True,
                logits_processor=LogitsProcessorList([force]),
                eos_token_id=self.eos_token_ids,
                pad_token_id=self.pad_token_id,
                return_dict_in_generate=True,
            )
        finally:
            if had_instance_update:
                decoder._update_model_kwargs_for_generation = previous_instance_update
            else:
                delattr(decoder, "_update_model_kwargs_for_generation")
        completion, _ = self._completion_from_sequence(
            self.torch, output.sequences[0], prompt_ids[0], candidate["sampled_token_ids"],
        )
        if force.index != targets.numel() or completion.numel() != targets.numel():
            raise RuntimeError("MiDasheng GRPO native replay token count mismatch")
        return (
            float(self.torch.stack(losses).mean().cpu()),
            float(self.torch.stack(errors).max().cpu()),
            float(self.torch.stack(kls).mean().cpu()),
        )

    def preflight(self) -> list[dict[str, Any]]:
        adapters = getattr(self.model, "peft_config", {})
        return [{
            "status": "ok",
            "backend": "MiDashengLMModel",
            "trainable_parameters": sum(parameter.numel() for parameter in self.trainable_parameters),
            "adapter_names": sorted(adapters),
            "has_reference_adapter": self.has_reference_adapter,
            "reference_trainable_parameters": sum(
                parameter.numel() for name, parameter in self.model.named_parameters()
                if parameter.requires_grad and ".reference." in name
            ),
            "model_mode": "eval" if not self.model.training else "train",
            "gradient_checkpointing": bool(getattr(self.model, "is_gradient_checkpointing", False)),
            "device": str(self.device),
            "dtype": str(next(self.model.parameters()).dtype),
            "eos_token_ids": self.eos_token_ids,
            "primary_eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "sequence_mode": "completion_only",
            "rollout_batch_size": int(self.cfg.get("generation", {}).get("rollout_batch_size", 1)),
            "replay_batch_size": int(self.cfg.get("training", {}).get("replay_batch_size", 1)),
            "adapter_audits": self.adapter_audits,
            "gpu_peak_memory_bytes": int(self.torch.cuda.max_memory_allocated(self.device)),
        }]


def main() -> None:
    parser = worker_parser(
        "MiDasheng structured caption production-candidate worker",
        ["rollout", "score", "grpo-update", "sft-update", "gradient-audit", "preflight"],
    )
    args = parser.parse_args()
    config, rows = load_job(args)
    distributed = DistributedContext.initialize(config)
    worker: MiDashengCaptionerWorker | None = None
    try:
        worker = MiDashengCaptionerWorker(
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
