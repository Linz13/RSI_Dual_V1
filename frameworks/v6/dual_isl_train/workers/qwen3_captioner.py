from __future__ import annotations

import contextlib
import gc
import json
import random
from pathlib import Path
from typing import Any

from dual_isl_train.adapters import audit_adapter_pair, load_reference_adapter_exact, promote_adapter_fp32
from dual_isl_train.distributed import DistributedContext, distributed_schedule, run_sharded_inference
from dual_isl_train.constants import CHECKPOINT_METADATA, FRAMEWORK_VERSION, SOURCE_DOMAIN_TARGET, TRAJECTORY_VERSION
from dual_isl_train.dual_space import (
    SYNTHESIZABLE_FIELDS, parse_synth_caption_text_detailed, synth_caption_json,
)
from dual_isl_train.render import caption_prompt, synth_caption_prompt
from dual_isl_train.schema import (
    caption_json, environment_character_boundary, parse_caption_text, parse_dual_caption_text,
    source_caption_json,
)
from dual_isl_train.telemetry import MetricLogger
from dual_isl_train.trajectory import mark_trajectory
from dual_isl_train.workers.common import load_job, validate_checkpoint_version, worker_parser, write_output
from dual_isl_train.workers.token_utils import completion_token_span


def _batches(values: list[Any], batch_size: int):
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    for start in range(0, len(values), batch_size):
        yield values[start:start + batch_size]


def _cuda_memory_snapshot(torch_module: Any, device: Any) -> dict[str, int]:
    cuda = torch_module.cuda
    return {
        "allocated_bytes": int(cuda.memory_allocated(device)),
        "reserved_bytes": int(cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(cuda.max_memory_allocated(device)),
    }


def _release_captioner_cuda_memory(worker: Any | None) -> tuple[None, dict[str, Any]]:
    """Drop the full Captioner before NCCL teardown and report released memory."""
    telemetry: dict[str, Any] = {
        "model_reference_cleared": False,
        "empty_cache_called": False,
    }
    if worker is None:
        telemetry["worker_initialized"] = False
        return None, telemetry

    telemetry["worker_initialized"] = True
    torch_module = worker.torch
    device = worker.device
    telemetry["device"] = str(device)
    try:
        telemetry["before"] = _cuda_memory_snapshot(torch_module, device)
    except Exception as exc:
        telemetry["before_error"] = repr(exc)

    # The worker remains in main()'s local scope until finally exits.  Clearing
    # its CUDA-heavy attributes explicitly makes their storage collectible now,
    # before ProcessGroupNCCL performs its own shutdown work.
    for attribute in ("ddp_model", "model"):
        if hasattr(worker, attribute):
            setattr(worker, attribute, None)
    telemetry["model_reference_cleared"] = True
    worker = None
    gc.collect()

    try:
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
            telemetry["empty_cache_called"] = True
            telemetry["after"] = _cuda_memory_snapshot(torch_module, device)
    except Exception as exc:
        telemetry["cleanup_error"] = repr(exc)
    return None, telemetry


def _close_captioner_worker(worker: Any | None, distributed: DistributedContext) -> dict[str, Any]:
    """Release Captioner CUDA state before destroying the process group."""
    worker, telemetry = _release_captioner_cuda_memory(worker)
    print("CAPTIONER_CUDA_CLEANUP " + json.dumps(telemetry, sort_keys=True), flush=True)
    distributed.close()
    return telemetry


def patch_initializer_range(config: Any) -> None:
    """Patch the incomplete local Qwen3-Omni config before model construction."""
    if not hasattr(config, "initializer_range"):
        config.initializer_range = 0.02
    for name in ("thinker_config", "talker_config", "code2wav_config"):
        child = getattr(config, name, None)
        if child is not None and not hasattr(child, "initializer_range"):
            child.initializer_range = 0.02


class Qwen3CaptionerWorker:
    """Trainable Qwen3-Omni Captioner worker with exact sampled-token replay.

    The update implementation is shared with the MiDaSheng worker because the
    Qwen thinker implements the same causal-LM contract.  Model loading and all
    multimodal input construction are Qwen-specific and follow the installed
    Qwen3-Omni backend rather than the thin Caption_Bench wrapper.
    """

    def __init__(
        self,
        config: dict[str, Any],
        checkpoint: str = "",
        target_checkpoint: str = "",
        distributed: DistributedContext | None = None,
    ):
        import torch
        from transformers import AutoConfig, Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        from qwen_omni_utils import process_mm_info

        self.torch = torch
        self.cfg = config["captioner"]
        self.seed = int(config["run"].get("seed", 42))
        self.parent_checkpoint = checkpoint or None
        self.distributed = distributed or DistributedContext()
        self.device = torch.device(self.distributed.device or str(self.cfg.get("device", "cuda:0")))
        if self.device.type != "cuda":
            raise ValueError("Qwen3 Captioner smoke training requires a CUDA device")
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
            raise ValueError(f"Unsupported Qwen3 Captioner dtype: {dtype_name}")

        model_path = str(self.cfg["model_path"])
        model_config = AutoConfig.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True,
        )
        patch_initializer_range(model_config)
        container = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            config=model_config,
            dtype=dtype,
            device_map={"": str(self.device)},
            attn_implementation=self.cfg.get("attn_implementation", "sdpa"),
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=True,
        )
        # This checkpoint has no talker.  Keeping only the thinker also makes it
        # explicit that every trainable and replayed likelihood is text-policy
        # likelihood, not a top-level wrapper side effect.
        self.model = container.thinker
        del container
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True,
        )
        self.tokenizer = self.processor.tokenizer
        eos_token = str(self.cfg.get("generation", {}).get("eos_token", "<|im_end|>"))
        self.eos_token_id = int(self.tokenizer.convert_tokens_to_ids(eos_token))
        if self.eos_token_id < 0 or self.tokenizer.eos_token_id != self.eos_token_id:
            raise RuntimeError(
                f"Qwen EOS mismatch: {eos_token}={self.eos_token_id}, "
                f"tokenizer.eos_token_id={self.tokenizer.eos_token_id}"
            )
        self.pad_token_id = int(self.tokenizer.pad_token_id)
        self.has_reference_adapter = False
        self.adapter_audits: dict[str, Any] = {}
        self._attach_or_load_lora(checkpoint, target_checkpoint)
        for peft_config in getattr(self.model, "peft_config", {}).values():
            peft_config.base_model_name_or_path = model_path
        if bool(self.cfg.get("gradient_checkpointing", False)) and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            # All base embeddings are frozen under LoRA.  Checkpointed blocks
            # still need a grad-requiring input so their adapter graph is kept.
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
        self.model.eval()
        self.freeze_batchnorm()
        self._audit_adapters()
        self._broadcast_current_adapter_parameters()

    def _attach_or_load_lora(self, checkpoint: str, target_checkpoint: str) -> None:
        from peft import PeftModel, get_peft_model

        if checkpoint and (Path(checkpoint) / "adapter_config.json").is_file():
            validate_checkpoint_version(checkpoint)
            self.model = PeftModel.from_pretrained(
                self.model, checkpoint, is_trainable=True, autocast_adapter_dtype=True,
            )
            promote_adapter_fp32(self.model, "default")
            from dual_isl_train.adapters import audit_adapter_checkpoint
            self.adapter_audits["policy"] = audit_adapter_checkpoint(self.model, checkpoint, "default")
            if not self.adapter_audits["policy"]["exact"]:
                raise RuntimeError(f"Qwen policy adapter load was not exact: {self.adapter_audits['policy']}")
        else:
            self.model = get_peft_model(self.model, self._lora_config(), autocast_adapter_dtype=True)
            promote_adapter_fp32(self.model, "default")
        if target_checkpoint and (Path(target_checkpoint) / "adapter_config.json").is_file():
            validate_checkpoint_version(target_checkpoint)
            self.adapter_audits["reference"] = load_reference_adapter_exact(
                self.model, target_checkpoint, "reference",
            )
            self.has_reference_adapter = True
            if checkpoint and Path(checkpoint).resolve() == Path(target_checkpoint).resolve():
                self.adapter_audits["policy_reference"] = audit_adapter_pair(self.model)
                if not self.adapter_audits["policy_reference"]["exact"]:
                    raise RuntimeError(
                        f"Qwen policy/reference adapters differ at round start: "
                        f"{self.adapter_audits['policy_reference']}"
                    )
        self.model.set_adapter("default")

    @contextlib.contextmanager
    def adapter_context(self, reference: bool):
        if not reference:
            self.model.set_adapter("default")
            yield
            return
        if self.has_reference_adapter:
            self.model.set_adapter("reference")
            try:
                yield
            finally:
                self.model.set_adapter("default")
        else:
            with self.model.disable_adapter():
                yield

    def freeze_batchnorm(self) -> None:
        for module in self.model.modules():
            if isinstance(module, self.torch.nn.modules.batchnorm._BatchNorm):
                module.eval()

    @property
    def trainable_parameters(self) -> list[Any]:
        self.model.set_adapter("default")
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def parameter_signature(self) -> float:
        values = self.trainable_parameters
        return sum((index + 1) * value.detach().double().sum().item() for index, value in enumerate(values))

    def _lora_config(self):
        from peft import LoraConfig

        lora = self.cfg.get("lora", {})
        targets = list(lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj"]))
        return LoraConfig(
            r=int(lora.get("r", 8)),
            lora_alpha=int(lora.get("alpha", 16)),
            lora_dropout=float(lora.get("dropout", 0.0)),
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        )

    def _audit_adapters(self) -> None:
        adapters = getattr(self.model, "peft_config", {})
        if "default" not in adapters:
            raise RuntimeError("Qwen3 current policy adapter 'default' was not attached")
        trainable_current = [
            name for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and ".default." in name
        ]
        if not trainable_current:
            raise RuntimeError("Qwen3 current policy adapter has no trainable parameters")
        if self.has_reference_adapter:
            trainable_reference = [
                name for name, parameter in self.model.named_parameters()
                if parameter.requires_grad and ".reference." in name
            ]
            if trainable_reference:
                raise RuntimeError("Qwen3 frozen reference adapter unexpectedly has trainable parameters")

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

    def _completion_span(self, audio_path: str, prompt: str, completion: str, prompt_inputs, full_inputs) -> tuple[int, int]:
        boundary = environment_character_boundary(completion)
        truncated = self._inputs(audio_path, prompt, completion[:boundary])
        return completion_token_span(
            prompt_inputs["input_ids"], full_inputs["input_ids"], truncated["input_ids"],
        )

    def _sampling_warpers(self, temperature: float):
        """Build the same sampling transform used by the configured rollout."""
        from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

        generation = self.cfg.get("generation", {})
        warpers = LogitsProcessorList()
        if temperature > 0 and temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(temperature))
        top_k = int(generation.get("top_k", 0))
        if top_k > 0:
            warpers.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1))
        top_p = float(generation.get("top_p", 1.0))
        if 0.0 < top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))
        return warpers

    def generate_batch(
        self,
        audio_path: str,
        prompt: str,
        seeds: list[int],
        *,
        temperature: float | None = None,
    ) -> list[dict[str, Any]]:
        """Generate one GRPO candidate batch from one shared audio prompt.

        Each row owns an independent RNG seeded by its recorded candidate seed.
        A custom processor samples and then forces that row's token, allowing the
        native Qwen generation loop to share one batched KV cache without losing
        per-candidate reproducibility or behavior-policy log probabilities.
        """
        if not seeds:
            return []
        from transformers import LogitsProcessorList

        generation = self.cfg.get("generation", {})
        temperature = float(generation.get("temperature", 1.0) if temperature is None else temperature)
        inputs = self._inputs(audio_path, prompt)
        self.model.set_adapter("default")
        self.model.eval()
        self.freeze_batchnorm()
        warpers = self._sampling_warpers(temperature)
        generators = []
        for seed in seeds:
            generator = self.torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
            generators.append(generator)
        sampled_ids: list[list[Any]] = [[] for _ in seeds]
        sampled_logprobs: list[list[Any]] = [[] for _ in seeds]

        class SampleAndForce:
            def __call__(processor_self, input_ids, scores):
                if scores.shape[0] != len(seeds):
                    raise RuntimeError(
                        f"Qwen rollout batch expanded to {scores.shape[0]} rows; expected {len(seeds)}"
                    )
                forced = self.torch.full_like(scores, -self.torch.inf)
                for row_index in range(len(seeds)):
                    row_scores = scores[row_index:row_index + 1]
                    if temperature > 0:
                        row_scores = warpers(input_ids[row_index:row_index + 1], row_scores)
                        # Match HF sampling dtype/order; cast only when values
                        # are persisted, as the former serial generate did.
                        row_logprobs = self.torch.log_softmax(row_scores, dim=-1)
                        target = self.torch.multinomial(
                            row_logprobs.exp(), num_samples=1, generator=generators[row_index],
                        ).squeeze(0).squeeze(0)
                    else:
                        row_logprobs = self.torch.log_softmax(row_scores, dim=-1)
                        target = row_logprobs.argmax(dim=-1).squeeze(0)
                    sampled_ids[row_index].append(target.detach())
                    sampled_logprobs[row_index].append(row_logprobs[0, target].detach())
                    forced[row_index, target] = 0
                return forced

        arguments: dict[str, Any] = {
            **inputs,
            "max_new_tokens": int(generation.get("max_new_tokens", 384)),
            "do_sample": True,
            "num_return_sequences": len(seeds),
            # Sampling has already happened independently inside SampleAndForce.
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "remove_invalid_values": True,
            "renormalize_logits": True,
            "return_dict_in_generate": True,
            "output_scores": False,
            "logits_processor": LogitsProcessorList([SampleAndForce()]),
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
        }
        with self.torch.no_grad():
            outputs = self.model.generate(**arguments)
        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
        prefix = inputs["input_ids"].shape[1]
        if sequences.shape[0] != len(seeds):
            raise RuntimeError(f"Qwen rollout returned {sequences.shape[0]} rows; expected {len(seeds)}")
        expected_prefix = inputs["input_ids"].expand(len(seeds), -1)
        if sequences.shape[1] < prefix or not self.torch.equal(sequences[:, :prefix], expected_prefix):
            raise RuntimeError("Qwen batched rollout did not preserve the expanded prompt token prefix")
        results = []
        for row_index, captured_ids in enumerate(sampled_ids):
            generated = sequences[row_index, prefix:]
            generated_ids = generated.detach().cpu().tolist()
            eos_positions = [index for index, token_id in enumerate(generated_ids) if token_id == self.eos_token_id]
            token_count = eos_positions[0] + 1 if eos_positions else len(generated_ids)
            token_ids = generated_ids[:token_count]
            replayed = generated[:token_count]
            expected = self.torch.stack(captured_ids[:token_count]).to(dtype=replayed.dtype)
            if replayed.shape[0] != expected.shape[0] or not self.torch.equal(replayed, expected):
                raise RuntimeError(f"Qwen batched rollout did not preserve candidate {row_index} token IDs")
            token_logprobs = self.torch.stack(sampled_logprobs[row_index][:token_count]).float().cpu().tolist()
            eos_positions = [index for index, token_id in enumerate(token_ids) if token_id == self.eos_token_id]
            terminated = bool(eos_positions and eos_positions[-1] == len(token_ids) - 1)
            first_eos = eos_positions[0] if eos_positions else None
            decoded = self.tokenizer.decode(
                token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
            ).strip()
            results.append({
                "raw_text": decoded,
                "sampled_token_ids": token_ids,
                "old_token_logprobs": token_logprobs,
                "finish_reason": "eos" if terminated else "length",
                "terminal_token_id": token_ids[-1] if token_ids else None,
                "terminated_by_eos": terminated,
                "post_eos_token_count": 0 if first_eos is None else len(token_ids) - first_eos - 1,
                "generated_token_count": len(token_ids),
                "eos_token_id": self.eos_token_id,
            })
        return results

    def generate(
        self,
        audio_path: str,
        prompt: str,
        *,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        return self.generate_batch(
            audio_path, prompt, [self.seed if seed is None else int(seed)], temperature=temperature,
        )[0]

    def sampled_token_logprobs(
        self,
        audio_path: str,
        prompt: str,
        sampled_token_ids: list[int],
        *,
        reference: bool = False,
        grad: bool = False,
    ):
        values = self.sampled_token_logprobs_batch(
            audio_path, prompt, [sampled_token_ids], reference=reference, grad=grad,
        )
        return values[0]

    def sampled_token_logprobs_batch(
        self,
        audio_path: str,
        prompt: str,
        sampled_token_ids_batch: list[list[int]],
        *,
        reference: bool = False,
        grad: bool = False,
    ) -> list[Any]:
        """Replay a candidate batch exactly inside the native HF loop.

        The capture processor records each conditional distribution before it
        forces the already-sampled ID.  This preserves Qwen's own KV-cache,
        multimodal RoPE and attention path without decode/re-tokenize drift.
        """
        if not sampled_token_ids_batch:
            return []
        from transformers import LogitsProcessorList

        self.model.eval()
        prompt_inputs = self._inputs(audio_path, prompt)
        empty = self.torch.empty(0, device=self.device, dtype=self.torch.float32)
        result: list[Any] = [empty for _ in sampled_token_ids_batch]
        active_indices = [index for index, values in enumerate(sampled_token_ids_batch) if values]
        if not active_indices:
            return result
        targets = [
            self.torch.tensor(
                sampled_token_ids_batch[index], device=self.device, dtype=prompt_inputs["input_ids"].dtype,
            )
            for index in active_indices
        ]
        max_tokens = max(int(values.numel()) for values in targets)
        for values in targets:
            if values.numel() < max_tokens and int(values[-1]) != self.eos_token_id:
                raise RuntimeError("A shorter Qwen replay trajectory must terminate with EOS")
        captured: list[list[Any]] = [[] for _ in targets]

        class CaptureAndForce:
            def __init__(processor_self):
                processor_self.index = 0

            def __call__(processor_self, _input_ids, scores):
                if scores.shape[0] != len(targets):
                    raise RuntimeError(
                        f"Qwen replay batch expanded to {scores.shape[0]} rows; expected {len(targets)}"
                    )
                index = processor_self.index
                forced = self.torch.full_like(scores, -self.torch.inf)
                logprobs = self.torch.log_softmax(scores.float(), dim=-1)
                for row_index, row_targets in enumerate(targets):
                    if index < row_targets.numel():
                        target = row_targets[index]
                        captured[row_index].append(logprobs[row_index, target])
                        forced[row_index, target] = 0
                    else:
                        forced[row_index, self.pad_token_id] = 0
                processor_self.index += 1
                return forced

        processor = CaptureAndForce()
        generation_model = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        generation = self.cfg.get("generation", {})
        arguments = {
            **prompt_inputs,
            "max_new_tokens": max_tokens,
            "do_sample": True,
            "num_return_sequences": len(targets),
            "temperature": float(generation.get("temperature", 1.0)),
            "top_p": float(generation.get("top_p", 1.0)),
            "top_k": int(generation.get("top_k", 0)),
            "remove_invalid_values": True,
            "renormalize_logits": True,
            "return_dict_in_generate": True,
            "logits_processor": LogitsProcessorList([processor]),
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
        }
        grad_context = contextlib.nullcontext() if grad else self.torch.no_grad()
        with grad_context, self.adapter_context(reference):
            generate_method = generation_model.generate
            if grad:
                undecorated = getattr(generate_method, "__wrapped__", None)
                if undecorated is None:
                    raise RuntimeError("Qwen generation method does not expose its no-grad wrapped implementation")
                outputs = undecorated(generation_model, **arguments)
            else:
                outputs = generate_method(**arguments)
        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
        prefix = prompt_inputs["input_ids"].shape[1]
        if sequences.shape[0] != len(targets):
            raise RuntimeError(f"Qwen replay returned {sequences.shape[0]} rows; expected {len(targets)}")
        expected_prefix = prompt_inputs["input_ids"].expand(len(targets), -1)
        if sequences.shape[1] < prefix or not self.torch.equal(sequences[:, :prefix], expected_prefix):
            raise RuntimeError("Qwen batched replay did not preserve the expanded prompt token prefix")
        for local_index, row_targets in enumerate(targets):
            replayed = sequences[local_index, prefix:prefix + row_targets.numel()]
            if replayed.shape[0] != row_targets.numel() or not self.torch.equal(replayed, row_targets):
                raise RuntimeError(f"Qwen native replay did not preserve candidate {local_index} token IDs")
            if len(captured[local_index]) != row_targets.numel():
                raise RuntimeError(f"Qwen native replay missed candidate {local_index} token likelihoods")
            result[active_indices[local_index]] = self.torch.stack(captured[local_index])
        return result

    def token_logprobs(
        self, audio_path: str, prompt: str, completion: str, *,
        reference: bool = False, grad: bool = False, mask_environment: bool = False,
    ):
        values, _ = self._completion_token_logprobs(
            audio_path, prompt, completion, reference=reference, grad=grad,
            mask_environment=mask_environment,
        )
        return values

    def _completion_token_logprobs(
        self, audio_path: str, prompt: str, completion: str, *,
        reference: bool = False, grad: bool = False, mask_environment: bool = False,
    ):
        prompt_inputs = self._inputs(audio_path, prompt)
        full_inputs = self._inputs(audio_path, prompt, completion)
        prefix, cutoff = completion_token_span(prompt_inputs["input_ids"], full_inputs["input_ids"])
        if mask_environment:
            prefix, cutoff = self._completion_span(audio_path, prompt, completion, prompt_inputs, full_inputs)
        full_ids = full_inputs["input_ids"]
        grad_context = contextlib.nullcontext() if grad else self.torch.no_grad()
        with grad_context, self.adapter_context(reference):
            outputs = self.model(**full_inputs)
        logits = outputs.logits[:, prefix - 1:cutoff - 1, :]
        targets = full_ids[:, prefix:cutoff]
        token_logps = self.torch.log_softmax(logits.float(), dim=-1).gather(
            -1, targets.unsqueeze(-1),
        ).squeeze(0).squeeze(-1)
        return token_logps, targets.squeeze(0)

    @staticmethod
    def _usable_value(value: Any) -> bool:
        if isinstance(value, str):
            return bool(value.strip()) and value.strip().casefold() != "unknown"
        if isinstance(value, list):
            return any(Qwen3CaptionerWorker._usable_value(item) for item in value)
        return value is not None

    @staticmethod
    def _value_content_spans(target: str, caption: dict[str, Any]) -> dict[str, list[tuple[int, int]]]:
        from dual_isl_train.schema import get_path

        spans: dict[str, list[tuple[int, int]]] = {}
        for field in SYNTHESIZABLE_FIELDS:
            value = get_path(caption, field)
            key = field.rsplit(".", 1)[-1]
            marker = json.dumps(key, ensure_ascii=False) + ":"
            key_at = target.find(marker)
            if key_at < 0:
                raise RuntimeError(f"Cannot locate serialized P_syn field {field}")
            if target.find(marker, key_at + len(marker)) >= 0:
                raise RuntimeError(f"Serialized P_syn field is ambiguous: {field}")
            value_at = key_at + len(marker)
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            if not target.startswith(encoded, value_at):
                raise RuntimeError(f"Serialized P_syn value mismatch for {field}")
            if not Qwen3CaptionerWorker._usable_value(value):
                continue
            field_spans: list[tuple[int, int]] = []
            if isinstance(value, str):
                if len(encoded) > 2:
                    field_spans.append((value_at + 1, value_at + len(encoded) - 1))
            elif isinstance(value, list):
                local = 0
                for item in value:
                    item_encoded = json.dumps(item, ensure_ascii=False)
                    found = encoded.find(item_encoded, local)
                    if found < 0:
                        continue
                    local = found + len(item_encoded)
                    if Qwen3CaptionerWorker._usable_value(item) and len(item_encoded) > 2:
                        field_spans.append((value_at + found + 1, value_at + found + len(item_encoded) - 1))
            if field_spans:
                spans[field] = field_spans
        return spans

    def synth_value_macro_logprob(
        self, audio_path: str, prompt: str, caption: dict[str, Any], *, reference: bool,
    ) -> tuple[float, dict[str, float], int]:
        target = synth_caption_json(caption)
        logps, completion_ids = self._completion_token_logprobs(
            audio_path, prompt, target, reference=reference,
        )
        encoded = self.tokenizer(
            target, add_special_tokens=False, return_offsets_mapping=True,
        )
        target_ids = list(encoded["input_ids"])
        offsets = [tuple(value) for value in encoded["offset_mapping"]]
        completion = completion_ids.detach().cpu().tolist()
        start = next(
            (index for index in range(len(completion) - len(target_ids) + 1)
             if completion[index:index + len(target_ids)] == target_ids),
            None,
        )
        if start is None:
            raise RuntimeError("P_syn target tokens cannot be aligned to multimodal completion tokens")
        spans = self._value_content_spans(target, caption)
        per_field: dict[str, float] = {}
        total_tokens = 0
        for field, field_spans in spans.items():
            positions = [
                index for index, (left, right) in enumerate(offsets)
                if right > left and any(right > begin and left < end for begin, end in field_spans)
            ]
            if not positions:
                continue
            values = logps[self.torch.tensor([start + index for index in positions], device=logps.device)]
            per_field[field] = float(values.mean().item())
            total_tokens += int(values.numel())
        if not per_field:
            raise RuntimeError("P_syn reconstruction has no usable value tokens")
        return sum(per_field.values()) / len(per_field), per_field, total_tokens

    def rollout(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output = []
        rollout_batch_size = int(self.cfg.get("generation", {}).get("rollout_batch_size", 1))
        replay_batch_size = int(self.cfg.get("training", {}).get("replay_batch_size", 1))
        for row in rows:
            group_size = int(row.get("group_size", 2))
            configured_seeds = row.get("candidate_seeds") or []
            seeds = [
                int(configured_seeds[index]) if index < len(configured_seeds) else self.seed + index
                for index in range(group_size)
            ]
            samples = []
            for seed_batch in _batches(seeds, rollout_batch_size):
                samples.extend(self.generate_batch(row["audio_path"], row["prompt"], seed_batch))
            if len(samples) != group_size:
                raise RuntimeError(f"Qwen rollout produced {len(samples)} candidates; expected {group_size}")

            policy_replays = []
            reference_replays = []
            for sample_batch in _batches(samples, replay_batch_size):
                trajectories = [sample["sampled_token_ids"] for sample in sample_batch]
                policy_replays.extend(self.sampled_token_logprobs_batch(
                    row["audio_path"], row["prompt"], trajectories, reference=False,
                ))
                reference_replays.extend(self.sampled_token_logprobs_batch(
                    row["audio_path"], row["prompt"], trajectories, reference=True,
                ))

            candidates = []
            for index, (generation_seed, sampled, policy_replay, ref_logprobs) in enumerate(zip(
                seeds, samples, policy_replays, reference_replays, strict=True,
            )):
                raw = sampled["raw_text"]
                schema = row.get("caption_schema", "full")
                parse_metadata: dict[str, Any] = {}
                if schema == "flat_audio4.v6.1":
                    from dual_isl_train.schema_v6 import admit
                    admission = admit(raw)
                    caption, errors = admission["caption"], admission.get("parse_errors", [])
                elif schema == "synth_v1":
                    caption, errors, parse_metadata = parse_synth_caption_text_detailed(raw)
                elif schema in {"dual_no_environment", "source_no_environment"}:
                    caption, errors = parse_dual_caption_text(raw)
                else:
                    caption, errors = parse_caption_text(raw)
                policy_reference_error = float((policy_replay.float() - ref_logprobs.float()).abs().max().cpu())
                candidate = {
                    "candidate_id": f"{row['id']}::{index}", "raw_text": raw,
                    "caption": caption, "parse_errors": errors, "generation_attempts": 1,
                    "generation_seed": generation_seed, **parse_metadata,
                    "generation_batch_size": sampled.get("generation_batch_size", 1),
                    "generation_batch_fallback": sampled.get("generation_batch_fallback"),
                    "sampled_token_ids": sampled["sampled_token_ids"],
                    "old_token_logprobs": sampled["old_token_logprobs"],
                    "ref_token_logprobs": ref_logprobs.float().cpu().tolist(),
                    "behavior_replay_max_abs_error": float(max(
                        abs(left - right) for left, right in zip(
                            sampled["old_token_logprobs"], policy_replay.float().cpu().tolist(), strict=True,
                        )
                    )),
                    "policy_reference_max_abs_error": policy_reference_error,
                    "behavior_logprob_mode": "generation_scores",
                    "trajectory_version": TRAJECTORY_VERSION,
                    "caption_schema": schema,
                    **{key: sampled[key] for key in (
                        "finish_reason", "terminal_token_id", "terminated_by_eos",
                        "post_eos_token_count", "generated_token_count", "eos_token_id",
                    )},
                }
                if caption is not None:
                    if schema == "flat_audio4.v6.1":
                        candidate["canonical_text"] = json.dumps(caption, ensure_ascii=False)
                    elif schema == "synth_v1":
                        candidate["canonical_text"] = synth_caption_json(caption)
                    elif schema in {"dual_no_environment", "source_no_environment"}:
                        candidate["canonical_text"] = source_caption_json(caption)
                    else:
                        candidate["canonical_text"] = caption_json(caption)
                candidates.append(mark_trajectory(candidate, "caption"))
            output.append({**row, "candidates": candidates})
        return output

    def score(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for row in rows:
            if row.get("score_mode") == "synth_values_macro":
                macro, per_field, token_count = self.synth_value_macro_logprob(
                    row["audio_path"], row.get("prompt") or synth_caption_prompt(),
                    row["target_caption"], reference=True,
                )
                output.append({
                    "candidate_id": row["candidate_id"],
                    "caption_target_logprob": macro,
                    "caption_target_logprob_macro": macro,
                    "caption_value_field_logprobs": per_field,
                    "caption_target_value_token_count": token_count,
                    "fixed_json_tokens_excluded": True,
                })
                continue
            target = (
                source_caption_json(row["target_caption"])
                if row.get("target_schema") == "source_no_environment"
                else caption_json(row["target_caption"])
            )
            mask_environment = bool(row.get("mask_environment", False))
            token_logps = self.token_logprobs(
                row["audio_path"], row.get("prompt") or caption_prompt(), target,
                reference=True, mask_environment=mask_environment,
            )
            output.append({
                "candidate_id": row["candidate_id"],
                "caption_target_logprob": float(token_logps.mean().item()) if token_logps.numel() else -1e9,
                "caption_target_token_count": int(token_logps.numel()),
                "environment_masked": mask_environment,
            })
        return output

    def _training_config(self, phase: str) -> dict[str, Any]:
        training = dict(self.cfg.get("training", {}))
        phases = training.pop("phases", {})
        if phase:
            training.update(phases.get(phase, {}))
        return training

    def _schedule(self, rows: list[dict[str, Any]], training: dict[str, Any]):
        epochs = int(training.get("min_epochs", training.get("epochs", 1)))
        rng = random.Random(self.seed)
        result = []
        for epoch in range(epochs):
            ordered = list(rows)
            if bool(training.get("shuffle", False)):
                rng.shuffle(ordered)
            result.extend((epoch, row) for row in ordered)
        return result

    def _optimizer(self, training: dict[str, Any]):
        return self.torch.optim.AdamW(
            self.trainable_parameters,
            lr=float(training.get("lr", 1e-6)),
            weight_decay=float(training.get("weight_decay", 0.01)),
        )

    def _save(self, checkpoint_out: str, details: dict[str, Any]) -> None:
        self.model.set_adapter("default")
        self.model.save_pretrained(checkpoint_out, selected_adapters=["default"])
        self.processor.save_pretrained(checkpoint_out)
        Path(checkpoint_out, CHECKPOINT_METADATA).write_text(
            json.dumps({
                "framework_version": FRAMEWORK_VERSION,
                "trajectory_version": TRAJECTORY_VERSION,
                "parent_checkpoint": self.parent_checkpoint,
                **details,
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )

    def _current_adapter_named_parameters(self) -> list[tuple[str, Any]]:
        """Return only trainable current-policy tensors, never reference tensors."""
        self.model.set_adapter("default")
        trainable = [(name, parameter) for name, parameter in self.model.named_parameters() if parameter.requires_grad]
        unexpected = [name for name, _ in trainable if ".default." not in name]
        if unexpected:
            raise RuntimeError(f"Qwen trainable parameters outside current adapter: {unexpected[:3]}")
        if not trainable:
            raise RuntimeError("Qwen current policy adapter has no trainable parameters")
        return trainable

    def _all_reduce_current_adapter_gradients(self) -> None:
        """Average native-generation gradients explicitly across all ranks.

        Native ``generate`` calls the underlying model directly and therefore
        cannot rely on a DDP reducer.  Every rank participates once per tensor,
        including with a zero gradient for locally unused MoE paths.
        """
        if not self.distributed.enabled:
            return
        import torch.distributed as dist

        for _, parameter in self._current_adapter_named_parameters():
            if parameter.grad is None:
                parameter.grad = self.torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(self.distributed.world_size)

    def _broadcast_current_adapter_parameters(self) -> None:
        """Give newly initialized LoRA tensors one identical rank-0 starting point."""
        if not self.distributed.enabled:
            return
        import torch.distributed as dist

        for _, parameter in self._current_adapter_named_parameters():
            dist.broadcast(parameter.data, src=0)

    def _verify_current_adapter_sync(self) -> dict[str, Any]:
        """Prove that every current-adapter tensor matches rank 0 before save."""
        named = self._current_adapter_named_parameters()
        local_layout = [(name, tuple(parameter.shape), str(parameter.dtype)) for name, parameter in named]
        layouts = self.distributed.gather_objects(local_layout)
        if any(layout != layouts[0] for layout in layouts[1:]):
            raise RuntimeError("Qwen current adapter layout differs across ranks")
        max_abs_diff = 0.0
        if self.distributed.enabled:
            import torch.distributed as dist

            for _, parameter in named:
                rank_zero = parameter.detach().clone()
                dist.broadcast(rank_zero, src=0)
                difference = (parameter.detach() - rank_zero).abs().max().float()
                dist.all_reduce(difference, op=dist.ReduceOp.MAX)
                max_abs_diff = max(max_abs_diff, float(difference.cpu()))
        signatures = self.distributed.gather_objects(self.parameter_signature())
        result = {
            "ok": max_abs_diff == 0.0,
            "checked_parameter_tensors": len(named),
            "max_abs_diff": max_abs_diff,
            "per_rank_signatures": signatures,
        }
        if not result["ok"]:
            raise RuntimeError(f"Qwen current adapter parameters diverged across ranks: {result}")
        return result

    def _finish_update(
        self, checkpoint_out: str, details: dict[str, Any], *, sample_ids: list[str], steps: int,
    ) -> list[dict[str, Any]]:
        """Gather auditable rank telemetry and save only after all ranks agree."""
        parameter_sync = self._verify_current_adapter_sync()
        rank_summary = {
            "rank": self.distributed.rank,
            "local_rank": self.distributed.local_rank,
            "cuda_device": self.torch.cuda.current_device(),
            "gpu_name": self.torch.cuda.get_device_name(self.torch.cuda.current_device()),
            "processed_items": len(sample_ids),
            "optimizer_steps": steps,
            "sample_ids": sample_ids,
            "mean_loss": details.get("mean_loss"),
            "max_grad_norm": details.get("max_grad_norm", details.get("grad_norm")),
            "parameter_before": details.get("parameter_before"),
            "parameter_after": details.get("parameter_after"),
            "parameter_changed": details.get("parameter_before") != details.get("parameter_after"),
            "gpu_peak_memory_bytes": int(self.torch.cuda.max_memory_allocated(self.device)),
        }
        per_rank = self.distributed.gather_objects(rank_summary)
        details["parameter_sync"] = parameter_sync
        details["distributed"] = {
            "enabled": self.distributed.enabled,
            "backend": self.distributed.backend or None,
            "world_size": self.distributed.world_size,
            "per_rank": per_rank,
        }
        if self.distributed.is_main:
            self._save(checkpoint_out, details)
            self.output_metrics = details["distributed"]
        self.distributed.barrier()
        status = "updated" if steps else "skipped"
        return [{"status": status, **details, "checkpoint": checkpoint_out}] if self.distributed.is_main else []

    @staticmethod
    def _detach_generation_cache(cache: Any) -> None:
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

    def _native_candidate_backward(
        self,
        group: dict[str, Any],
        candidate: dict[str, Any],
        *,
        clip: float,
        kl_beta: float,
        candidate_count: int,
    ) -> tuple[float, float, float]:
        return self._native_candidates_backward(
            group, [candidate], clip=clip, kl_beta=kl_beta, candidate_count=candidate_count,
        )[0]

    def _native_candidates_backward(
        self,
        group: dict[str, Any],
        candidates: list[dict[str, Any]],
        *,
        clip: float,
        kl_beta: float,
        candidate_count: int,
    ) -> list[tuple[float, float, float]]:
        """Backprop one candidate microbatch with detached KV boundaries.

        Every candidate keeps its original token-mean objective and the group
        keeps its original candidate mean.  Batching changes only how the model
        forward/replay work is scheduled.
        """
        if not candidates:
            return []
        from transformers import LogitsProcessorList

        prompt_inputs = self._inputs(group["audio_path"], group["prompt"])
        sampled = [
            self.torch.tensor(
                candidate["sampled_token_ids"],
                device=self.device,
                dtype=prompt_inputs["input_ids"].dtype,
            )
            for candidate in candidates
        ]
        old = [
            self.torch.tensor(
                candidate["old_token_logprobs"], device=self.device, dtype=self.torch.float32,
            )
            for candidate in candidates
        ]
        ref = [
            self.torch.tensor(
                candidate["ref_token_logprobs"], device=self.device, dtype=self.torch.float32,
            )
            for candidate in candidates
        ]
        advantages = [
            self.torch.tensor(float(candidate["advantage"]), device=self.device, dtype=self.torch.float32)
            for candidate in candidates
        ]
        max_tokens = max(int(values.numel()) for values in sampled)
        for row_index, (row_sampled, row_old, row_ref) in enumerate(zip(sampled, old, ref, strict=True)):
            if row_sampled.numel() != row_old.numel() or row_sampled.numel() != row_ref.numel() or not row_sampled.numel():
                raise RuntimeError(f"Qwen Captioner GRPO trajectory {row_index} lengths are invalid")
            if row_sampled.numel() < max_tokens and int(row_sampled[-1]) != self.eos_token_id:
                raise RuntimeError("A shorter Qwen GRPO trajectory must terminate with EOS")
        token_losses: list[list[Any]] = [[] for _ in candidates]
        replay_errors: list[list[Any]] = [[] for _ in candidates]
        kl_values: list[list[Any]] = [[] for _ in candidates]
        backward_chunk_size = int(self.cfg.get("training", {}).get("grpo_backward_chunk_tokens", 16))
        if backward_chunk_size < 1:
            raise ValueError("captioner.training.grpo_backward_chunk_tokens must be >= 1")
        pending_losses = []

        class ForceAndBackward:
            def __init__(processor_self, worker):
                processor_self.worker = worker
                processor_self.index = 0

            def __call__(processor_self, _input_ids, scores):
                index = processor_self.index
                if index >= max_tokens:
                    raise RuntimeError("Qwen GRPO replay generated beyond the sampled trajectory")
                if scores.shape[0] != len(candidates):
                    raise RuntimeError(
                        f"Qwen GRPO replay batch expanded to {scores.shape[0]} rows; expected {len(candidates)}"
                    )
                logprobs = processor_self.worker.torch.log_softmax(scores.float(), dim=-1)
                forced = processor_self.worker.torch.full_like(scores, -processor_self.worker.torch.inf)
                for row_index, row_sampled in enumerate(sampled):
                    if index >= row_sampled.numel():
                        forced[row_index, processor_self.worker.pad_token_id] = 0
                        continue
                    target = row_sampled[index]
                    current = logprobs[row_index, target]
                    ratio = processor_self.worker.torch.exp(
                        processor_self.worker.torch.clamp(current - old[row_index][index], -10, 10)
                    )
                    policy = -processor_self.worker.torch.minimum(
                        ratio * advantages[row_index],
                        processor_self.worker.torch.clamp(ratio, 1 - clip, 1 + clip) * advantages[row_index],
                    )
                    log_ratio = ref[row_index][index] - current
                    kl = (
                        processor_self.worker.torch.exp(
                            processor_self.worker.torch.clamp(log_ratio, -10, 10)
                        ) - log_ratio - 1
                    )
                    loss = policy + kl_beta * kl
                    pending_losses.append(loss / (candidate_count * row_sampled.numel()))
                    token_losses[row_index].append(loss.detach())
                    kl_values[row_index].append(kl.detach())
                    replay_errors[row_index].append((current.detach() - old[row_index][index]).abs())
                    forced[row_index, target] = 0
                if (index + 1) % backward_chunk_size == 0 or index + 1 == max_tokens:
                    processor_self.worker.torch.stack(pending_losses).sum().backward()
                    pending_losses.clear()
                processor_self.index += 1
                return forced

        generation_model = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        had_instance_update = "_update_model_kwargs_for_generation" in generation_model.__dict__
        previous_instance_update = generation_model.__dict__.get("_update_model_kwargs_for_generation")
        original_update = generation_model._update_model_kwargs_for_generation
        generation_index = 0

        def detach_update(outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1):
            nonlocal generation_index
            updated = original_update(outputs, model_kwargs, is_encoder_decoder, num_new_tokens)
            # HF updates the cache immediately before the logits processor.  A
            # boundary here starts the next bounded graph; the preceding chunk
            # is backpropagated by the processor before that next forward.
            if (generation_index + 1) % backward_chunk_size == 0:
                self._detach_generation_cache(updated.get("past_key_values"))
            generation_index += 1
            return updated

        generation_model._update_model_kwargs_for_generation = detach_update
        generation = self.cfg.get("generation", {})
        try:
            undecorated = getattr(generation_model.generate, "__wrapped__", None)
            if undecorated is None:
                raise RuntimeError("Qwen generation method does not expose its no-grad wrapped implementation")
            outputs = undecorated(
                generation_model,
                **prompt_inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                num_return_sequences=len(candidates),
                temperature=float(generation.get("temperature", 1.0)),
                top_p=float(generation.get("top_p", 1.0)),
                top_k=int(generation.get("top_k", 0)),
                remove_invalid_values=True,
                renormalize_logits=True,
                return_dict_in_generate=True,
                logits_processor=LogitsProcessorList([ForceAndBackward(self)]),
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
            )
        finally:
            if had_instance_update:
                generation_model._update_model_kwargs_for_generation = previous_instance_update
            else:
                delattr(generation_model, "_update_model_kwargs_for_generation")
        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
        prefix = prompt_inputs["input_ids"].shape[1]
        if sequences.shape[0] != len(candidates):
            raise RuntimeError(f"Qwen GRPO replay returned {sequences.shape[0]} rows; expected {len(candidates)}")
        expected_prefix = prompt_inputs["input_ids"].expand(len(candidates), -1)
        if sequences.shape[1] < prefix or not self.torch.equal(sequences[:, :prefix], expected_prefix):
            raise RuntimeError("Qwen batched GRPO replay did not preserve the expanded prompt token prefix")
        results = []
        for row_index, row_sampled in enumerate(sampled):
            replayed = sequences[row_index, prefix:prefix + row_sampled.numel()]
            if replayed.shape[0] != row_sampled.numel() or not self.torch.equal(replayed, row_sampled):
                raise RuntimeError(f"Qwen GRPO replay did not preserve candidate {row_index} token IDs")
            if len(token_losses[row_index]) != row_sampled.numel():
                raise RuntimeError(f"Qwen GRPO replay missed candidate {row_index} tokens")
            results.append((
                float(self.torch.stack(token_losses[row_index]).mean().cpu()),
                float(self.torch.stack(replay_errors[row_index]).max().cpu()),
                float(self.torch.stack(kl_values[row_index]).mean().cpu()),
            ))
        return results

    def _grpo_epoch_schedule(self, rows: list[dict[str, Any]], training: dict[str, Any], epoch: int):
        if self.distributed.enabled:
            schedule = distributed_schedule(
                rows, {"epochs": 1, "shuffle": bool(training.get("shuffle", False))},
                self.seed + epoch, self.distributed.rank, self.distributed.world_size,
            )
            return [(step, epoch, group, padded) for step, _, group, padded in schedule]
        ordered = list(rows)
        if bool(training.get("shuffle", False)):
            random.Random(self.seed + epoch).shuffle(ordered)
        return [(step, epoch, group, False) for step, group in enumerate(ordered)]

    def grpo_update(self, rows: list[dict[str, Any]], checkpoint_out: str, phase: str = "") -> list[dict[str, Any]]:
        """GRPO with one configurable same-group replay microbatch at a time."""
        training = self._training_config(phase)
        usable = [row for row in rows if any(not item.get("skip_update", True) for item in row.get("candidates", []))]
        before = self.parameter_signature()
        if not usable:
            return self._finish_update(checkpoint_out, {
                "update": "grpo", "steps": 0, "parameter_before": before, "parameter_after": before,
                "reason": "no non-flat valid groups",
            }, sample_ids=[], steps=0)
        optimizer = self._optimizer(training)
        clip = float(training.get("clip_range", 0.2))
        kl_beta = float(training.get("kl_beta", 0.02))
        grad_clip = float(training.get("grad_clip", 1.0))
        replay_batch_size = int(training.get("replay_batch_size", 1))
        logs, grad_norms, sample_ids, update_replay_errors = [], [], [], []
        self.model.set_adapter("default")
        self.model.eval()
        metric_path = Path(checkpoint_out) / "training_metrics"
        if self.distributed.enabled:
            metric_path = metric_path / f"rank_{self.distributed.rank:03d}"
        metric_context = MetricLogger(metric_path)
        epoch_mean_kls: list[float] = []
        group_use_counts: dict[str, int] = {}
        max_epochs = int(training.get("epochs", 1))
        if max_epochs != 1:
            raise ValueError("DualISL-Train Captioner GRPO must run exactly one epoch")
        with metric_context as metrics:
          for epoch in range(max_epochs):
            epoch_kls: list[float] = []
            schedule = self._grpo_epoch_schedule(usable, training, epoch)
            for local_step, _, group, padded in schedule:
                step = len(logs)
                candidates = [
                    candidate for candidate in group["candidates"]
                    if not candidate.get("skip_update") and candidate.get("sampled_token_ids")
                ]
                if not candidates:
                    raise RuntimeError(f"Usable Captioner GRPO group {group['id']} produced no candidate loss")
                optimizer.zero_grad(set_to_none=True)
                candidate_losses = []
                for candidate_batch in _batches(candidates, replay_batch_size):
                    batch_results = self._native_candidates_backward(
                        group, candidate_batch, clip=clip, kl_beta=kl_beta,
                        candidate_count=len(candidates),
                    )
                    for candidate_loss, replay_error, mean_kl in batch_results:
                        candidate_losses.append(candidate_loss)
                        update_replay_errors.append(replay_error)
                        epoch_kls.append(mean_kl)
                if self.distributed.enabled:
                    active = self.distributed.gather_objects(not padded)
                    active_count = sum(bool(value) for value in active)
                    scale = 0.0 if padded else self.distributed.world_size / max(active_count, 1)
                    for _, parameter in self._current_adapter_named_parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(scale)
                self._all_reduce_current_adapter_gradients()
                grad_norm = float(self.torch.nn.utils.clip_grad_norm_(self.trainable_parameters, grad_clip).detach().cpu())
                grad_norms.append(grad_norm)
                optimizer.step()
                loss_value = self.distributed.mean(sum(candidate_losses) / len(candidate_losses))
                logs.append(loss_value)
                if not padded:
                    sample_id = str(group["id"])
                    sample_ids.append(sample_id)
                    group_use_counts[sample_id] = group_use_counts.get(sample_id, 0) + 1
                metrics.log(
                    step, loss=loss_value, grad_norm=grad_norm, update="grpo", epoch=epoch,
                    sample_id=group["id"], candidates=len(candidates), padded=padded,
                )
            epoch_mean = self.distributed.mean(sum(epoch_kls) / max(len(epoch_kls), 1))
            epoch_mean_kls.append(epoch_mean)
        after = self.parameter_signature()
        return self._finish_update(checkpoint_out, {
            "update": "grpo", "steps": len(logs), "mean_loss": sum(logs) / len(logs),
            "max_grad_norm": max(grad_norms), "parameter_before": before, "parameter_after": after,
            "update_behavior_replay_max_abs_error": max(update_replay_errors),
            "cache_gradient_mode": "native_generation_detached_kv_boundaries",
            "gradient_sync_mode": "explicit_current_adapter_all_reduce_mean",
            "grpo_backward_chunk_tokens": int(training.get("grpo_backward_chunk_tokens", 16)),
            "replay_batch_size": replay_batch_size,
            "epochs_completed": len(epoch_mean_kls), "epoch_mean_kl": epoch_mean_kls,
            "group_use_counts": group_use_counts,
        }, sample_ids=sample_ids, steps=len(logs))

    def sft_update(self, rows: list[dict[str, Any]], checkpoint_out: str, phase: str = "") -> list[dict[str, Any]]:
        """SFT with explicit averaging for dynamically routed MoE adapters."""
        for row in rows:
            if row.get("target_origin") != SOURCE_DOMAIN_TARGET:
                raise ValueError(f"Refusing Captioner SFT row {row.get('id')}: target_origin must be source_domain")
        training = self._training_config(phase)
        before = self.parameter_signature()
        if not rows:
            return self._finish_update(checkpoint_out, {
                "update": "sft", "steps": 0, "parameter_before": before, "parameter_after": before,
                "reason": "empty SFT manifest",
            }, sample_ids=[], steps=0)
        optimizer = self._optimizer(training)
        grad_clip = float(training.get("grad_clip", 1.0))
        logs, grad_norms, sample_ids = [], [], []
        gradient_families = {"audio": 0, "text": 0}
        self.model.set_adapter("default")
        self.model.train()
        self.freeze_batchnorm()
        original_use_cache = getattr(self.model.config, "use_cache", None)
        if original_use_cache is not None:
            self.model.config.use_cache = False
        schedule = (
            distributed_schedule(rows, training, self.seed, self.distributed.rank, self.distributed.world_size)
            if self.distributed.enabled else
            [(step, epoch, row, False) for step, (epoch, row) in enumerate(self._schedule(rows, training))]
        )
        metric_path = Path(checkpoint_out) / "training_metrics"
        if self.distributed.enabled:
            metric_path = metric_path / f"rank_{self.distributed.rank:03d}"
        with MetricLogger(metric_path) as metrics:
            for step, epoch, row, padded in schedule:
                target_schema = row.get("target_schema")
                target = (
                    synth_caption_json(row["caption"])
                    if target_schema == "synth_v1" else
                    source_caption_json(row["caption"])
                    if target_schema == "source_no_environment" else
                    caption_json(row["caption"])
                )
                prompt = row.get("prompt") or caption_prompt()
                prompt_inputs = self._inputs(row["audio_path"], prompt)
                full_inputs = self._inputs(row["audio_path"], prompt, target)
                labels = full_inputs["input_ids"].clone()
                prefix, cutoff = completion_token_span(prompt_inputs["input_ids"], full_inputs["input_ids"])
                labels[:, :prefix] = -100
                labels[:, cutoff:] = -100
                if bool(row.get("mask_environment", False)):
                    prefix, cutoff = self._completion_span(
                        row["audio_path"], prompt, target, prompt_inputs, full_inputs,
                    )
                    cutoff = min(cutoff, labels.shape[1])
                    labels[:, cutoff:] = -100
                optimizer.zero_grad(set_to_none=True)
                outputs = self.model(**full_inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                if self.distributed.enabled:
                    active = self.distributed.gather_objects(not padded)
                    active_count = sum(bool(value) for value in active)
                    scale = 0.0 if padded else self.distributed.world_size / max(active_count, 1)
                    for _, parameter in self._current_adapter_named_parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(scale)
                step_families = self._gradient_family_counts()
                for family, count in step_families.items():
                    gradient_families[family] += count
                self._all_reduce_current_adapter_gradients()
                grad_norm = float(self.torch.nn.utils.clip_grad_norm_(self.trainable_parameters, grad_clip).detach().cpu())
                grad_norms.append(grad_norm)
                optimizer.step()
                value = self.distributed.mean(float(loss.detach().cpu()))
                logs.append(value)
                if not padded:
                    sample_ids.append(str(row["id"]))
                metrics.log(
                    step, loss=value, grad_norm=grad_norm, update="sft", epoch=epoch,
                    sample_id=row["id"], environment_masked=bool(row.get("mask_environment", False)), padded=padded,
                )
        if original_use_cache is not None:
            self.model.config.use_cache = original_use_cache
        gradient_audit = self._audit_sft_gradient_families(gradient_families, len(sample_ids))
        after = self.parameter_signature()
        return self._finish_update(checkpoint_out, {
            "update": "sft", "steps": len(logs), "mean_loss": sum(logs) / len(logs),
            "max_grad_norm": max(grad_norms), "parameter_before": before, "parameter_after": after,
            "gradient_sync_mode": "explicit_current_adapter_all_reduce_mean",
            **gradient_audit,
        }, sample_ids=sample_ids, steps=len(logs))

    def _audit_sft_gradient_families(self, counts: dict[str, int], active_samples: int) -> dict[str, Any]:
        # Counts are measured before all-reduce, after padding gradients are zeroed.
        # A rank assigned only padding must still synchronize and save with its peers;
        # it cannot be required to contribute its own nonzero gradient.
        per_rank = self.distributed.gather_objects({
            "rank": self.distributed.rank, "active_samples": active_samples,
            "gradient_family_nonzero": dict(counts),
        })
        active = [item for item in per_rank if item["active_samples"] > 0]
        failed = [item for item in active if any(
            item["gradient_family_nonzero"][family] < 1 for family in ("audio", "text")
        )]
        # All ranks reach the collective before any raises, including on real faults.
        if not active or failed:
            raise RuntimeError(f"Captioner SFT gradient connectivity failed: {failed or per_rank}")
        return {
            "gradient_family_nonzero": {
                family: sum(item["gradient_family_nonzero"][family] for item in active)
                for family in ("audio", "text")
            },
            "gradient_family_nonzero_per_rank": per_rank,
        }

    def _gradient_family_counts(self) -> dict[str, int]:
        result = {"audio": 0, "text": 0}
        for name, parameter in self._current_adapter_named_parameters():
            gradient = parameter.grad
            if gradient is None or not bool(self.torch.isfinite(gradient).all()) or not bool(gradient.abs().max() > 0):
                continue
            family = "audio" if "audio" in name.lower() else "text"
            result[family] += 1
        return result

    def gradient_audit(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            raise ValueError("Gradient audit requires one paired row")
        row = rows[0]
        target_schema = row.get("target_schema")
        target = (
            synth_caption_json(row["caption"])
            if target_schema == "synth_v1" else
            source_caption_json(row["caption"])
            if target_schema == "source_no_environment" else
            caption_json(row["caption"])
        )
        prompt = row.get("prompt") or caption_prompt()
        prompt_inputs = self._inputs(row["audio_path"], prompt)
        full_inputs = self._inputs(row["audio_path"], prompt, target)
        labels = full_inputs["input_ids"].clone()
        prefix, _ = completion_token_span(prompt_inputs["input_ids"], full_inputs["input_ids"])
        labels[:, :prefix] = -100
        original_use_cache = getattr(self.model.config, "use_cache", None)
        if original_use_cache is not None:
            self.model.config.use_cache = False
        self.model.set_adapter("default")
        self.model.train()
        self.freeze_batchnorm()
        self.model.zero_grad(set_to_none=True)
        loss = self.model(**full_inputs, labels=labels).loss
        loss.backward()
        families = self._gradient_family_counts()
        reference_gradients = sum(
            1 for name, parameter in self.model.named_parameters()
            if ".reference." in name and parameter.grad is not None and bool(parameter.grad.abs().max() > 0)
        )
        self.model.zero_grad(set_to_none=True)
        if original_use_cache is not None:
            self.model.config.use_cache = original_use_cache
        ok = families["audio"] > 0 and families["text"] > 0 and reference_gradients == 0
        if not ok:
            raise RuntimeError(
                f"Qwen gradient audit failed: families={families}, reference_gradients={reference_gradients}"
            )
        return [{
            "status": "ok", "loss": float(loss.detach().cpu()),
            "gradient_family_nonzero": families,
            "reference_gradient_tensors": reference_gradients,
        }]

    @staticmethod
    def _native_grpo_loss_value(candidate: dict[str, Any], *, clip: float, kl_beta: float) -> float:
        """The step-start exact replay has ratio=1; report its auditable objective value."""
        import math

        advantage = float(candidate["advantage"])
        policy = -min(advantage, max(1 - clip, min(1 + clip, 1.0)) * advantage)
        values = []
        for current, reference in zip(candidate["old_token_logprobs"], candidate["ref_token_logprobs"]):
            log_ratio = max(-10.0, min(10.0, float(reference) - float(current)))
            values.append(math.exp(log_ratio) - log_ratio - 1.0)
        return policy + kl_beta * sum(values) / len(values)

    def preflight(self) -> list[dict[str, Any]]:
        adapters = getattr(self.model, "peft_config", {})
        reference_trainable = sum(
            parameter.numel() for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and ".reference." in name
        )
        return [{
            "status": "ok",
            "backend": "Qwen3OmniMoeThinkerForConditionalGeneration",
            "trainable_parameters": sum(parameter.numel() for parameter in self.trainable_parameters),
            "adapter_names": sorted(adapters),
            "has_reference_adapter": self.has_reference_adapter,
            "reference_trainable_parameters": reference_trainable,
            "model_mode": "eval" if not self.model.training else "train",
            "gradient_checkpointing": bool(getattr(self.model, "is_gradient_checkpointing", False)),
            "device": str(self.device),
            "dtype": str(next(self.model.parameters()).dtype),
            "eos_token_id": self.eos_token_id,
            "eos_token": self.tokenizer.eos_token,
            "pad_token_id": self.pad_token_id,
            "rollout_batch_size": int(self.cfg.get("generation", {}).get("rollout_batch_size", 1)),
            "replay_batch_size": int(self.cfg.get("training", {}).get("replay_batch_size", 1)),
            "adapter_audits": self.adapter_audits,
            "gpu_peak_memory_bytes": int(self.torch.cuda.max_memory_allocated(self.device)),
        }]


def main() -> None:
    parser = worker_parser(
        "Qwen3-Omni structured caption worker",
        ["rollout", "score", "grpo-update", "sft-update", "gradient-audit", "preflight"],
    )
    args = parser.parse_args()
    config, rows = load_job(args)
    distributed = DistributedContext.initialize(config)
    worker: Qwen3CaptionerWorker | None = None
    try:
        worker = Qwen3CaptionerWorker(config, args.checkpoint_in, args.target_checkpoint, distributed)
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
