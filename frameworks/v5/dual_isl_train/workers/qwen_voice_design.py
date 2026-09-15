from __future__ import annotations

import contextlib
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

from dual_isl_train.adapters import (
    audit_adapter_checkpoint, audit_adapter_pair, load_reference_adapter_exact, promote_adapter_fp32,
)
from dual_isl_train.constants import CHECKPOINT_METADATA, FRAMEWORK_VERSION, SOURCE_DOMAIN_TARGET, TRAJECTORY_VERSION
from dual_isl_train.distributed import (
    DistributedContext, cost_bucketed_distributed_schedule, distributed_schedule, run_sharded_inference,
)
from dual_isl_train.io import atomic_json, sha256_file
from dual_isl_train.render import render_qwen_request
from dual_isl_train.synchronization import audit_parameter_sync, parameter_delta, snapshot_parameters
from dual_isl_train.telemetry import MetricLogger, ProgressLogger
from dual_isl_train.trajectory import mark_trajectory
from dual_isl_train.workers.common import load_job, validate_checkpoint_version, worker_parser, write_output


def main_generation_logprobs(torch, generation_result, codes):
    """Gather main-talker log-probs from the actual processed generation scores."""
    targets = torch.as_tensor(codes, dtype=torch.long)
    if targets.ndim != 2 or targets.shape[1] != 16:
        raise ValueError(f"Expected generated codec codes [T,16], got {tuple(targets.shape)}")
    scores = getattr(generation_result, "scores", None)
    if not scores or len(scores) < targets.shape[0]:
        raise RuntimeError(
            f"Qwen generation exposed {0 if scores is None else len(scores)} main scores for "
            f"{targets.shape[0]} retained codec frames"
        )
    values = []
    for index, target in enumerate(targets[:, 0].tolist()):
        logits = scores[index]
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise RuntimeError(f"Unexpected Qwen main generation score shape: {tuple(logits.shape)}")
        values.append(torch.log_softmax(logits.float(), dim=-1)[0, int(target)])
    return torch.stack(values)


def sub_generation_logprobs(torch, generation_results, codes):
    """Gather all 15 sub-talker log-probs from their actual generation scores."""
    targets = torch.as_tensor(codes, dtype=torch.long)
    if targets.ndim != 2 or targets.shape[1] != 16:
        raise ValueError(f"Expected generated codec codes [T,16], got {tuple(targets.shape)}")
    if len(generation_results) < targets.shape[0]:
        raise RuntimeError(
            f"Qwen generation exposed {len(generation_results)} sub-talker traces for "
            f"{targets.shape[0]} retained codec frames"
        )
    frames = []
    for frame_index, result in enumerate(generation_results[:targets.shape[0]]):
        scores = getattr(result, "scores", None)
        if not scores or len(scores) < 15:
            raise RuntimeError(
                f"Qwen sub-talker generation exposed {0 if scores is None else len(scores)} "
                f"scores for frame {frame_index}; expected 15"
            )
        values = []
        for codebook_index, target in enumerate(targets[frame_index, 1:].tolist()):
            logits = scores[codebook_index]
            if logits.ndim != 2 or logits.shape[0] != 1:
                raise RuntimeError(f"Unexpected Qwen sub generation score shape: {tuple(logits.shape)}")
            values.append(torch.log_softmax(logits.float(), dim=-1)[0, int(target)])
        frames.append(torch.stack(values))
    return torch.stack(frames)


def candidate_backward_scale(candidate_count: int, world_size: int, active_count: int, padded: bool) -> float:
    """Scale one candidate so sequential backward equals the old group mean."""
    if candidate_count < 1:
        raise ValueError("candidate_count must be >= 1")
    if world_size < 1 or active_count < 1 or active_count > world_size:
        raise ValueError("active_count must be within [1, world_size]")
    return 0.0 if padded else world_size / active_count / candidate_count


def heartbeat_frame_due(processed_frames: int, total_frames: int, interval: int) -> bool:
    if total_frames < 1 or processed_frames < 1 or processed_frames > total_frames:
        raise ValueError("processed_frames must be within [1, total_frames]")
    if interval < 1:
        raise ValueError("heartbeat interval must be >= 1")
    return processed_frames == total_frames or processed_frames % interval == 0


from .tts_batch import BatchedTTSMixin


class QwenVoiceDesignWorker(BatchedTTSMixin):
    """Qwen3-TTS VoiceDesign worker with explicit 16-codebook policy likelihoods."""

    def __init__(
        self, config: dict[str, Any], checkpoint: str = "", target_checkpoint: str = "",
        distributed: DistributedContext | None = None,
    ):
        import torch
        from qwen_tts import Qwen3TTSModel

        self.torch = torch
        self.cfg = config["tts"]
        self.sync_cfg = config.get("synchronization", {})
        self.seed = int(config["run"].get("seed", 42))
        self.parent_checkpoint = checkpoint or None
        self.distributed = distributed or DistributedContext()
        self.device = torch.device(self.distributed.device or str(self.cfg.get("device", "cuda:0")))
        self.ddp_replay = None
        self.replay_module = None
        self._heartbeat_frames = int(self.cfg.get("training", {}).get("heartbeat_frames", 32))
        self.output_metrics: dict[str, Any] = {}
        dtype_name = str(self.cfg.get("dtype", "bfloat16"))
        self.dtype = getattr(torch, dtype_name)
        self.wrapper = Qwen3TTSModel.from_pretrained(
            self.cfg["model_path"], device_map=str(self.device), dtype=self.dtype,
            attn_implementation=self.cfg.get("attn_implementation", "flash_attention_2"),
            local_files_only=True,
        )
        if str(self.wrapper.model.tts_model_type) != "voice_design":
            raise ValueError(f"Expected Qwen VoiceDesign, got {self.wrapper.model.tts_model_type!r}")
        if int(self.wrapper.model.config.talker_config.num_code_groups) != int(self.cfg.get("num_codebooks", 16)):
            raise ValueError("Qwen model codebook count does not match config")
        self.adapter_audits: dict[str, Any] = {}
        self.policy = self._attach_or_load_lora(self.wrapper.model, checkpoint, target_checkpoint)
        self.wrapper.model = self.policy
        self.has_reference_adapter = bool(target_checkpoint and (Path(target_checkpoint) / "adapter_config.json").is_file())
        self.policy.set_adapter("default")
        self._update_snapshot = None
        self._parameter_sync_before = None
        self._broadcast_trainable_parameters()
        if bool(self.cfg.get("gradient_checkpointing", True)):
            core = self.core
            checkpointing_kwargs = {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
            if hasattr(core, "gradient_checkpointing_enable"):
                core.gradient_checkpointing_enable(**checkpointing_kwargs)
            if hasattr(core.talker, "gradient_checkpointing_enable"):
                core.talker.gradient_checkpointing_enable(**checkpointing_kwargs)

    @property
    def core(self):
        return self.policy.get_base_model()

    def _ensure_ddp(self) -> None:
        if not self.distributed.enabled or self.ddp_replay is not None:
            return
        import weakref
        from torch.nn.parallel import DistributedDataParallel

        worker_ref = weakref.ref(self)

        class ReplayModule(self.torch.nn.Module):
            def __init__(module_self, talker):
                super().__init__()
                module_self.talker = talker

            def forward(module_self, mode, request, codes, processed_policy=False, progress_callback=None):
                worker = worker_ref()
                if worker is None:
                    raise RuntimeError("TTS replay worker was released")
                if mode == "incremental":
                    return worker._incremental_trajectory_impl(
                        request, codes, talker=module_self.talker, progress_callback=progress_callback,
                    )
                if mode == "incremental_group":
                    return tuple(
                        worker._incremental_trajectory_impl(item["request"], item["codes"], talker=module_self.talker)
                        for item in request
                    )
                if mode == "teacher":
                    return worker._trajectory_impl(
                        request, codes, talker=module_self.talker, processed_policy=processed_policy,
                    )
                raise ValueError(f"Unknown replay mode: {mode}")

        self.replay_module = ReplayModule(self.core.talker)
        self.ddp_replay = DistributedDataParallel(
            self.replay_module,
            device_ids=[self.distributed.local_rank], output_device=self.distributed.local_rank,
            broadcast_buffers=False, find_unused_parameters=False, static_graph=False,
        )

    def _lora_config(self):
        from peft import LoraConfig
        lora = self.cfg.get("lora", {})
        targets = "|".join([
            r"^talker\.model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$",
            r"^talker\.model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)$",
            r"^talker\.code_predictor\.model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$",
            r"^talker\.code_predictor\.model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)$",
        ])
        return LoraConfig(
            r=int(lora.get("r", 16)), lora_alpha=int(lora.get("alpha", 32)),
            lora_dropout=float(lora.get("dropout", 0.05)), target_modules=targets, bias="none",
        )

    def _attach_or_load_lora(self, model, checkpoint: str, target_checkpoint: str):
        from peft import PeftModel, get_peft_model
        if checkpoint and (Path(checkpoint) / "adapter_config.json").is_file():
            validate_checkpoint_version(checkpoint)
            policy = PeftModel.from_pretrained(
                model, checkpoint, is_trainable=True, autocast_adapter_dtype=True,
            )
            promote_adapter_fp32(policy, "default")
            self.adapter_audits["policy"] = audit_adapter_checkpoint(policy, checkpoint, "default")
            if not self.adapter_audits["policy"]["exact"]:
                raise RuntimeError(f"TTS policy adapter load was not exact: {self.adapter_audits['policy']}")
        else:
            policy = get_peft_model(model, self._lora_config(), autocast_adapter_dtype=True)
            promote_adapter_fp32(policy, "default")
        if target_checkpoint and (Path(target_checkpoint) / "adapter_config.json").is_file():
            validate_checkpoint_version(target_checkpoint)
            self.adapter_audits["reference"] = load_reference_adapter_exact(
                policy, target_checkpoint, "reference",
            )
            if checkpoint and Path(checkpoint).resolve() == Path(target_checkpoint).resolve():
                self.adapter_audits["policy_reference"] = audit_adapter_pair(policy)
                if not self.adapter_audits["policy_reference"]["exact"]:
                    raise RuntimeError(
                        f"TTS policy/reference adapters differ at round start: "
                        f"{self.adapter_audits['policy_reference']}"
                    )
        return policy

    @contextlib.contextmanager
    def adapter_context(self, reference: bool):
        if not reference:
            self.policy.set_adapter("default")
            yield
            return
        if self.has_reference_adapter:
            self.policy.set_adapter("reference")
            try:
                yield
            finally:
                self.policy.set_adapter("default")
        else:
            with self.policy.disable_adapter():
                yield

    @property
    def trainable_parameters(self) -> list[Any]:
        self.policy.set_adapter("default")
        return [parameter for parameter in self.policy.parameters() if parameter.requires_grad]

    def _trainable_named_parameters(self) -> list[tuple[str, Any]]:
        self.policy.set_adapter("default")
        values = [(name, parameter) for name, parameter in self.policy.named_parameters() if parameter.requires_grad]
        if not values:
            raise RuntimeError("TTS current adapter has no trainable parameters")
        return values

    def _broadcast_trainable_parameters(self) -> None:
        if not self.distributed.enabled:
            return
        import torch.distributed as dist

        for _, parameter in self._trainable_named_parameters():
            dist.broadcast(parameter.data, src=0)

    def _sync_audit(self) -> dict[str, Any]:
        values = self.sync_cfg
        return audit_parameter_sync(
            self._trainable_named_parameters(), self.distributed,
            rtol=float(values.get("rtol", 1e-6)), atol=float(values.get("atol", 1e-7)),
        )

    def _begin_update_audit(self) -> None:
        self._parameter_sync_before = self._sync_audit()
        self._update_snapshot = snapshot_parameters(self._trainable_named_parameters())

    def parameter_signature(self) -> float:
        parameters = self.trainable_parameters
        return sum((index + 1) * parameter.detach().double().sum().item() for index, parameter in enumerate(parameters)) if parameters else 0.0

    def _tokenize_request(self, request: dict[str, str]):
        input_id = self.wrapper._tokenize_texts([self.wrapper._build_assistant_text(request["text"])])[0]
        instruct = request.get("instruct", "")
        instruct_id = self.wrapper._tokenize_texts([self.wrapper._build_instruct_text(instruct)])[0] if instruct else None
        return input_id, instruct_id

    def _conditioning(self, request: dict[str, str], *, talker=None):
        """Reproduce the non-streaming VoiceDesign prefix used by Qwen inference."""
        torch = self.torch
        core = self.core
        talker = talker or core.talker
        input_id, instruct_id = self._tokenize_request(request)
        input_id = input_id.to(self.device)
        parts = []
        if instruct_id is not None:
            instruct_id = instruct_id.to(self.device)
            parts.append(talker.text_projection(talker.get_text_embeddings()(instruct_id)))
        language = request.get("language", "Auto")
        if language.lower() == "auto":
            language_id = None
        else:
            language_id = core.config.talker_config.codec_language_id[language.lower()]
        special = torch.tensor(
            [[core.config.tts_bos_token_id, core.config.tts_eos_token_id, core.config.tts_pad_token_id]],
            device=self.device, dtype=input_id.dtype,
        )
        tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(talker.get_text_embeddings()(special)).chunk(3, dim=1)
        if language_id is None:
            codec_prefill = [[
                core.config.talker_config.codec_nothink_id,
                core.config.talker_config.codec_think_bos_id,
                core.config.talker_config.codec_think_eos_id,
            ]]
        else:
            codec_prefill = [[
                core.config.talker_config.codec_think_id,
                core.config.talker_config.codec_think_bos_id,
                language_id,
                core.config.talker_config.codec_think_eos_id,
            ]]
        codec_ids = torch.tensor(codec_prefill, device=self.device, dtype=input_id.dtype)
        codec_embed_0 = talker.get_input_embeddings()(codec_ids)
        codec_embed_1 = talker.get_input_embeddings()(torch.tensor(
            [[core.config.talker_config.codec_pad_id, core.config.talker_config.codec_bos_id]],
            device=self.device, dtype=input_id.dtype,
        ))
        codec_embed = torch.cat([codec_embed_0, codec_embed_1], dim=1)
        role = talker.text_projection(talker.get_text_embeddings()(input_id[:, :3]))
        prefix_codec = torch.cat([
            tts_pad_embed.expand(-1, codec_embed.shape[1] - 2, -1), tts_bos_embed,
        ], dim=1) + codec_embed[:, :-1]
        prefix = torch.cat([role, prefix_codec], dim=1)
        text_part = torch.cat([
            talker.text_projection(talker.get_text_embeddings()(input_id[:, 3:-5])), tts_eos_embed,
        ], dim=1)
        text_part = text_part + talker.get_input_embeddings()(torch.full(
            (1, text_part.shape[1]), core.config.talker_config.codec_pad_id,
            device=self.device, dtype=input_id.dtype,
        ))
        codec_bos = tts_pad_embed + talker.get_input_embeddings()(torch.tensor(
            [[core.config.talker_config.codec_bos_id]], device=self.device, dtype=input_id.dtype,
        ))
        prefix = torch.cat([prefix, text_part, codec_bos], dim=1)
        parts.append(prefix)
        return torch.cat(parts, dim=1), tts_pad_embed

    def _frame_embeddings(self, codes, tts_pad_embed, *, talker=None):
        talker = talker or self.core.talker
        # Match Qwen generation's cat(...).sum(1) reduction order exactly.  A
        # left-associated BF16 sum is measurably different and breaks replay.
        pieces = [talker.get_input_embeddings()(codes[:, :, 0])]
        pieces.extend(
            talker.code_predictor.get_input_embeddings()[codebook - 1](codes[:, :, codebook])
            for codebook in range(1, 16)
        )
        return self.torch.stack(pieces, dim=2).sum(2) + tts_pad_embed

    def _incremental_trajectory_impl(
        self, request: dict[str, str], codes, *, talker, progress_callback=None,
    ):
        """Differentiable KV-cache replay executed inside the DDP forward boundary."""
        torch = self.torch
        codes = torch.as_tensor(codes, dtype=torch.long, device=self.device)
        if codes.ndim == 2:
            codes = codes.unsqueeze(0)
        if codes.ndim != 3 or codes.shape[0] != 1 or codes.shape[2] != 16:
            raise ValueError(f"Expected codec codes [T,16], got {tuple(codes.shape)}")
        prefix, tts_pad_embed = self._conditioning(request, talker=talker)
        attention = torch.ones(prefix.shape[:2], dtype=torch.long, device=self.device)
        output = talker.model(
            inputs_embeds=prefix, attention_mask=attention, use_cache=True, output_hidden_states=True,
        )
        cache = output.past_key_values
        hidden = output.last_hidden_state[:, -1, :]
        logits = talker.codec_head(hidden)
        main_values, sub_values = [], []
        vocab_size = int(self.core.config.talker_config.vocab_size)
        eos = int(self.core.config.talker_config.codec_eos_token_id)
        suppressed = [index for index in range(vocab_size - 1024, vocab_size) if index != eos]
        total_frames = int(codes.shape[1])
        heartbeat_interval = self._heartbeat_frames
        for frame_index in range(total_frames):
            processed = logits.float().clone()
            processed[..., suppressed] = -torch.inf
            if frame_index < 2:
                processed[..., eos] = -torch.inf
            target_main = codes[:, frame_index, 0]
            main_values.append(
                torch.log_softmax(processed, dim=-1).gather(-1, target_main.unsqueeze(-1)).squeeze(-1)
            )
            frame_codes = codes[:, frame_index, :]
            predictor = talker.code_predictor
            predictor_inputs = torch.cat([
                hidden.unsqueeze(1), talker.get_input_embeddings()(target_main.unsqueeze(-1)),
            ], dim=1)
            predictor_output = predictor(
                inputs_embeds=predictor_inputs, use_cache=True, output_hidden_states=True,
            )
            predictor_cache = predictor_output.past_key_values
            predictor_logits = predictor_output.logits[:, -1, :]
            predictor_attention = torch.ones((1, 2), dtype=torch.long, device=self.device)
            codebook_values = []
            generation_steps = predictor_output.generation_steps
            for sub_index in range(15):
                target_sub = frame_codes[:, sub_index + 1]
                codebook_values.append(
                    torch.log_softmax(predictor_logits.float(), dim=-1)
                    .gather(-1, target_sub.unsqueeze(-1)).squeeze(-1)
                )
                if sub_index < 14:
                    predictor_attention = torch.cat([
                        predictor_attention, torch.ones((1, 1), dtype=torch.long, device=self.device),
                    ], dim=1)
                    predictor_output = predictor(
                        input_ids=target_sub.unsqueeze(-1), attention_mask=predictor_attention,
                        past_key_values=predictor_cache, use_cache=True,
                        generation_steps=generation_steps, output_hidden_states=True,
                    )
                    predictor_cache = predictor_output.past_key_values
                    predictor_logits = predictor_output.logits[:, -1, :]
                    generation_steps = predictor_output.generation_steps
            sub_values.append(torch.stack(codebook_values, dim=1))
            if frame_index + 1 < codes.shape[1]:
                frame_embedding = self._frame_embeddings(
                    codes[:, frame_index:frame_index + 1, :], tts_pad_embed, talker=talker,
                )
                attention = torch.cat([
                    attention, torch.ones((1, 1), dtype=torch.long, device=self.device),
                ], dim=1)
                output = talker.model(
                    inputs_embeds=frame_embedding, attention_mask=attention,
                    past_key_values=cache, use_cache=True, output_hidden_states=True,
                )
                cache = output.past_key_values
                hidden = output.last_hidden_state[:, -1, :]
                logits = talker.codec_head(hidden)
            processed_frames = frame_index + 1
            if progress_callback is not None and heartbeat_frame_due(
                processed_frames, total_frames, heartbeat_interval,
            ):
                progress_callback(processed_frames, total_frames)
        return torch.cat(main_values, dim=0), torch.cat(sub_values, dim=0)

    def incremental_trajectory_logprobs(
        self, request: dict[str, str], codes, *, reference: bool = False, grad: bool = False,
    ):
        grad_context = contextlib.nullcontext() if grad else self.torch.no_grad()
        with grad_context, self.adapter_context(reference):
            if grad and self.distributed.enabled:
                self._ensure_ddp()
                return self.ddp_replay("incremental", request, codes, False)
            return self._incremental_trajectory_impl(request, codes, talker=self.core.talker)

    def incremental_group_logprobs(self, candidates: list[dict[str, Any]]):
        """Enter DDP exactly once for one GRPO group on every rank."""
        payload = [
            {"request": candidate["request"], "codes": candidate["codec_codes"]}
            for candidate in candidates
        ]
        with self.adapter_context(False):
            if self.distributed.enabled:
                self._ensure_ddp()
                return self.ddp_replay("incremental_group", payload, None, False)
            return tuple(
                self._incremental_trajectory_impl(item["request"], item["codes"], talker=self.core.talker)
                for item in payload
            )

    def incremental_candidate_logprobs(self, candidate: dict[str, Any], progress_callback=None):
        """Replay one candidate so its graph can be released after backward."""
        with self.adapter_context(False):
            if self.distributed.enabled:
                self._ensure_ddp()
                return self.ddp_replay(
                    "incremental", candidate["request"], candidate["codec_codes"], False,
                    progress_callback,
                )
            return self._incremental_trajectory_impl(
                candidate["request"], candidate["codec_codes"], talker=self.core.talker,
                progress_callback=progress_callback,
            )

    def _trajectory_impl(self, request: dict[str, str], codes, *, talker, processed_policy: bool = False):
        torch = self.torch
        codes = torch.as_tensor(codes, dtype=torch.long, device=self.device)
        if codes.ndim == 2:
            codes = codes.unsqueeze(0)
        if codes.ndim != 3 or codes.shape[0] != 1 or codes.shape[2] != 16:
            raise ValueError(f"Expected codec codes [T,16], got {tuple(codes.shape)}")
        prefix, tts_pad_embed = self._conditioning(request, talker=talker)
        frame_embeddings = self._frame_embeddings(codes, tts_pad_embed, talker=talker)
        inputs = torch.cat([prefix, frame_embeddings], dim=1)
        attention = torch.ones(inputs.shape[:2], dtype=torch.long, device=self.device)
        output = talker(
            inputs_embeds=inputs, attention_mask=attention, output_hidden_states=True, use_cache=False,
        )
        length = codes.shape[1]
        start = prefix.shape[1] - 1
        main_logits = output.logits[:, start:start + length, :]
        main_targets = codes[:, :, 0]
        if processed_policy:
            vocab_size = int(self.core.config.talker_config.vocab_size)
            eos = int(self.core.config.talker_config.codec_eos_token_id)
            suppressed = [index for index in range(vocab_size - 1024, vocab_size) if index != eos]
            main_logits = main_logits.float().clone()
            main_logits[..., suppressed] = -self.torch.inf
            main_logits[:, :min(2, length), eos] = -self.torch.inf
        main_logps = torch.log_softmax(main_logits.float(), dim=-1).gather(
            -1, main_targets.unsqueeze(-1),
        ).squeeze(0).squeeze(-1)
        hidden = output.hidden_states[0][-1][:, start:start + length, :].squeeze(0)
        sub_logits, _ = talker.forward_sub_talker_finetune(codes.squeeze(0), hidden)
        sub_targets = codes.squeeze(0)[:, 1:]
        sub_logps = torch.log_softmax(sub_logits.float(), dim=-1).gather(
            -1, sub_targets.unsqueeze(-1),
        ).squeeze(-1)
        return main_logps, sub_logps

    def trajectory_logprobs(
        self, request: dict[str, str], codes, *, reference: bool = False,
        grad: bool = False, processed_policy: bool = False,
    ):
        grad_context = contextlib.nullcontext() if grad else self.torch.no_grad()
        with grad_context, self.adapter_context(reference):
            if grad and self.distributed.enabled:
                self._ensure_ddp()
                return self.ddp_replay("teacher", request, codes, processed_policy)
            return self._trajectory_impl(
                request, codes, talker=self.core.talker, processed_policy=processed_policy,
            )

    def encode_audio(self, audio_path: str):
        encoded = self.core.speech_tokenizer.encode(audio_path)
        codes = encoded.audio_codes[0]
        if codes.ndim != 2 or codes.shape[1] != 16:
            raise ValueError(f"Tokenizer returned invalid codes for {audio_path}: {tuple(codes.shape)}")
        return codes.long().cpu()

    def _subtalker_logprobs_from_hidden(self, codes, hidden):
        codes = self.torch.as_tensor(codes, dtype=self.torch.long, device=self.device)
        hidden = hidden.to(self.device)
        sub_logits, _ = self.core.talker.forward_sub_talker_finetune(codes, hidden)
        return self.torch.log_softmax(sub_logits.float(), dim=-1).gather(-1, codes[:, 1:].unsqueeze(-1)).squeeze(-1)

    def _generate_with_main_scores(self, **kwargs):
        """Capture main and sub HF generation results discarded by Qwen's wrapper."""
        talker = self.core.talker
        had_instance_method = "generate" in talker.__dict__
        previous_instance_method = talker.__dict__.get("generate")
        original_generate = talker.generate
        predictor = talker.code_predictor
        original_predictor_generate = predictor.generate
        captured: dict[str, Any] = {"sub": []}

        def capture(*args, **generation_kwargs):
            generation_kwargs["output_scores"] = True
            generation_kwargs["return_dict_in_generate"] = True
            result = original_generate(*args, **generation_kwargs)
            captured["result"] = result
            return result

        def capture_sub(*args, **generation_kwargs):
            generation_kwargs["output_scores"] = True
            generation_kwargs["return_dict_in_generate"] = True
            result = original_predictor_generate(*args, **generation_kwargs)
            captured["sub"].append(result)
            return result

        talker.generate = capture
        predictor.generate = capture_sub
        try:
            output = self.core.generate(**kwargs)
        finally:
            delattr(predictor, "generate")
            if had_instance_method:
                talker.generate = previous_instance_method
            else:
                delattr(talker, "generate")
        if "result" not in captured:
            raise RuntimeError("Qwen wrapper did not invoke the main talker generation method")
        return output, captured["result"], captured["sub"]

    def prepare_codecs(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cache_dir = Path(self.cfg["codec_cache_dir"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_identity = hashlib.sha256(str(self.cfg.get("tokenizer_path") or self.cfg["model_path"]).encode()).hexdigest()
        output = []
        for row in rows:
            audio_path = str(row["audio_path"])
            audio_hash = sha256_file(audio_path)
            cache_path = cache_dir / f"{row['id']}.{audio_hash[:16]}.json"
            valid = False
            if cache_path.is_file():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                valid = cached.get("audio_sha256") == audio_hash and cached.get("tokenizer_identity") == tokenizer_identity
            if not valid:
                codes = self.encode_audio(audio_path)
                atomic_json(cache_path, {
                    "id": row["id"], "audio_path": audio_path, "audio_sha256": audio_hash,
                    "tokenizer_identity": tokenizer_identity, "shape": list(codes.shape), "codec_codes": codes.tolist(),
                })
            output.append({"id": row["id"], "audio_path": audio_path, "codec_path": str(cache_path), "cache_reused": valid})
        return output

    @staticmethod
    def _load_codes(row: dict[str, Any]):
        if row.get("codec_codes"):
            return row["codec_codes"]
        codec_path = row.get("codec_path")
        if not codec_path:
            raise ValueError(f"Row {row.get('id')} has neither codec_codes nor codec_path")
        return json.loads(Path(codec_path).read_text(encoding="utf-8"))["codec_codes"]

    def score_reconstruction(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sub_weight = float(self.cfg.get("training", {}).get("sub_talker_weight", 0.3))
        output = []
        for row in rows:
            main, sub = self.trajectory_logprobs(row["request"], self._load_codes(row), reference=True)
            score = float(main.mean().item() + sub_weight * sub.mean().item())
            output.append({
                "candidate_id": row["candidate_id"], "tts_target_logprob": score,
                "tts_main_logprob": float(main.mean().item()), "tts_sub_logprob": float(sub.mean().item()),
                "codec_frames": int(main.numel()), "codebooks": int(sub.shape[1] + 1),
            })
        return output

    def rollout(self, rows, output_path):
        if int(self.cfg.get("generation", {}).get("rollout_batch_size", 1)) > 1:
            return self.rollout_v5(rows, output_path)
        return self._serial_rollout(rows, output_path)

    def _serial_rollout(self, rows: list[dict[str, Any]], output_path: str) -> list[dict[str, Any]]:
        import soundfile as sf
        generation = self.cfg.get("generation", {})
        audio_dir = Path(output_path).parent / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        output = []
        self.policy.set_adapter("default")
        self.policy.eval()
        from ..reference_replay import RolloutReferenceReplay
        reference_replay = RolloutReferenceReplay(self)
        for row in rows:
            request = row.get("request") or render_qwen_request(row["caption"])
            input_id, instruct_id = self._tokenize_request(request)
            candidates = []
            for index in range(int(row.get("group_size", 2))):
                seeds = row.get("candidate_seeds") or []
                generation_seed = int(seeds[index]) if index < len(seeds) else self.seed + index
                self.torch.manual_seed(generation_seed)
                if self.torch.cuda.is_available():
                    self.torch.cuda.manual_seed(generation_seed)
                with self.torch.no_grad(), self.adapter_context(False):
                    (codes_list, _hidden_list), main_generation, sub_generations = self._generate_with_main_scores(
                        input_ids=[input_id.to(self.device)], instruct_ids=[instruct_id.to(self.device) if instruct_id is not None else None],
                        languages=[request["language"]], non_streaming_mode=True,
                        max_new_tokens=int(generation.get("max_new_tokens", 512)),
                        do_sample=True, top_k=int(generation.get("top_k", 50)),
                        top_p=float(generation.get("top_p", 1.0)), temperature=float(generation.get("temperature", 0.9)),
                        subtalker_dosample=True, subtalker_top_k=int(generation.get("subtalker_top_k", 50)),
                        subtalker_top_p=float(generation.get("subtalker_top_p", 1.0)),
                        subtalker_temperature=float(generation.get("subtalker_temperature", 0.9)),
                        repetition_penalty=float(generation.get("repetition_penalty", 1.0)),
                    )
                codes = codes_list[0]
                generation_main = main_generation_logprobs(self.torch, main_generation, codes).to(self.device)
                generation_sub = sub_generation_logprobs(self.torch, sub_generations, codes).to(self.device)
                with self.torch.no_grad():
                    wavs, sample_rate = self.core.speech_tokenizer.decode([{"audio_codes": codes}])
                candidate_id = f"{row['id']}::{index}"
                audio_path = audio_dir / f"{candidate_id.replace('::', '_')}.wav"
                sf.write(audio_path, wavs[0], sample_rate)
                replay_main, replay_sub = self.incremental_trajectory_logprobs(request, codes)
                (ref_main, ref_sub), reference_mode = reference_replay.reference(
                    request, codes, (replay_main, replay_sub),
                )
                candidate = {
                    "candidate_id": candidate_id, "audio_path": str(audio_path), "sample_rate": int(sample_rate),
                    "generation_seed": generation_seed,
                    "request": request, "codec_codes": codes.long().cpu().tolist(),
                    "old_main_logprobs": generation_main.cpu().tolist(), "old_sub_logprobs": generation_sub.cpu().tolist(),
                    "ref_main_logprobs": ref_main.cpu().tolist(), "ref_sub_logprobs": ref_sub.cpu().tolist(),
                    "codec_frames": int(codes.shape[0]), "codebooks": int(codes.shape[1]),
                    "main_behavior_replay_max_abs_error": float((generation_main - replay_main).abs().max().item()),
                    "subtalker_behavior_replay_max_abs_error": float((generation_sub - replay_sub).abs().max().item()),
                    "main_policy_reference_max_abs_error": float((replay_main - ref_main).abs().max().item()),
                    "subtalker_policy_reference_max_abs_error": float((replay_sub - ref_sub).abs().max().item()),
                    "behavior_logprob_mode": "processed_generation_scores", "trajectory_version": TRAJECTORY_VERSION,
                    "reference_replay_mode": reference_mode,
                    "policy_reference_error_source": (
                        "reused_policy_values" if reference_mode == "reused" else "independent_reference_replay"
                    ),
                }
                candidates.append(mark_trajectory(candidate, "tts"))
            output.append({**row, "request": request, "candidates": candidates})
        self.output_metrics["reference_replay"] = reference_replay.metrics
        return output

    def _training_config(self, phase: str) -> dict[str, Any]:
        training = dict(self.cfg.get("training", {}))
        phases = training.pop("phases", {})
        if phase:
            training.update(phases.get(phase, {}))
        return training

    def _schedule(self, rows: list[dict[str, Any]], training: dict[str, Any]):
        steps = int(training.get("min_epochs", training.get("epochs", 1))) * len(rows)
        rng = random.Random(self.seed)
        result = []
        epoch = 0
        while len(result) < steps and rows:
            ordered = list(rows)
            if bool(training.get("shuffle", False)):
                rng.shuffle(ordered)
            result.extend((epoch, row) for row in ordered)
            epoch += 1
        return result[:steps]

    def _grpo_epoch_schedule(self, rows: list[dict[str, Any]], training: dict[str, Any], epoch: int):
        if self.distributed.enabled:
            schedule_config = {"epochs": 1, "shuffle": bool(training.get("shuffle", False))}
            if str(training.get("schedule", "length_bucketed")) == "length_bucketed":
                schedule = cost_bucketed_distributed_schedule(
                    rows, schedule_config, self.seed + epoch,
                    self.distributed.rank, self.distributed.world_size, self._grpo_group_cost,
                )
            else:
                schedule = distributed_schedule(
                    rows, schedule_config, self.seed + epoch,
                    self.distributed.rank, self.distributed.world_size,
                )
            return [(step, epoch, group, padded) for step, _, group, padded in schedule]
        ordered = list(rows)
        if bool(training.get("shuffle", False)):
            random.Random(self.seed + epoch).shuffle(ordered)
        return [(step, epoch, group, False) for step, group in enumerate(ordered)]

    @staticmethod
    def _grpo_group_cost(group: dict[str, Any]) -> int:
        return sum(
            len(candidate.get("codec_codes") or candidate.get("old_main_logprobs") or [])
            for candidate in group.get("candidates", [])
            if not candidate.get("skip_update")
        )

    def _cuda_memory_snapshot(self) -> dict[str, int]:
        if not self.torch.cuda.is_available():
            return {"allocated_bytes": 0, "reserved_bytes": 0, "peak_allocated_bytes": 0}
        return {
            "allocated_bytes": int(self.torch.cuda.memory_allocated(self.device)),
            "reserved_bytes": int(self.torch.cuda.memory_reserved(self.device)),
            "peak_allocated_bytes": int(self.torch.cuda.max_memory_allocated(self.device)),
        }

    def _optimizer(self, training: dict[str, Any]):
        return self.torch.optim.AdamW(
            self.trainable_parameters, lr=float(training.get("lr", 1e-6)),
            weight_decay=float(training.get("weight_decay", 0.01)),
        )

    def _save(self, checkpoint_out: str, details: dict[str, Any]) -> None:
        self.policy.set_adapter("default")
        self.policy.save_pretrained(checkpoint_out, selected_adapters=["default"])
        Path(checkpoint_out, CHECKPOINT_METADATA).write_text(json.dumps({
            "framework_version": FRAMEWORK_VERSION,
            "trajectory_version": TRAJECTORY_VERSION,
            "parent_checkpoint": self.parent_checkpoint,
            **details,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _finish_update(
        self, checkpoint_out: str, details: dict[str, Any], *, sample_ids: list[str], steps: int,
    ) -> list[dict[str, Any]]:
        sync_after = self._sync_audit()
        delta = parameter_delta(self._update_snapshot or {}, self._trainable_named_parameters())
        if steps > 0 and not delta["ok"]:
            raise RuntimeError(f"TTS optimizer ran {steps} steps without a verified parameter change: {delta}")
        details["parameter_sync_before"] = self._parameter_sync_before
        details["parameter_sync_after"] = sync_after
        details["parameter_delta"] = delta
        details.setdefault(
            "gradient_sync_mode",
            "standard_ddp_replay_forward" if self.distributed.enabled else "single_process",
        )
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
            "parameter_delta": delta,
            "gpu_peak_memory_bytes": int(self.torch.cuda.max_memory_allocated()),
        }
        per_rank = self.distributed.gather_objects(rank_summary)
        details["distributed"] = {
            "enabled": self.distributed.enabled,
            "backend": self.distributed.backend or None,
            "world_size": self.distributed.world_size,
            "per_rank": per_rank,
        }
        if self.distributed.is_main:
            self._save(checkpoint_out, details)
            self.output_metrics = {
                "parameter_sync_before": self._parameter_sync_before,
                "parameter_sync_after": sync_after,
                "parameter_delta": delta,
                "distributed": details["distributed"],
            }
        self.distributed.barrier()
        status = "updated" if steps else "skipped"
        return [{"status": status, **details, "checkpoint": checkpoint_out}] if self.distributed.is_main else []

    @staticmethod
    def _clipped_loss(torch, current, old, ref, advantage, clip: float):
        ratio = torch.exp(torch.clamp(current - old, -10, 10))
        policy = -torch.minimum(ratio * advantage, torch.clamp(ratio, 1 - clip, 1 + clip) * advantage).mean()
        log_ratio = ref - current
        kl = (torch.exp(torch.clamp(log_ratio, -10, 10)) - log_ratio - 1).mean()
        return policy, kl

    def grpo_update(self, rows: list[dict[str, Any]], checkpoint_out: str, phase: str = "") -> list[dict[str, Any]]:
        training = self._training_config(phase)
        usable = [row for row in rows if any(not item.get("skip_update", True) for item in row.get("candidates", []))]
        if not usable:
            before = self.parameter_signature()
            return self._finish_update(checkpoint_out, {
                "update": "grpo", "steps": 0, "parameter_before": before, "parameter_after": before,
                "reason": "no non-flat valid groups",
            }, sample_ids=[], steps=0)
        self._ensure_ddp()
        self._begin_update_audit()
        before = self.parameter_signature()
        optimizer = self._optimizer(training)
        clip = float(training.get("clip_range", 0.2))
        kl_beta = float(training.get("kl_beta", 0.02))
        sub_weight = float(training.get("sub_talker_weight", self.cfg.get("training", {}).get("sub_talker_weight", 0.3)))
        grad_clip = float(training.get("grad_clip", 1.0))
        logs, grad_norms = [], []
        self.policy.set_adapter("default")
        # Keep rollout and replay dropout behavior identical while retaining grads.
        self.policy.eval()
        metric_context = MetricLogger(Path(checkpoint_out) / "training_metrics") if self.distributed.is_main else contextlib.nullcontext(None)
        sample_ids = []
        epoch_mean_kls: list[float] = []
        group_use_counts: dict[str, int] = {}
        max_epochs = int(training.get("epochs", 1))
        if max_epochs != 1:
            raise ValueError("DualISL-Train TTS GRPO must run exactly one epoch")
        self._heartbeat_frames = int(training.get("heartbeat_frames", 32))
        if self._heartbeat_frames < 1:
            raise ValueError("TTS GRPO heartbeat_frames must be >= 1")
        progress = ProgressLogger(
            Path(checkpoint_out) / "training_progress", self.distributed.rank,
        )
        with metric_context as metrics:
          for epoch in range(max_epochs):
            epoch_kls: list[float] = []
            schedule = self._grpo_epoch_schedule(usable, training, epoch)
            for _, _, group, padded in schedule:
                step = len(logs)
                step_started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                train_candidates = [
                    candidate for candidate in group["candidates"] if not candidate.get("skip_update")
                ]
                if not train_candidates:
                    raise RuntimeError(f"Usable Qwen GRPO group {group['id']} produced no candidate loss")
                if self.distributed.enabled:
                    active = self.distributed.gather_objects(not padded)
                    active_count = sum(bool(value) for value in active)
                else:
                    active_count = 1
                backward_scale = candidate_backward_scale(
                    len(train_candidates), self.distributed.world_size if self.distributed.enabled else 1,
                    active_count, padded,
                )
                group_codec_frames = self._grpo_group_cost(group)
                progress.log(
                    "step_start", step=step, epoch=epoch, sample_id=str(group["id"]),
                    padded=padded, candidates=len(train_candidates), codec_frames=group_codec_frames,
                    **self._cuda_memory_snapshot(),
                )
                local_loss_value = 0.0
                for candidate_index, candidate in enumerate(train_candidates):
                    candidate_started = time.perf_counter()
                    candidate_id = str(candidate["candidate_id"])
                    codec_frames = len(candidate.get("codec_codes") or [])
                    progress.log(
                        "candidate_start", step=step, epoch=epoch, sample_id=str(group["id"]),
                        candidate_id=candidate_id, candidate_index=candidate_index,
                        candidates=len(train_candidates), codec_frames=codec_frames, padded=padded,
                        **self._cuda_memory_snapshot(),
                    )

                    def frame_progress(processed_frames, total_frames, *, _candidate_id=candidate_id, _index=candidate_index):
                        progress.log(
                            "frame_progress", step=step, epoch=epoch, sample_id=str(group["id"]),
                            candidate_id=_candidate_id, candidate_index=_index,
                            processed_frames=processed_frames, total_frames=total_frames, padded=padded,
                            **self._cuda_memory_snapshot(),
                        )

                    synchronize = candidate_index + 1 == len(train_candidates)
                    sync_context = (
                        contextlib.nullcontext()
                        if not self.distributed.enabled or synchronize
                        else self.ddp_replay.no_sync()
                    )
                    with sync_context:
                        current_main, current_sub = self.incremental_candidate_logprobs(
                            candidate, progress_callback=frame_progress,
                        )
                        old_main = self.torch.tensor(candidate["old_main_logprobs"], device=self.device)
                        old_sub = self.torch.tensor(candidate["old_sub_logprobs"], device=self.device)
                        ref_main = self.torch.tensor(candidate["ref_main_logprobs"], device=self.device)
                        ref_sub = self.torch.tensor(candidate["ref_sub_logprobs"], device=self.device)
                        if (
                            current_main.shape != old_main.shape or current_sub.shape != old_sub.shape
                            or current_main.shape != ref_main.shape or current_sub.shape != ref_sub.shape
                        ):
                            raise ValueError(f"Trajectory log-prob shape mismatch for {candidate_id}")
                        advantage = self.torch.tensor(float(candidate["advantage"]), device=self.device)
                        main_policy, main_kl = self._clipped_loss(
                            self.torch, current_main, old_main, ref_main, advantage, clip,
                        )
                        sub_policy, sub_kl = self._clipped_loss(
                            self.torch, current_sub, old_sub, ref_sub, advantage, clip,
                        )
                        candidate_loss = (
                            main_policy + sub_weight * sub_policy
                            + kl_beta * (main_kl + sub_weight * sub_kl)
                        )
                        candidate_kl = float((main_kl + sub_weight * sub_kl).detach().cpu())
                        candidate_loss_value = float(candidate_loss.detach().cpu())
                        progress.log(
                            "candidate_loss_ready", step=step, epoch=epoch,
                            sample_id=str(group["id"]), candidate_id=candidate_id,
                            candidate_index=candidate_index, candidates=len(train_candidates),
                            codec_frames=codec_frames, padded=padded,
                            loss=candidate_loss_value, kl=candidate_kl,
                            **self._cuda_memory_snapshot(),
                        )
                        (candidate_loss * backward_scale).backward()
                    local_loss_value += candidate_loss_value * backward_scale
                    epoch_kls.append(candidate_kl)
                    if self.torch.cuda.is_available():
                        self.torch.cuda.synchronize(self.device)
                    progress.log(
                        "candidate_complete", step=step, epoch=epoch, sample_id=str(group["id"]),
                        candidate_id=candidate_id, candidate_index=candidate_index,
                        candidates=len(train_candidates), codec_frames=codec_frames, padded=padded,
                        loss=candidate_loss_value, kl=candidate_kl,
                        elapsed_seconds=time.perf_counter() - candidate_started,
                        **self._cuda_memory_snapshot(),
                    )
                    del (
                        current_main, current_sub, old_main, old_sub, ref_main, ref_sub, advantage,
                        main_policy, main_kl, sub_policy, sub_kl, candidate_loss,
                    )
                grad_norm = float(self.torch.nn.utils.clip_grad_norm_(self.trainable_parameters, grad_clip).detach().cpu())
                grad_norms.append(grad_norm)
                optimizer.step()
                value = self.distributed.mean(local_loss_value)
                logs.append(value)
                if not padded:
                    sample_id = str(group["id"])
                    sample_ids.append(sample_id)
                    group_use_counts[sample_id] = group_use_counts.get(sample_id, 0) + 1
                progress.log(
                    "step_complete", step=step, epoch=epoch, sample_id=str(group["id"]),
                    padded=padded, candidates=len(train_candidates), codec_frames=group_codec_frames,
                    loss=value, grad_norm=grad_norm,
                    elapsed_seconds=time.perf_counter() - step_started,
                    **self._cuda_memory_snapshot(),
                )
                if metrics is not None:
                    metrics.log(
                        step, loss=value, grad_norm=grad_norm, update="grpo", epoch=epoch,
                        sample_id=group["id"], candidates=len(train_candidates), padded=padded,
                        codec_frames=group_codec_frames,
                        elapsed_seconds=time.perf_counter() - step_started,
                    )
            epoch_mean = self.distributed.mean(sum(epoch_kls) / max(len(epoch_kls), 1))
            epoch_mean_kls.append(epoch_mean)
        after = self.parameter_signature()
        return self._finish_update(checkpoint_out, {
            "update": "grpo", "steps": len(logs), "mean_loss": sum(logs) / len(logs),
            "max_grad_norm": max(grad_norms), "parameter_before": before, "parameter_after": after,
            "epochs_completed": len(epoch_mean_kls), "epoch_mean_kl": epoch_mean_kls,
            "group_use_counts": group_use_counts,
            "gradient_sync_mode": (
                "standard_ddp_candidate_accumulation_last_candidate_sync"
                if self.distributed.enabled else "single_process_candidate_accumulation"
            ),
            "grpo_schedule": str(training.get("schedule", "length_bucketed")),
            "heartbeat_frames": int(training.get("heartbeat_frames", 32)),
        }, sample_ids=sample_ids, steps=len(logs))

    def sft_update(self, rows: list[dict[str, Any]], checkpoint_out: str, phase: str = "") -> list[dict[str, Any]]:
        for row in rows:
            if row.get("target_origin") != SOURCE_DOMAIN_TARGET:
                raise ValueError(f"Refusing TTS SFT row {row.get('id')}: target_origin must be source_domain")
            if row.get("audio_path") != row.get("source_audio_path"):
                raise ValueError(f"Refusing TTS SFT row {row.get('id')}: target audio is not the source-domain audio")
        training = self._training_config(phase)
        if not rows:
            before = self.parameter_signature()
            return self._finish_update(checkpoint_out, {
                "update": "sft", "steps": 0, "parameter_before": before, "parameter_after": before,
                "reason": "empty SFT manifest",
            }, sample_ids=[], steps=0)
        self._ensure_ddp()
        self._begin_update_audit()
        before = self.parameter_signature()
        optimizer = self._optimizer(training)
        sub_weight = float(training.get("sub_talker_weight", self.cfg.get("training", {}).get("sub_talker_weight", 0.3)))
        grad_clip = float(training.get("grad_clip", 1.0))
        logs, grad_norms = [], []
        self.policy.set_adapter("default")
        self.policy.train()
        schedule = (
            distributed_schedule(rows, training, self.seed, self.distributed.rank, self.distributed.world_size)
            if self.distributed.enabled else
            [(step, epoch, row, False) for step, (epoch, row) in enumerate(self._schedule(rows, training))]
        )
        metric_context = MetricLogger(Path(checkpoint_out) / "training_metrics") if self.distributed.is_main else contextlib.nullcontext(None)
        sample_ids = []
        with metric_context as metrics:
            for step, epoch, row, padded in schedule:
                optimizer.zero_grad(set_to_none=True)
                main, sub = self.trajectory_logprobs(row["request"], self._load_codes(row), grad=True)
                loss = -main.mean() - sub_weight * sub.mean()
                if self.distributed.enabled:
                    active = self.distributed.gather_objects(not padded)
                    active_count = sum(bool(value) for value in active)
                    loss = loss * (0.0 if padded else self.distributed.world_size / max(active_count, 1))
                loss.backward()
                grad_norm = float(self.torch.nn.utils.clip_grad_norm_(self.trainable_parameters, grad_clip).detach().cpu())
                grad_norms.append(grad_norm)
                optimizer.step()
                value = self.distributed.mean(float(loss.detach().cpu()))
                logs.append(value)
                if not padded:
                    sample_ids.append(str(row["id"]))
                if metrics is not None:
                    metrics.log(step, loss=value, main_nll=float(-main.mean().detach().cpu()), sub_nll=float(-sub.mean().detach().cpu()), grad_norm=grad_norm, update="sft", epoch=epoch, sample_id=row["id"], padded=padded)
        after = self.parameter_signature()
        return self._finish_update(checkpoint_out, {
            "update": "sft", "steps": len(logs), "mean_loss": sum(logs) / len(logs),
            "max_grad_norm": max(grad_norms), "parameter_before": before, "parameter_after": after,
        }, sample_ids=sample_ids, steps=len(logs))


def main() -> None:
    actions = ["generate-audio", "prepare-codecs", "rollout", "score-reconstruction", "grpo-update", "sft-update", "preflight"]
    parser = worker_parser("Qwen3-TTS VoiceDesign dual-learning worker", actions)
    args = parser.parse_args()
    config, rows = load_job(args)
    distributed = DistributedContext.initialize(config)
    try:
        worker = QwenVoiceDesignWorker(config, args.checkpoint_in, args.target_checkpoint, distributed)
        extra_metrics: dict[str, Any] = {}
        inference_action = False
        if args.action == "prepare-codecs":
            output = worker.prepare_codecs(rows)
        elif args.action == "generate-audio":
            inference_action = distributed.enabled
            function = lambda values: worker.generate_audio(values, args.output)
            if distributed.enabled:
                from .tts_batch import synthesis_assignment
                owners, placement = synthesis_assignment(
                    rows, int(worker.cfg.get("generation", {}).get("synthesis_batch_size", 4)),
                    distributed.world_size,
                )
                output, extra_metrics = run_sharded_inference(
                    rows, args.output, distributed, function, owners=owners,
                )
                extra_metrics["synthesis_placement"] = placement
            else:
                output = function(rows)
        elif args.action == "rollout":
            inference_action = distributed.enabled
            function = lambda values: worker.rollout(values, args.output)
            if distributed.enabled:
                from ..rollout_balance import assignment_plan
                owners, placement = assignment_plan(
                    rows, distributed.world_size,
                    worker.cfg.get("generation", {}).get("rollout_schedule", "round_robin"),
                )
                output, extra_metrics = run_sharded_inference(rows, args.output, distributed, function, owners=owners)
                extra_metrics["rollout_placement"] = placement
            else:
                output = function(rows)
            replay_metrics = {"rank": distributed.rank, **worker.output_metrics["reference_replay"]}
            replay_metrics["batch_fallback_groups"] = worker.output_metrics.get("batch_fallback_groups", 0)
            extra_metrics["reference_replay_per_rank"] = distributed.gather_objects(replay_metrics)
        elif args.action == "score-reconstruction":
            inference_action = distributed.enabled
            if distributed.enabled:
                output, extra_metrics = run_sharded_inference(rows, args.output, distributed, worker.score_reconstruction)
            else:
                output = worker.score_reconstruction(rows)
        elif args.action == "grpo-update":
            output = worker.grpo_update(rows, args.checkpoint_out, args.training_phase)
            extra_metrics = worker.output_metrics
        elif args.action == "sft-update":
            output = worker.sft_update(rows, args.checkpoint_out, args.training_phase)
            extra_metrics = worker.output_metrics
        else:
            output = [{
                "status": "ok", "tts_model_type": str(worker.core.tts_model_type),
                "num_codebooks": int(worker.core.config.talker_config.num_code_groups),
                "trainable_parameters": sum(parameter.numel() for parameter in worker.trainable_parameters),
                "has_reference_adapter": worker.has_reference_adapter,
                "adapter_audits": worker.adapter_audits,
            }]
        extra_metrics["adapter_audits"] = worker.adapter_audits
        if not distributed.enabled or (distributed.is_main and not inference_action):
            write_output(args.output, output, extra_metrics)
        elif distributed.is_main:
            atomic_json(str(args.output) + ".metrics.json", extra_metrics)
    finally:
        distributed.close()


if __name__ == "__main__":
    main()
