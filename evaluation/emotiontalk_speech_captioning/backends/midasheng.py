from __future__ import annotations

import os
from pathlib import Path

import torch

from model_adapter_utils import describe_adapter, load_peft_adapter

from .base import CaptionBackend, CaptionRequest


DEFAULT_MODEL = "/data/L202500147/Caption/models/MiDashengLM-7B-1021-BF16"


class MiDashengBackend(CaptionBackend):
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        max_new_tokens: int = 128,
        adapter_dir: str | None = None,
        attn_backend: str | None = "sdpa",
        **_: object,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("MiDasheng evaluation requires a visible CUDA GPU")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(f"BF16 is not supported by {torch.cuda.get_device_name(0)}")

        self.model_path = str(Path(model_path).resolve())
        self.max_new_tokens = max_new_tokens
        self.attn_backend = attn_backend
        model_kwargs: dict[str, object] = {
            "trust_remote_code": True,
            "local_files_only": True,
            "dtype": torch.bfloat16,
            "device_map": {"": "cuda:0"},
            "low_cpu_mem_usage": True,
        }
        if attn_backend is not None:
            model_kwargs["attn_implementation"] = attn_backend
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, **model_kwargs
        )
        descriptor = describe_adapter(adapter_dir, self.model_path)
        self.adapter = None
        if descriptor is not None:
            self.model, self.adapter = load_peft_adapter(self.model, descriptor)
        self.model.requires_grad_(False)
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()

        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True, local_files_only=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, local_files_only=True
        )
        configured_eos = self.model.generation_config.eos_token_id
        if configured_eos is None:
            configured_eos = self.tokenizer.eos_token_id
        self.eos_token_ids = (
            [int(value) for value in configured_eos]
            if isinstance(configured_eos, (list, tuple))
            else [int(configured_eos)]
        )
        self.pad_token_id = int(
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.eos_token_ids[0]
        )

    def generate(self, request: CaptionRequest) -> str:
        return self.generate_batch([request])[0]

    def generate_batch(self, requests: list[CaptionRequest]) -> list[str]:
        conversations = [[{
            "role": "user",
            "content": [
                {"type": "text", "text": request.prompt},
                {"type": "audio", "path": str(Path(request.audio_path).resolve())},
            ],
        }] for request in requests]
        values = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            add_special_tokens=True,
            return_dict=True,
            padding=True,
        )
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        prepared: dict[str, object] = {}
        for key, value in values.items():
            if hasattr(value, "to"):
                value = value.to(device)
                if getattr(value, "is_floating_point", lambda: False)():
                    value = value.to(dtype=dtype)
            prepared[key] = value

        with torch.inference_mode():
            output = self.model.generate(
                **prepared,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=self.eos_token_ids,
                pad_token_id=self.pad_token_id,
                return_dict_in_generate=True,
            )
        sequences = output.sequences.detach().long()
        prompt_ids = prepared["input_ids"].detach().long()
        if (
            sequences.shape[1] >= prompt_ids.shape[1]
            and torch.equal(sequences[:, : prompt_ids.shape[1]], prompt_ids)
        ):
            sequences = sequences[:, prompt_ids.shape[1]:]
        predictions = [value.strip() for value in self.tokenizer.batch_decode(
            sequences.cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )]
        if len(predictions) != len(requests):
            raise RuntimeError(
                f"MiDasheng output count mismatch: {len(predictions)} != {len(requests)}"
            )
        return predictions

    def identity(self) -> dict[str, object]:
        config = Path(self.model_path) / "config.json"
        return {
            "backend": "midasheng",
            "model_path": self.model_path,
            "model_config_mtime_ns": config.stat().st_mtime_ns if config.exists() else None,
            "dtype": "bfloat16",
            "deterministic": True,
            "max_new_tokens": self.max_new_tokens,
            "attn_backend": self.attn_backend,
            "adapter": self.adapter,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
