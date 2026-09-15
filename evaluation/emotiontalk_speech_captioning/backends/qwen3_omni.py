from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from model_adapter_utils import describe_adapter, load_peft_adapter

from .base import CaptionBackend, CaptionRequest


DEFAULT_MODEL = "/data/L202500147/Caption/models/Qwen3-Omni-30B-A3B-Captioner"


class Qwen3OmniBackend(CaptionBackend):
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        max_new_tokens: int = 128,
        adapter_dir: str | None = None,
        attn_backend: str | None = None,
        **_: object,
    ) -> None:
        from qwen_omni_utils import process_mm_info
        from transformers import AutoConfig, Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        self.model_path = str(Path(model_path).resolve())
        self.max_new_tokens = max_new_tokens
        self.attn_backend = attn_backend
        self._process_mm_info = process_mm_info
        config = AutoConfig.from_pretrained(self.model_path, local_files_only=True, trust_remote_code=True)
        if not hasattr(config, "initializer_range"):
            config.initializer_range = 0.02
        for name in ("thinker_config", "talker_config", "code2wav_config"):
            child = getattr(config, name, None)
            if child is not None and not hasattr(child, "initializer_range"):
                child.initializer_range = 0.02
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=True
        )
        model_kwargs: dict[str, Any] = {
            "config": config,
            "dtype": torch.bfloat16,
            "device_map": {"": "cuda:0"},
            "low_cpu_mem_usage": True,
            "local_files_only": True,
            "trust_remote_code": True,
        }
        if attn_backend is not None:
            model_kwargs["attn_implementation"] = attn_backend
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            self.model_path, **model_kwargs
        )
        descriptor = describe_adapter(adapter_dir, self.model_path)
        self.adapter = None
        if descriptor is not None:
            self.model.thinker, self.adapter = load_peft_adapter(
                self.model.thinker, descriptor
            )
        self.model.requires_grad_(False)
        self.model.eval()

    def generate(self, request: CaptionRequest) -> str:
        return self.generate_batch([request])[0]

    def generate_batch(self, requests: list[CaptionRequest]) -> list[str]:
        conversations = [[{
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(Path(request.audio_path).resolve())},
                {"type": "text", "text": request.prompt},
            ],
        }] for request in requests]
        text = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )
        audios, _, _ = self._process_mm_info(
            conversations, use_audio_in_video=False
        )
        inputs = self.processor(
            text=text,
            audio=audios,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        for key in ("input_features", "pixel_values", "pixel_values_videos"):
            if key in inputs:
                inputs[key] = inputs[key].to(dtype=dtype)
        with torch.inference_mode():
            text_ids, _ = self.model.generate(
                **inputs,
                return_audio=False,
                thinker_max_new_tokens=self.max_new_tokens,
                thinker_do_sample=False,
                thinker_return_dict_in_generate=True,
            )
        sequences = text_ids.sequences if hasattr(text_ids, "sequences") else text_ids
        prompt_length = inputs["input_ids"].shape[1]
        output_ids = sequences[:, prompt_length:]
        predictions = [value.strip() for value in self.processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )]
        if len(predictions) != len(requests):
            raise RuntimeError(
                f"Qwen3-Omni output count mismatch: {len(predictions)} != {len(requests)}"
            )
        return predictions

    def identity(self) -> dict[str, object]:
        config = Path(self.model_path) / "config.json"
        return {
            "backend": "qwen3_omni",
            "model_path": self.model_path,
            "model_config_mtime_ns": config.stat().st_mtime_ns if config.exists() else None,
            "dtype": "bfloat16",
            "deterministic": True,
            "max_new_tokens": self.max_new_tokens,
            "attn_backend": self.attn_backend,
            "adapter": self.adapter,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
