from __future__ import annotations

from qwen25_caption_backend import Qwen25Captioner
from .base import CaptionBackend


class Qwen25OmniBackend(CaptionBackend):
    def __init__(self, max_new_tokens=128, **kwargs):
        self.captioner = Qwen25Captioner(**kwargs)
        self.max_new_tokens = max_new_tokens

    def generate(self, request):
        return self.generate_batch([request])[0]

    def generate_batch(self, requests):
        return self.captioner.generate_batch(
            [request.audio_path for request in requests],
            [request.prompt for request in requests], self.max_new_tokens,
        )

    def identity(self):
        return {"backend": "qwen25", "model_path": self.captioner.model_path,
                "adapter": self.captioner.adapter, "thinker_only": True,
                "attn_backend": self.captioner.attn_backend,
                "max_new_tokens": self.max_new_tokens, "deterministic": True}
