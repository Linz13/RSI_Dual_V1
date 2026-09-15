from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CaptionRequest:
    id: str
    task: str
    audio_path: str
    prompt: str


class CaptionBackend(ABC):
    @abstractmethod
    def generate(self, request: CaptionRequest) -> str:
        raise NotImplementedError

    def generate_batch(self, requests: list[CaptionRequest]) -> list[str]:
        """Generate a batch, with a safe scalar fallback for simple backends."""
        return [self.generate(request) for request in requests]

    @abstractmethod
    def identity(self) -> dict[str, Any]:
        raise NotImplementedError
