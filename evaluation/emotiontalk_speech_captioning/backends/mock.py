from __future__ import annotations

from .base import CaptionBackend, CaptionRequest


class MockBackend(CaptionBackend):
    def __init__(self, **_: object) -> None:
        pass

    def generate(self, request: CaptionRequest) -> str:
        return f"Mock {request.task} caption for {request.id}."

    def identity(self) -> dict[str, object]:
        return {"backend": "mock", "version": 1}
