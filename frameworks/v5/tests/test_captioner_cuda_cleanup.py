from __future__ import annotations

from dual_isl_train.workers.qwen3_captioner import _close_captioner_worker


class _FakeCuda:
    def __init__(self, events: list[str]):
        self.events = events
        self.allocated = 75_000
        self.reserved = 79_000

    @staticmethod
    def is_available() -> bool:
        return True

    def memory_allocated(self, _device: object) -> int:
        return self.allocated

    def memory_reserved(self, _device: object) -> int:
        return self.reserved

    @staticmethod
    def max_memory_allocated(_device: object) -> int:
        return 80_000

    def empty_cache(self) -> None:
        self.events.append("empty_cache")
        self.allocated = 10_000
        self.reserved = 12_000


class _FakeTorch:
    def __init__(self, events: list[str]):
        self.cuda = _FakeCuda(events)


class _FakeWorker:
    def __init__(self, events: list[str]):
        self.torch = _FakeTorch(events)
        self.device = "cuda:1"
        self.model = object()
        self.ddp_model = object()


class _FakeDistributed:
    def __init__(self, worker: _FakeWorker, events: list[str]):
        self.worker = worker
        self.events = events

    def close(self) -> None:
        assert self.worker.model is None
        assert self.worker.ddp_model is None
        self.events.append("close_process_group")


def test_captioner_releases_cuda_memory_before_process_group_close(capsys):
    events: list[str] = []
    worker = _FakeWorker(events)
    distributed = _FakeDistributed(worker, events)

    telemetry = _close_captioner_worker(worker, distributed)  # type: ignore[arg-type]

    assert events == ["empty_cache", "close_process_group"]
    assert telemetry == {
        "model_reference_cleared": True,
        "empty_cache_called": True,
        "worker_initialized": True,
        "device": "cuda:1",
        "before": {
            "allocated_bytes": 75_000,
            "reserved_bytes": 79_000,
            "peak_allocated_bytes": 80_000,
        },
        "after": {
            "allocated_bytes": 10_000,
            "reserved_bytes": 12_000,
            "peak_allocated_bytes": 80_000,
        },
    }
    assert "CAPTIONER_CUDA_CLEANUP" in capsys.readouterr().out


def test_captioner_close_handles_worker_initialization_failure(capsys):
    events: list[str] = []

    class Distributed:
        def close(self) -> None:
            events.append("close_process_group")

    telemetry = _close_captioner_worker(None, Distributed())  # type: ignore[arg-type]

    assert telemetry == {
        "model_reference_cleared": False,
        "empty_cache_called": False,
        "worker_initialized": False,
    }
    assert events == ["close_process_group"]
    assert "CAPTIONER_CUDA_CLEANUP" in capsys.readouterr().out
