"""Run the production SFT/backward/all-reduce path on tiny CPU models using Gloo."""
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dual_isl_train.distributed import DistributedContext
from dual_isl_train.workers.qwen3_captioner import Qwen3CaptionerWorker
from dual_isl_train.workers.mock import mock_caption


class CPUContext(DistributedContext):
    @property
    def device(self):
        return "cpu"


def tiny_worker(context, broken_rank=None):
    import torch

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.audio = torch.nn.ModuleDict({"default": torch.nn.Linear(1, 1, bias=False)})
            self.text = torch.nn.ModuleDict({"default": torch.nn.Linear(1, 1, bias=False)})
            with torch.no_grad():
                self.audio["default"].weight.fill_(0.25)
                self.text["default"].weight.fill_(-0.1)
            self.config = SimpleNamespace(use_cache=True)

        def set_adapter(self, name):
            assert name == "default"

        def forward(self, input_ids, feature, labels):
            assert labels.tolist() == [[-100, -100, 3, 4]]
            audio = self.audio["default"](feature)
            if context.rank == broken_rank:
                audio = audio.detach()
            prediction = audio + self.text["default"](torch.ones_like(feature))
            return SimpleNamespace(loss=((prediction - 1) ** 2).mean())

    class Worker(Qwen3CaptionerWorker):
        def __init__(self):
            self.torch, self.distributed, self.seed = torch, context, 42
            self.model = Model()

        def _training_config(self, phase):
            return {"epochs": 1, "shuffle": False, "grad_clip": 100.0}

        def _optimizer(self, training):
            return torch.optim.SGD(self.trainable_parameters, lr=0.01)

        def parameter_signature(self):
            return sum(p.detach().double().sum().item() for p in self.trainable_parameters)

        def freeze_batchnorm(self):
            pass

        def _inputs(self, audio_path, prompt, completion=None):
            return {"input_ids": torch.tensor([[1, 2, 3, 4] if completion else [1, 2]]),
                    "feature": torch.tensor([[float(audio_path)]])}

        def _finish_update(self, checkpoint_out, details, *, sample_ids, steps):
            return [{**details, "sample_ids": sample_ids,
                     "weights": [p.detach().item() for p in self.trainable_parameters]}]

    return Worker()


def run_rank(rank, world, row_count, rendezvous, output, broken_rank):
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(seconds=30))
    try:
        worker = tiny_worker(CPUContext(True, rank, rank, world, "gloo"), broken_rank)
        rows = [{"id": str(i), "audio_path": str(i+1), "caption": mock_caption(0),
                 "target_origin": "source_domain"} for i in range(row_count)]
        try:
            result = worker.sft_update(rows, str(Path(output) / "checkpoint"))[0]
        except RuntimeError as exc:
            result = {"error": str(exc)}
        (Path(output) / f"rank_{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world,row_count,broken_rank", [(8, 4, None), (8, 10, None), (8, 4, 0), (1, 1, None)])
def test_sft_real_collectives_handle_padding_and_connectivity(tmp_path, world, row_count, broken_rank):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    # CPU/Gloo only: exercise real sft_update, gradient scaling and reduction without GPUs/models.
    torch.multiprocessing.start_processes(run_rank,
        args=(world, row_count, "file://" + str(tmp_path / "rendezvous"), str(tmp_path), broken_rank),
        nprocs=world, join=True, start_method="fork")
    results = [json.loads((tmp_path / f"rank_{rank}.json").read_text()) for rank in range(world)]
    if broken_rank is not None:
        assert all("gradient connectivity failed" in r.get("error", "") for r in results)
        return
    assert all("error" not in r for r in results), results
    # Independent single-process reference: each synchronous step averages only real rows.
    weights = torch.tensor([0.25, -0.1], requires_grad=True)
    optimizer = torch.optim.SGD([weights], lr=0.01)
    for offset in range(0, row_count, world):
        x = torch.arange(offset + 1, min(offset + world, row_count) + 1, dtype=torch.float32)
        optimizer.zero_grad()
        ((weights[0] * x + weights[1] - 1) ** 2).mean().backward()
        torch.nn.utils.clip_grad_norm_([weights], 100.0)
        optimizer.step()
    for r in results:
        assert r["weights"] == pytest.approx(weights.detach().tolist(), abs=1e-7)
        assert r["parameter_before"] != r["parameter_after"]
        assert all(value > 0 for value in r["gradient_family_nonzero"].values())
        audit = r["gradient_family_nonzero_per_rank"]
        assert sum(item["active_samples"] for item in audit) == row_count
        assert sum(item["active_samples"] == 0 for item in audit) == max(world - row_count, 0)
    assert sorted(sid for r in results for sid in r["sample_ids"]) == sorted(map(str, range(row_count)))


def test_sft_audit_rejects_empty_global_activity():
    worker = Qwen3CaptionerWorker.__new__(Qwen3CaptionerWorker)
    worker.distributed = DistributedContext()
    with pytest.raises(RuntimeError, match="gradient connectivity failed"):
        worker._audit_sft_gradient_families({"audio": 0, "text": 0}, 0)
