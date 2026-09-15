import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dual_isl_train.workers.tts_batch import BatchedTTSMixin, synthesis_assignment
from dual_isl_train.io import stable_hash


def jobs(count):
    return [{"id": f"sample::{i}", "candidate_id": f"sample::{i}",
             "request": {"text": "a" * ((i * 17) % 31 + 1)}, "generation_seed": 80 + i}
            for i in range(count)]


def batches(rows, size=4):
    ordered = sorted(rows, key=lambda r: (len(r["request"]["text"]), r["id"]))
    return [ordered[i:i + size] for i in range(0, len(ordered), size)]


@pytest.mark.parametrize("world", [1, 4, 8])
@pytest.mark.parametrize("count", [0, 1, 3, 4, 5, 31, 32, 33, 457])
def test_distributed_batches_keep_original_members_order_and_rng(world, count):
    rows = jobs(count)
    owners, plan = synthesis_assignment(rows, 4, world)
    expected = {tuple(r["id"] for r in batch): stable_hash([r["generation_seed"] for r in batch])
                for batch in batches(rows)}
    actual = {}
    for rank in range(world):
        for batch in batches([r for r, owner in zip(rows, owners) if owner == rank]):
            ids = tuple(r["id"] for r in batch)
            assert ids not in actual
            actual[ids] = stable_hash([r["generation_seed"] for r in batch])
    assert actual == expected
    assert plan["batches"] == len(expected)
    assert synthesis_assignment(rows, 4, world) == (owners, plan)


def test_unique_ids_and_positive_sizes_required():
    with pytest.raises(ValueError, match="unique"):
        synthesis_assignment(jobs(1) * 2, 4, 8)
    for size, world in [(0, 8), (4, 0)]:
        with pytest.raises(ValueError, match="positive"):
            synthesis_assignment(jobs(1), size, world)


def test_rank_journals_and_waveform_mapping_with_empty_ranks(tmp_path):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    rows = jobs(5)
    owners, _ = synthesis_assignment(rows, 4, 8)
    class Worker(BatchedTTSMixin):
        def _v5_generate(self, batch, capture=False):
            assert not capture
            return ([r["generation_seed"] for r in batch], None), 123
    outputs = []
    for rank in range(8):
        worker = Worker()
        worker.cfg = {"generation": {"synthesis_batch_size": 4}}
        worker.distributed = SimpleNamespace(rank=rank)
        worker.policy = SimpleNamespace(eval=lambda: None)
        worker.torch = SimpleNamespace(no_grad=contextlib.nullcontext,
                                       cuda=SimpleNamespace(OutOfMemoryError=RuntimeError))
        worker.core = SimpleNamespace(speech_tokenizer=SimpleNamespace(
            decode=lambda values: ([np.full(50, c["audio_codes"] / 100, dtype=np.float32) for c in values], 24000)))
        local = [r for r, owner in zip(rows, owners) if owner == rank]
        out = worker.generate_audio(local, str(tmp_path / "output.jsonl"))
        assert [r["id"] for r in out] == [r["id"] for r in local]
        journal = tmp_path / f"rank_{rank:03d}.ready.jsonl"
        assert journal.exists() == bool(local)
        if local:
            assert {json.loads(l)["id"] for l in journal.read_text().splitlines()} == {r["id"] for r in local}
        outputs.extend(out)
    assert {r["id"] for r in outputs} == {r["id"] for r in rows}
    for row in outputs:
        wav, _ = sf.read(row["audio_path"])
        assert wav.mean() == pytest.approx(row["generation_seed"] / 100, abs=0.0001)


@pytest.mark.parametrize("action,distributed,expected", [
    ("generate-audio", True, True), ("rollout", True, True),
    ("generate-audio", False, False), ("prepare-codecs", True, False), ("preflight", True, False),
])
def test_worker_launcher_dispatches_synthesis_to_torchrun(tmp_path, monkeypatch, action, distributed, expected):
    from dual_isl_train.stages import run_worker
    calls = []
    monkeypatch.setattr("subprocess.run", lambda command, **kwargs: calls.append(command))
    run_worker(python="/fixture/python", module="fixture.worker", action=action,
               config_path="config.yaml", input_path=tmp_path / "in.jsonl", output_path=tmp_path / "out.jsonl",
               checkpoint_in=None, target_checkpoint=None, checkpoint_out=None,
               log_path=tmp_path / "log", project_root=tmp_path,
               distributed={"enabled": distributed, "world_size": 8})
    assert ("torch.distributed.run" in calls[0]) == expected
    if expected:
        assert "--nproc-per-node=8" in calls[0]


def test_serial_rollout_setting_bypasses_batched_attempts():
    from dual_isl_train.workers.qwen_voice_design import QwenVoiceDesignWorker
    calls = []
    worker = SimpleNamespace(cfg={"generation": {"rollout_batch_size": 1}},
                             _serial_rollout=lambda *args: calls.append("serial") or ["ok"],
                             rollout_v5=lambda *args: pytest.fail("batch retry should not run"))
    assert QwenVoiceDesignWorker.rollout(worker, [], "out") == ["ok"]
    assert calls == ["serial"]


def test_worker_stage_makes_new_checkpoint_shared_before_resume(tmp_path, monkeypatch):
    from dual_isl_train.stages import run_worker
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "adapter_model.safetensors"
    weights.write_bytes(b"fixture")
    weights.chmod(0o600)
    monkeypatch.setenv("DUALISL_SHARED_WRITABLE", "1")
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: None)
    run_worker(python="/fixture/python", module="fixture.worker", action="sft-update",
               config_path="config.yaml", input_path=tmp_path / "in.jsonl", output_path=tmp_path / "out.jsonl",
               checkpoint_in=None, target_checkpoint=None, checkpoint_out=checkpoint,
               log_path=tmp_path / "log", project_root=tmp_path)
    assert weights.stat().st_mode & 0o666 == 0o666
    assert weights.read_bytes() == b"fixture"


def _gloo_synthesis_rank(rank, world, directory, count):
    """Real collectives and shard merging; only GPU telemetry/model calls are stubbed."""
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    from dual_isl_train.distributed import DistributedContext, run_sharded_inference
    dist.init_process_group("gloo", init_method="file://" + str(Path(directory) / "rendezvous"),
                            rank=rank, world_size=world, timeout=timedelta(seconds=45))
    torch.cuda.synchronize = lambda: None
    torch.cuda.current_device = lambda: 0
    torch.cuda.get_device_name = lambda device: "CPU test double"
    torch.cuda.max_memory_allocated = lambda: 0
    try:
        rows = jobs(count)
        owners, _ = synthesis_assignment(rows, 4, world)
        def generate(local):
            index = {}
            for batch in batches(local):
                seed = stable_hash([r["generation_seed"] for r in batch])
                for row in batch:
                    path = Path(directory) / (stable_hash(row["id"]) + ".fixture")
                    path.write_text(seed)
                    index[row["id"]] = {"id": row["id"], "candidate_id": row["candidate_id"],
                                        "audio_path": str(path), "batch_seed": seed}
                    with (Path(directory) / f"rank_{rank:03d}.ready.jsonl").open("a") as stream:
                        stream.write(json.dumps(index[row["id"]]) + "\n")
            return [index[row["id"]] for row in local]
        output, metrics = run_sharded_inference(rows, Path(directory) / "merged.jsonl",
            DistributedContext(True, rank, rank, world, "gloo"), generate, owners=owners)
        if rank == 0:
            assert [r["id"] for r in output] == [r["id"] for r in rows]
            assert sum(r["processed_rows"] for r in metrics["per_rank"]) == count
            assert len(metrics["per_rank"]) == world
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("count", [5, 37])
def test_eight_process_synthesis_merge_and_journals(tmp_path, count):
    torch = pytest.importorskip("torch")
    import torch.multiprocessing as mp
    mp.start_processes(_gloo_synthesis_rank, args=(8, str(tmp_path), count), nprocs=8,
                       join=True, start_method="fork")
    output = [json.loads(l) for l in (tmp_path / "merged.jsonl").read_text().splitlines()]
    expected = {r["id"]: stable_hash([v["generation_seed"] for v in batch])
                for batch in batches(jobs(count)) for r in batch}
    assert {r["id"]: r["batch_seed"] for r in output} == expected
    ready = [json.loads(l) for p in tmp_path.glob("*.ready.jsonl") for l in p.read_text().splitlines()]
    assert sorted(r["id"] for r in ready) == sorted(expected)
    assert all(Path(r["audio_path"]).is_file() for r in ready)
