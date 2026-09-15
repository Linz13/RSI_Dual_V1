from __future__ import annotations

import math
import os
import random
from datetime import timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .io import atomic_json, read_jsonl, write_jsonl


@dataclass
class DistributedContext:
    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str = ""

    @classmethod
    def initialize(cls, config: dict[str, Any]) -> "DistributedContext":
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size <= 1:
            return cls()
        requested = config.get("distributed", {})
        if not bool(requested.get("enabled", False)):
            raise RuntimeError("torchrun worker started but distributed.enabled is false")
        expected = int(requested.get("world_size", world_size))
        if expected != world_size:
            raise RuntimeError(f"torchrun WORLD_SIZE={world_size} does not match configured world_size={expected}")
        import torch
        import torch.distributed as dist

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        backend = str(requested.get("backend", "nccl"))
        timeout_seconds = int(requested.get("timeout_seconds", 1800))
        if timeout_seconds < 1:
            raise ValueError("distributed.timeout_seconds must be >= 1")
        dist.init_process_group(backend=backend, timeout=timedelta(seconds=timeout_seconds))
        return cls(True, dist.get_rank(), local_rank, dist.get_world_size(), backend)

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def device(self) -> str:
        return f"cuda:{self.local_rank}" if self.enabled else ""

    def barrier(self) -> None:
        if self.enabled:
            import torch.distributed as dist
            dist.barrier()

    def gather_objects(self, value: Any) -> list[Any]:
        if not self.enabled:
            return [value]
        import torch.distributed as dist
        output: list[Any] = [None for _ in range(self.world_size)]
        dist.all_gather_object(output, value)
        return output

    def mean(self, value: float) -> float:
        if not self.enabled:
            return float(value)
        import torch
        import torch.distributed as dist
        tensor = torch.tensor(float(value), device=self.device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float((tensor / self.world_size).item())

    def close(self) -> None:
        if self.enabled:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()


def row_group_key(row: dict[str, Any]) -> str:
    if row.get("group_id") is not None:
        return str(row["group_id"])
    if row.get("id") is not None:
        return str(row["id"])
    candidate_id = str(row.get("candidate_id", ""))
    return candidate_id.split("::", 1)[0] if candidate_id else ""


def shard_indexed_rows(rows: list[dict[str, Any]], rank: int, world_size: int) -> list[tuple[int, dict[str, Any]]]:
    """Assign complete sample groups round-robin while retaining input indices."""
    owners: dict[str, int] = {}
    next_owner = 0
    output: list[tuple[int, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        key = row_group_key(row)
        if key not in owners:
            owners[key] = next_owner % world_size
            next_owner += 1
        if owners[key] == rank:
            output.append((index, row))
    return output


def distributed_schedule(
    rows: list[dict[str, Any]], training: dict[str, Any], seed: int,
    rank: int, world_size: int,
) -> list[tuple[int, int, dict[str, Any], bool]]:
    """Build equal-length per-rank schedules without splitting a row/group."""
    if not rows:
        return []
    rng = random.Random(seed)
    assignments: list[tuple[int, dict[str, Any], bool]] = []
    if "epochs" in training:
        for epoch in range(int(training["epochs"])):
            ordered = list(rows)
            if bool(training.get("shuffle", False)):
                rng.shuffle(ordered)
            padded = math.ceil(len(ordered) / world_size) * world_size
            for index in range(padded):
                assignments.append((epoch, ordered[index % len(ordered)], index >= len(ordered)))
    else:
        target = int(training.get("update_steps", 100)) * world_size
        epoch = 0
        while len(assignments) < target:
            ordered = list(rows)
            if bool(training.get("shuffle", False)):
                rng.shuffle(ordered)
            assignments.extend((epoch, row, False) for row in ordered)
            epoch += 1
        assignments = assignments[:target]
    return [
        (global_step, epoch, row, padded)
        for global_step, offset in enumerate(range(rank, len(assignments), world_size))
        for epoch, row, padded in [assignments[offset]]
    ]


def cost_bucketed_distributed_schedule(
    rows: list[dict[str, Any]], training: dict[str, Any], seed: int,
    rank: int, world_size: int, cost: Callable[[dict[str, Any]], int | float],
) -> list[tuple[int, int, dict[str, Any], bool]]:
    """Group similarly expensive rows into deterministic synchronized steps.

    DDP step time is controlled by the slowest rank.  Random round-robin order
    spreads long trajectories across many steps, so every one becomes a
    separate straggler.  This schedule sorts by a caller-provided cost, packs
    adjacent rows into world-size buckets, shuffles the bucket order, and
    rotates rank ownership between buckets.  Every real row is still consumed
    exactly once per epoch; only optimizer batch composition and order change.
    """
    if not rows:
        return []
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if "epochs" not in training:
        raise ValueError("cost-bucketed scheduling requires an explicit epoch count")

    rng = random.Random(seed)
    assignments: list[tuple[int, dict[str, Any], bool]] = []
    for epoch in range(int(training["epochs"])):
        # The random key makes equal-cost ordering deterministic without making
        # input order a hidden scheduling dependency.
        decorated = [(float(cost(row)), rng.random(), row) for row in rows]
        decorated.sort(key=lambda item: (-item[0], item[1]))
        ordered = [(row, False) for _, _, row in decorated]
        padded = math.ceil(len(ordered) / world_size) * world_size
        while len(ordered) < padded:
            # Repeat the cheapest row for DDP shape/collective parity.  Its
            # gradient is explicitly zeroed by the caller through `padded`.
            ordered.append((ordered[-1][0], True))

        buckets = [ordered[offset:offset + world_size] for offset in range(0, len(ordered), world_size)]
        if bool(training.get("shuffle", False)):
            rng.shuffle(buckets)
        for step, bucket in enumerate(buckets):
            rotation = (step + epoch) % world_size
            rotated = bucket[rotation:] + bucket[:rotation]
            assignments.extend((epoch, row, is_padding) for row, is_padding in rotated)

    return [
        (global_step, epoch, row, padded)
        for global_step, offset in enumerate(range(rank, len(assignments), world_size))
        for epoch, row, padded in [assignments[offset]]
    ]


def _rank_output_path(output_path: str | Path, rank: int) -> Path:
    output = Path(output_path)
    return output.parent / f"{output.name}.rank{rank:03d}.jsonl"


def run_sharded_inference(
    rows: list[dict[str, Any]], output_path: str | Path, context: DistributedContext,
    function: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    *, owners: list[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if owners is not None:
        if len(owners) != len(rows) or any(type(r) is not int or not 0 <= r < context.world_size for r in owners):
            raise ValueError("Invalid inference ownership plan")
        group_owners = {}
        for row, owner in zip(rows, owners):
            key = row_group_key(row)
            if key in group_owners and group_owners[key] != owner:
                raise ValueError("Inference plan split a complete group")
            group_owners[key] = owner
        indexed = [(i, row) for i, row in enumerate(rows) if owners[i] == context.rank]
    else:
        indexed = shard_indexed_rows(rows, context.rank, context.world_size)
    local_rows = [row for _, row in indexed]
    import time
    import torch
    torch.cuda.synchronize()
    started = time.perf_counter()
    outputs = function(local_rows)
    torch.cuda.synchronize()
    compute_seconds = time.perf_counter() - started
    if len(outputs) != len(indexed):
        raise RuntimeError(f"Distributed worker rank {context.rank} returned {len(outputs)} rows for {len(indexed)} inputs")
    shard = _rank_output_path(output_path, context.rank)
    write_jsonl(shard, [
        {"input_index": index, "output": output}
        for (index, _), output in zip(indexed, outputs, strict=True)
    ])
    import torch
    rank_metrics = {
        "rank": context.rank,
        "local_rank": context.local_rank,
        "cuda_device": torch.cuda.current_device(),
        "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "processed_rows": len(local_rows),
        "sample_ids": [row_group_key(row) for row in local_rows],
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "shard_path": str(shard),
        "inference_seconds": compute_seconds,
    }
    atomic_json(str(shard) + ".metrics.json", rank_metrics)
    context.barrier()
    merged: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    if context.is_main:
        indexed_outputs = []
        for rank in range(context.world_size):
            rank_shard = _rank_output_path(output_path, rank)
            indexed_outputs.extend(read_jsonl(rank_shard))
            all_metrics.append(__import__("json").loads(Path(str(rank_shard) + ".metrics.json").read_text(encoding="utf-8")))
        indexed_outputs.sort(key=lambda item: int(item["input_index"]))
        merged = [item["output"] for item in indexed_outputs]
        write_jsonl(output_path, merged)
    context.barrier()
    return merged, {
        "distributed": True,
        "backend": context.backend,
        "world_size": context.world_size,
        "per_rank": all_metrics if context.is_main else [],
    }
