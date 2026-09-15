from __future__ import annotations

"""CPU DDP probe for the TTS candidate-by-candidate gradient accumulation contract."""

import contextlib
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def build_model() -> DistributedDataParallel:
    module = torch.nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[0.25, -0.5, 0.75]]))
    return DistributedDataParallel(module, broadcast_buffers=False)


def rank_batch(rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    count = 4 if rank == 0 else 3
    offset = rank * 0.125
    inputs = torch.arange(count * 3, dtype=torch.float32).reshape(count, 3) / 10 + offset
    targets = torch.linspace(-0.3, 0.6, count, dtype=torch.float32).unsqueeze(1)
    return inputs, targets


def update_group(
    model: DistributedDataParallel, inputs: torch.Tensor, targets: torch.Tensor,
    padded: bool, active_count: int,
) -> None:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    optimizer.zero_grad(set_to_none=True)
    losses = (model(inputs) - targets).square().reshape(-1)
    scale = 0.0 if padded else dist.get_world_size() / active_count
    (losses.mean() * scale).backward()
    optimizer.step()


def update_candidates(
    model: DistributedDataParallel, inputs: torch.Tensor, targets: torch.Tensor,
    padded: bool, active_count: int,
) -> None:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    optimizer.zero_grad(set_to_none=True)
    count = inputs.shape[0]
    scale = 0.0 if padded else dist.get_world_size() / active_count / count
    for index in range(count):
        sync = index + 1 == count
        with contextlib.nullcontext() if sync else model.no_sync():
            loss = (model(inputs[index:index + 1]) - targets[index:index + 1]).square().mean()
            (loss * scale).backward()
    optimizer.step()


def run_case(padded_rank_one: bool) -> float:
    rank = dist.get_rank()
    inputs, targets = rank_batch(rank)
    padded = padded_rank_one and rank == 1
    active_count = 1 if padded_rank_one else dist.get_world_size()
    group_model = build_model()
    candidate_model = build_model()
    update_group(group_model, inputs, targets, padded, active_count)
    update_candidates(candidate_model, inputs, targets, padded, active_count)
    difference = float(max(
        (left - right).abs().max().item()
        for left, right in zip(group_model.parameters(), candidate_model.parameters(), strict=True)
    ))
    maximum = torch.tensor(difference)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return float(maximum.item())


def main() -> None:
    dist.init_process_group("gloo")
    try:
        differences = {
            "all_active": run_case(False),
            "rank_one_padded": run_case(True),
        }
        tolerance = 1e-6
        if dist.get_rank() == 0:
            print({"world_size": dist.get_world_size(), "max_abs_differences": differences})
        if max(differences.values()) > tolerance:
            raise RuntimeError(f"Candidate DDP accumulation differs from group mean: {differences}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        raise SystemExit("Run with torch.distributed.run and at least two processes")
    main()
