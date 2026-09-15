from __future__ import annotations

from typing import Any, Iterable


def numerical_difference(left: Any, right: Any, *, rtol: float, atol: float) -> dict[str, Any]:
    torch = __import__("torch")
    left = left.detach().float()
    right = right.detach().float()
    if left.shape != right.shape:
        return {"ok": False, "shape_mismatch": True, "max_abs_diff": float("inf"), "mean_abs_diff": float("inf"), "nonfinite_count": 0}
    difference = (left - right).abs()
    nonfinite = int((~torch.isfinite(left)).sum().item() + (~torch.isfinite(right)).sum().item())
    return {
        "ok": nonfinite == 0 and bool(torch.allclose(left, right, rtol=rtol, atol=atol, equal_nan=False)),
        "shape_mismatch": False,
        "max_abs_diff": float(difference.max().item()) if difference.numel() else 0.0,
        "mean_abs_diff": float(difference.double().mean().item()) if difference.numel() else 0.0,
        "nonfinite_count": nonfinite,
    }


def trainable_named_parameters(model: Any) -> list[tuple[str, Any]]:
    values = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not values:
        raise RuntimeError("No trainable parameters found for synchronization audit")
    return values


def snapshot_parameters(named: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    return {name: parameter.detach().float().cpu().clone() for name, parameter in named}


def parameter_delta(snapshot: dict[str, Any], named: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    total, count, maximum, changed = 0.0, 0, 0.0, 0
    missing = []
    for name, parameter in named:
        before = snapshot.get(name)
        if before is None:
            missing.append(name)
            continue
        difference = (parameter.detach().float().cpu() - before).abs()
        local_max = float(difference.max()) if difference.numel() else 0.0
        maximum = max(maximum, local_max)
        total += float(difference.double().sum())
        count += int(difference.numel())
        changed += int(local_max > 0.0)
    return {
        "ok": not missing and changed > 0,
        "changed_parameter_tensors": changed,
        "checked_parameter_tensors": len(snapshot),
        "max_abs_diff": maximum,
        "mean_abs_diff": total / max(count, 1),
        "missing_tensors": missing,
    }


def audit_parameter_sync(
    named: list[tuple[str, Any]], distributed: Any, *, rtol: float = 1e-6, atol: float = 1e-7,
) -> dict[str, Any]:
    """Numerically compare every trainable tensor with rank 0 without repairing it."""
    torch = __import__("torch")
    layout = [(name, tuple(parameter.shape), str(parameter.dtype)) for name, parameter in named]
    layouts = distributed.gather_objects(layout)
    if any(item != layouts[0] for item in layouts[1:]):
        raise RuntimeError("Trainable parameter layout differs across ranks")
    local_sum = 0.0
    local_count = 0
    local_max = 0.0
    local_nonfinite = 0
    local_failed: list[str] = []
    if distributed.enabled:
        import torch.distributed as dist

        for name, parameter in named:
            current = parameter.detach().float()
            reference = current.clone()
            dist.broadcast(reference, src=0)
            difference = (current - reference).abs()
            comparison = numerical_difference(current, reference, rtol=rtol, atol=atol)
            nonfinite = int(comparison["nonfinite_count"])
            local_nonfinite += nonfinite
            local_sum += float(difference.double().sum().item())
            local_count += int(difference.numel())
            local_max = max(local_max, float(difference.max().item()) if difference.numel() else 0.0)
            if not comparison["ok"]:
                local_failed.append(name)
    else:
        for _, parameter in named:
            current = parameter.detach()
            local_nonfinite += int((~torch.isfinite(current)).sum().item())
            local_count += int(current.numel())
    local = {
        "rank": distributed.rank,
        "max_abs_diff": local_max,
        "mean_abs_diff": local_sum / max(local_count, 1),
        "nonfinite_count": local_nonfinite,
        "failed_tensors": local_failed[:10],
    }
    per_rank = distributed.gather_objects(local)
    global_count = sum(local_count_item for local_count_item in distributed.gather_objects(local_count))
    global_sum = sum(local_sum_item for local_sum_item in distributed.gather_objects(local_sum))
    report = {
        "ok": all(not item["failed_tensors"] and item["nonfinite_count"] == 0 for item in per_rank),
        "rtol": rtol,
        "atol": atol,
        "checked_parameter_tensors": len(named),
        "global_max_abs_diff": max((item["max_abs_diff"] for item in per_rank), default=0.0),
        "global_mean_abs_diff": global_sum / max(global_count, 1),
        "nonfinite_count": sum(item["nonfinite_count"] for item in per_rank),
        "per_rank": per_rank,
    }
    if not report["ok"]:
        raise RuntimeError(f"Meaningful cross-rank parameter divergence detected: {report}")
    return report

