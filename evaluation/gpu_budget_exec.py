#!/usr/bin/env python3
"""Run a Python benchmark on one visible GPU with a PyTorch allocator budget."""
import argparse
from pathlib import Path
import runpy
import sys


def configure_budget(torch, budget_gib):
    if budget_gib <= 0 or torch.cuda.device_count() != 1:
        raise RuntimeError("A positive budget and exactly one visible GPU are required")
    total = torch.cuda.get_device_properties(0).total_memory
    budget = int(budget_gib * 1024 ** 3)
    if budget > total:
        raise RuntimeError("GPU allocator budget exceeds physical GPU memory")
    torch.cuda.set_per_process_memory_fraction(budget / total, 0)
    free, _ = torch.cuda.mem_get_info(0)
    if free < budget + 1024 ** 3:
        raise RuntimeError("GPU free memory changed: need allocator budget plus 1 GiB headroom")
    print(f"[MEMORY] PyTorch budget={budget_gib:g} GiB; free={free / 1024 ** 3:.1f} GiB", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-memory-gib", type=float, required=True)
    parser.add_argument("script", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    import torch
    configure_budget(torch, args.gpu_memory_gib)
    script = args.script.resolve()
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *args.arguments]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
