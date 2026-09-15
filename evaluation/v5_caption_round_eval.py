#!/usr/bin/env python3
"""Evaluate one committed V5 Captioner round on three GPUs with bounded allocation."""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import signal

import midasheng_v4_v5_eval as legacy
from v5_tts_dsd_eval import check_gpu_memory, parse_gpus

ROOT = Path(__file__).resolve().parent
VERSION = "v5-caption-round-three-gpu-v1"
GUARD = ROOT / "gpu_budget_exec.py"


def discover(round_index):
    if round_index < 0:
        raise ValueError("round-index must be nonnegative")
    candidates = legacy.existing.discover_reward_candidates(
        legacy.CAPTION, [(legacy.V5, "midasheng_rewardv5", [round_index])], True, True)
    if len(candidates) != 1 or candidates[0].status != "ready":
        raise RuntimeError("Requested V5 round is not ready: " + str([asdict(c) for c in candidates]))
    legacy.validate_candidates(candidates)
    c = candidates[0]
    summary = legacy.existing.read_json(Path(c.adapter).parents[1] / "summary.json")
    if summary["output_captioner"] != {"path": c.adapter, "sha256": c.detail}:
        raise RuntimeError("Round summary and committed Captioner differ")
    return candidates


class BudgetRunner(legacy.Runner):
    def __init__(self, output, batch_size, budget):
        super().__init__(output)
        self.batch_size, self.budget = batch_size, budget

    def prepare_command(self, command):
        command = list(command)
        if "--batch-size" in command:
            command[command.index("--batch-size") + 1] = str(self.batch_size)
        # These four commands load GPU models. The remaining scorers are CPU only.
        gpu_scripts = {legacy.ET / "run_inference.py", legacy.ET / "evaluate.py",
                       legacy.PSC / "run_content_scheme_a.py", legacy.SC / "run_midasheng.py"}
        if Path(command[1]) in gpu_scripts:
            command = [command[0], str(GUARD), "--gpu-memory-gib", str(self.budget), *command[1:]]
        return command

    def run_command(self, command, env, log):
        actual = self.prepare_command(command)
        log.write("BUDGET_COMMAND " + json.dumps(actual) + "\n")
        log.flush()
        return super().run_command(actual, env, log)


def inventory(output, round_index, batch_size):
    candidates = discover(round_index)
    resources = legacy.resource_identity()
    for path in (Path(__file__), GUARD, ROOT / "v5_tts_dsd_eval.py",
                 ROOT / "run_v5_caption_round.sh"):
        resources[str(path)] = legacy.existing.sha256_file(path)
    record = {"version": VERSION, "candidates": [asdict(c) for c in candidates],
              "resources": resources, "protocol": {
                  "round_index": round_index, "emotiontalk": "standard_public",
                  "paraspeechcaps": "Scheme A attr6", "stylecap": "speaker-open MCQ",
                  "attention": "sdpa", "batch_size": batch_size, "max_new_tokens_emotiontalk": 128}}
    path = output / "inventory.json"
    if path.exists() and legacy.existing.read_json(path) != record:
        raise RuntimeError("Checkpoint, batch, code or data changed; use a new output directory")
    return record, candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "smoke", "run", "status"))
    parser.add_argument("--round-index", type=int, default=1, help="Zero based; default is the second round")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--gpus", default=os.getenv("EVAL_GPUS", "0,1,2"))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("EVAL_BATCH_SIZE", "1")))
    parser.add_argument("--gpu-memory-gib", type=float, default=float(os.getenv("EVAL_GPU_MEMORY_GIB", "26")))
    args = parser.parse_args()
    gpus = parse_gpus(args.gpus)
    if len(gpus) != 3 or args.batch_size < 1 or args.gpu_memory_gib <= 0:
        raise ValueError("Select exactly three GPUs, a positive batch size and GPU memory budget")
    os.umask(0)
    output = (args.output_root or ROOT / f"midasheng_v5_round{args.round_index:03d}_caption_3gpu_run01").resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".launcher.lock").open("a") as lock:
        if args.mode != "status":
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record, candidates = inventory(output, args.round_index, args.batch_size)
        if args.mode == "status":
            print(json.dumps(legacy.summarize(output, candidates), ensure_ascii=False, indent=2))
            return 0
        report = {**legacy.cpu_check(candidates), "round_index": args.round_index,
                  "gpus_requested": gpus, "batch_size": args.batch_size,
                  "gpu_memory_gib": args.gpu_memory_gib}
        legacy.write_json(output / "inventory.json", record)
        legacy.write_json(output / "cpu_preflight.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if args.mode == "check":
            return 0
        check_gpu_memory(gpus, args.gpu_memory_gib)
        runner = BudgetRunner(output, args.batch_size, args.gpu_memory_gib)
        def terminate(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, terminate)
        try:
            runner.stage(candidates, "smoke" if args.mode == "smoke" else "full", gpus)
        finally:
            runner.cancel()
            legacy.summarize(output, candidates)
        print(f"[REPORT] {output / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Stopped; completed results preserved. Rerun the same command to resume.", flush=True)
        raise SystemExit(130)
