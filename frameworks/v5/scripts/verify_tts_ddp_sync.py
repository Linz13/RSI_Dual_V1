from __future__ import annotations

import argparse
import json
from pathlib import Path


METADATA = "dual_isl_train_training.json"


def verify(path: Path, expected_world_size: int, require_memory_bounded_grpo: bool = False) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    distributed = value.get("distributed", {})
    errors = []
    mode = value.get("gradient_sync_mode")
    allowed_modes = {
        "standard_ddp_replay_forward",
        "standard_ddp_candidate_accumulation_last_candidate_sync",
    }
    if mode not in allowed_modes:
        errors.append(f"unsupported gradient_sync_mode: {mode!r}")
    if (
        require_memory_bounded_grpo and value.get("update") == "grpo"
        and mode != "standard_ddp_candidate_accumulation_last_candidate_sync"
    ):
        errors.append("GRPO checkpoint did not use memory-bounded candidate accumulation")
    if not distributed.get("enabled"):
        errors.append("distributed training was not enabled")
    if int(distributed.get("world_size", 0)) != expected_world_size:
        errors.append(f"world_size is {distributed.get('world_size')}, expected {expected_world_size}")
    if len(distributed.get("per_rank", [])) != expected_world_size:
        errors.append("per_rank audit count does not match world_size")
    for key in ("parameter_sync_before", "parameter_sync_after"):
        audit = value.get(key) or {}
        if not audit.get("ok"):
            errors.append(f"{key} failed: {audit}")
        if int(audit.get("nonfinite_count", -1)) != 0:
            errors.append(f"{key} contains non-finite parameters")
    if int(value.get("steps", 0)) > 0 and not (value.get("parameter_delta") or {}).get("ok"):
        errors.append("optimizer steps ran without a verified parameter change")
    return {"checkpoint_metadata": str(path), "ok": not errors, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-world-size", required=True, type=int)
    parser.add_argument("--require-memory-bounded-grpo", action="store_true")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    paths = sorted(run_dir.glob(f"round_*/checkpoints/tts_*/{METADATA}"))
    if not paths:
        raise SystemExit(f"No TTS checkpoint metadata found below {run_dir}")
    reports = [
        verify(path, args.expected_world_size, args.require_memory_bounded_grpo)
        for path in paths
    ]
    print(json.dumps({"ok": all(item["ok"] for item in reports), "reports": reports}, indent=2))
    raise SystemExit(0 if all(item["ok"] for item in reports) else 2)


if __name__ == "__main__":
    main()
