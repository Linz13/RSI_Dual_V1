from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def last_record(path: Path) -> dict | None:
    if not path.is_file():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else None


def human_gib(value: int | float | None) -> str:
    return f"{float(value or 0) / (1024 ** 3):.1f}GiB"


def latest_round(run_dir: Path) -> Path:
    rounds = sorted(path for path in run_dir.glob("round_*" ) if path.is_dir())
    if not rounds:
        raise SystemExit(f"No round directory found below {run_dir}")
    return rounds[-1]


def snapshot(run_dir: Path) -> None:
    round_dir = latest_round(run_dir)
    checkpoint = round_dir / "checkpoints" / "tts_after_grpo"
    progress_dir = checkpoint / "training_progress"
    metrics_path = checkpoint / "training_metrics" / "metrics.jsonl"
    metrics = []
    if metrics_path.is_file():
        metrics = [
            json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"\n[{now}] round={round_dir.name} completed_global_steps={len(metrics)}", flush=True)
    if metrics:
        last = metrics[-1]
        print(
            "last_loss "
            f"step={last.get('step')} loss={last.get('loss')} grad_norm={last.get('grad_norm')} "
            f"elapsed={last.get('elapsed_seconds')}s sample={last.get('sample_id')}",
            flush=True,
        )
    paths = sorted(progress_dir.glob("rank_*.jsonl"))
    if not paths:
        print(f"No heartbeat files yet below {progress_dir}", flush=True)
        return
    print("rank step event               cand frames       loss       kl allocated reserved peak sample", flush=True)
    for path in paths:
        row = last_record(path) or {}
        frames = "-"
        if row.get("total_frames") is not None:
            frames = f"{row.get('processed_frames', 0)}/{row['total_frames']}"
        candidate = row.get("candidate_index")
        candidate_text = "-" if candidate is None else str(candidate)
        loss = "-" if row.get("loss") is None else f"{float(row['loss']):.5g}"
        kl = "-" if row.get("kl") is None else f"{float(row['kl']):.4g}"
        print(
            f"{int(row.get('rank', -1)):>4} {int(row.get('step', -1)):>4} "
            f"{str(row.get('event', '-')):<19} {candidate_text:>4} {frames:>11} "
            f"{loss:>10} {kl:>8} "
            f"{human_gib(row.get('allocated_bytes')):>9} "
            f"{human_gib(row.get('reserved_bytes')):>8} "
            f"{human_gib(row.get('peak_allocated_bytes')):>6} "
            f"{row.get('sample_id', '-')}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor rank-level TTS GRPO heartbeat and completed losses")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--interval", type=float, default=0.0, help="Refresh interval; 0 prints once")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    try:
        while True:
            snapshot(run_dir)
            if args.interval <= 0:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
