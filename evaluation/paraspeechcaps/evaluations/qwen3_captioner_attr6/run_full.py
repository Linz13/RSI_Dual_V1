#!/usr/bin/env python3
"""Run resumable inference followed by scoring, suitable for nohup execution."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from common import DEFAULT_RUN_ROOT


EVAL_ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = DEFAULT_RUN_ROOT / "runtime"
COMPLETION_PATH = RUNTIME_DIR / "completion.json"
PID_PATH = RUNTIME_DIR / "full_run.pid"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_completion(payload: dict[str, object]) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    temporary = COMPLETION_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(COMPLETION_PATH)


def main() -> int:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(f"{os.getpid()}\n", encoding="utf-8")
    started_at = utc_now()
    started = time.perf_counter()
    base = {
        "started_at_utc": started_at,
        "python": sys.executable,
    }
    write_completion({**base, "status": "running"})
    try:
        subprocess.run(
            [sys.executable, str(EVAL_ROOT / "run_qwen3_captioner.py"), "--resume"],
            cwd=EVAL_ROOT,
            check=True,
        )
        subprocess.run(
            [sys.executable, str(EVAL_ROOT / "score.py")],
            cwd=EVAL_ROOT,
            check=True,
        )
    except BaseException as exc:
        write_completion(
            {
                **base,
                "status": "failed",
                "finished_at_utc": utc_now(),
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        raise

    write_completion(
        {
            **base,
            "status": "completed",
            "finished_at_utc": utc_now(),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "predictions": str(DEFAULT_RUN_ROOT / "outputs" / "predictions.jsonl"),
            "summary": str(DEFAULT_RUN_ROOT / "reports" / "summary.json"),
            "report": str(DEFAULT_RUN_ROOT / "reports" / "final_report.md"),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
