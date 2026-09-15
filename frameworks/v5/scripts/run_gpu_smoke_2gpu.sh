#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python -m dual_isl_train train --config configs/gpu_smoke_2gpu.yaml
python scripts/verify_tts_ddp_sync.py --run-dir runs/gpu_smoke_2gpu --expected-world-size 2
