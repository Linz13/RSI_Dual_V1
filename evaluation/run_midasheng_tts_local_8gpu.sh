#!/usr/bin/env bash
# Original MiDasheng: completed rounds 6 and 9, local TTS evaluation only.
set -euo pipefail
BENCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DRIVER_PYTHON="${DRIVER_PYTHON:-/data/L202500147/miniconda3/envs/qwen3-tts/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "$DRIVER_PYTHON" "$BENCH_DIR/midasheng_tts_local_eval.py" "$@"
