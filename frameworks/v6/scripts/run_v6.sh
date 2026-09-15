#!/usr/bin/env bash
set -euo pipefail
umask 000
export DUALISL_SHARED_WRITABLE=1
V6_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V6_MODE="${1:-check}"
V6_RUN="${2:-$V6_ROOT/runs/midasheng_v6_8gpu_run01}"
if [[ $# -gt 2 || "$V6_RUN" != /* ]]; then
  echo 'Usage: bash scripts/run_v6.sh [check|prepare|batch-smoke|gpu-smoke|train|resume|verify|status|dashboard] ABS_RUN_DIR' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="${DUALISL_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
export PYTHONPATH="$V6_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
V6_PY="${DUALISL_DRIVER_PYTHON:-$V6_ROOT/../../miniconda3/envs/dualisl-critic/bin/python}"
cd "$V6_ROOT"
exec "$V6_PY" -m scripts.v6_launcher "$V6_MODE" "$V6_RUN"
