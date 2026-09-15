#!/usr/bin/env bash
set -euo pipefail
# This workspace is shared across users/containers on the cluster.
umask 000
export DUALISL_SHARED_WRITABLE=1
V5_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V5_MODE="${1:-check}"
V5_RUN="${2:-$V5_ROOT/runs/midasheng_v5_run01}"
if [[ $# -gt 2 || "$V5_RUN" != /* ]]; then
  echo "Usage: bash scripts/run_v5.sh [check|gpu-smoke|train|resume|verify|dashboard] ABS_RUN_DIR" >&2
  exit 2
fi
export PYTHONPATH="$V5_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${DUALISL_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
V5_PY="${DUALISL_DRIVER_PYTHON:-$V5_ROOT/../../miniconda3/envs/dualisl-critic/bin/python}"
cd "$V5_ROOT"
if [[ "$V5_MODE" == dashboard ]]; then
  V5_LABEL="$("$V5_PY" - "$V5_RUN" <<'PY'
import hashlib,sys
print('v5_' + hashlib.sha256(sys.argv[1].encode()).hexdigest()[:12])
PY
)"
  exec bash scripts/run_training_dashboard.sh "$V5_RUN" "$V5_LABEL" "${DUALISL_DASHBOARD_PORT:-6007}"
fi
exec "$V5_PY" -m scripts.v5_launcher "$V5_MODE" "$V5_RUN"
