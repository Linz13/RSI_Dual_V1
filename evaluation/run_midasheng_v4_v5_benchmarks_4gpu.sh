#!/usr/bin/env bash
set -Eeuo pipefail
umask 000
BENCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
exec "${BENCH_ROOT}/../../miniconda3/envs/dualisl-critic/bin/python" \
  "${BENCH_ROOT}/midasheng_v4_v5_eval.py" "$@"
