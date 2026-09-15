#!/usr/bin/env bash
set -euo pipefail
umask 000
V6_BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
exec "$V6_BENCH_ROOT/../../miniconda3/envs/emergent-tts-eval/bin/python" \
  "$V6_BENCH_ROOT/v6_round_benchmarks.py" "$@"
