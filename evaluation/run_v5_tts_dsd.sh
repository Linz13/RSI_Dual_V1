#!/usr/bin/env bash
set -euo pipefail
BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER_DIR="$(cd -- "${BENCHMARK_DIR}/../.." && pwd)"
exec "${CLUSTER_DIR}/miniconda3/envs/emergent-tts-eval/bin/python" \
  "${BENCHMARK_DIR}/v5_tts_dsd_eval.py" "$@"
