#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CAPTION_ROOT="$(cd -- "${ROOT}/../.." && pwd)"
CLUSTER_ROOT="$(cd -- "${CAPTION_ROOT}/.." && pwd)"

PYTHON="${MIDASHENG_PYTHON:-${CLUSTER_ROOT}/miniconda3/envs/midasheng-captioner/bin/python}"
MODEL_DIR="${MIDASHENG_MODEL_DIR:-${CAPTION_ROOT}/models/MiDashengLM-7B-1021-BF16}"
ADAPTER_DIR="${MIDASHENG_ADAPTER_DIR:-${CAPTION_ROOT}/DualISL_Train/runs/dual_recursive_4gpu_h100_midasheng_20260830_run01/round_002/checkpoints/caption_final}"
OUTPUT_DIR="${MIDASHENG_OUTPUT_DIR:-${ROOT}/runs/midasheng4_round2_full_20260831_run01}"
GPU="${MIDASHENG_GPU:-0}"

[[ -x "${PYTHON}" ]] || { echo "Missing MiDasheng Python: ${PYTHON}" >&2; exit 1; }
[[ -d "${MODEL_DIR}" ]] || { echo "Missing MiDasheng model: ${MODEL_DIR}" >&2; exit 1; }
if [[ "${ADAPTER_DIR}" != "none" && "${ADAPTER_DIR}" != "base" && ! -d "${ADAPTER_DIR}" ]]; then
  echo "Missing MiDasheng adapter: ${ADAPTER_DIR}" >&2
  exit 1
fi
[[ -f "${ROOT}/data/benchmark.jsonl" ]] || { echo "Missing benchmark JSONL" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"${PYTHON}" "${ROOT}/run_midasheng.py" \
  --benchmark "${ROOT}/data/benchmark.jsonl" \
  --model-dir "${MODEL_DIR}" \
  --adapter-dir "${ADAPTER_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --resume \
  "$@"

"${PYTHON}" "${ROOT}/evaluate.py" \
  --benchmark "${ROOT}/data/benchmark.jsonl" \
  --predictions "${OUTPUT_DIR}/predictions.jsonl" \
  --output "${OUTPUT_DIR}/evaluation_summary.json"
