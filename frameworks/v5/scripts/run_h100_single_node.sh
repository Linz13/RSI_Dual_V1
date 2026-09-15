#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: DUALISL_RUN_DIR=/absolute/run/path $0 <smoke|train> <4|7|8>" >&2
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

MODE=$1
GPU_COUNT=$2
case "${MODE}:${GPU_COUNT}" in
  smoke:4|smoke:7|smoke:8|train:4|train:7|train:8) ;;
  *)
    usage
    exit 2
    ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARED_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
if [[ "${MODE}" == "smoke" ]]; then
  CONFIG="${PROJECT_ROOT}/configs/gpu_smoke_${GPU_COUNT}gpu_h100.yaml"
  DEFAULT_RUN_DIR="${PROJECT_ROOT}/runs/gpu_smoke_${GPU_COUNT}gpu_h100"
else
  CONFIG="${PROJECT_ROOT}/configs/train_${GPU_COUNT}gpu_h100.yaml"
  DEFAULT_RUN_DIR="${PROJECT_ROOT}/runs/dual_recursive_${GPU_COUNT}gpu_h100"
fi
RUN_DIR="${DUALISL_RUN_DIR:-${DEFAULT_RUN_DIR}}"
DRIVER_PYTHON="${DUALISL_DRIVER_PYTHON:-${SHARED_ROOT}/miniconda3/envs/qwen3-tts/bin/python}"

if [[ "${RUN_DIR}" != /* ]]; then
  echo "DUALISL_RUN_DIR must be an absolute path: ${RUN_DIR}" >&2
  exit 2
fi
if [[ ! -x "${DRIVER_PYTHON}" ]]; then
  echo "Driver Python is not executable: ${DRIVER_PYTHON}" >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must be set by the scheduler or caller." >&2
  exit 2
fi

export DUALISL_RUN_DIR="${RUN_DIR}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

VISIBLE_GPU_COUNT=$("${DRIVER_PYTHON}" -c 'import torch; print(torch.cuda.device_count())')
if [[ "${VISIBLE_GPU_COUNT}" != "${GPU_COUNT}" ]]; then
  echo "Expected ${GPU_COUNT} visible GPUs, but PyTorch sees ${VISIBLE_GPU_COUNT}." >&2
  exit 2
fi

mkdir -p "${RUN_DIR}"
exec 9>"${RUN_DIR}/.launcher.lock"
if ! flock -n 9; then
  echo "Another launcher already holds ${RUN_DIR}/.launcher.lock" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
"${DRIVER_PYTHON}" -m dual_isl_train validate-config --config "${CONFIG}" >/dev/null

ACTION=train
if [[ -f "${RUN_DIR}/run_state.json" ]]; then
  ACTION=resume
elif find "${RUN_DIR}" -mindepth 1 -maxdepth 1 ! -name .launcher.lock -print -quit | grep -q .; then
  echo "Refusing a non-empty run directory without run_state.json: ${RUN_DIR}" >&2
  exit 2
fi

echo "DualISL mode=${MODE} GPUs=${GPU_COUNT} action=${ACTION} run_dir=${RUN_DIR}"
"${DRIVER_PYTHON}" -m dual_isl_train "${ACTION}" --config "${CONFIG}"
"${DRIVER_PYTHON}" scripts/verify_tts_ddp_sync.py \
  --run-dir "${RUN_DIR}" --expected-world-size "${GPU_COUNT}" \
  --require-memory-bounded-grpo
