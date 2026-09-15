#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARED_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/gpu_smoke_1gpu.yaml"
RUN_DIR="${DUALISL_RUN_DIR:-${PROJECT_ROOT}/runs/gpu_smoke_1gpu}"
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
  echo "CUDA_VISIBLE_DEVICES must expose exactly one GPU." >&2
  exit 2
fi

export DUALISL_RUN_DIR="${RUN_DIR}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

VISIBLE_GPU_COUNT=$("${DRIVER_PYTHON}" -c 'import torch; print(torch.cuda.device_count())')
if [[ "${VISIBLE_GPU_COUNT}" != "1" ]]; then
  echo "Expected 1 visible GPU, but PyTorch sees ${VISIBLE_GPU_COUNT}." >&2
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

echo "DualISL mode=smoke GPUs=1 action=${ACTION} run_dir=${RUN_DIR}"
"${DRIVER_PYTHON}" -m dual_isl_train "${ACTION}" --config "${CONFIG}"
