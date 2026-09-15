#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: DUALISL_RUN_DIR=/absolute/run/path $0 <smoke|train> 4" >&2
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

MODE=$1
GPU_COUNT=$2
case "${MODE}:${GPU_COUNT}" in
  smoke:4|train:4) ;;
  *)
    usage
    exit 2
    ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARED_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
WORKER_PATH="${PROJECT_ROOT}/scripts/midasheng_captioner_candidate.py"
if [[ "${MODE}" == "smoke" ]]; then
  CONFIG="${PROJECT_ROOT}/configs/gpu_smoke_4gpu_h100_midasheng_candidate.yaml"
  DEFAULT_RUN_DIR="${PROJECT_ROOT}/runs/gpu_smoke_4gpu_h100_midasheng_candidate"
else
  CONFIG="${PROJECT_ROOT}/configs/train_4gpu_h100_midasheng_candidate.yaml"
  DEFAULT_RUN_DIR="${PROJECT_ROOT}/runs/dual_recursive_4gpu_h100_midasheng_candidate"
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
if [[ ! -f "${WORKER_PATH}" ]]; then
  echo "MiDasheng candidate worker is missing: ${WORKER_PATH}" >&2
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
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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

WORKER_HASH=$(sha256sum "${WORKER_PATH}" | awk '{print $1}')
WORKER_HASH_FILE="${RUN_DIR}/midasheng_candidate_worker.sha256"
ACTION=train
if [[ -f "${RUN_DIR}/run_state.json" ]]; then
  ACTION=resume
  if [[ ! -f "${WORKER_HASH_FILE}" ]]; then
    echo "Refusing resume without ${WORKER_HASH_FILE}" >&2
    exit 2
  fi
  RECORDED_WORKER_HASH=$(awk 'NR == 1 {print $1}' "${WORKER_HASH_FILE}")
  if [[ "${RECORDED_WORKER_HASH}" != "${WORKER_HASH}" ]]; then
    echo "Refusing resume: MiDasheng candidate worker hash changed." >&2
    echo "recorded=${RECORDED_WORKER_HASH} current=${WORKER_HASH}" >&2
    exit 2
  fi
elif find "${RUN_DIR}" -mindepth 1 -maxdepth 1 \
    ! -name .launcher.lock ! -name midasheng_candidate_worker.sha256 \
    -print -quit | grep -q .; then
  echo "Refusing a non-empty run directory without run_state.json: ${RUN_DIR}" >&2
  exit 2
else
  printf '%s  %s\n' "${WORKER_HASH}" "${WORKER_PATH}" >"${WORKER_HASH_FILE}"
  chmod 0644 "${WORKER_HASH_FILE}"
fi

cd "${PROJECT_ROOT}"
"${DRIVER_PYTHON}" -m dual_isl_train validate-config --config "${CONFIG}" >/dev/null

echo "DualISL captioner=midasheng mode=${MODE} GPUs=${GPU_COUNT} action=${ACTION} run_dir=${RUN_DIR}"
"${DRIVER_PYTHON}" -m dual_isl_train "${ACTION}" --config "${CONFIG}"
"${DRIVER_PYTHON}" scripts/verify_tts_ddp_sync.py \
  --run-dir "${RUN_DIR}" --expected-world-size "${GPU_COUNT}" \
  --require-memory-bounded-grpo
