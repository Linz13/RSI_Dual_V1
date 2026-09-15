#!/usr/bin/env bash
set -euo pipefail

# Runs live on shared storage. Keep newly created logs/checkpoints writable by
# every server user, matching this project's deliberately shared permissions.
umask 000

usage() {
  echo "Usage:" >&2
  echo "  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DUALISL_RUN_DIR=/absolute/run/path $0 smoke 8" >&2
  echo "  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DUALISL_RUN_DIR=/absolute/run/path $0 train 8 <rounds>" >&2
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 2
fi

MODE=$1
GPU_COUNT=$2
case "${MODE}:${GPU_COUNT}" in
  smoke:8)
    if [[ $# -ne 2 ]]; then
      usage
      exit 2
    fi
    TRAINING_ROUNDS=1
    ;;
  train:8)
    if [[ $# -ne 3 || ! ${3:-} =~ ^[1-9][0-9]*$ ]]; then
      echo "Training rounds must be a positive integer." >&2
      usage
      exit 2
    fi
    TRAINING_ROUNDS=$3
    ;;
  *)
    usage
    exit 2
    ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARED_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
WORKER_PATH="${PROJECT_ROOT}/scripts/midasheng_captioner_candidate.py"
MIDASHENG_MODEL="${SHARED_ROOT}/Caption/models/MiDashengLM-7B-1021-BF16"
MIDASHENG_PYTHON="${SHARED_ROOT}/miniconda3/envs/midasheng-captioner/bin/python"
DRIVER_PYTHON="${DUALISL_DRIVER_PYTHON:-${SHARED_ROOT}/miniconda3/envs/qwen3-tts/bin/python}"

if [[ "${MODE}" == "smoke" ]]; then
  CONFIG="${PROJECT_ROOT}/configs/gpu_smoke_8gpu_h100_midasheng_reward_v2.yaml"
  DEFAULT_RUN_DIR="${PROJECT_ROOT}/runs/gpu_smoke_8gpu_h100_midasheng_reward_v2"
else
  CONFIG="${PROJECT_ROOT}/configs/train_8gpu_h100_midasheng_reward_v2.yaml"
  DEFAULT_RUN_DIR="${PROJECT_ROOT}/runs/dual_recursive_8gpu_h100_midasheng_reward_v2"
fi
RUN_DIR="${DUALISL_RUN_DIR:-${DEFAULT_RUN_DIR}}"

if [[ "${RUN_DIR}" != /* ]]; then
  echo "DUALISL_RUN_DIR must be an absolute path: ${RUN_DIR}" >&2
  exit 2
fi
if [[ ! -x "${DRIVER_PYTHON}" ]]; then
  echo "Driver Python is not executable: ${DRIVER_PYTHON}" >&2
  exit 2
fi
if [[ ! -x "${MIDASHENG_PYTHON}" ]]; then
  echo "MiDasheng Python is not executable: ${MIDASHENG_PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${WORKER_PATH}" ]]; then
  echo "MiDasheng worker is missing: ${WORKER_PATH}" >&2
  exit 2
fi
if [[ ! -d "${MIDASHENG_MODEL}" ]]; then
  echo "MiDasheng model is missing: ${MIDASHENG_MODEL}" >&2
  exit 2
fi
if [[ -n "${DUALISL_CAPTION_ADAPTER:-}" || -n "${DUALISL_TTS_ADAPTER:-}" ]]; then
  echo "RewardV2 base run refuses DUALISL_CAPTION_ADAPTER/DUALISL_TTS_ADAPTER overrides." >&2
  echo "Unset them so the run starts from the two base models and fits a fresh v2 calibration." >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must be set by the scheduler or caller." >&2
  exit 2
fi

export DUALISL_RUN_DIR="${RUN_DIR}"
export DUALISL_ROUNDS="${TRAINING_ROUNDS}"
export DUALISL_SHARED_WRITABLE=1
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${PROJECT_ROOT}"
"${DRIVER_PYTHON}" -m dual_isl_train validate-config --config "${CONFIG}" >/dev/null
"${DRIVER_PYTHON}" - "${CONFIG}" "${TRAINING_ROUNDS}" <<'PY'
import sys

from dual_isl_train.config import load_config

config = load_config(sys.argv[1])
expected_rounds = int(sys.argv[2])
distributed = config["distributed"]
if (
    not bool(distributed.get("enabled"))
    or int(distributed.get("world_size", 0)) != 8
    or distributed.get("backend") != "nccl"
):
    raise RuntimeError("RewardV2 MiDasheng launcher requires enabled 8-rank NCCL")
if int(config["training"].get("round_offset", 0)) != 0:
    raise RuntimeError("RewardV2 base run requires training.round_offset=0")
if int(config["training"].get("rounds", 0)) != expected_rounds:
    raise RuntimeError("DUALISL_ROUNDS was not applied to training.rounds")
if config["captioner"].get("adapter_path") or config["tts"].get("adapter_path"):
    raise RuntimeError("RewardV2 base run must not load old adapters")
if config["reward"]["calibration"]["method"] != "round0_dual_counterfactual_zscore":
    raise RuntimeError("RewardV2 calibration method is not enabled")
if config["captioner"]["worker_module"] != "scripts.midasheng_captioner_candidate":
    raise RuntimeError("Configured Captioner is not MiDasheng")
PY

VISIBLE_GPU_COUNT=$("${DRIVER_PYTHON}" -c 'import torch; print(torch.cuda.device_count())')
if [[ "${VISIBLE_GPU_COUNT}" != "${GPU_COUNT}" ]]; then
  echo "Expected ${GPU_COUNT} visible GPUs, but PyTorch sees ${VISIBLE_GPU_COUNT}." >&2
  exit 2
fi

mkdir -p "${RUN_DIR}"
chmod a+rwx "${RUN_DIR}"
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
    echo "Refusing resume: MiDasheng worker hash changed." >&2
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
  chmod 0666 "${WORKER_HASH_FILE}"
fi

echo "DualISL RewardV2 captioner=midasheng start=base mode=${MODE} rounds=${TRAINING_ROUNDS} GPUs=8 action=${ACTION} run_dir=${RUN_DIR}"
"${DRIVER_PYTHON}" -m dual_isl_train "${ACTION}" --config "${CONFIG}"
"${DRIVER_PYTHON}" scripts/verify_tts_ddp_sync.py \
  --run-dir "${RUN_DIR}" --expected-world-size 8 \
  --require-memory-bounded-grpo
