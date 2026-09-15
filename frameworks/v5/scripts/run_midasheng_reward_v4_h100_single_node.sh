#!/usr/bin/env bash
set -euo pipefail
umask 000

usage() {
  echo "Usage: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DUALISL_RUN_DIR=/absolute/V4/runs/name $0 smoke 8 | replay-smoke 8 | train 8 <rounds>" >&2
}
if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 2
fi
MODE=$1
GPU_COUNT=$2
case "${MODE}:${GPU_COUNT}" in
  replay-smoke:8)
    if [[ $# -ne 2 ]]; then usage; exit 2; fi
    TRAINING_ROUNDS=2
    CONFIG_STEM=replay_smoke
    ;;
  smoke:8)
    if [[ $# -ne 2 ]]; then usage; exit 2; fi
    TRAINING_ROUNDS=1
    CONFIG_STEM=gpu_smoke
    ;;
  train:8)
    if [[ $# -ne 3 || ! ${3:-} =~ ^[1-9][0-9]*$ ]]; then usage; exit 2; fi
    TRAINING_ROUNDS=$3
    CONFIG_STEM=train
    ;;
  *) usage; exit 2 ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARED_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
DRIVER_PYTHON="${DUALISL_DRIVER_PYTHON:-${SHARED_ROOT}/miniconda3/envs/qwen3-tts/bin/python}"
CONFIG="${PROJECT_ROOT}/configs/${CONFIG_STEM}_8gpu_h100_midasheng_reward_v4.yaml"
RUN_DIR="${DUALISL_RUN_DIR:-${PROJECT_ROOT}/runs/${CONFIG_STEM}_8gpu_h100_midasheng_reward_v4}"
if [[ "${RUN_DIR}" != /* ]]; then
  echo "DUALISL_RUN_DIR must be absolute." >&2
  exit 2
fi
if [[ ! -x "${DRIVER_PYTHON}" ]]; then
  echo "Driver Python is not executable: ${DRIVER_PYTHON}" >&2
  exit 2
fi
if [[ -n "${DUALISL_CAPTION_ADAPTER:-}" || -n "${DUALISL_TTS_ADAPTER:-}" ]]; then
  echo "Unset DUALISL_CAPTION_ADAPTER and DUALISL_TTS_ADAPTER: this V4 run starts from base." >&2
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

# Check the destination and identity before creating files or touching permissions.
"${DRIVER_PYTHON}" -m scripts.reward_v4_run_guard \
  --config "${CONFIG}" --mode "${MODE}" --rounds "${TRAINING_ROUNDS}" >/dev/null
"${DRIVER_PYTHON}" - "${CONFIG}" <<'PY'
import os
import sys
from pathlib import Path
from dual_isl_train.config import load_config

config = load_config(sys.argv[1])
for role in ("captioner", "tts", "critics"):
    interpreter = config[role]["python"]
    if not os.access(interpreter, os.X_OK):
        raise RuntimeError(f"Missing executable {role} Python: {interpreter}")
for role, key in (("captioner", "model_path"), ("tts", "model_path"),
                  ("tts", "tokenizer_path"), ("critics", "whisper_model")):
    if not Path(config[role][key]).exists():
        raise RuntimeError(f"Missing {role}.{key}: {config[role][key]}")
for key in ("paired_path", "audio_only_path", "caption_only_path"):
    if not Path(config["data"][key]).is_file():
        raise RuntimeError(f"Missing data.{key}: {config['data'][key]}")
PY
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to the eight GPUs allocated to this job." >&2
  exit 2
fi
VISIBLE_GPU_COUNT=$("${DRIVER_PYTHON}" -c 'import torch; print(torch.cuda.device_count())')
if [[ "${VISIBLE_GPU_COUNT}" != "8" ]]; then
  echo "Expected 8 visible GPUs; PyTorch sees ${VISIBLE_GPU_COUNT}." >&2
  exit 2
fi

mkdir -p "${RUN_DIR}"
exec 9>"${RUN_DIR}/.launcher.lock"
if ! flock -n 9; then
  echo "Another launcher already holds ${RUN_DIR}/.launcher.lock" >&2
  exit 2
fi
ACTION=$("${DRIVER_PYTHON}" -m scripts.reward_v4_run_guard \
  --config "${CONFIG}" --mode "${MODE}" --rounds "${TRAINING_ROUNDS}" --reserve)
chmod a+rwx "${RUN_DIR}"
echo "DualISL RewardV4 captioner=MiDasheng-7B start=base mode=${MODE} rounds=${TRAINING_ROUNDS} GPUs=8 action=${ACTION} run_dir=${RUN_DIR}"
"${DRIVER_PYTHON}" -m dual_isl_train "${ACTION}" --config "${CONFIG}"
"${DRIVER_PYTHON}" scripts/verify_tts_ddp_sync.py \
  --run-dir "${RUN_DIR}" --expected-world-size 8 --require-memory-bounded-grpo
if [[ "${MODE}" == "replay-smoke" ]]; then
  "${DRIVER_PYTHON}" scripts/verify_tts_reference_reuse.py --run-dir "${RUN_DIR}"
fi
