#!/usr/bin/env bash
set -euo pipefail

# Common guarded launcher for the two new base-model Captioner branches.
# Public wrappers intentionally keep model-specific command names.
umask 000

usage() {
  echo "Usage:" >&2
  echo "  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DUALISL_RUN_DIR=/absolute/run/path $0 <midasheng_0p6b|qwen2_5_omni_3b> smoke 8" >&2
  echo "  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DUALISL_RUN_DIR=/absolute/run/path $0 <midasheng_0p6b|qwen2_5_omni_3b> train 8 <rounds>" >&2
}

if [[ $# -lt 3 || $# -gt 4 ]]; then
  usage
  exit 2
fi

MODEL_KIND=$1
MODE=$2
GPU_COUNT=$3
case "${MODE}:${GPU_COUNT}" in
  smoke:8)
    [[ $# -eq 3 ]] || { usage; exit 2; }
    TRAINING_ROUNDS=1
    ;;
  train:8)
    [[ $# -eq 4 && ${4:-} =~ ^[1-9][0-9]*$ ]] || {
      echo "Training rounds must be a positive integer." >&2
      usage
      exit 2
    }
    TRAINING_ROUNDS=$4
    ;;
  *)
    usage
    exit 2
    ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARED_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
DRIVER_PYTHON="${DUALISL_DRIVER_PYTHON:-${SHARED_ROOT}/miniconda3/envs/qwen3-tts/bin/python}"

case "${MODEL_KIND}" in
  midasheng_0p6b)
    WORKER_PATH="${PROJECT_ROOT}/scripts/midasheng_0p6b_captioner_candidate.py"
    EXPECTED_WORKER="scripts.midasheng_0p6b_captioner_candidate"
    EXPECTED_MODEL="${SHARED_ROOT}/Caption/models/MiDashengLM-0.6B-FP32"
    EXPECTED_REPO="midasheng/midashenglm-0.6b-fp32"
    EXPECTED_PYTHON="${SHARED_ROOT}/miniconda3/envs/midasheng-0p6b-captioner/bin/python"
    WORKER_HASH_FILE_NAME="midasheng_0p6b_captioner_worker.sha256"
    if [[ "${MODE}" == "smoke" ]]; then
      CONFIG="${PROJECT_ROOT}/configs/gpu_smoke_8gpu_h100_midasheng_0p6b_reward_v2.yaml"
      DEFAULT_RUN_DIR="${PROJECT_ROOT}/runs/gpu_smoke_8gpu_h100_midasheng_0p6b_reward_v2"
    else
      CONFIG="${PROJECT_ROOT}/configs/train_8gpu_h100_midasheng_0p6b_reward_v2.yaml"
      DEFAULT_RUN_DIR="${PROJECT_ROOT}/runs/dual_recursive_8gpu_h100_midasheng_0p6b_reward_v2"
    fi
    ;;
  qwen2_5_omni_3b)
    WORKER_PATH="${PROJECT_ROOT}/scripts/qwen2_5_omni_3b_captioner_candidate.py"
    EXPECTED_WORKER="scripts.qwen2_5_omni_3b_captioner_candidate"
    EXPECTED_MODEL="${SHARED_ROOT}/Caption/models/Qwen2.5-Omni-3B"
    EXPECTED_REPO="Qwen/Qwen2.5-Omni-3B"
    EXPECTED_PYTHON="${SHARED_ROOT}/miniconda3/envs/qwen2_5-omni-3b-captioner/bin/python"
    WORKER_HASH_FILE_NAME="qwen2_5_omni_3b_captioner_worker.sha256"
    if [[ "${MODE}" == "smoke" ]]; then
      CONFIG="${PROJECT_ROOT}/configs/gpu_smoke_8gpu_h100_qwen2_5_omni_3b_reward_v2.yaml"
      DEFAULT_RUN_DIR="${PROJECT_ROOT}/runs/gpu_smoke_8gpu_h100_qwen2_5_omni_3b_reward_v2"
    else
      CONFIG="${PROJECT_ROOT}/configs/train_8gpu_h100_qwen2_5_omni_3b_reward_v2.yaml"
      DEFAULT_RUN_DIR="${PROJECT_ROOT}/runs/dual_recursive_8gpu_h100_qwen2_5_omni_3b_reward_v2"
    fi
    ;;
  *)
    echo "Unknown model kind: ${MODEL_KIND}" >&2
    usage
    exit 2
    ;;
esac

RUN_DIR="${DUALISL_RUN_DIR:-${DEFAULT_RUN_DIR}}"
if [[ "${RUN_DIR}" != /* ]]; then
  echo "DUALISL_RUN_DIR must be an absolute path: ${RUN_DIR}" >&2
  exit 2
fi
for executable in "${DRIVER_PYTHON}" "${EXPECTED_PYTHON}"; do
  [[ -x "${executable}" ]] || { echo "Python is not executable: ${executable}" >&2; exit 2; }
done
[[ -f "${WORKER_PATH}" ]] || { echo "Captioner worker is missing: ${WORKER_PATH}" >&2; exit 2; }
[[ -d "${EXPECTED_MODEL}" ]] || { echo "Captioner model is missing: ${EXPECTED_MODEL}" >&2; exit 2; }
[[ -f "${EXPECTED_MODEL}/.download_provenance.json" ]] || {
  echo "Model provenance marker is missing: ${EXPECTED_MODEL}/.download_provenance.json" >&2
  exit 2
}
if [[ -n "${DUALISL_CAPTION_ADAPTER:-}" || -n "${DUALISL_TTS_ADAPTER:-}" ]]; then
  echo "New RewardV2 base runs refuse Captioner/TTS adapter overrides." >&2
  exit 2
fi
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || {
  echo "CUDA_VISIBLE_DEVICES must be set by the scheduler or caller." >&2
  exit 2
}

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
"${DRIVER_PYTHON}" - "${CONFIG}" "${TRAINING_ROUNDS}" "${MODEL_KIND}" "${EXPECTED_MODEL}" "${EXPECTED_REPO}" "${EXPECTED_PYTHON}" "${EXPECTED_WORKER}" <<'PY'
import json, pathlib, sys
from dual_isl_train.config import load_config

config = load_config(sys.argv[1])
expected_rounds = int(sys.argv[2])
kind, expected_model, expected_repo, expected_python, expected_worker = sys.argv[3:]
distributed = config["distributed"]
if not (bool(distributed.get("enabled")) and int(distributed.get("world_size", 0)) == 8 and distributed.get("backend") == "nccl"):
    raise RuntimeError("New RewardV2 launcher requires enabled 8-rank NCCL")
if int(config["training"].get("round_offset", 0)) != 0:
    raise RuntimeError("New base run requires training.round_offset=0")
if int(config["training"].get("rounds", 0)) != expected_rounds:
    raise RuntimeError("DUALISL_ROUNDS was not applied to training.rounds")
if config["captioner"].get("adapter_path") or config["tts"].get("adapter_path"):
    raise RuntimeError("New base run must not load old adapters")
if config["reward"]["calibration"]["method"] != "round0_dual_counterfactual_zscore":
    raise RuntimeError("Fresh RewardV2 calibration is not enabled")
if config["captioner"].get("worker_module") != expected_worker:
    raise RuntimeError(f"Unexpected Captioner worker for {kind}")
if pathlib.Path(config["captioner"].get("model_path", "")).resolve() != pathlib.Path(expected_model).resolve():
    raise RuntimeError("Configured Captioner model path is not the expected downloaded model")
if pathlib.Path(config["captioner"].get("python", "")).resolve() != pathlib.Path(expected_python).resolve():
    raise RuntimeError("Configured Captioner environment is not the isolated branch environment")
provenance = json.loads((pathlib.Path(expected_model) / ".download_provenance.json").read_text(encoding="utf-8"))
if provenance.get("state") != "complete" or provenance.get("repo_id") != expected_repo:
    raise RuntimeError(f"Incomplete or mismatched model provenance: {provenance}")
if config["tts"].get("codec_cache_dir", "").find("$DUALISL_RUN_DIR") >= 0:
    raise RuntimeError("Config codec cache was not expanded")
print(f"validated {kind}: model={expected_repo} worker={expected_worker}")
PY

VISIBLE_GPU_COUNT=$(${DRIVER_PYTHON} -c 'import torch; print(torch.cuda.device_count())')
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
WORKER_HASH_FILE="${RUN_DIR}/${WORKER_HASH_FILE_NAME}"
ACTION=train
if [[ -f "${RUN_DIR}/run_state.json" ]]; then
  ACTION=resume
  [[ -f "${WORKER_HASH_FILE}" ]] || { echo "Refusing resume without ${WORKER_HASH_FILE}" >&2; exit 2; }
  RECORDED_WORKER_HASH=$(awk 'NR == 1 {print $1}' "${WORKER_HASH_FILE}")
  [[ "${RECORDED_WORKER_HASH}" == "${WORKER_HASH}" ]] || {
    echo "Refusing resume: ${MODEL_KIND} worker hash changed." >&2
    exit 2
  }
elif find "${RUN_DIR}" -mindepth 1 -maxdepth 1 \
    ! -name .launcher.lock ! -name "${WORKER_HASH_FILE_NAME}" \
    -print -quit | grep -q .; then
  echo "Refusing a non-empty run directory without run_state.json: ${RUN_DIR}" >&2
  exit 2
else
  printf '%s  %s\n' "${WORKER_HASH}" "${WORKER_PATH}" >"${WORKER_HASH_FILE}"
  chmod 0666 "${WORKER_HASH_FILE}"
fi

echo "DualISL RewardV2 captioner=${MODEL_KIND} start=base mode=${MODE} rounds=${TRAINING_ROUNDS} GPUs=8 action=${ACTION} run_dir=${RUN_DIR}"
"${DRIVER_PYTHON}" -m dual_isl_train "${ACTION}" --config "${CONFIG}"
"${DRIVER_PYTHON}" scripts/verify_tts_ddp_sync.py \
  --run-dir "${RUN_DIR}" --expected-world-size 8 \
  --require-memory-bounded-grpo
