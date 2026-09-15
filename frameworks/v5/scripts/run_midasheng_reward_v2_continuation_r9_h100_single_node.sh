#!/usr/bin/env bash
set -euo pipefail

# Runs live on shared storage. Keep new continuation artifacts readable and
# writable across the training servers used by this project.
umask 000

usage() {
  echo "Usage:" >&2
  echo "  DUALISL_CONFIRM_EXCLUSIVE_GPUS=YES CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\" >&2
  echo "  DUALISL_RUN_DIR=/absolute/new/run/path $0 <smoke|train> 8" >&2
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

MODE=$1
GPU_COUNT=$2
case "${MODE}:${GPU_COUNT}" in
  smoke:8|train:8) ;;
  *)
    usage
    exit 2
    ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARED_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
SOURCE_RUN="${PROJECT_ROOT}/runs/dual_recursive_8gpu_h100_midasheng_reward_v2_20260901_run01"
SOURCE_ROUND=9
ROUND_OFFSET=10
SOURCE_CAPTION="${SOURCE_RUN}/round_009/checkpoints/caption_final"
SOURCE_TTS="${SOURCE_RUN}/round_009/checkpoints/tts_final"
SOURCE_COMMIT="${SOURCE_RUN}/round_009/commit.json"
SOURCE_LATEST="${SOURCE_RUN}/latest.json"
SOURCE_CALIBRATION="${SOURCE_RUN}/reward_calibration.json"
SOURCE_REWARD_ANCHOR="${SOURCE_RUN}/reward_anchor.json"
WORKER_PATH="${PROJECT_ROOT}/scripts/midasheng_captioner_candidate.py"
MIDASHENG_MODEL="${SHARED_ROOT}/Caption/models/MiDashengLM-7B-1021-BF16"
TTS_MODEL="${SHARED_ROOT}/Caption/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
MIDASHENG_PYTHON="${SHARED_ROOT}/miniconda3/envs/midasheng-captioner/bin/python"
DRIVER_PYTHON="${DUALISL_DRIVER_PYTHON:-${SHARED_ROOT}/miniconda3/envs/qwen3-tts/bin/python}"

if [[ "${MODE}" == "smoke" ]]; then
  CONFIG="${PROJECT_ROOT}/configs/gpu_smoke_8gpu_h100_midasheng_reward_v2_continuation_r9.yaml"
  EXPECTED_ROUNDS=1
else
  CONFIG="${PROJECT_ROOT}/configs/train_8gpu_h100_midasheng_reward_v2_continuation_r9.yaml"
  EXPECTED_ROUNDS=10
fi

if [[ -z "${DUALISL_RUN_DIR:-}" ]]; then
  echo "DUALISL_RUN_DIR is required and must name a new absolute run directory." >&2
  usage
  exit 2
fi
RUN_DIR="${DUALISL_RUN_DIR}"
if [[ "${RUN_DIR}" != /* ]]; then
  echo "DUALISL_RUN_DIR must be an absolute path: ${RUN_DIR}" >&2
  exit 2
fi
if [[ "${RUN_DIR}" == "${SOURCE_RUN}" ]]; then
  echo "Refusing to overwrite or resume the completed source run: ${SOURCE_RUN}" >&2
  exit 2
fi
if [[ "${DUALISL_CONFIRM_EXCLUSIVE_GPUS:-}" != "YES" ]]; then
  echo "Set DUALISL_CONFIRM_EXCLUSIVE_GPUS=YES only after the previous 8-GPU jobs have stopped." >&2
  exit 2
fi
if [[ -n "${DUALISL_CAPTION_ADAPTER:-}" || -n "${DUALISL_TTS_ADAPTER:-}" || -n "${DUALISL_ROUNDS:-}" ]]; then
  echo "Refusing pre-existing DUALISL_CAPTION_ADAPTER, DUALISL_TTS_ADAPTER, or DUALISL_ROUNDS overrides." >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must be set by the scheduler or caller." >&2
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
if [[ ! -d "${MIDASHENG_MODEL}" || ! -d "${TTS_MODEL}" ]]; then
  echo "A required base model directory is missing." >&2
  exit 2
fi
for path in "${SOURCE_COMMIT}" "${SOURCE_LATEST}" "${SOURCE_CALIBRATION}" "${SOURCE_REWARD_ANCHOR}"; do
  if [[ ! -r "${path}" ]]; then
    echo "Required source lineage file is missing or unreadable: ${path}" >&2
    exit 2
  fi
done
for checkpoint in "${SOURCE_CAPTION}" "${SOURCE_TTS}"; do
  if [[ ! -r "${checkpoint}/adapter_config.json" || ! -r "${checkpoint}/adapter_model.safetensors" ]]; then
    echo "Continuation checkpoint is missing or unreadable: ${checkpoint}" >&2
    echo "Fix its file permissions from the account/server that created the source run, then retry." >&2
    exit 2
  fi
done

export DUALISL_RUN_DIR="${RUN_DIR}"
export DUALISL_CAPTION_ADAPTER="${SOURCE_CAPTION}"
export DUALISL_TTS_ADAPTER="${SOURCE_TTS}"
export DUALISL_ROUNDS="${EXPECTED_ROUNDS}"
export DUALISL_SHARED_WRITABLE=1
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${PROJECT_ROOT}"
"${DRIVER_PYTHON}" -m dual_isl_train validate-config --config "${CONFIG}" >/dev/null
"${DRIVER_PYTHON}" - \
  "${CONFIG}" "${EXPECTED_ROUNDS}" "${ROUND_OFFSET}" \
  "${SOURCE_RUN}" "${SOURCE_ROUND}" "${SOURCE_CAPTION}" "${SOURCE_TTS}" \
  "${SOURCE_COMMIT}" "${SOURCE_LATEST}" "${SOURCE_CALIBRATION}" \
  "${SOURCE_REWARD_ANCHOR}" "${MIDASHENG_MODEL}" "${TTS_MODEL}" <<'PY'
import json
import sys
from pathlib import Path

from dual_isl_train.checkpoints import checkpoint_record
from dual_isl_train.config import load_config
from dual_isl_train.rewards import validate_calibration
from dual_isl_train.workers.common import validate_checkpoint_version

(
    config_path, expected_rounds, round_offset, source_run, source_round,
    caption_checkpoint, tts_checkpoint, commit_path, latest_path,
    calibration_path, reward_anchor_path, caption_model, tts_model,
) = sys.argv[1:]
expected_rounds = int(expected_rounds)
round_offset = int(round_offset)
source_round = int(source_round)

config = load_config(config_path)
distributed = config["distributed"]
if (
    not bool(distributed.get("enabled"))
    or int(distributed.get("world_size", 0)) != 8
    or distributed.get("backend") != "nccl"
):
    raise RuntimeError("RewardV2 MiDasheng continuation requires enabled 8-rank NCCL")
if int(config["training"].get("rounds", 0)) != expected_rounds:
    raise RuntimeError("Configured continuation round count does not match the launcher")
if int(config["training"].get("round_offset", -1)) != round_offset:
    raise RuntimeError("Configured continuation round offset does not match the launcher")
if Path(config["captioner"]["adapter_path"]).resolve() != Path(caption_checkpoint).resolve():
    raise RuntimeError("Configured Captioner checkpoint is not the committed source r9 checkpoint")
if Path(config["tts"]["adapter_path"]).resolve() != Path(tts_checkpoint).resolve():
    raise RuntimeError("Configured TTS checkpoint is not the committed source r9 checkpoint")
if Path(config["reward"]["calibration"]["initial_path"]).resolve() != Path(calibration_path).resolve():
    raise RuntimeError("Configured calibration is not the source run's frozen round-0 calibration")
if config["reward"].get("anchor") != {
    "captioner_adapter_path": "", "tts_adapter_path": "",
}:
    raise RuntimeError("Continuation must preserve the source run's base-model reward anchors")
if config["reward"]["calibration"]["method"] != "round0_dual_counterfactual_zscore":
    raise RuntimeError("RewardV2 calibration method is not enabled")
if config["captioner"]["worker_module"] != "scripts.midasheng_captioner_candidate":
    raise RuntimeError("Configured Captioner is not MiDasheng-7B")
if Path(config["captioner"]["model_path"]).resolve() != Path(caption_model).resolve():
    raise RuntimeError("Configured Captioner base model changed")
if Path(config["tts"]["model_path"]).resolve() != Path(tts_model).resolve():
    raise RuntimeError("Configured TTS base model changed")

with open(commit_path, encoding="utf-8") as handle:
    commit = json.load(handle)
with open(latest_path, encoding="utf-8") as handle:
    latest = json.load(handle)
if commit != latest:
    raise RuntimeError("Source latest.json does not exactly match the final r9 commit")
if int(commit.get("round", -1)) != source_round or commit.get("same_round_start") is not True:
    raise RuntimeError("Source r9 commit is not a valid same-round dual checkpoint commit")
for role, checkpoint in (("captioner", caption_checkpoint), ("tts", tts_checkpoint)):
    validate_checkpoint_version(checkpoint)
    actual = checkpoint_record(checkpoint)
    if commit.get(role) != actual:
        raise RuntimeError(f"Source {role} checkpoint path/hash does not match the r9 commit")

with open(calibration_path, encoding="utf-8") as handle:
    calibration = json.load(handle)
validate_calibration(calibration)
if (
    calibration.get("method") != "round0_dual_counterfactual_zscore"
    or calibration.get("version") != 2
    or calibration.get("fitted_round") != 0
    or calibration.get("frozen_across_rounds") is not True
):
    raise RuntimeError("Continuation requires the original frozen RewardV2 round-0 calibration")

with open(reward_anchor_path, encoding="utf-8") as handle:
    reward_anchor = json.load(handle)
expected_anchor = {
    "kind": "frozen_recursive_start_anchor",
    "captioner_adapter": None,
    "tts_adapter": None,
    "captioner_model_path": str(Path(caption_model).resolve()),
    "tts_model_path": str(Path(tts_model).resolve()),
    "updated_during_training": False,
}
if reward_anchor != expected_anchor:
    raise RuntimeError("Source reward_anchor.json is not the original base-model C_0/T_0 anchor")
if Path(source_run).resolve() in Path(config["run"]["output_dir"]).resolve().parents or (
    Path(config["run"]["output_dir"]).resolve() == Path(source_run).resolve()
):
    raise RuntimeError("Continuation output must not overwrite the completed source run")
PY

VISIBLE_GPU_COUNT=$("${DRIVER_PYTHON}" -c 'import torch; print(torch.cuda.device_count())')
if [[ "${VISIBLE_GPU_COUNT}" != "${GPU_COUNT}" ]]; then
  echo "Expected ${GPU_COUNT} visible GPUs, but PyTorch sees ${VISIBLE_GPU_COUNT}." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required for the exclusive-GPU preflight." >&2
  exit 2
fi
MAX_USED_MIB="${DUALISL_MAX_USED_MIB:-2048}"
if [[ ! "${MAX_USED_MIB}" =~ ^[0-9]+$ ]]; then
  echo "DUALISL_MAX_USED_MIB must be a non-negative integer." >&2
  exit 2
fi
mapfile -t USED_MIB < <(
  nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-gpu=memory.used --format=csv,noheader,nounits \
    | sed 's/[[:space:]]//g'
)
if [[ ${#USED_MIB[@]} -ne 8 ]]; then
  echo "Expected 8 memory readings from nvidia-smi, got ${#USED_MIB[@]}." >&2
  exit 2
fi
for index in "${!USED_MIB[@]}"; do
  if [[ ! "${USED_MIB[$index]}" =~ ^[0-9]+$ || ${USED_MIB[$index]} -gt ${MAX_USED_MIB} ]]; then
    echo "GPU ${index} is not idle enough: used=${USED_MIB[$index]} MiB, limit=${MAX_USED_MIB} MiB." >&2
    echo "Stop the previous jobs and wait for their CUDA contexts to exit before retrying." >&2
    exit 2
  fi
done

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
  if [[ ! -r "${WORKER_HASH_FILE}" ]]; then
    echo "Refusing resume without readable ${WORKER_HASH_FILE}" >&2
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

echo "DualISL RewardV2 captioner=midasheng continuation=r9 rounds=${EXPECTED_ROUNDS} offset=${ROUND_OFFSET} mode=${MODE} GPUs=8 action=${ACTION} run_dir=${RUN_DIR}"
"${DRIVER_PYTHON}" -m dual_isl_train "${ACTION}" --config "${CONFIG}"
"${DRIVER_PYTHON}" scripts/verify_tts_ddp_sync.py \
  --run-dir "${RUN_DIR}" --expected-world-size 8 \
  --require-memory-bounded-grpo
