#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARED_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
SOURCE_RUN_DIR="${DUALISL_SOURCE_RUN_DIR:?Set DUALISL_SOURCE_RUN_DIR to the stopped 7-GPU run}"
STRESS_DIR="${DUALISL_STRESS_DIR:?Set DUALISL_STRESS_DIR to a new absolute directory}"
TTS_PYTHON="${SHARED_ROOT}/miniconda3/envs/qwen3-tts/bin/python"
CONFIG="${PROJECT_ROOT}/configs/gpu_smoke_8gpu_h100.yaml"
SOURCE_INPUT="${SOURCE_RUN_DIR}/round_000/training/round_000_tts_grpo.input.jsonl"

if [[ "${SOURCE_RUN_DIR}" != /* || "${STRESS_DIR}" != /* ]]; then
  echo "DUALISL_SOURCE_RUN_DIR and DUALISL_STRESS_DIR must be absolute paths" >&2
  exit 2
fi
if [[ ! -f "${SOURCE_INPUT}" ]]; then
  echo "Missing source TTS GRPO input: ${SOURCE_INPUT}" >&2
  exit 2
fi
if [[ ! -r "${SOURCE_INPUT}" ]]; then
  echo "Source TTS GRPO input is not readable from this server: ${SOURCE_INPUT}" >&2
  echo "On the old 7-GPU server, run: chmod 0644 '${SOURCE_INPUT}'" >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must expose exactly eight GPUs" >&2
  exit 2
fi
if [[ "$(CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${TTS_PYTHON}" -c 'import torch; print(torch.cuda.device_count())')" != "8" ]]; then
  echo "The TTS environment does not see exactly eight GPUs" >&2
  exit 2
fi

mkdir -p "${STRESS_DIR}"
exec 9>"${STRESS_DIR}/.launcher.lock"
if ! flock -n 9; then
  echo "Another stress launcher holds ${STRESS_DIR}/.launcher.lock" >&2
  exit 2
fi
if find "${STRESS_DIR}" -mindepth 1 -maxdepth 1 ! -name .launcher.lock -print -quit | grep -q .; then
  echo "Refusing non-empty stress directory: ${STRESS_DIR}" >&2
  exit 2
fi

export DUALISL_RUN_DIR="${STRESS_DIR}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

INPUT="${STRESS_DIR}/round_000/training/round_000_tts_grpo.input.jsonl"
OUTPUT="${STRESS_DIR}/round_000/training/round_000_tts_grpo.output.jsonl"
CHECKPOINT="${STRESS_DIR}/round_000/checkpoints/tts_after_grpo"
mkdir -p "$(dirname "${INPUT}")" "$(dirname "${CHECKPOINT}")"

cd "${PROJECT_ROOT}"
"${TTS_PYTHON}" scripts/prepare_tts_grpo_stress_input.py \
  --input "${SOURCE_INPUT}" --output "${INPUT}" --world-size 8
"${TTS_PYTHON}" -m torch.distributed.run --standalone --nproc-per-node=8 \
  --module dual_isl_train.workers.qwen_voice_design grpo-update \
  --config "${CONFIG}" --input "${INPUT}" --output "${OUTPUT}" \
  --checkpoint-out "${CHECKPOINT}" --training-phase grpo
"${TTS_PYTHON}" scripts/verify_tts_ddp_sync.py \
  --run-dir "${STRESS_DIR}" --expected-world-size 8 --require-memory-bounded-grpo
