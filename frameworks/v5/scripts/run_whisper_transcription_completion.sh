#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARED_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
CRITIC_PYTHON="${DUALISL_CRITIC_PYTHON:-${SHARED_ROOT}/miniconda3/envs/dualisl-critic/bin/python}"
SCOPE="${1:-all}"

if [[ "${SCOPE}" != "all" && "${SCOPE}" != "training-required" ]]; then
  echo "Usage: CUDA_VISIBLE_DEVICES=<one GPU> $0 [all|training-required]" >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must expose exactly one GPU." >&2
  exit 2
fi
if [[ ! -x "${CRITIC_PYTHON}" ]]; then
  echo "Critic Python is not executable: ${CRITIC_PYTHON}" >&2
  exit 2
fi

VISIBLE_GPU_COUNT=$("${CRITIC_PYTHON}" -c 'import torch; print(torch.cuda.device_count())')
if [[ "${VISIBLE_GPU_COUNT}" != "1" ]]; then
  echo "Expected exactly one visible GPU, but PyTorch sees ${VISIBLE_GPU_COUNT}." >&2
  exit 2
fi

mkdir -p "${PROJECT_ROOT}/runs" "${PROJECT_ROOT}/reports"
exec 9>"${PROJECT_ROOT}/runs/whisper_transcription_completion.lock"
if ! flock -n 9; then
  echo "Another Whisper completion process is already running." >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "${PROJECT_ROOT}"
"${CRITIC_PYTHON}" scripts/fill_missing_transcriptions_whisper.py \
  --scope "${SCOPE}" 2>&1 | tee -a "reports/whisper_transcription_completion.console.log"
