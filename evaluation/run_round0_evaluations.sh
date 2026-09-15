#!/usr/bin/env bash
set -euo pipefail

# Resumable evaluation driver for committed DualISL checkpoints.
# Paid Gemini calls are deliberately isolated behind *-paid subcommands.

BENCHMARK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CAPTION_ROOT="$(cd -- "${BENCHMARK_ROOT}/.." && pwd)"
CLUSTER_ROOT="$(cd -- "${CAPTION_ROOT}/.." && pwd)"

PSC_ROOT="${BENCHMARK_ROOT}/paraspeechcaps/evaluations/qwen3_captioner_attr6"
TTS_EVAL_ROOT="${BENCHMARK_ROOT}/EmergentTTS-Eval-public/qwen3_voice_design"
TRAIN_ROOT="${CAPTION_ROOT}/DualISL_Train/runs"
ROUND_COMPLETION_ROOT="${BENCHMARK_ROOT}/round_completion_eval_runs_20260831"
PSC_REMAINING_ROOT="${ROUND_COMPLETION_ROOT}/paraspeechcaps"
PSC_REMAINING_SMOKE_ROOT="${PSC_REMAINING_ROOT}/shared_smoke"
PSC_REMAINING_SMOKE_MANIFEST="${PSC_REMAINING_SMOKE_ROOT}/manifest.jsonl"

QWEN_CAPTION_PY="${CLUSTER_ROOT}/miniconda3/envs/qwen3-captioner/bin/python"
MIDASHENG_PY="${CLUSTER_ROOT}/miniconda3/envs/midasheng-captioner/bin/python"
QWEN_CAPTION_BASE="${CAPTION_ROOT}/models/Qwen3-Omni-30B-A3B-Captioner"
MIDASHENG_BASE="${CAPTION_ROOT}/models/MiDashengLM-7B-1021-BF16"
TTS_BASE="${CAPTION_ROOT}/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign"d

QWEN8_RUN="${TRAIN_ROOT}/dual_recursive_8gpu_h100_20260829_run01"
MIDASHENG4_RUN="${TRAIN_ROOT}/dual_recursive_4gpu_h100_midasheng_20260830_run01"
QWEN8_CAPTION_ADAPTER="${QWEN8_RUN}/round_000/checkpoints/caption_final"
QWEN8_TTS_ADAPTER="${QWEN8_RUN}/round_000/checkpoints/tts_final"
QWEN8_ROUND1_CAPTION_ADAPTER="${QWEN8_RUN}/round_001/checkpoints/caption_final"
QWEN8_ROUND1_TTS_ADAPTER="${QWEN8_RUN}/round_001/checkpoints/tts_final"
QWEN8_ROUND2_CAPTION_ADAPTER="${QWEN8_RUN}/round_002/checkpoints/caption_final"
QWEN8_ROUND2_TTS_ADAPTER="${QWEN8_RUN}/round_002/checkpoints/tts_final"
MIDASHENG4_CAPTION_ADAPTER="${MIDASHENG4_RUN}/round_000/checkpoints/caption_final"
MIDASHENG4_TTS_ADAPTER="${MIDASHENG4_RUN}/round_000/checkpoints/tts_final"
MIDASHENG4_ROUND1_CAPTION_ADAPTER="${MIDASHENG4_RUN}/round_001/checkpoints/caption_final"
MIDASHENG4_ROUND1_TTS_ADAPTER="${MIDASHENG4_RUN}/round_001/checkpoints/tts_final"
MIDASHENG4_ROUND2_CAPTION_ADAPTER="${MIDASHENG4_RUN}/round_002/checkpoints/caption_final"
MIDASHENG4_ROUND2_TTS_ADAPTER="${MIDASHENG4_RUN}/round_002/checkpoints/tts_final"

PSC_MANIFEST="${PSC_ROOT}/runs/new_cluster_default/manifest.jsonl"
PSC_SMOKE_ROOT="${PSC_ROOT}/runs/round0_eval_shared_smoke_20260830"
PSC_SMOKE_MANIFEST="${PSC_SMOKE_ROOT}/manifest.jsonl"

usage() {
  cat <<'EOF'
Usage: bash run_round0_evaluations.sh COMMAND [TTS_BRANCH]

Commands, in recommended order:
  check
  caption-smoke
  caption-full
  caption-remaining-smoke
  caption-remaining-full
  caption-content-smoke-8gpu
  caption-content-full-8gpu
  tts-smoke-local qwen8_r0
  tts-smoke-paid  qwen8_r0
  tts-smoke-local qwen8_r1
  tts-smoke-paid  qwen8_r1
  tts-smoke-local qwen8_r2
  tts-smoke-paid  qwen8_r2
  tts-smoke-local midasheng4_r0
  tts-smoke-paid  midasheng4_r0
  tts-smoke-local midasheng4_r1
  tts-smoke-paid  midasheng4_r1
  tts-smoke-local midasheng4_r2
  tts-smoke-paid  midasheng4_r2
  tts-full-local  qwen8_r0
  tts-full-local  qwen8_r1
  tts-full-local  qwen8_r2
  tts-full-local  midasheng4_r0
  tts-full-local  midasheng4_r1
  tts-full-local  midasheng4_r2
  tts-full-paid   qwen8_r0
  tts-full-paid   qwen8_r1
  tts-full-paid   qwen8_r2
  tts-full-paid   midasheng4_r0
  tts-full-paid   midasheng4_r1
  tts-full-paid   midasheng4_r2

The *-paid commands call the configured Gemini API. All other commands do not.
The caption-remaining-* commands run three cases in parallel on physical GPUs 0, 1, and 2.
Override them with CAPTION_GPU_QWEN8_R2, CAPTION_GPU_MIDASHENG4_R1, and
CAPTION_GPU_MIDASHENG4_R2. The three selected GPU IDs must be distinct.
The caption-content-*-8gpu commands run the separate Content-only Scheme A
protocol for all eight Caption candidates concurrently on physical GPUs 0..7.
Override their order with CONTENT_SCHEME_A_GPUS=0,1,2,3,4,5,6,7.
Set TTS_GPU=N to select the physical GPU for a *-local TTS command (default: 0).
TTS_BRANCH is qwen8_r0/r1/r2 or midasheng4_r0/r1/r2.
EOF
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing required directory: $1" >&2
    exit 1
  fi
}

require_executable() {
  if [[ ! -x "$1" ]]; then
    echo "Missing executable: $1" >&2
    exit 1
  fi
}

adapter_for_tts_branch() {
  case "${1:-}" in
    qwen8|qwen8_r0) printf '%s\n' "${QWEN8_TTS_ADAPTER}" ;;
    qwen8_r1) printf '%s\n' "${QWEN8_ROUND1_TTS_ADAPTER}" ;;
    qwen8_r2) printf '%s\n' "${QWEN8_ROUND2_TTS_ADAPTER}" ;;
    midasheng4|midasheng4_r0) printf '%s\n' "${MIDASHENG4_TTS_ADAPTER}" ;;
    midasheng4_r1) printf '%s\n' "${MIDASHENG4_ROUND1_TTS_ADAPTER}" ;;
    midasheng4_r2) printf '%s\n' "${MIDASHENG4_ROUND2_TTS_ADAPTER}" ;;
    *)
      echo "Second argument must be qwen8_r0/r1/r2 or midasheng4_r0/r1/r2." >&2
      usage >&2
      exit 2
      ;;
  esac
}

label_for_tts_branch() {
  case "${1:-}" in
    qwen8|qwen8_r0) printf '%s\n' qwen8_round0 ;;
    qwen8_r1) printf '%s\n' qwen8_round1 ;;
    qwen8_r2) printf '%s\n' qwen8_round2 ;;
    midasheng4|midasheng4_r0) printf '%s\n' midasheng4_round0 ;;
    midasheng4_r1) printf '%s\n' midasheng4_round1 ;;
    midasheng4_r2) printf '%s\n' midasheng4_round2 ;;
    *)
      echo "Second argument must be qwen8_r0/r1/r2 or midasheng4_r0/r1/r2." >&2
      exit 2
      ;;
  esac
}

tts_run_root() {
  local branch="$1"
  local size="$2"
  local label
  label="$(label_for_tts_branch "${branch}")"
  local evaluation_date=20260830
  case "${branch}" in
    qwen8_r2|midasheng4_r1|midasheng4_r2) evaluation_date=20260831 ;;
  esac
  printf '%s\n' "${TTS_EVAL_ROOT}/runs/eval_${label}_${size}_${evaluation_date}_run01"
}

tts_paid_stage_root() {
  local branch="$1"
  local size="$2"
  local run_root
  run_root="$(tts_run_root "${branch}" "${size}")"

  # Every full paid evaluation runs on the API server and writes beside the
  # GPU server's local run. Some historical local run directories are not
  # writable across QuarkFS projects even when the numeric UID is root.
  if [[ "${size}" == "full" ]]; then
    printf '%s\n' "${TTS_EVAL_ROOT}/runs/$(basename "${run_root}")_paid_evaluation"
    return
  fi

  case "${branch}" in
    qwen8_r2|midasheng4_r1|midasheng4_r2)
      printf '%s\n' "${TTS_EVAL_ROOT}/runs/$(basename "${run_root}")_paid_evaluation"
      ;;
    *)
      printf '%s\n' "${run_root}/paid_evaluation"
      ;;
  esac
}

check_inputs() {
  require_executable "${QWEN_CAPTION_PY}"
  require_executable "${MIDASHENG_PY}"
  require_file "${PSC_MANIFEST}"
  require_dir "${QWEN_CAPTION_BASE}"
  require_dir "${MIDASHENG_BASE}"
  require_dir "${TTS_BASE}"
  require_dir "${QWEN8_CAPTION_ADAPTER}"
  require_dir "${QWEN8_TTS_ADAPTER}"
  require_dir "${QWEN8_ROUND1_CAPTION_ADAPTER}"
  require_dir "${QWEN8_ROUND1_TTS_ADAPTER}"
  require_dir "${QWEN8_ROUND2_CAPTION_ADAPTER}"
  require_dir "${QWEN8_ROUND2_TTS_ADAPTER}"
  require_file "${QWEN8_RUN}/round_000/commit.json"
  require_file "${QWEN8_RUN}/round_001/commit.json"
  require_file "${QWEN8_RUN}/round_002/commit.json"
  require_dir "${MIDASHENG4_CAPTION_ADAPTER}"
  require_dir "${MIDASHENG4_TTS_ADAPTER}"
  require_dir "${MIDASHENG4_ROUND1_CAPTION_ADAPTER}"
  require_dir "${MIDASHENG4_ROUND1_TTS_ADAPTER}"
  require_dir "${MIDASHENG4_ROUND2_CAPTION_ADAPTER}"
  require_dir "${MIDASHENG4_ROUND2_TTS_ADAPTER}"
  require_file "${MIDASHENG4_RUN}/round_000/commit.json"
  require_file "${MIDASHENG4_RUN}/round_001/commit.json"
  require_file "${MIDASHENG4_RUN}/round_002/commit.json"
  require_file "${QWEN8_CAPTION_ADAPTER}/adapter_model.safetensors"
  require_file "${QWEN8_ROUND1_CAPTION_ADAPTER}/adapter_model.safetensors"
  require_file "${QWEN8_ROUND2_CAPTION_ADAPTER}/adapter_model.safetensors"
  require_file "${MIDASHENG4_CAPTION_ADAPTER}/adapter_model.safetensors"
  require_file "${MIDASHENG4_ROUND1_CAPTION_ADAPTER}/adapter_model.safetensors"
  require_file "${MIDASHENG4_ROUND2_CAPTION_ADAPTER}/adapter_model.safetensors"

  local failed=0
  local weight
  for weight in \
    "${QWEN8_TTS_ADAPTER}/adapter_model.safetensors" \
    "${QWEN8_ROUND1_TTS_ADAPTER}/adapter_model.safetensors" \
    "${QWEN8_ROUND2_TTS_ADAPTER}/adapter_model.safetensors" \
    "${MIDASHENG4_TTS_ADAPTER}/adapter_model.safetensors" \
    "${MIDASHENG4_ROUND1_TTS_ADAPTER}/adapter_model.safetensors" \
    "${MIDASHENG4_ROUND2_TTS_ADAPTER}/adapter_model.safetensors"
  do
    if [[ ! -r "${weight}" ]]; then
      echo "Unreadable TTS adapter: ${weight}" >&2
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "Run the chmod 0644 commands from README step 0 on the owner/training server, then retry check." >&2
    exit 1
  fi

  "${QWEN_CAPTION_PY}" - "${BENCHMARK_ROOT}" \
    "${QWEN_CAPTION_BASE}" "${QWEN8_CAPTION_ADAPTER}" \
    "${QWEN8_ROUND1_CAPTION_ADAPTER}" "${QWEN8_ROUND2_CAPTION_ADAPTER}" \
    "${MIDASHENG_BASE}" "${MIDASHENG4_CAPTION_ADAPTER}" \
    "${MIDASHENG4_ROUND1_CAPTION_ADAPTER}" "${MIDASHENG4_ROUND2_CAPTION_ADAPTER}" \
    "${TTS_BASE}" "${QWEN8_TTS_ADAPTER}" "${QWEN8_ROUND1_TTS_ADAPTER}" \
    "${QWEN8_ROUND2_TTS_ADAPTER}" "${MIDASHENG4_TTS_ADAPTER}" \
    "${MIDASHENG4_ROUND1_TTS_ADAPTER}" "${MIDASHENG4_ROUND2_TTS_ADAPTER}" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from model_adapter_utils import describe_adapter

for base_index, adapter_index in (
    (2, 3), (2, 4), (2, 5),
    (6, 7), (6, 8), (6, 9),
    (10, 11), (10, 12), (10, 13), (10, 14), (10, 15), (10, 16),
):
    descriptor = describe_adapter(Path(sys.argv[adapter_index]), Path(sys.argv[base_index]))
    print(f"OK adapter={descriptor['path']}")
    print(f"   sha256={descriptor['weights_sha256']}")
PY

  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
  local gpu0_used_mib
  gpu0_used_mib="$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
  if [[ ! "${gpu0_used_mib}" =~ ^[0-9]+$ ]] || (( gpu0_used_mib > 2048 )); then
    echo "GPU 0 is not sufficiently idle (memory.used=${gpu0_used_mib} MiB)." >&2
    echo "Wait for the current GPU job to finish, then rerun check." >&2
    exit 1
  fi
  echo "[DONE] All models, adapters, environments, manifest, and TTS permissions passed."
}

prepare_caption_smoke_manifest() {
  mkdir -p "${PSC_SMOKE_ROOT}"
  sed -n '1p' "${PSC_MANIFEST}" > "${PSC_SMOKE_MANIFEST}"
}

prepare_caption_remaining_smoke_manifest() {
  umask 000
  mkdir -p "${PSC_REMAINING_SMOKE_ROOT}"
  sed -n '1p' "${PSC_MANIFEST}" > "${PSC_REMAINING_SMOKE_MANIFEST}"
}

run_caption_case() {
  local name="$1"
  local python="$2"
  local runner="$3"
  local model="$4"
  local adapter="$5"
  local manifest="$6"
  local run_root="$7"
  local score_program="$8"
  local caption_gpu="${9:-${CAPTION_GPU:-0}}"

  if [[ ! "${caption_gpu}" =~ ^[0-9]+$ ]]; then
    echo "Caption GPU must be a non-negative physical GPU index; got: ${caption_gpu}" >&2
    return 2
  fi

  local adapter_args=()
  if [[ -n "${adapter}" ]]; then
    adapter_args=(--adapter-dir "${adapter}")
  fi
  "${python}" "${runner}" \
    --manifest "${manifest}" \
    --model-dir "${model}" \
    "${adapter_args[@]}" \
    --output-dir "${run_root}/outputs" \
    --gpu-list "${caption_gpu}" \
    --resume
  "${QWEN_CAPTION_PY}" "${score_program}" \
    --manifest "${manifest}" \
    --predictions "${run_root}/outputs/predictions.jsonl" \
    --output-dir "${run_root}/reports"
  echo "[DONE] Caption case ${name}: ${run_root}"
}

run_cross_server_caption_case() {
  local run_root="$7"
  (
    umask 000
    mkdir -p "${run_root}/outputs" "${run_root}/reports"
    run_caption_case "$@"
  )
}

run_caption_remaining_parallel() {
  local size="$1"
  local manifest="$2"
  local score_program="$3"
  local qwen_gpu="${CAPTION_GPU_QWEN8_R2:-0}"
  local midasheng_r1_gpu="${CAPTION_GPU_MIDASHENG4_R1:-1}"
  local midasheng_r2_gpu="${CAPTION_GPU_MIDASHENG4_R2:-2}"
  local log_root="${PSC_REMAINING_ROOT}/logs/${size}"
  local gpu

  for gpu in "${qwen_gpu}" "${midasheng_r1_gpu}" "${midasheng_r2_gpu}"; do
    if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
      echo "Caption GPU IDs must be non-negative integers; got: ${gpu}" >&2
      return 2
    fi
  done
  if [[ "${qwen_gpu}" == "${midasheng_r1_gpu}" || \
        "${qwen_gpu}" == "${midasheng_r2_gpu}" || \
        "${midasheng_r1_gpu}" == "${midasheng_r2_gpu}" ]]; then
    echo "The three parallel Caption cases must use distinct physical GPU IDs." >&2
    echo "Selected GPUs: qwen8_r2=${qwen_gpu}, midasheng4_r1=${midasheng_r1_gpu}, midasheng4_r2=${midasheng_r2_gpu}" >&2
    return 2
  fi

  umask 000
  mkdir -p "${PSC_REMAINING_ROOT}" "${log_root}"

  local -a names=(qwen8_round2 midasheng4_round1 midasheng4_round2)
  local -a gpus=("${qwen_gpu}" "${midasheng_r1_gpu}" "${midasheng_r2_gpu}")
  local -a logs=(
    "${log_root}/qwen8_round2.log"
    "${log_root}/midasheng4_round1.log"
    "${log_root}/midasheng4_round2.log"
  )
  local -a pids=()

  echo "[START] Parallel Caption ${size} evaluations"
  echo "[START] qwen8_round2 -> physical GPU ${qwen_gpu}; log=${logs[0]}"
  echo "[START] midasheng4_round1 -> physical GPU ${midasheng_r1_gpu}; log=${logs[1]}"
  echo "[START] midasheng4_round2 -> physical GPU ${midasheng_r2_gpu}; log=${logs[2]}"

  run_cross_server_caption_case \
    qwen8_round2 "${QWEN_CAPTION_PY}" "${PSC_ROOT}/run_qwen3_captioner.py" \
    "${QWEN_CAPTION_BASE}" "${QWEN8_ROUND2_CAPTION_ADAPTER}" "${manifest}" \
    "${PSC_REMAINING_ROOT}/eval_qwen8_round2_${size}_20260831_run01" \
    "${score_program}" "${qwen_gpu}" >"${logs[0]}" 2>&1 &
  pids+=("$!")

  run_cross_server_caption_case \
    midasheng4_round1 "${MIDASHENG_PY}" "${PSC_ROOT}/run_midasheng_captioner.py" \
    "${MIDASHENG_BASE}" "${MIDASHENG4_ROUND1_CAPTION_ADAPTER}" "${manifest}" \
    "${PSC_REMAINING_ROOT}/eval_midasheng4_round1_${size}_20260831_run01" \
    "${score_program}" "${midasheng_r1_gpu}" >"${logs[1]}" 2>&1 &
  pids+=("$!")

  run_cross_server_caption_case \
    midasheng4_round2 "${MIDASHENG_PY}" "${PSC_ROOT}/run_midasheng_captioner.py" \
    "${MIDASHENG_BASE}" "${MIDASHENG4_ROUND2_CAPTION_ADAPTER}" "${manifest}" \
    "${PSC_REMAINING_ROOT}/eval_midasheng4_round2_${size}_20260831_run01" \
    "${score_program}" "${midasheng_r2_gpu}" >"${logs[2]}" 2>&1 &
  pids+=("$!")

  terminate_caption_children() {
    trap - INT TERM
    echo "[STOP] Terminating parallel Caption evaluations..." >&2
    local pid
    for pid in "${pids[@]}"; do
      kill "${pid}" 2>/dev/null || true
    done
    wait || true
    exit 130
  }
  trap terminate_caption_children INT TERM

  local failed=0
  local index status
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      echo "[DONE] Caption ${size} ${names[$index]} on physical GPU ${gpus[$index]}"
    else
      status=$?
      echo "[FAILED] Caption ${size} ${names[$index]} exited with status ${status}; log=${logs[$index]}" >&2
      failed=1
    fi
  done
  trap - INT TERM

  if (( failed != 0 )); then
    echo "[STOP] At least one parallel Caption ${size} evaluation failed." >&2
    return 1
  fi
  echo "[DONE] All parallel Caption ${size} evaluations completed."
}

run_caption_smoke() {
  prepare_caption_smoke_manifest
  run_caption_case \
    qwen8 "${QWEN_CAPTION_PY}" "${PSC_ROOT}/run_qwen3_captioner.py" \
    "${QWEN_CAPTION_BASE}" "${QWEN8_CAPTION_ADAPTER}" "${PSC_SMOKE_MANIFEST}" \
    "${PSC_ROOT}/runs/eval_qwen8_round0_smoke_20260830_run01" \
    "${PSC_ROOT}/score_smoke.py"
  run_caption_case \
    qwen8_round1 "${QWEN_CAPTION_PY}" "${PSC_ROOT}/run_qwen3_captioner.py" \
    "${QWEN_CAPTION_BASE}" "${QWEN8_ROUND1_CAPTION_ADAPTER}" "${PSC_SMOKE_MANIFEST}" \
    "${PSC_ROOT}/runs/eval_qwen8_round1_smoke_20260830_run01" \
    "${PSC_ROOT}/score_smoke.py"
  run_caption_case \
    midasheng_base "${MIDASHENG_PY}" "${PSC_ROOT}/run_midasheng_captioner.py" \
    "${MIDASHENG_BASE}" "" "${PSC_SMOKE_MANIFEST}" \
    "${PSC_ROOT}/runs/eval_midasheng_base_smoke_20260830_run01" \
    "${PSC_ROOT}/score_smoke.py"
  run_caption_case \
    midasheng4 "${MIDASHENG_PY}" "${PSC_ROOT}/run_midasheng_captioner.py" \
    "${MIDASHENG_BASE}" "${MIDASHENG4_CAPTION_ADAPTER}" "${PSC_SMOKE_MANIFEST}" \
    "${PSC_ROOT}/runs/eval_midasheng4_round0_smoke_20260830_run01" \
    "${PSC_ROOT}/score_smoke.py"
}

run_caption_full() {
  run_caption_case \
    qwen8 "${QWEN_CAPTION_PY}" "${PSC_ROOT}/run_qwen3_captioner.py" \
    "${QWEN_CAPTION_BASE}" "${QWEN8_CAPTION_ADAPTER}" "${PSC_MANIFEST}" \
    "${PSC_ROOT}/runs/eval_qwen8_round0_full_20260830_run01" \
    "${PSC_ROOT}/score.py"
  run_caption_case \
    qwen8_round1 "${QWEN_CAPTION_PY}" "${PSC_ROOT}/run_qwen3_captioner.py" \
    "${QWEN_CAPTION_BASE}" "${QWEN8_ROUND1_CAPTION_ADAPTER}" "${PSC_MANIFEST}" \
    "${PSC_ROOT}/runs/eval_qwen8_round1_full_20260830_run01" \
    "${PSC_ROOT}/score.py"
  run_caption_case \
    midasheng_base "${MIDASHENG_PY}" "${PSC_ROOT}/run_midasheng_captioner.py" \
    "${MIDASHENG_BASE}" "" "${PSC_MANIFEST}" \
    "${PSC_ROOT}/runs/eval_midasheng_base_full_20260830_run01" \
    "${PSC_ROOT}/score.py"
  run_caption_case \
    midasheng4 "${MIDASHENG_PY}" "${PSC_ROOT}/run_midasheng_captioner.py" \
    "${MIDASHENG_BASE}" "${MIDASHENG4_CAPTION_ADAPTER}" "${PSC_MANIFEST}" \
    "${PSC_ROOT}/runs/eval_midasheng4_round0_full_20260830_run01" \
    "${PSC_ROOT}/score.py"
}

run_caption_remaining_smoke() {
  prepare_caption_remaining_smoke_manifest
  run_caption_remaining_parallel \
    smoke "${PSC_REMAINING_SMOKE_MANIFEST}" "${PSC_ROOT}/score_smoke.py"
}

run_caption_remaining_full() {
  run_caption_remaining_parallel full "${PSC_MANIFEST}" "${PSC_ROOT}/score.py"
}

run_tts_local() {
  local branch="$1"
  local size="$2"
  local tts_gpu="${TTS_GPU:-0}"
  if [[ ! "${tts_gpu}" =~ ^[0-9]+$ ]]; then
    echo "TTS_GPU must be a non-negative physical GPU index; got: ${tts_gpu}" >&2
    exit 2
  fi
  local adapter
  adapter="$(adapter_for_tts_branch "${branch}")"
  local run_root
  run_root="$(tts_run_root "${branch}" "${size}")"
  local sample_args=()
  local batch_size=8
  if [[ "${size}" == "smoke" ]]; then
    sample_args=(--num-samples 1)
    batch_size=1
  fi
  require_file "${adapter}/adapter_model.safetensors"
  if [[ ! -r "${adapter}/adapter_model.safetensors" ]]; then
    echo "TTS adapter is not readable: ${adapter}/adapter_model.safetensors" >&2
    echo "Run README step 0 on the owner/training server first." >&2
    exit 1
  fi
  umask 000
  mkdir -p "${run_root}"
  chmod 2777 "${run_root}"
  echo "[START] TTS ${size} local evaluation for ${branch} on physical GPU ${tts_gpu}"
  (
    cd "${TTS_EVAL_ROOT}"
    CUDA_VISIBLE_DEVICES="${tts_gpu}" bash run_inference.sh \
      "${sample_args[@]}" \
      --model-path "${TTS_BASE}" \
      --adapter-dir "${adapter}" \
      --output-dir "${run_root}/audios" \
      --batch-size "${batch_size}"
    CUDA_VISIBLE_DEVICES="${tts_gpu}" bash run_staged_local_metrics.sh \
      "${sample_args[@]}" \
      --audio-dir "${run_root}/audios" \
      --output-dir "${run_root}/staged_evaluation" \
      --workers 1 \
      --devices cuda:0
  )
  echo "[DONE] TTS ${size} generation and local metrics for ${branch}: ${run_root}"
}

run_tts_paid() {
  local branch="$1"
  local size="$2"
  adapter_for_tts_branch "${branch}" >/dev/null
  local run_root
  run_root="$(tts_run_root "${branch}" "${size}")"
  local local_stage_dir="${run_root}/staged_evaluation"
  local local_metrics="${local_stage_dir}/local_metrics.jsonl"
  # Keep paid results separate from the local stage. On shared storage the
  # existing staged_evaluation directory may belong to the worker that ran the
  # local metrics and therefore be read-only from this server.
  # Full paid checkpoints use a sibling under the world-writable TTS runs root
  # instead of trying to create files inside the GPU server's run directory.
  local paid_stage_dir
  paid_stage_dir="$(tts_paid_stage_root "${branch}" "${size}")"
  local sample_args=()
  local workers=128
  if [[ "${size}" == "smoke" ]]; then
    sample_args=(--num-samples 1)
    workers=1
  fi
  require_file "${run_root}/generation_manifest.json"
  require_file "${local_metrics}"

  # Fail before making any paid API request if checkpoints cannot be saved.
  umask 000
  mkdir -p "${paid_stage_dir}/final"
  chmod 2777 "${paid_stage_dir}" "${paid_stage_dir}/final"
  if [[ ! -w "${paid_stage_dir}" || ! -w "${paid_stage_dir}/final" ]]; then
    echo "[ERROR] Paid evaluation output is not writable: ${paid_stage_dir}" >&2
    echo "[ERROR] Run the paid stage on the server that owns this directory, or remove/rename it there and retry." >&2
    return 1
  fi
  if [[ -e "${paid_stage_dir}/judge_results.jsonl" && ! -w "${paid_stage_dir}/judge_results.jsonl" ]]; then
    echo "[ERROR] Gemini checkpoint is not appendable: ${paid_stage_dir}/judge_results.jsonl" >&2
    echo "[ERROR] Run the paid stage on the server that owns this file, or remove/rename it there and retry." >&2
    return 1
  fi

  (
    cd "${TTS_EVAL_ROOT}"
    bash run_staged_gemini_judge.sh \
      "${sample_args[@]}" \
      --audio-dir "${run_root}/audios" \
      --output-dir "${paid_stage_dir}" \
      --workers "${workers}"
    bash run_staged_score.sh \
      "${sample_args[@]}" \
      --stage-dir "${paid_stage_dir}" \
      --local-metrics "${local_metrics}" \
      --audio-dir "${run_root}/audios"
  )
  echo "[DONE] TTS ${size} Gemini judge and final score for ${branch}: ${paid_stage_dir}"
}

command="${1:-}"
case "${command}" in
  check) check_inputs ;;
  caption-smoke) run_caption_smoke ;;
  caption-full) run_caption_full ;;
  caption-remaining-smoke) run_caption_remaining_smoke ;;
  caption-remaining-full) run_caption_remaining_full ;;
  caption-content-smoke-8gpu)
    bash "${BENCHMARK_ROOT}/run_caption_content_scheme_a_8gpu.sh" smoke
    ;;
  caption-content-full-8gpu)
    bash "${BENCHMARK_ROOT}/run_caption_content_scheme_a_8gpu.sh" full
    ;;
  tts-smoke-local) run_tts_local "${2:-}" smoke ;;
  tts-smoke-paid) run_tts_paid "${2:-}" smoke ;;
  tts-full-local) run_tts_local "${2:-}" full ;;
  tts-full-paid) run_tts_paid "${2:-}" full ;;
  -h|--help|help|"") usage ;;
  *)
    echo "Unknown command: ${command}" >&2
    usage >&2
    exit 2
    ;;
esac
