#!/usr/bin/env bash
set -euo pipefail

# Run the three full local TTS evaluations on separate GPUs. Paid Gemini stages
# are opt-in and start only after every local evaluation exits successfully.

BENCHMARK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="${BENCHMARK_ROOT}/run_round0_evaluations.sh"
TTS_EVAL_ROOT="${BENCHMARK_ROOT}/EmergentTTS-Eval-public/qwen3_voice_design"

usage() {
  cat <<'EOF'
Usage: bash run_round0_tts_parallel.sh [--remaining] [--with-paid]

Environment overrides:
  TTS_GPU_QWEN8_R0       physical GPU for qwen8_r0       (default: 0)
  TTS_GPU_QWEN8_R1       physical GPU for qwen8_r1       (default: 1)
  TTS_GPU_MIDASHENG4_R0  physical GPU for midasheng4_r0  (default: 2)
  TTS_GPU_QWEN8_R2       physical GPU for qwen8_r2       (default: 0)
  TTS_GPU_MIDASHENG4_R1  physical GPU for midasheng4_r1  (default: 1)
  TTS_GPU_MIDASHENG4_R2  physical GPU for midasheng4_r2  (default: 2)
  ROUND0_TTS_LOG_ROOT    directory for per-stage logs

Without --with-paid, only the three local GPU evaluations run.
With --with-paid, the three paid Gemini stages run sequentially, but only after
all local evaluations complete successfully.
With --remaining, the three branches are qwen8_r2, midasheng4_r1, and
midasheng4_r2 instead of the already evaluated qwen8_r0/r1 and midasheng4_r0.
EOF
}

with_paid=0
remaining=0
for argument in "$@"; do
  case "${argument}" in
    --with-paid) with_paid=1 ;;
    --remaining) remaining=1 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: ${argument}" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${DRIVER}" ]]; then
  echo "Missing evaluation driver: ${DRIVER}" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable." >&2
  exit 1
fi

if (( remaining != 0 )); then
  branches=(qwen8_r2 midasheng4_r1 midasheng4_r2)
  gpus=(
    "${TTS_GPU_QWEN8_R2:-0}"
    "${TTS_GPU_MIDASHENG4_R1:-1}"
    "${TTS_GPU_MIDASHENG4_R2:-2}"
  )
else
  branches=(qwen8_r0 qwen8_r1 midasheng4_r0)
  gpus=(
    "${TTS_GPU_QWEN8_R0:-0}"
    "${TTS_GPU_QWEN8_R1:-1}"
    "${TTS_GPU_MIDASHENG4_R0:-2}"
  )
fi

declare -A visible_gpu=()
while IFS= read -r gpu_index; do
  gpu_index="${gpu_index//[[:space:]]/}"
  if [[ -n "${gpu_index}" ]]; then
    visible_gpu["${gpu_index}"]=1
  fi
done < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)

declare -A assigned_gpu=()
for index in "${!branches[@]}"; do
  branch="${branches[$index]}"
  gpu="${gpus[$index]}"
  if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU index for ${branch}: ${gpu}" >&2
    exit 2
  fi
  if [[ -z "${visible_gpu[$gpu]:-}" ]]; then
    echo "GPU ${gpu} for ${branch} is not visible to nvidia-smi." >&2
    exit 1
  fi
  if [[ -n "${assigned_gpu[$gpu]:-}" ]]; then
    echo "GPU ${gpu} is assigned more than once; use three distinct GPUs." >&2
    exit 2
  fi
  assigned_gpu["${gpu}"]="${branch}"
done

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_root="${ROUND0_TTS_LOG_ROOT:-${BENCHMARK_ROOT}/round0_tts_logs_${timestamp}}"
umask 000
mkdir -p "${log_root}"
chmod 2777 "${log_root}"
exec > >(tee -a "${log_root}/orchestrator.log") 2>&1

echo "[START] Parallel local TTS evaluations"
echo "[START] Logs: ${log_root}"
for index in "${!branches[@]}"; do
  echo "[START] ${branches[$index]} -> physical GPU ${gpus[$index]}"
done

pids=()
for index in "${!branches[@]}"; do
  branch="${branches[$index]}"
  gpu="${gpus[$index]}"
  TTS_GPU="${gpu}" bash "${DRIVER}" tts-full-local "${branch}" \
    >"${log_root}/${branch}.local.log" 2>&1 &
  pids+=("$!")
done

terminate_children() {
  trap - INT TERM
  echo "[STOP] Terminating local evaluation children..." >&2
  for pid in "${pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait || true
  exit 130
}
trap terminate_children INT TERM

local_failed=0
for index in "${!pids[@]}"; do
  branch="${branches[$index]}"
  pid="${pids[$index]}"
  if wait "${pid}"; then
    echo "[DONE] Local ${branch}"
  else
    status=$?
    echo "[FAILED] Local ${branch} exited with status ${status}; log=${log_root}/${branch}.local.log" >&2
    local_failed=1
  fi
done
trap - INT TERM

if (( local_failed != 0 )); then
  echo "[STOP] At least one local evaluation failed; paid stages were not started." >&2
  exit 1
fi

if (( with_paid == 0 )); then
  echo "[DONE] All local evaluations completed. Paid stages were not requested."
  exit 0
fi

if [[ -z "${JUDGER_API_KEY:-}" && ! -r "${TTS_EVAL_ROOT}/local_judger_config.py" ]]; then
  echo "Missing Gemini configuration: set JUDGER_API_KEY or provide local_judger_config.py." >&2
  exit 1
fi

echo "[START] All local evaluations succeeded; starting paid stages sequentially."
for branch in "${branches[@]}"; do
  paid_log="${log_root}/${branch}.paid.log"
  echo "[START] Paid ${branch}; log=${paid_log}"
  if bash "${DRIVER}" tts-full-paid "${branch}" >"${paid_log}" 2>&1; then
    echo "[DONE] Paid ${branch}"
  else
    status=$?
    echo "[FAILED] Paid ${branch} exited with status ${status}; log=${paid_log}" >&2
    exit "${status}"
  fi
done

echo "[ALL DONE] $(date -Is)"
