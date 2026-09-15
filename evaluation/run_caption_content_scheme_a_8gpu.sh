#!/usr/bin/env bash
set -euo pipefail

# Run the independent-field (content-only Scheme A) Caption benchmark for all
# eight comparison candidates. One complete model is loaded on each GPU.

BENCHMARK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CAPTION_ROOT="$(cd -- "${BENCHMARK_ROOT}/.." && pwd)"
CLUSTER_ROOT="$(cd -- "${CAPTION_ROOT}/.." && pwd)"
EVAL_ROOT="${BENCHMARK_ROOT}/paraspeechcaps/evaluations/qwen3_captioner_attr6"
TRAIN_ROOT="${CAPTION_ROOT}/DualISL_Train/runs"

QWEN_PY="${CLUSTER_ROOT}/miniconda3/envs/qwen3-captioner/bin/python"
MIDASHENG_PY="${CLUSTER_ROOT}/miniconda3/envs/midasheng-captioner/bin/python"
QWEN_BASE="${CAPTION_ROOT}/models/Qwen3-Omni-30B-A3B-Captioner"
MIDASHENG_BASE="${CAPTION_ROOT}/models/MiDashengLM-7B-1021-BF16"
QWEN_RUN="${TRAIN_ROOT}/dual_recursive_8gpu_h100_20260829_run01"
MIDASHENG_RUN="${TRAIN_ROOT}/dual_recursive_4gpu_h100_midasheng_20260830_run01"
MANIFEST="${EVAL_ROOT}/runs/new_cluster_default/manifest.jsonl"
RUNNER="${EVAL_ROOT}/run_content_scheme_a.py"
SCORER="${EVAL_ROOT}/score_content_scheme_a.py"
OUTPUT_ROOT="${CONTENT_SCHEME_A_ROOT:-${BENCHMARK_ROOT}/round_completion_eval_runs_20260831/paraspeechcaps_content_scheme_a_v1}"

usage() {
  cat <<'EOF'
Usage: bash run_caption_content_scheme_a_8gpu.sh {smoke|full} [--dry-run]

All eight candidates run concurrently, one per physical GPU. The default GPU
order is 0,1,2,3,4,5,6,7 for:
  qwen_base, qwen8_r0, qwen8_r1, qwen8_r2,
  midasheng_base, midasheng4_r0, midasheng4_r1, midasheng4_r2

Override the mapping with, for example:
  CONTENT_SCHEME_A_GPUS=7,6,5,4,3,2,1,0 bash ... full
EOF
}

size="${1:-}"
dry_run=0
if [[ "${2:-}" == "--dry-run" ]]; then
  dry_run=1
elif [[ -n "${2:-}" ]]; then
  echo "Unknown argument: ${2}" >&2
  usage >&2
  exit 2
fi
if [[ "${size}" != "smoke" && "${size}" != "full" ]]; then
  usage >&2
  exit 2
fi

names=(
  qwen_base qwen8_r0 qwen8_r1 qwen8_r2
  midasheng_base midasheng4_r0 midasheng4_r1 midasheng4_r2
)
backends=(qwen3 qwen3 qwen3 qwen3 midasheng midasheng midasheng midasheng)
pythons=(
  "${QWEN_PY}" "${QWEN_PY}" "${QWEN_PY}" "${QWEN_PY}"
  "${MIDASHENG_PY}" "${MIDASHENG_PY}" "${MIDASHENG_PY}" "${MIDASHENG_PY}"
)
models=(
  "${QWEN_BASE}" "${QWEN_BASE}" "${QWEN_BASE}" "${QWEN_BASE}"
  "${MIDASHENG_BASE}" "${MIDASHENG_BASE}" "${MIDASHENG_BASE}" "${MIDASHENG_BASE}"
)
adapters=(
  ""
  "${QWEN_RUN}/round_000/checkpoints/caption_final"
  "${QWEN_RUN}/round_001/checkpoints/caption_final"
  "${QWEN_RUN}/round_002/checkpoints/caption_final"
  ""
  "${MIDASHENG_RUN}/round_000/checkpoints/caption_final"
  "${MIDASHENG_RUN}/round_001/checkpoints/caption_final"
  "${MIDASHENG_RUN}/round_002/checkpoints/caption_final"
)
attn_backends=(
  flash_attention_2 flash_attention_2 flash_attention_2 flash_attention_2
  sdpa sdpa sdpa sdpa
)

IFS=',' read -r -a gpus <<< "${CONTENT_SCHEME_A_GPUS:-0,1,2,3,4,5,6,7}"
if (( ${#gpus[@]} != ${#names[@]} )); then
  echo "CONTENT_SCHEME_A_GPUS must contain exactly eight comma-separated GPU IDs." >&2
  exit 2
fi

declare -A seen_gpus=()
for gpu in "${gpus[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
    echo "GPU IDs must be non-negative integers; got: ${gpu}" >&2
    exit 2
  fi
  if [[ -n "${seen_gpus[${gpu}]:-}" ]]; then
    echo "All eight candidates require distinct GPU IDs; duplicate: ${gpu}" >&2
    exit 2
  fi
  seen_gpus["${gpu}"]=1
done

if (( dry_run == 0 )); then
  for path in "${QWEN_PY}" "${MIDASHENG_PY}"; do
    if [[ ! -x "${path}" ]]; then
      echo "Missing Python executable: ${path}" >&2
      exit 1
    fi
  done
  for path in "${MANIFEST}" "${RUNNER}" "${SCORER}"; do
    if [[ ! -f "${path}" ]]; then
      echo "Missing required file: ${path}" >&2
      exit 1
    fi
  done
  for path in "${models[@]}"; do
    if [[ ! -d "${path}" ]]; then
      echo "Missing model directory: ${path}" >&2
      exit 1
    fi
  done
  for path in "${adapters[@]}"; do
    if [[ -n "${path}" && ! -r "${path}/adapter_model.safetensors" ]]; then
      echo "Missing or unreadable adapter weights: ${path}/adapter_model.safetensors" >&2
      exit 1
    fi
  done
  mapfile -t visible_gpus < <(
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits | sed 's/[[:space:]]//g'
  )
  if (( ${#visible_gpus[@]} == 0 )); then
    echo "No visible NVIDIA GPU was detected." >&2
    exit 1
  fi
  for gpu in "${gpus[@]}"; do
    found=0
    for visible in "${visible_gpus[@]}"; do
      if [[ "${gpu}" == "${visible}" ]]; then
        found=1
        break
      fi
    done
    if (( found == 0 )); then
      echo "Requested physical GPU ${gpu} is not visible; visible=${visible_gpus[*]}" >&2
      exit 1
    fi
  done
fi

sample_args=()
if [[ "${size}" == "smoke" ]]; then
  sample_args=(--max-samples 1)
fi

if (( dry_run != 0 )); then
  echo "[DRY RUN] Scheme A ${size}; output_root=${OUTPUT_ROOT}"
  for index in "${!names[@]}"; do
    echo "[MAP] ${names[$index]} -> physical GPU ${gpus[$index]}; backend=${backends[$index]}; adapter=${adapters[$index]:-BASE}"
  done
  exit 0
fi

umask 000
log_root="${OUTPUT_ROOT}/logs/${size}"
mkdir -p "${OUTPUT_ROOT}" "${log_root}"
chmod 2777 "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/logs" "${log_root}"

pids=()
logs=()
echo "[START] Content-only Scheme A ${size}: eight candidates on eight GPUs"
for index in "${!names[@]}"; do
  name="${names[$index]}"
  gpu="${gpus[$index]}"
  run_root="${OUTPUT_ROOT}/eval_${name}_${size}_20260831_run01"
  log_path="${log_root}/${name}.log"
  adapter_args=()
  if [[ -n "${adapters[$index]}" ]]; then
    adapter_args=(--adapter-dir "${adapters[$index]}")
  fi
  mkdir -p "${run_root}/outputs" "${run_root}/reports"
  chmod 2777 "${run_root}" "${run_root}/outputs" "${run_root}/reports"
  echo "[START] ${name} -> physical GPU ${gpu}; log=${log_path}"
  (
    set -e
    CUDA_VISIBLE_DEVICES="${gpu}" "${pythons[$index]}" "${RUNNER}" \
      --manifest "${MANIFEST}" \
      --output-dir "${run_root}/outputs" \
      --candidate-name "${name}" \
      --backend "${backends[$index]}" \
      --model-dir "${models[$index]}" \
      "${adapter_args[@]}" \
      --attn-backend "${attn_backends[$index]}" \
      --resume \
      "${sample_args[@]}"
    "${QWEN_PY}" "${SCORER}" \
      --manifest "${run_root}/outputs/selected_manifest.jsonl" \
      --predictions "${run_root}/outputs/predictions.jsonl" \
      --output-dir "${run_root}/reports"
    echo "[DONE] ${name}: ${run_root}"
  ) >"${log_path}" 2>&1 &
  pids+=("$!")
  logs+=("${log_path}")
done

terminate_children() {
  trap - INT TERM
  echo "[STOP] Terminating Scheme A evaluation children..." >&2
  for pid in "${pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait || true
  exit 130
}
trap terminate_children INT TERM

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "[DONE] ${names[$index]} on physical GPU ${gpus[$index]}"
  else
    status=$?
    echo "[FAILED] ${names[$index]} exited with status ${status}; log=${logs[$index]}" >&2
    failed=1
  fi
done
trap - INT TERM

if (( failed != 0 )); then
  echo "[STOP] One or more Scheme A evaluations failed. Fix the logged cause and rerun the same command to resume." >&2
  exit 1
fi
echo "[DONE] All eight Content-only Scheme A ${size} evaluations completed: ${OUTPUT_ROOT}"
