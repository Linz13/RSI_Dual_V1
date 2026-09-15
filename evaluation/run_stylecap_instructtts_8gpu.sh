#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluate the two three-round DualISL branches plus their base models on:
#   1) the derived StyleCap/PromptSpeech MCQ (Captioner), and
#   2) official-data InstructTTSEval (TTS generation + separate Gemini judge).
# No command except instruct-*-paid sends a Gemini request.

BENCHMARK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CAPTION_ROOT="$(cd -- "${BENCHMARK_ROOT}/.." && pwd)"
CLUSTER_ROOT="$(cd -- "${CAPTION_ROOT}/.." && pwd)"

STYLE_ROOT="${BENCHMARK_ROOT}/stylecap_promptspeech_mcq"
INSTRUCT_DRIVER="${BENCHMARK_ROOT}/run_instructtts_evaluations.sh"
INSTRUCT_ROOT="${BENCHMARK_ROOT}/InstructTTSEval-public"
INSTRUCT_PIPELINE="${INSTRUCT_ROOT}/qwen3_voice_design"

QWEN_PY="${CLUSTER_ROOT}/miniconda3/envs/qwen3-captioner/bin/python"
MIDASHENG_PY="${CLUSTER_ROOT}/miniconda3/envs/midasheng-captioner/bin/python"
TTS_PY="${CLUSTER_ROOT}/miniconda3/envs/qwen3-tts/bin/python"
EVAL_PY="${CLUSTER_ROOT}/miniconda3/envs/emergent-tts-eval/bin/python"

QWEN_BASE="${CAPTION_ROOT}/models/Qwen3-Omni-30B-A3B-Captioner"
MIDASHENG_BASE="${CAPTION_ROOT}/models/MiDashengLM-7B-1021-BF16"
TTS_BASE="${CAPTION_ROOT}/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
QWEN_RUN="${CAPTION_ROOT}/DualISL_Train/runs/dual_recursive_8gpu_h100_20260829_run01"
MIDASHENG_RUN="${CAPTION_ROOT}/DualISL_Train/runs/dual_recursive_4gpu_h100_midasheng_20260830_run01"

RUN_TAG="${EVAL_RUN_TAG:-20260901_run01}"
LOG_ROOT="${BENCHMARK_ROOT}/stylecap_instructtts_eval_runs_${RUN_TAG}/logs"
DEFAULT_JUDGE_MODEL="models/gemini-2.5-pro"

STYLE_NAMES=(
  qwen_base qwen8_round0 qwen8_round1 qwen8_round2
  midasheng_base midasheng4_round0 midasheng4_round1 midasheng4_round2
)
STYLE_BACKENDS=(qwen3 qwen3 qwen3 qwen3 midasheng midasheng midasheng midasheng)
STYLE_PYTHONS=(
  "${QWEN_PY}" "${QWEN_PY}" "${QWEN_PY}" "${QWEN_PY}"
  "${MIDASHENG_PY}" "${MIDASHENG_PY}" "${MIDASHENG_PY}" "${MIDASHENG_PY}"
)
STYLE_MODELS=(
  "${QWEN_BASE}" "${QWEN_BASE}" "${QWEN_BASE}" "${QWEN_BASE}"
  "${MIDASHENG_BASE}" "${MIDASHENG_BASE}" "${MIDASHENG_BASE}" "${MIDASHENG_BASE}"
)
STYLE_ADAPTERS=(
  base
  "${QWEN_RUN}/round_000/checkpoints/caption_final"
  "${QWEN_RUN}/round_001/checkpoints/caption_final"
  "${QWEN_RUN}/round_002/checkpoints/caption_final"
  base
  "${MIDASHENG_RUN}/round_000/checkpoints/caption_final"
  "${MIDASHENG_RUN}/round_001/checkpoints/caption_final"
  "${MIDASHENG_RUN}/round_002/checkpoints/caption_final"
)
STYLE_ATTN=(
  flash_attention_2 flash_attention_2 flash_attention_2 flash_attention_2
  sdpa sdpa sdpa sdpa
)

INSTRUCT_BRANCHES=(
  base qwen8_r0 qwen8_r1 qwen8_r2
  midasheng4_r0 midasheng4_r1 midasheng4_r2
)

usage() {
  cat <<'EOF'
Usage: bash run_stylecap_instructtts_8gpu.sh COMMAND

Read-only / no GPU:
  check                       Validate data, all 12 adapters, and CPU tests.
  paths                       Print expected result locations.

Local GPU inference (never calls Gemini):
  stylecap-smoke              8 Captioner cases, four MCQs per case.
  stylecap-full               8 Captioner cases, 3,112 MCQs per case.
  instruct-smoke-local        7 unique TTS cases, six WAVs per case.
  instruct-full-local         7 unique TTS cases, 6,000 WAVs per case.
  smoke-local                 check + both smoke suites + dry judges.
  full-local                  check + StyleCap full + Instruct full generation.

Gemini stage (no GPU required):
  instruct-smoke-dry-run      Offline simulated judge for all seven smoke cases.
  instruct-smoke-paid         Real Gemini judge for all seven smoke cases.
  instruct-full-paid          Real Gemini judge for all seven full cases.
  instruct-score-full         Recompute all seven full summaries from checkpoints.

GPU mapping is read from EVAL_GPUS (default 0,1,2,3,4,5,6,7).
StyleCap uses all eight GPUs. InstructTTSEval uses seven because the TTS base is
shared by the Qwen/MiDasheng training branches and is evaluated only once.
The suites run sequentially: StyleCap first, then InstructTTSEval. Per-GPU batch
sizes default to 4 for StyleCap and 8 for InstructTTSEval; override with
STYLECAP_QWEN_BATCH_SIZE, STYLECAP_MIDASHENG_BATCH_SIZE, or INSTRUCTTTS_BATCH_SIZE.

Paid commands additionally require CONFIRM_PAID=YES. The exact paper model,
gemini-2.5-pro-preview-05-06, was shut down by Google; this launcher therefore
defaults to models/gemini-2.5-pro and records the run as a non-exact reproduction.
Gemini judging runs one model case at a time with 128 request workers by default,
so total API concurrency is capped at 128. Override INSTRUCTTTS_GEMINI_WORKERS or
INSTRUCTTTS_JUDGE_MODEL/BACKEND/TAG only deliberately. Recorded failures are
scored as incomplete and skipped on resume; set INSTRUCTTTS_RETRY_FAILED=YES to
retry them explicitly.
EOF
}

require_file() {
  [[ -f "$1" ]] || { printf 'Missing required file: %s\n' "$1" >&2; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { printf 'Missing required directory: %s\n' "$1" >&2; exit 1; }
}

require_executable() {
  [[ -x "$1" ]] || { printf 'Missing executable: %s\n' "$1" >&2; exit 1; }
}

adapter_for_instruct_branch() {
  case "$1" in
    base) printf '\n' ;;
    qwen8_r0) printf '%s\n' "${QWEN_RUN}/round_000/checkpoints/tts_final" ;;
    qwen8_r1) printf '%s\n' "${QWEN_RUN}/round_001/checkpoints/tts_final" ;;
    qwen8_r2) printf '%s\n' "${QWEN_RUN}/round_002/checkpoints/tts_final" ;;
    midasheng4_r0) printf '%s\n' "${MIDASHENG_RUN}/round_000/checkpoints/tts_final" ;;
    midasheng4_r1) printf '%s\n' "${MIDASHENG_RUN}/round_001/checkpoints/tts_final" ;;
    midasheng4_r2) printf '%s\n' "${MIDASHENG_RUN}/round_002/checkpoints/tts_final" ;;
    *) printf 'Unknown InstructTTSEval branch: %s\n' "$1" >&2; return 2 ;;
  esac
}

resolve_gpus() {
  local required="$1"
  local specification="${EVAL_GPUS:-0,1,2,3,4,5,6,7}"
  IFS=',' read -r -a GPUS <<<"${specification}"
  if (( ${#GPUS[@]} < required )); then
    printf 'COMMAND requires at least %d GPU IDs; EVAL_GPUS=%s\n' \
      "${required}" "${specification}" >&2
    exit 2
  fi
  local seen="," gpu
  for gpu in "${GPUS[@]:0:required}"; do
    if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
      printf 'GPU IDs must be non-negative integers; got %s\n' "${gpu}" >&2
      exit 2
    fi
    if [[ "${seen}" == *",${gpu},"* ]]; then
      printf 'EVAL_GPUS contains a duplicate GPU ID: %s\n' "${gpu}" >&2
      exit 2
    fi
    seen+="${gpu},"
  done
  if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L >/dev/null 2>&1; then
    printf 'No working NVIDIA driver is visible on this host.\n' >&2
    exit 1
  fi
}

check_adapter_files() {
  local adapter="$1"
  require_file "${adapter}/adapter_config.json"
  require_file "${adapter}/adapter_model.safetensors"
  [[ -r "${adapter}/adapter_model.safetensors" ]] || {
    printf 'Adapter weights are not readable: %s\n' "${adapter}" >&2
    exit 1
  }
}

check_all() {
  require_executable "${QWEN_PY}"
  require_executable "${MIDASHENG_PY}"
  require_executable "${TTS_PY}"
  require_executable "${EVAL_PY}"
  require_file "${STYLE_ROOT}/data/benchmark.jsonl"
  require_file "${INSTRUCT_DRIVER}"
  require_dir "${QWEN_BASE}"
  require_dir "${MIDASHENG_BASE}"
  require_dir "${TTS_BASE}"

  local adapter
  for adapter in "${STYLE_ADAPTERS[@]}"; do
    [[ "${adapter}" == base ]] || check_adapter_files "${adapter}"
  done
  local branch
  local -a tts_check_args=()
  for branch in "${INSTRUCT_BRANCHES[@]}"; do
    adapter="$(adapter_for_instruct_branch "${branch}")"
    if [[ -n "${adapter}" ]]; then
      check_adapter_files "${adapter}"
      tts_check_args+=(--adapter-dir "${adapter}")
    fi
  done
  for round in 000 001 002; do
    require_file "${QWEN_RUN}/round_${round}/commit.json"
    require_file "${MIDASHENG_RUN}/round_${round}/commit.json"
  done

  "${QWEN_PY}" "${STYLE_ROOT}/validate_benchmark.py"
  "${QWEN_PY}" -m unittest discover -s "${STYLE_ROOT}/tests" -p 'test_*.py'
  "${TTS_PY}" "${INSTRUCT_PIPELINE}/check.py" \
    --model-path "${TTS_BASE}" "${tts_check_args[@]}"
  "${EVAL_PY}" -m unittest discover \
    -s "${INSTRUCT_PIPELINE}/tests" -p 'test_common_data_judge.py'
  "${TTS_PY}" -m unittest discover \
    -s "${INSTRUCT_PIPELINE}/tests" -p 'test_generation.py'
  printf '[DONE] Static data, checkpoint, and CPU checks passed. No Gemini request was sent.\n'
}

style_output_dir() {
  local name="$1" size="$2"
  # Reuse the already-completed and identity-compatible MiDasheng r2 result.
  if [[ "${name}" == midasheng4_round2 ]]; then
    printf '%s\n' "${STYLE_ROOT}/runs/midasheng4_round2_${size}_20260831_run01"
  else
    printf '%s\n' "${STYLE_ROOT}/runs/${name}_${size}_${RUN_TAG}"
  fi
}

run_style_case() {
  local index="$1" size="$2" gpu="$3"
  local name="${STYLE_NAMES[$index]}"
  local output
  output="$(style_output_dir "${name}" "${size}")"
  local -a selection=()
  local batch_size
  if [[ "${STYLE_BACKENDS[$index]}" == qwen3 ]]; then
    batch_size="${STYLECAP_QWEN_BATCH_SIZE:-4}"
  else
    batch_size="${STYLECAP_MIDASHENG_BATCH_SIZE:-4}"
  fi
  if [[ ! "${batch_size}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'StyleCap batch size must be a positive integer; got %s\n' \
      "${batch_size}" >&2
    return 2
  fi
  # Historical MiDasheng r2 artifacts were generated one question at a time;
  # retain that identity when resuming them instead of forcing recomputation.
  if [[ "${name}" == midasheng4_round2 && -f "${output}/generations.jsonl" ]]; then
    batch_size=1
  fi
  if [[ "${size}" == smoke ]]; then
    selection=(--max-questions 4)
  fi
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${gpu}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "${STYLE_PYTHONS[$index]}" "${STYLE_ROOT}/run_midasheng.py" \
      --backend "${STYLE_BACKENDS[$index]}" \
      --benchmark "${STYLE_ROOT}/data/benchmark.jsonl" \
      --model-dir "${STYLE_MODELS[$index]}" \
      --adapter-dir "${STYLE_ADAPTERS[$index]}" \
      --output-dir "${output}" \
      --attn-backend "${STYLE_ATTN[$index]}" \
      --batch-size "${batch_size}" \
      --resume "${selection[@]}"
  if [[ "${size}" == full ]]; then
    "${STYLE_PYTHONS[$index]}" "${STYLE_ROOT}/evaluate.py" \
      --benchmark "${STYLE_ROOT}/data/benchmark.jsonl" \
      --predictions "${output}/predictions.jsonl" \
      --output "${output}/evaluation_summary.json"
  fi
}

wait_for_jobs() {
  local suite="$1"
  local failed=0 index status
  for index in "${!JOB_PIDS[@]}"; do
    if wait "${JOB_PIDS[$index]}"; then
      printf '[DONE] %s %s on GPU %s\n' \
        "${suite}" "${JOB_NAMES[$index]}" "${JOB_GPUS[$index]}"
    else
      status=$?
      printf '[FAILED] %s %s exited %d; log=%s\n' \
        "${suite}" "${JOB_NAMES[$index]}" "${status}" "${JOB_LOGS[$index]}" >&2
      failed=1
    fi
  done
  trap - INT TERM
  (( failed == 0 )) || return 1
}

stop_jobs() {
  trap - INT TERM
  printf '[STOP] Terminating evaluation launchers. Outputs remain resumable.\n' >&2
  local pid
  for pid in "${JOB_PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait || true
  exit 130
}

run_style_all() {
  local size="$1"
  resolve_gpus 8
  local log_dir="${LOG_ROOT}/stylecap_${size}"
  mkdir -p "${log_dir}"
  JOB_PIDS=(); JOB_NAMES=(); JOB_GPUS=(); JOB_LOGS=()
  local index log
  for index in "${!STYLE_NAMES[@]}"; do
    log="${log_dir}/${STYLE_NAMES[$index]}.log"
    printf '[START] StyleCap %s -> GPU %s; log=%s\n' \
      "${STYLE_NAMES[$index]}" "${GPUS[$index]}" "${log}"
    run_style_case "${index}" "${size}" "${GPUS[$index]}" >"${log}" 2>&1 &
    JOB_PIDS+=("$!"); JOB_NAMES+=("${STYLE_NAMES[$index]}")
    JOB_GPUS+=("${GPUS[$index]}"); JOB_LOGS+=("${log}")
  done
  trap stop_jobs INT TERM
  wait_for_jobs "StyleCap-${size}"
}

run_instruct_local_case() {
  local branch="$1" size="$2" gpu="$3"
  TTS_GPU="${gpu}" bash "${INSTRUCT_DRIVER}" "${size}-local" "${branch}"
}

run_instruct_local_all() {
  local size="$1"
  resolve_gpus 7
  local log_dir="${LOG_ROOT}/instruct_${size}_local"
  mkdir -p "${log_dir}"
  JOB_PIDS=(); JOB_NAMES=(); JOB_GPUS=(); JOB_LOGS=()
  local index branch log
  for index in "${!INSTRUCT_BRANCHES[@]}"; do
    branch="${INSTRUCT_BRANCHES[$index]}"
    log="${log_dir}/${branch}.log"
    printf '[START] InstructTTSEval %s -> GPU %s; log=%s\n' \
      "${branch}" "${GPUS[$index]}" "${log}"
    run_instruct_local_case "${branch}" "${size}" "${GPUS[$index]}" >"${log}" 2>&1 &
    JOB_PIDS+=("$!"); JOB_NAMES+=("${branch}")
    JOB_GPUS+=("${GPUS[$index]}"); JOB_LOGS+=("${log}")
  done
  trap stop_jobs INT TERM
  wait_for_jobs "InstructTTSEval-${size}-local"
}

run_instruct_dry_all() {
  local branch
  for branch in "${INSTRUCT_BRANCHES[@]}"; do
    printf '[START] InstructTTSEval dry judge: %s\n' "${branch}"
    bash "${INSTRUCT_DRIVER}" smoke-judge-dry-run "${branch}"
  done
}

run_instruct_paid_all() {
  local size="$1"
  if [[ "${CONFIRM_PAID:-}" != YES ]]; then
    printf 'Paid Gemini calls are blocked. Set CONFIRM_PAID=YES explicitly.\n' >&2
    return 2
  fi
  local model="${INSTRUCTTTS_JUDGE_MODEL:-${DEFAULT_JUDGE_MODEL}}"
  local backend="${INSTRUCTTTS_JUDGE_BACKEND:-auto}"
  local tag="${INSTRUCTTTS_JUDGE_TAG:-evaluation_gemini_2_5_pro}"
  local log_dir="${LOG_ROOT}/instruct_${size}_paid_${tag}"
  mkdir -p "${log_dir}"
  local branch
  for branch in "${INSTRUCT_BRANCHES[@]}"; do
    printf '[START] Paid InstructTTSEval %s model=%s backend=%s\n' \
      "${branch}" "${model}" "${backend}"
    CONFIRM_PAID=YES \
    INSTRUCTTTS_JUDGE_MODEL="${model}" \
    INSTRUCTTTS_JUDGE_BACKEND="${backend}" \
    INSTRUCTTTS_JUDGE_TAG="${tag}" \
    INSTRUCTTTS_GEMINI_WORKERS="${INSTRUCTTTS_GEMINI_WORKERS:-128}" \
      bash "${INSTRUCT_DRIVER}" "${size}-paid" "${branch}" \
      2>&1 | tee "${log_dir}/${branch}.log"
  done
}

score_instruct_full_all() {
  local model="${INSTRUCTTTS_JUDGE_MODEL:-${DEFAULT_JUDGE_MODEL}}"
  local backend="${INSTRUCTTTS_JUDGE_BACKEND:-auto}"
  local tag="${INSTRUCTTTS_JUDGE_TAG:-evaluation_gemini_2_5_pro}"
  local branch
  for branch in "${INSTRUCT_BRANCHES[@]}"; do
    INSTRUCTTTS_JUDGE_MODEL="${model}" \
    INSTRUCTTTS_JUDGE_BACKEND="${backend}" \
    INSTRUCTTTS_JUDGE_TAG="${tag}" \
      bash "${INSTRUCT_DRIVER}" score "${branch}" full
  done
}

print_paths() {
  local size name branch
  for size in smoke full; do
    printf 'StyleCap %s:\n' "${size}"
    for name in "${STYLE_NAMES[@]}"; do
      style_output_dir "${name}" "${size}"
    done
  done
  for size in smoke full; do
    printf 'InstructTTSEval %s:\n' "${size}"
    for branch in "${INSTRUCT_BRANCHES[@]}"; do
      printf '%s\n' "${INSTRUCT_PIPELINE}/runs/${branch}/${size}_bilingual_seed42"
    done
  done
  printf 'Logs:\n%s\n' "${LOG_ROOT}"
}

umask 000
command="${1:-}"
case "${command}" in
  check) check_all ;;
  paths) print_paths ;;
  stylecap-smoke) run_style_all smoke ;;
  stylecap-full) run_style_all full ;;
  instruct-smoke-local) run_instruct_local_all smoke ;;
  instruct-full-local) run_instruct_local_all full ;;
  instruct-smoke-dry-run) run_instruct_dry_all ;;
  instruct-smoke-paid) run_instruct_paid_all smoke ;;
  instruct-full-paid) run_instruct_paid_all full ;;
  instruct-score-full) score_instruct_full_all ;;
  smoke-local)
    check_all
    run_style_all smoke
    run_instruct_local_all smoke
    run_instruct_dry_all
    ;;
  full-local)
    check_all
    run_style_all full
    run_instruct_local_all full
    ;;
  -h|--help|help|'') usage ;;
  *)
    printf 'Unknown command: %s\n' "${command}" >&2
    usage >&2
    exit 2
    ;;
esac
