#!/usr/bin/env bash
set -euo pipefail

# Local InstructTTSEval deployment. Only *-paid commands can call Gemini.

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CAPTION_DIR="$(cd -- "${BENCHMARK_DIR}/.." && pwd)"
CLUSTER_DIR="$(cd -- "${CAPTION_DIR}/.." && pwd)"
EVAL_ROOT="${BENCHMARK_DIR}/InstructTTSEval-public"
PIPELINE_ROOT="${EVAL_ROOT}/qwen3_voice_design"
DATA_SOURCE="${EVAL_ROOT}/data/source"
DATA_MANIFESTS="${EVAL_ROOT}/data/manifests"

QWEN_PY="${CLUSTER_DIR}/miniconda3/envs/qwen3-tts/bin/python"
EVAL_PY="${CLUSTER_DIR}/miniconda3/envs/emergent-tts-eval/bin/python"
TTS_BASE="${CAPTION_DIR}/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
TRAIN_ROOT="${CAPTION_DIR}/DualISL_Train/runs"
QWEN8_RUN="${TRAIN_ROOT}/dual_recursive_8gpu_h100_20260829_run01"
MIDASHENG4_RUN="${TRAIN_ROOT}/dual_recursive_4gpu_h100_midasheng_20260830_run01"

OFFICIAL_COMMIT="b7e4120c7cee179a3ce1b99819cf13d8e6f199ce"
DATASET_REVISION="b12cdf288e78cfddb1d1975fd0e05d6fd61ac0d2"
EN_BYTES=347034080
ZH_BYTES=287103717
EN_SHA256="7837c45c1906ceaa130d7bfa7102df20006bc70975456585970cdd01f8d0b826"
ZH_SHA256="c22f59ece17668b398db099974aa262f87b57c46f89c1b3f9d01e64877b95075"
JUDGE_MODEL="${INSTRUCTTTS_JUDGE_MODEL:-models/gemini-2.5-pro}"
JUDGE_BACKEND="${INSTRUCTTTS_JUDGE_BACKEND:-auto}"
JUDGE_TAG="${INSTRUCTTTS_JUDGE_TAG:-evaluation}"
GEMINI_WORKERS="${INSTRUCTTTS_GEMINI_WORKERS:-128}"

if [[ ! "${JUDGE_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'INSTRUCTTTS_JUDGE_TAG must contain only letters, digits, dot, underscore, or dash.\n' >&2
  exit 2
fi
if [[ ! "${GEMINI_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'INSTRUCTTTS_GEMINI_WORKERS must be a positive integer; got %s\n' \
    "${GEMINI_WORKERS}" >&2
  exit 2
fi

usage() {
  printf '%s\n' \
    'Usage: bash run_instructtts_evaluations.sh COMMAND [BRANCH] [SIZE]' \
    '' \
    'No Gemini requests:' \
    '  prepare-data' \
    '  check' \
    '  smoke-local BRANCH' \
    '  smoke-judge-dry-run BRANCH' \
    '  full-local BRANCH' \
    '' \
    'Paid Gemini requests (also require CONFIRM_PAID=YES):' \
    '  smoke-paid BRANCH' \
    '  full-paid BRANCH' \
    '' \
    'Scoring:' \
    '  score BRANCH smoke|full' \
    '' \
    'BRANCH: base, qwen8_r0/r1/r2, midasheng4_r0/r1/r2' \
    'Set TTS_GPU=N to choose the physical GPU for local generation (default 0).' \
    'INSTRUCTTTS_BATCH_SIZE defaults to 8; INSTRUCTTTS_GEMINI_WORKERS defaults to 128.' \
    'Paid runs keep recorded failures by default; set INSTRUCTTTS_RETRY_FAILED=YES to retry them.'
}

require_file() {
  if [[ ! -f "$1" ]]; then
    printf 'Missing file: %s\n' "$1" >&2
    exit 1
  fi
}

require_executable() {
  if [[ ! -x "$1" ]]; then
    printf 'Missing executable: %s\n' "$1" >&2
    exit 1
  fi
}

adapter_for_branch() {
  case "${1:-}" in
    base) printf '\n' ;;
    qwen8_r0) printf '%s\n' "${QWEN8_RUN}/round_000/checkpoints/tts_final" ;;
    qwen8_r1) printf '%s\n' "${QWEN8_RUN}/round_001/checkpoints/tts_final" ;;
    qwen8_r2) printf '%s\n' "${QWEN8_RUN}/round_002/checkpoints/tts_final" ;;
    midasheng4_r0) printf '%s\n' "${MIDASHENG4_RUN}/round_000/checkpoints/tts_final" ;;
    midasheng4_r1) printf '%s\n' "${MIDASHENG4_RUN}/round_001/checkpoints/tts_final" ;;
    midasheng4_r2) printf '%s\n' "${MIDASHENG4_RUN}/round_002/checkpoints/tts_final" ;;
    *)
      printf 'Unknown branch: %s\n' "${1:-}" >&2
      usage >&2
      exit 2
      ;;
  esac
}

run_root() {
  local branch="$1"
  local size="$2"
  printf '%s\n' "${PIPELINE_ROOT}/runs/${branch}/${size}_bilingual_seed42"
}

download_split() {
  local language="$1"
  local expected_bytes="$2"
  local expected_sha256="$3"
  local output="${DATA_SOURCE}/${language}.parquet"
  mkdir -p "${DATA_SOURCE}"
  "${EVAL_PY}" "${PIPELINE_ROOT}/download_data.py" \
    --url "https://huggingface.co/datasets/CaasiHUANG/InstructTTSEval/resolve/${DATASET_REVISION}/${language}.parquet?download=true" \
    --output "${output}" \
    --expected-bytes "${expected_bytes}" \
    --expected-sha256 "${expected_sha256}" \
    --workers 8
}

prepare_data() {
  require_executable "${EVAL_PY}"
  download_split en "${EN_BYTES}" "${EN_SHA256}"
  download_split zh "${ZH_BYTES}" "${ZH_SHA256}"
  "${EVAL_PY}" "${PIPELINE_ROOT}/prepare_data.py"
  printf '[DONE] Prepared pinned EN/ZH manifests.\n'
}

check_deployment() {
  require_executable "${QWEN_PY}"
  require_executable "${EVAL_PY}"
  require_file "${EVAL_ROOT}/data/metadata.json"
  require_file "${EVAL_ROOT}/data/manifests/en.jsonl"
  require_file "${EVAL_ROOT}/data/manifests/zh.jsonl"
  require_file "${TTS_BASE}/model.safetensors"
  local adapters=()
  local branch adapter
  for branch in qwen8_r0 qwen8_r1 qwen8_r2 midasheng4_r0 midasheng4_r1 midasheng4_r2; do
    adapter="$(adapter_for_branch "${branch}")"
    require_file "${adapter}/adapter_model.safetensors"
    if [[ ! -r "${adapter}/adapter_model.safetensors" ]]; then
      printf 'Unreadable adapter weights: %s\n' "${adapter}/adapter_model.safetensors" >&2
      exit 1
    fi
    adapters+=(--adapter-dir "${adapter}")
  done
  "${QWEN_PY}" "${PIPELINE_ROOT}/check.py" \
    --model-path "${TTS_BASE}" "${adapters[@]}"
  "${EVAL_PY}" -c \
    'import importlib.metadata as m; [print(f"[CHECK] {p}={m.version(p)}") for p in ("google-genai", "datasets", "pyarrow", "json-repair")]'
  "${EVAL_PY}" "${PIPELINE_ROOT}/judge.py" --check-config
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
  printf '[DONE] Deployment checks passed without sending a Gemini request.\n'
}

run_local() {
  local branch="$1"
  local size="$2"
  local adapter
  adapter="$(adapter_for_branch "${branch}")"
  local output
  output="$(run_root "${branch}" "${size}")"
  local gpu="${TTS_GPU:-0}"
  if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
    printf 'TTS_GPU must be a non-negative integer; got %s\n' "${gpu}" >&2
    exit 2
  fi
  local selection=()
  local batch_size="${INSTRUCTTTS_BATCH_SIZE:-8}"
  if [[ ! "${batch_size}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'INSTRUCTTTS_BATCH_SIZE must be a positive integer; got %s\n' \
      "${batch_size}" >&2
    exit 2
  fi
  if [[ "${size}" == smoke ]]; then
    selection=(--num-samples-per-language 1)
  fi
  # A resumable output directory is identity-locked, including its configured
  # batch size. Reuse the recorded value so older complete/partial runs are not
  # rejected merely because the launcher's default changed later.
  local manifest="${output}/generation_manifest.json"
  if [[ -f "${manifest}" ]]; then
    local recorded_batch_size
    recorded_batch_size="$("${EVAL_PY}" - "${manifest}" <<'PY'
import json
import sys

try:
    value = json.load(open(sys.argv[1], encoding="utf-8"))["generation_identity"]["batch_size"]
except (KeyError, OSError, TypeError, ValueError):
    value = ""
print(value if isinstance(value, int) and value > 0 else "")
PY
)"
    if [[ -n "${recorded_batch_size}" && "${recorded_batch_size}" != "${batch_size}" ]]; then
      printf '[RESUME] %s %s uses recorded batch size %s (requested %s)\n' \
        "${branch}" "${size}" "${recorded_batch_size}" "${batch_size}"
      batch_size="${recorded_batch_size}"
    fi
  fi
  local adapter_args=()
  if [[ -n "${adapter}" ]]; then
    adapter_args=(--adapter-dir "${adapter}")
  fi
  mkdir -p "${output}"
  printf '[START] %s %s local generation on physical GPU %s\n' \
    "${branch}" "${size}" "${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${QWEN_PY}" "${PIPELINE_ROOT}/generate.py" \
    --model-path "${TTS_BASE}" \
    "${adapter_args[@]}" \
    --manifest-dir "${DATA_MANIFESTS}" \
    --output-dir "${output}" \
    --languages en zh \
    "${selection[@]}" \
    --batch-size "${batch_size}" \
    --device cuda:0
  printf '[DONE] Audio and manifest: %s\n' "${output}"
}

run_dry_judge() {
  local branch="$1"
  local output
  output="$(run_root "${branch}" smoke)"
  require_file "${output}/generation_manifest.json"
  "${EVAL_PY}" "${PIPELINE_ROOT}/judge.py" \
    --generation-manifest "${output}/generation_manifest.json" \
    --output-dir "${output}/dry_run" \
    --dry-run \
    --model "${JUDGE_MODEL}" \
    --backend "${JUDGE_BACKEND}" \
    --workers 2
  "${EVAL_PY}" "${PIPELINE_ROOT}/score.py" \
    --generation-manifest "${output}/generation_manifest.json" \
    --judge-results "${output}/dry_run/judge_results.jsonl" \
    --output "${output}/dry_run/summary.json" \
    --allow-dry-run
}

require_paid_confirmation() {
  if [[ "${CONFIRM_PAID:-}" != YES ]]; then
    printf 'Paid Gemini calls are blocked. Set CONFIRM_PAID=YES explicitly.\n' >&2
    exit 2
  fi
}

run_paid() {
  local branch="$1"
  local size="$2"
  require_paid_confirmation
  adapter_for_branch "${branch}" >/dev/null
  local output
  output="$(run_root "${branch}" "${size}")"
  require_file "${output}/generation_manifest.json"
  local retry_failed="${INSTRUCTTTS_RETRY_FAILED:-NO}"
  local -a retry_args=()
  case "${retry_failed}" in
    NO) ;;
    YES) retry_args=(--retry-failed) ;;
    *)
      printf 'INSTRUCTTTS_RETRY_FAILED must be YES or NO; got %s\n' \
        "${retry_failed}" >&2
      exit 2
      ;;
  esac
  "${EVAL_PY}" "${PIPELINE_ROOT}/judge.py" \
    --generation-manifest "${output}/generation_manifest.json" \
    --output-dir "${output}/${JUDGE_TAG}" \
    --model "${JUDGE_MODEL}" \
    --backend "${JUDGE_BACKEND}" \
    --workers "${GEMINI_WORKERS}" \
    --allow-incomplete \
    "${retry_args[@]}" \
    --confirm-paid
  score_results "${branch}" "${size}"
}

score_results() {
  local branch="$1"
  local size="$2"
  adapter_for_branch "${branch}" >/dev/null
  if [[ "${size}" != smoke && "${size}" != full ]]; then
    printf 'SIZE must be smoke or full; got %s\n' "${size}" >&2
    exit 2
  fi
  local output
  output="$(run_root "${branch}" "${size}")"
  "${EVAL_PY}" "${PIPELINE_ROOT}/score.py" \
    --generation-manifest "${output}/generation_manifest.json" \
    --judge-results "${output}/${JUDGE_TAG}/judge_results.jsonl" \
    --output "${output}/${JUDGE_TAG}/summary.json" \
    --allow-incomplete
}

# Evaluation artifacts are intentionally shared across the GPU generation and
# API-judge servers, which can use different numeric UIDs on QuarkFS.
umask 000
command="${1:-}"
case "${command}" in
  prepare-data) prepare_data ;;
  check) check_deployment ;;
  smoke-local) run_local "${2:-}" smoke ;;
  smoke-judge-dry-run) run_dry_judge "${2:-}" ;;
  full-local) run_local "${2:-}" full ;;
  smoke-paid) run_paid "${2:-}" smoke ;;
  full-paid) run_paid "${2:-}" full ;;
  score) score_results "${2:-}" "${3:-}" ;;
  -h|--help|help|'') usage ;;
  *)
    printf 'Unknown command: %s\n' "${command}" >&2
    usage >&2
    exit 2
    ;;
esac
