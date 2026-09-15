#!/usr/bin/env bash
set -Eeuo pipefail

# Run the eight Captioner candidates on EmotionTalk, one model per physical GPU.
umask 000

BENCHMARK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CAPTION_ROOT="$(cd -- "${BENCHMARK_ROOT}/.." && pwd)"
CLUSTER_ROOT="$(cd -- "${CAPTION_ROOT}/.." && pwd)"
EMOTION_ROOT="${BENCHMARK_ROOT}/emotiontalk_speech_captioning"
SMOKE_DRIVER="${EMOTION_ROOT}/run_emotiontalk_smoke.sh"

QWEN_PY="${QWEN_PY:-${CLUSTER_ROOT}/miniconda3/envs/qwen3-captioner/bin/python}"
MIDASHENG_PY="${MIDASHENG_PY:-${CLUSTER_ROOT}/miniconda3/envs/midasheng-captioner/bin/python}"
METRICS_PY="${METRICS_PY:-${CLUSTER_ROOT}/miniconda3/envs/emotiontalk-metrics/bin/python}"
QWEN_MODEL="${QWEN_MODEL:-${CAPTION_ROOT}/models/Qwen3-Omni-30B-A3B-Captioner}"
MIDASHENG_MODEL="${MIDASHENG_MODEL:-${CAPTION_ROOT}/models/MiDashengLM-7B-1021-BF16}"
QWEN_RUN="${QWEN_RUN:-${CAPTION_ROOT}/DualISL_Train/runs/dual_recursive_8gpu_h100_20260829_run01}"
MIDASHENG_RUN="${MIDASHENG_RUN:-${CAPTION_ROOT}/DualISL_Train/runs/dual_recursive_4gpu_h100_midasheng_20260830_run01}"

RUN_TAG="${EMOTIONTALK_RUN_TAG:-20260901_run01}"
RUN_ROOT="${EMOTIONTALK_RUN_ROOT:-${EMOTION_ROOT}/runs/eight_models_${RUN_TAG}}"
LOG_ROOT="${RUN_ROOT}/logs"
MAX_NEW_TOKENS="${EMOTIONTALK_MAX_NEW_TOKENS:-128}"

METRICS_CACHE="${EMOTIONTALK_METRICS_CACHE:-${EMOTION_ROOT}/cache/aac_metrics}"
HF_CACHE="${EMOTIONTALK_HF_CACHE:-${EMOTION_ROOT}/cache/huggingface}"
XDG_CACHE="${EMOTIONTALK_XDG_CACHE:-${EMOTION_ROOT}/cache/xdg}"
METRICS_BIN="$(dirname "${METRICS_PY}")"
METRICS_PREFIX="$(dirname "${METRICS_BIN}")"
export JAVA_HOME="${JAVA_HOME:-${METRICS_PREFIX}}"
export PATH="${METRICS_BIN}:${PATH}"
export AAC_METRICS_CACHE_PATH="${METRICS_CACHE}"
export HF_HOME="${HF_CACHE}"
export HF_DATASETS_CACHE="${HF_CACHE}/datasets"
export TRANSFORMERS_CACHE="${HF_CACHE}/transformers"
export SENTENCE_TRANSFORMERS_HOME="${HF_CACHE}/sentence_transformers"
export XDG_CACHE_HOME="${XDG_CACHE}"
export HF_HUB_DISABLE_XET=1

MODEL_NAMES=(
  qwen_base qwen8_round0 qwen8_round1 qwen8_round2
  midasheng_base midasheng4_round0 midasheng4_round1 midasheng4_round2
)
BACKENDS=(
  qwen3_omni qwen3_omni qwen3_omni qwen3_omni
  midasheng midasheng midasheng midasheng
)
PYTHONS=(
  "${QWEN_PY}" "${QWEN_PY}" "${QWEN_PY}" "${QWEN_PY}"
  "${MIDASHENG_PY}" "${MIDASHENG_PY}" "${MIDASHENG_PY}" "${MIDASHENG_PY}"
)
MODELS=(
  "${QWEN_MODEL}" "${QWEN_MODEL}" "${QWEN_MODEL}" "${QWEN_MODEL}"
  "${MIDASHENG_MODEL}" "${MIDASHENG_MODEL}" "${MIDASHENG_MODEL}" "${MIDASHENG_MODEL}"
)
ADAPTERS=(
  base
  "${QWEN_RUN}/round_000/checkpoints/caption_final"
  "${QWEN_RUN}/round_001/checkpoints/caption_final"
  "${QWEN_RUN}/round_002/checkpoints/caption_final"
  base
  "${MIDASHENG_RUN}/round_000/checkpoints/caption_final"
  "${MIDASHENG_RUN}/round_001/checkpoints/caption_final"
  "${MIDASHENG_RUN}/round_002/checkpoints/caption_final"
)
ATTN_BACKENDS=(
  flash_attention_2 flash_attention_2 flash_attention_2 flash_attention_2
  sdpa sdpa sdpa sdpa
)

usage() {
  cat <<'EOF'
Usage: bash run_emotiontalk_8gpu.sh COMMAND

Commands:
  check            Validate data, environments, models, adapters, and tests.
  prepare          Regenerate official manifests and prepare metric resources.
  paths            Print run and log locations.
  smoke-inference  Run 1 utterance x 4 tasks for all eight models.
  smoke-score      Score and aggregate all eight smoke runs.
  smoke            check + smoke-inference + smoke-score.
  full-inference   Run 1,929 utterances x 4 tasks for all eight models.
  full-score       Score and aggregate all eight full runs.
  full             check + full-inference + full-score.

GPU mapping comes from EVAL_GPUS (default 0,1,2,3,4,5,6,7).
The order is qwen_base, qwen8_round0/1/2, midasheng_base,
midasheng4_round0/1/2. Override batch sizes with
EMOTIONTALK_QWEN_BATCH_SIZE or EMOTIONTALK_MIDASHENG_BATCH_SIZE.
Every inference output is append-only and resumable.
EOF
}

require_file() { [[ -f "$1" ]] || { printf 'Missing file: %s\n' "$1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { printf 'Missing directory: %s\n' "$1" >&2; exit 1; }; }
require_executable() { [[ -x "$1" ]] || { printf 'Missing executable: %s\n' "$1" >&2; exit 1; }; }

resolve_gpus() {
  local specification="${EVAL_GPUS:-0,1,2,3,4,5,6,7}"
  IFS=',' read -r -a GPUS <<<"${specification}"
  if (( ${#GPUS[@]} != 8 )); then
    printf 'Exactly 8 GPU IDs are required; EVAL_GPUS=%s\n' "${specification}" >&2
    exit 2
  fi
  local seen="," gpu
  for gpu in "${GPUS[@]}"; do
    if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
      printf 'GPU IDs must be non-negative integers; got %s\n' "${gpu}" >&2
      exit 2
    fi
    if [[ "${seen}" == *",${gpu},"* ]]; then
      printf 'Duplicate GPU ID: %s\n' "${gpu}" >&2
      exit 2
    fi
    seen+="${gpu},"
  done
  command -v nvidia-smi >/dev/null
  local visible_count available="," available_id
  visible_count="$(nvidia-smi -L | wc -l)"
  if (( visible_count < 8 )); then
    printf 'Eight visible GPUs are required; nvidia-smi reports %s\n' "${visible_count}" >&2
    exit 1
  fi
  while IFS= read -r available_id; do
    available+=",${available_id},"
  done < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
  for gpu in "${GPUS[@]}"; do
    if [[ "${available}" != *",${gpu},"* ]]; then
      printf 'GPU %s from EVAL_GPUS is not reported by nvidia-smi\n' "${gpu}" >&2
      exit 1
    fi
  done
}

check_all() {
  require_executable "${QWEN_PY}"
  require_executable "${MIDASHENG_PY}"
  require_executable "${METRICS_PY}"
  require_file "${SMOKE_DRIVER}"
  require_file "${EMOTION_ROOT}/run_inference.py"
  require_file "${EMOTION_ROOT}/evaluate.py"
  require_file "${EMOTION_ROOT}/summarize_eight_models.py"
  require_dir "${QWEN_MODEL}"
  require_dir "${MIDASHENG_MODEL}"
  "${QWEN_PY}" - <<'PY'
import peft, qwen_omni_utils, torch, transformers
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
print(f"OK Qwen environment: torch={torch.__version__} transformers={transformers.__version__} peft={peft.__version__}")
PY
  "${MIDASHENG_PY}" - <<'PY'
import peft, torch, transformers
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
print(f"OK MiDasheng environment: torch={torch.__version__} transformers={transformers.__version__} peft={peft.__version__}")
PY
  local adapter
  for adapter in "${ADAPTERS[@]}"; do
    if [[ "${adapter}" != base ]]; then
      require_file "${adapter}/adapter_config.json"
      require_file "${adapter}/adapter_model.safetensors"
      [[ -r "${adapter}/adapter_model.safetensors" ]] || {
        printf 'Unreadable adapter weights: %s\n' "${adapter}" >&2
        exit 1
      }
    fi
  done
  bash "${SMOKE_DRIVER}" check
  "${QWEN_PY}" - "${BENCHMARK_ROOT}" "${QWEN_MODEL}" "${MIDASHENG_MODEL}" \
    "${ADAPTERS[@]}" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from model_adapter_utils import describe_adapter
qwen, midasheng = Path(sys.argv[2]), Path(sys.argv[3])
for index, value in enumerate(sys.argv[4:]):
    if value == "base":
        continue
    base = qwen if index < 4 else midasheng
    descriptor = describe_adapter(Path(value), base)
    print(f"OK adapter={descriptor['path']} sha256={descriptor['weights_sha256']}")
PY
  printf '[DONE] EmotionTalk data, environments, models, and six adapters passed.\n'
}

prepare_all() {
  bash "${SMOKE_DRIVER}" prepare-data
  bash "${SMOKE_DRIVER}" prepare-metrics
  check_all
}

model_output() {
  printf '%s/%s/%s\n' "${RUN_ROOT}" "$2" "$1"
}

validate_predictions() {
  local output="$1" expected="$2" smoke="${3:-false}"
  "${QWEN_PY}" - "${output}/predictions.jsonl" "${expected}" "${smoke}" <<'PY'
import json, sys
from collections import Counter
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
expected = int(sys.argv[2])
smoke = sys.argv[3] == "true"
counts = Counter(row.get("task") for row in rows)
assert counts == {task: expected for task in ("speaker", "style", "emotion", "overall")}, counts
assert all(isinstance(row.get("prediction"), str) and row["prediction"].strip() for row in rows)
assert all(set(row) == {"id", "task", "prediction"} for row in rows)
if smoke:
    assert all("\n" not in row["prediction"] and "\r" not in row["prediction"] for row in rows)
    assert all(any("\u3400" <= char <= "\u9fff" for char in row["prediction"]) for row in rows)
print(f"validated {len(rows)} predictions: {dict(counts)}")
PY
}

run_inference_case() {
  local index="$1" size="$2" gpu="$3"
  local name="${MODEL_NAMES[$index]}"
  local output batch expected
  output="$(model_output "${name}" "${size}")"
  if [[ "${BACKENDS[$index]}" == qwen3_omni ]]; then
    batch="${EMOTIONTALK_QWEN_BATCH_SIZE:-4}"
  else
    batch="${EMOTIONTALK_MIDASHENG_BATCH_SIZE:-4}"
  fi
  [[ "${batch}" =~ ^[1-9][0-9]*$ ]] || { printf 'Invalid batch size: %s\n' "${batch}" >&2; return 2; }
  local -a adapter_args=() selection=()
  [[ "${ADAPTERS[$index]}" == base ]] || adapter_args=(--adapter-dir "${ADAPTERS[$index]}")
  if [[ "${size}" == smoke ]]; then
    selection=(--max-samples 1)
    expected=1
  else
    expected=1929
  fi
  mkdir -p "${output}"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "${PYTHONS[$index]}" "${EMOTION_ROOT}/run_inference.py" \
      --backend "${BACKENDS[$index]}" \
      --tasks all \
      --gpu "${gpu}" \
      --model-path "${MODELS[$index]}" \
      "${adapter_args[@]}" \
      --attn-backend "${ATTN_BACKENDS[$index]}" \
      --batch-size "${batch}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --output-dir "${output}" \
      --resume "${selection[@]}"
  validate_predictions "${output}" "${expected}" "$([[ "${size}" == smoke ]] && echo true || echo false)"
}

wait_for_jobs() {
  local suite="$1" failed=0 index status
  for index in "${!JOB_PIDS[@]}"; do
    if wait "${JOB_PIDS[$index]}"; then
      printf '[DONE] %s %s GPU=%s\n' "${suite}" "${JOB_NAMES[$index]}" "${JOB_GPUS[$index]}"
    else
      status=$?
      printf '[FAILED] %s %s exit=%s log=%s\n' \
        "${suite}" "${JOB_NAMES[$index]}" "${status}" "${JOB_LOGS[$index]}" >&2
      failed=1
    fi
  done
  trap - INT TERM
  (( failed == 0 ))
}

stop_jobs() {
  trap - INT TERM
  printf '[STOP] Terminating launchers; append-only outputs remain resumable.\n' >&2
  local pid
  for pid in "${JOB_PIDS[@]:-}"; do
    # The background shell normally has the model/metric process as its direct
    # child. Signal both so Ctrl-C cannot leave an orphan consuming a GPU.
    pkill -TERM -P "${pid}" 2>/dev/null || true
    kill "${pid}" 2>/dev/null || true
  done
  wait || true
  exit 130
}

run_inference_all() {
  local size="$1"
  resolve_gpus
  local log_dir="${LOG_ROOT}/inference_${size}"
  mkdir -p "${log_dir}"
  JOB_PIDS=(); JOB_NAMES=(); JOB_GPUS=(); JOB_LOGS=()
  local index log
  for index in "${!MODEL_NAMES[@]}"; do
    log="${log_dir}/${MODEL_NAMES[$index]}.log"
    printf '[START] inference %s GPU=%s log=%s\n' "${MODEL_NAMES[$index]}" "${GPUS[$index]}" "${log}"
    run_inference_case "${index}" "${size}" "${GPUS[$index]}" >"${log}" 2>&1 &
    JOB_PIDS+=("$!"); JOB_NAMES+=("${MODEL_NAMES[$index]}")
    JOB_GPUS+=("${GPUS[$index]}"); JOB_LOGS+=("${log}")
  done
  trap stop_jobs INT TERM
  wait_for_jobs "EmotionTalk-${size}-inference"
}

run_score_case() {
  local index="$1" size="$2" gpu="$3"
  local name="${MODEL_NAMES[$index]}" output expected
  output="$(model_output "${name}" "${size}")"
  local -a partial=()
  if [[ "${size}" == smoke ]]; then
    partial=(--allow-partial)
    expected=1
  else
    expected=1929
  fi
  validate_predictions "${output}" "${expected}" "$([[ "${size}" == smoke ]] && echo true || echo false)"
  mkdir -p "${RUN_ROOT}/metrics_tmp/${size}/${name}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  AAC_METRICS_TMP_PATH="${RUN_ROOT}/metrics_tmp/${size}/${name}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 AAC_METRICS_DEVICE=cuda_if_available \
    "${METRICS_PY}" "${EMOTION_ROOT}/evaluate.py" \
      --predictions "${output}/predictions.jsonl" \
      --references "${EMOTION_ROOT}/data/test_references.jsonl" \
      --profile standard_public \
      "${partial[@]}" --require-all-metrics \
      --device cuda_if_available \
      --output-dir "${output}/metrics_standard_public"
}

run_score_all() {
  local size="$1"
  resolve_gpus
  local log_dir="${LOG_ROOT}/score_${size}"
  mkdir -p "${log_dir}"
  JOB_PIDS=(); JOB_NAMES=(); JOB_GPUS=(); JOB_LOGS=()
  local index log
  for index in "${!MODEL_NAMES[@]}"; do
    log="${log_dir}/${MODEL_NAMES[$index]}.log"
    printf '[START] score %s GPU=%s log=%s\n' "${MODEL_NAMES[$index]}" "${GPUS[$index]}" "${log}"
    run_score_case "${index}" "${size}" "${GPUS[$index]}" >"${log}" 2>&1 &
    JOB_PIDS+=("$!"); JOB_NAMES+=("${MODEL_NAMES[$index]}")
    JOB_GPUS+=("${GPUS[$index]}"); JOB_LOGS+=("${log}")
  done
  trap stop_jobs INT TERM
  wait_for_jobs "EmotionTalk-${size}-score"
  local expected=1929
  [[ "${size}" == smoke ]] && expected=1
  "${QWEN_PY}" "${EMOTION_ROOT}/summarize_eight_models.py" \
    --run-root "${RUN_ROOT}/${size}" --expected-count "${expected}"
}

print_paths() {
  printf 'Run root: %s\nLogs: %s\n' "${RUN_ROOT}" "${LOG_ROOT}"
  local size name
  for size in smoke full; do
    for name in "${MODEL_NAMES[@]}"; do model_output "${name}" "${size}"; done
    printf '%s/%s/comparison.md\n' "${RUN_ROOT}" "${size}"
  done
}

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${METRICS_CACHE}" "${HF_CACHE}" "${XDG_CACHE}"
command="${1:-}"
case "${command}" in
  check) check_all ;;
  prepare) prepare_all ;;
  paths) print_paths ;;
  smoke-inference) run_inference_all smoke ;;
  smoke-score) run_score_all smoke ;;
  smoke) check_all; run_inference_all smoke; run_score_all smoke ;;
  full-inference) run_inference_all full ;;
  full-score) run_score_all full ;;
  full) check_all; run_inference_all full; run_score_all full ;;
  -h|--help|help|'') usage ;;
  *) printf 'Unknown command: %s\n' "${command}" >&2; usage >&2; exit 2 ;;
esac
