#!/usr/bin/env bash
set -Eeuo pipefail

# Dynamically evaluate later Captioner checkpoints. There is one global queue:
# a GPU that finishes any (benchmark, candidate) task immediately takes the next
# eligible task, subject only to the Qwen concurrency cap.

umask 000
BENCHMARK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CAPTION_ROOT="$(cd -- "${BENCHMARK_ROOT}/.." && pwd)"
CLUSTER_ROOT="$(cd -- "${CAPTION_ROOT}/.." && pwd)"
HELPER="${BENCHMARK_ROOT}/later_caption_eval.py"
OUTPUT_ROOT="${LATER_CAPTION_EVAL_ROOT:-${BENCHMARK_ROOT}/later_caption_eval_runs_20260902_run01}"

EMOTION_ROOT="${BENCHMARK_ROOT}/emotiontalk_speech_captioning"
PARA_ROOT="${BENCHMARK_ROOT}/paraspeechcaps/evaluations/qwen3_captioner_attr6"
STYLE_ROOT="${BENCHMARK_ROOT}/stylecap_promptspeech_mcq"
QWEN_PY="${QWEN_PY:-${CLUSTER_ROOT}/miniconda3/envs/qwen3-captioner/bin/python}"
MIDASHENG_PY="${MIDASHENG_PY:-${CLUSTER_ROOT}/miniconda3/envs/midasheng-captioner/bin/python}"
QWEN25_PY="${QWEN25_PY:-${CLUSTER_ROOT}/miniconda3/envs/qwen2_5-omni-3b-captioner/bin/python}"
METRICS_PY="${METRICS_PY:-${CLUSTER_ROOT}/miniconda3/envs/emotiontalk-metrics/bin/python}"

METRICS_CACHE="${EMOTIONTALK_METRICS_CACHE:-${EMOTION_ROOT}/cache/aac_metrics}"
HF_CACHE="${EMOTIONTALK_HF_CACHE:-${EMOTION_ROOT}/cache/huggingface}"
XDG_CACHE="${EMOTIONTALK_XDG_CACHE:-${EMOTION_ROOT}/cache/xdg}"
METRICS_PREFIX="$(dirname "$(dirname "${METRICS_PY}")")"
export JAVA_HOME="${JAVA_HOME:-${METRICS_PREFIX}}"
export PATH="$(dirname "${METRICS_PY}"):${PATH}"
export AAC_METRICS_CACHE_PATH="${METRICS_CACHE}"
export HF_HOME="${HF_CACHE}"
export HF_DATASETS_CACHE="${HF_CACHE}/datasets"
export TRANSFORMERS_CACHE="${HF_CACHE}/transformers"
export SENTENCE_TRANSFORMERS_HOME="${HF_CACHE}/sentence_transformers"
export XDG_CACHE_HOME="${XDG_CACHE}"
export HF_HUB_DISABLE_XET=1

usage() {
  cat <<'EOF'
Usage: bash run_later_caption_benchmarks_8gpu.sh COMMAND [SUITE]

Read-only / no GPU:
  inventory                         List ready/pending/invalid checkpoints.
  paths                             Print output and summary paths.
  summarize                         Merge historical base/r0/r1/r2 and later results.

Validation and dynamic GPU evaluation:
  check                             Validate Bash, 8 GPUs, environments, inputs, and identities.
  smoke-ready {all|paraspeechcaps|stylecap|emotiontalk}
  full-ready  {all|paraspeechcaps|stylecap|emotiontalk}

EVAL_GPUS defaults to 0,1,2,3,4,5,6,7. QWEN_EVAL_CONCURRENCY defaults
to 4. Tasks are globally ordered EmotionTalk > StyleCap > ParaSpeechCaps,
with Qwen before MiDasheng at the same benchmark priority. Bash 5.1+ uses
wait -n -p directly; Bash 5.0 uses a wait -n compatibility path. A completed task
immediately releases its GPU to the next eligible task; there is no batch or
benchmark barrier. Repeating a command resumes partial output and skips tasks
whose counts and adapter identity are already complete.
EOF
}

require_file() { [[ -f "$1" ]] || { printf 'Missing required file: %s\n' "$1" >&2; return 1; }; }
require_dir() { [[ -d "$1" ]] || { printf 'Missing required directory: %s\n' "$1" >&2; return 1; }; }
require_executable() { [[ -x "$1" ]] || { printf 'Missing executable: %s\n' "$1" >&2; return 1; }; }

resolve_gpus() {
  local specification="${EVAL_GPUS:-0,1,2,3,4,5,6,7}"
  IFS=',' read -r -a GPUS <<<"${specification}"
  if (( ${#GPUS[@]} != 8 )); then
    printf 'Exactly 8 GPU IDs are required; EVAL_GPUS=%s\n' "${specification}" >&2
    return 2
  fi
  local seen="," gpu visible="," item
  for gpu in "${GPUS[@]}"; do
    [[ "${gpu}" =~ ^[0-9]+$ ]] || { printf 'Invalid GPU ID: %s\n' "${gpu}" >&2; return 2; }
    [[ "${seen}" != *",${gpu},"* ]] || { printf 'Duplicate GPU ID: %s\n' "${gpu}" >&2; return 2; }
    seen+="${gpu},"
  done
  command -v nvidia-smi >/dev/null || { printf 'nvidia-smi is unavailable.\n' >&2; return 1; }
  while IFS= read -r item; do
    item="${item//[[:space:]]/}"
    visible+="${item},"
  done < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
  for gpu in "${GPUS[@]}"; do
    [[ "${visible}" == *",${gpu},"* ]] || { printf 'Requested GPU %s is not visible.\n' "${gpu}" >&2; return 1; }
  done
}

check_all() {
  if (( BASH_VERSINFO[0] < 5 )); then
    printf 'Bash >= 5.0 is required for dynamic wait -n scheduling; found %s.\n' "${BASH_VERSION}" >&2
    return 1
  fi
  resolve_gpus
  if [[ "${LATER_CAPTION_PROFILE:-legacy}" == qwen25_v1_v2 ]]; then
    require_executable "${QWEN25_PY}"
    "${QWEN25_PY}" - <<'PY'
import peft, torch, transformers, qwen_omni_utils, flash_attn
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.is_bf16_supported(), "BF16 unavailable"
print(f"OK Qwen2.5 environment: torch={torch.__version__} transformers={transformers.__version__}")
PY
    "${QWEN25_PY}" -m unittest discover -s "${BENCHMARK_ROOT}/tests" -p 'test_qwen25_caption_backend.py'
  fi
  require_executable "${QWEN_PY}"
  require_executable "${MIDASHENG_PY}"
  require_executable "${METRICS_PY}"
  require_file "${HELPER}"
  require_file "${EMOTION_ROOT}/run_inference.py"
  require_file "${EMOTION_ROOT}/evaluate.py"
  require_file "${EMOTION_ROOT}/data/test_references.jsonl"
  require_file "${PARA_ROOT}/runs/new_cluster_default/manifest.jsonl"
  require_file "${PARA_ROOT}/run_content_scheme_a.py"
  require_file "${PARA_ROOT}/score_content_scheme_a.py"
  require_file "${STYLE_ROOT}/data/benchmark.jsonl"
  require_file "${STYLE_ROOT}/run_midasheng.py"
  require_file "${STYLE_ROOT}/evaluate.py"
  "${QWEN_PY}" "${HELPER}" validate --historical
  LATER_CAPTION_PROFILE=legacy "${QWEN_PY}" -m unittest "${BENCHMARK_ROOT}/tests/test_later_caption_eval.py"
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
  bash "${EMOTION_ROOT}/run_emotiontalk_smoke.sh" check
  "${QWEN_PY}" -m unittest discover -s "${PARA_ROOT}/tests" -p 'test_*.py'
  "${QWEN_PY}" "${STYLE_ROOT}/validate_benchmark.py"
  "${QWEN_PY}" -m unittest discover -s "${STYLE_ROOT}/tests" -p 'test_*.py'
  printf '[DONE] Dynamic evaluation preflight passed.\n'
}

validate_emotion_predictions() {
  local output="$1" expected="$2"
  "${QWEN_PY}" - "${output}/predictions.jsonl" "${expected}" <<'PY'
import json, sys
from collections import Counter
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
expected = int(sys.argv[2])
counts = Counter(row.get("task") for row in rows)
assert counts == {task: expected for task in ("speaker", "style", "emotion", "overall")}, counts
assert all(isinstance(row.get("prediction"), str) and row["prediction"].strip() for row in rows)
print(f"validated {len(rows)} EmotionTalk predictions")
PY
}

run_emotiontalk() {
  local size="$1" gpu="$2" candidate="$3" family="$4" python="$5" model="$6" adapter="$7" attn="$8" output="$9"
  local backend expected batch
  local -a selection=() partial=()
  if [[ "${family}" == qwen ]]; then
    backend=qwen3_omni; batch="${EMOTIONTALK_QWEN_BATCH_SIZE:-4}"
  elif [[ "${family}" == qwen25 ]]; then
    backend=qwen25; batch="${EMOTIONTALK_QWEN25_BATCH_SIZE:-1}"
  else
    backend=midasheng; batch="${EMOTIONTALK_MIDASHENG_BATCH_SIZE:-4}"
  fi
  if [[ "${size}" == smoke ]]; then
    selection=(--max-samples 1); partial=(--allow-partial); expected=1
  else
    expected=1929
  fi
  mkdir -p "${output}" "${OUTPUT_ROOT}/metrics_tmp/${size}/${candidate}"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "${python}" "${EMOTION_ROOT}/run_inference.py" \
      --backend "${backend}" --tasks all --gpu "${gpu}" --model-path "${model}" \
      --adapter-dir "${adapter}" --attn-backend "${attn}" --batch-size "${batch}" \
      --max-new-tokens "${EMOTIONTALK_MAX_NEW_TOKENS:-128}" --output-dir "${output}" \
      --resume "${selection[@]}"
  validate_emotion_predictions "${output}" "${expected}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  AAC_METRICS_TMP_PATH="${OUTPUT_ROOT}/metrics_tmp/${size}/${candidate}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 AAC_METRICS_DEVICE=cuda_if_available \
    "${METRICS_PY}" "${EMOTION_ROOT}/evaluate.py" \
      --predictions "${output}/predictions.jsonl" \
      --references "${EMOTION_ROOT}/data/test_references.jsonl" \
      --profile standard_public "${partial[@]}" --device cuda_if_available \
      --output-dir "${output}/metrics_standard_public"
}

run_paraspeechcaps() {
  local size="$1" gpu="$2" candidate="$3" family="$4" python="$5" model="$6" adapter="$7" attn="$8" output="$9"
  local backend
  local -a selection=()
  [[ "${family}" == qwen ]] && backend=qwen3 || backend=midasheng
  [[ "${family}" == qwen25 ]] && backend=qwen25
  [[ "${size}" == smoke ]] && selection=(--max-samples 1)
  mkdir -p "${output}/outputs" "${output}/reports"
  CUDA_VISIBLE_DEVICES="${gpu}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "${python}" "${PARA_ROOT}/run_content_scheme_a.py" \
      --manifest "${PARA_ROOT}/runs/new_cluster_default/manifest.jsonl" \
      --output-dir "${output}/outputs" --candidate-name "${candidate}" \
      --backend "${backend}" --model-dir "${model}" --adapter-dir "${adapter}" \
      --attn-backend "${attn}" --resume "${selection[@]}"
  "${QWEN_PY}" "${PARA_ROOT}/score_content_scheme_a.py" \
    --manifest "${output}/outputs/selected_manifest.jsonl" \
    --predictions "${output}/outputs/predictions.jsonl" \
    --output-dir "${output}/reports"
}

run_stylecap() {
  local size="$1" gpu="$2" candidate="$3" family="$4" python="$5" model="$6" adapter="$7" attn="$8" output="$9"
  local backend batch
  local -a selection=()
  if [[ "${family}" == qwen ]]; then
    backend=qwen3; batch="${STYLECAP_QWEN_BATCH_SIZE:-4}"
  elif [[ "${family}" == qwen25 ]]; then
    backend=qwen25; batch="${STYLECAP_QWEN25_BATCH_SIZE:-1}"
  else
    backend=midasheng; batch="${STYLECAP_MIDASHENG_BATCH_SIZE:-4}"
  fi
  [[ "${size}" == smoke ]] && selection=(--max-questions 4)
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${gpu}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "${python}" "${STYLE_ROOT}/run_midasheng.py" \
      --backend "${backend}" --benchmark "${STYLE_ROOT}/data/benchmark.jsonl" \
      --model-dir "${model}" --adapter-dir "${adapter}" --output-dir "${output}" \
      --attn-backend "${attn}" --batch-size "${batch}" --resume "${selection[@]}"
  if [[ "${size}" == full ]]; then
    "${python}" "${STYLE_ROOT}/evaluate.py" \
      --benchmark "${STYLE_ROOT}/data/benchmark.jsonl" \
      --predictions "${output}/predictions.jsonl" \
      --output "${output}/evaluation_summary.json"
  fi
}

run_task() {
  local size="$1" suite="$2" gpu="$3" candidate="$4" family="$5" python="$6" model="$7" adapter="$8" attn="$9" output="${10}"
  case "${suite}" in
    emotiontalk) run_emotiontalk "${size}" "${gpu}" "${candidate}" "${family}" "${python}" "${model}" "${adapter}" "${attn}" "${output}" ;;
    paraspeechcaps) run_paraspeechcaps "${size}" "${gpu}" "${candidate}" "${family}" "${python}" "${model}" "${adapter}" "${attn}" "${output}" ;;
    stylecap) run_stylecap "${size}" "${gpu}" "${candidate}" "${family}" "${python}" "${model}" "${adapter}" "${attn}" "${output}" ;;
    *) printf 'Unknown suite: %s\n' "${suite}" >&2; return 2 ;;
  esac
  "${QWEN_PY}" "${HELPER}" verify-task --output-root "${OUTPUT_ROOT}" \
    --size "${size}" --suite "${suite}" --candidate "${candidate}"
}

declare -a TASK_LINES=() TASK_PENDING=() FREE_GPUS=()
declare -A PID_GPU=() PID_TASK=() PID_FAMILY=() PID_LOG=()
RUNNING=0
QWEN_RUNNING=0
QWEN_LIMIT=4
ANY_FAILED=0

find_eligible_task() {
  local index family
  for index in "${!TASK_LINES[@]}"; do
    [[ "${TASK_PENDING[$index]}" == 1 ]] || continue
    IFS=$'\t' read -r _ _ _ family _ <<<"${TASK_LINES[$index]}"
    if [[ "${family}" == qwen ]] && (( QWEN_RUNNING >= QWEN_LIMIT )); then
      continue
    fi
    printf '%s\n' "${index}"
    return 0
  done
  return 1
}

launch_task() {
  local size="$1" index="$2" gpu="$3"
  local priority suite candidate family python model adapter attn output pid log stamp
  IFS=$'\t' read -r priority suite candidate family python model adapter attn output <<<"${TASK_LINES[$index]}"
  stamp="$(date -u +%Y%m%dT%H%M%S)_${RANDOM}"
  log="${OUTPUT_ROOT}/logs/${size}/${suite}/${candidate}_${stamp}.log"
  mkdir -p "$(dirname "${log}")" "${output}"
  printf '[START] priority=%s suite=%s candidate=%s family=%s GPU=%s log=%s\n' \
    "${priority}" "${suite}" "${candidate}" "${family}" "${gpu}" "${log}"
  (
    set -Eeuo pipefail
    run_task "${size}" "${suite}" "${gpu}" "${candidate}" "${family}" \
      "${python}" "${model}" "${adapter}" "${attn}" "${output}"
  ) >"${log}" 2>&1 &
  pid=$!
  PID_GPU["${pid}"]="${gpu}"
  PID_TASK["${pid}"]="${suite}/${candidate}"
  PID_FAMILY["${pid}"]="${family}"
  PID_LOG["${pid}"]="${log}"
  TASK_PENDING[$index]=0
  ((RUNNING += 1))
  if [[ "${family}" == qwen ]]; then
    ((QWEN_RUNNING += 1))
  fi
}

terminate_tree() {
  local parent="$1" child
  while IFS= read -r child; do
    [[ -n "${child}" ]] && terminate_tree "${child}"
  done < <(pgrep -P "${parent}" 2>/dev/null || true)
  kill -TERM "${parent}" 2>/dev/null || true
}

stop_all() {
  trap - INT TERM
  printf '[STOP] Terminating all evaluation process trees; existing outputs remain resumable.\n' >&2
  local pid
  for pid in "${!PID_GPU[@]}"; do
    terminate_tree "${pid}"
  done
  wait || true
  exit 130
}

record_failure() {
  local task="$1" gpu="$2" status="$3" log="$4"
  printf '{"timestamp_utc":"%s","task":"%s","gpu":"%s","exit_code":%s,"log":"%s"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${task}" "${gpu}" "${status}" "${log}" \
    >>"${OUTPUT_ROOT}/failures.jsonl"
}

finish_running_task() {
  local done_pid="$1" status="$2"
  local gpu task family log
  gpu="${PID_GPU[${done_pid}]}"; task="${PID_TASK[${done_pid}]}"
  family="${PID_FAMILY[${done_pid}]}"; log="${PID_LOG[${done_pid}]}"
  unset 'PID_GPU['"${done_pid}"']' 'PID_TASK['"${done_pid}"']' \
    'PID_FAMILY['"${done_pid}"']' 'PID_LOG['"${done_pid}"']'
  RUNNING=$((RUNNING - 1))
  if [[ "${family}" == qwen ]]; then
    QWEN_RUNNING=$((QWEN_RUNNING - 1))
  fi
  FREE_GPUS+=("${gpu}")
  if (( status == 0 )); then
    printf '[DONE] %s GPU=%s; GPU immediately returned to queue\n' "${task}" "${gpu}"
  else
    printf '[FAILED] %s GPU=%s exit=%s log=%s; continuing queue\n' "${task}" "${gpu}" "${status}" "${log}" >&2
    record_failure "${task}" "${gpu}" "${status}" "${log}"
    ANY_FAILED=1
  fi
}

run_dynamic_queue() {
  local size="$1" suite_selection="$2"
  resolve_gpus
  QWEN_LIMIT="${QWEN_EVAL_CONCURRENCY:-4}"
  [[ "${QWEN_LIMIT}" =~ ^[1-8]$ ]] || { printf 'QWEN_EVAL_CONCURRENCY must be an integer from 1 to 8.\n' >&2; return 2; }
  mkdir -p "${OUTPUT_ROOT}/inventory" "${OUTPUT_ROOT}/logs" "${METRICS_CACHE}" "${HF_CACHE}" "${XDG_CACHE}"
  # QuarkFS may allow cross-UID writes while rejecting chmod by a non-owner.
  # Permissions are therefore best-effort; actual writability is the gate.
  chmod 2777 "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/inventory" "${OUTPUT_ROOT}/logs" 2>/dev/null || true
  local writable_dir
  for writable_dir in "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/inventory" "${OUTPUT_ROOT}/logs"; do
    [[ -w "${writable_dir}" ]] || {
      printf 'Output directory is not writable on this server: %s\n' "${writable_dir}" >&2
      return 1
    }
  done
  command -v flock >/dev/null || { printf 'flock is required to protect GPUs from overlapping schedulers.\n' >&2; return 1; }
  exec {SCHEDULER_LOCK_FD}>"${OUTPUT_ROOT}/.scheduler.lock"
  flock -n "${SCHEDULER_LOCK_FD}" || {
    printf 'Another scheduler already holds %s/.scheduler.lock\n' "${OUTPUT_ROOT}" >&2
    return 1
  }
  local snapshot task_file
  snapshot="${OUTPUT_ROOT}/inventory/${size}_${suite_selection}_$(date -u +%Y%m%dT%H%M%S)_pid$$.json"
  if [[ -n "${LATER_CAPTION_INVENTORY_FILE:-}" ]]; then
    cp -- "${LATER_CAPTION_INVENTORY_FILE}" "${snapshot}"
  else
    "${QWEN_PY}" "${HELPER}" inventory --json >"${snapshot}"
  fi
  task_file="$(mktemp)"
  trap 'rm -f -- "${task_file}"' RETURN
  local -a task_args=(tasks --output-root "${OUTPUT_ROOT}" --size "${size}" \
    --suite "${suite_selection}" --inventory-file "${snapshot}")
  [[ "${size}" == full ]] && task_args+=(--require-smoke)
  "${QWEN_PY}" "${HELPER}" "${task_args[@]}" >"${task_file}"
  mapfile -t TASK_LINES <"${task_file}"
  rm -f -- "${task_file}"
  trap - RETURN
  if (( ${#TASK_LINES[@]} == 0 )); then
    printf '[DONE] No incomplete %s tasks for suite=%s.\n' "${size}" "${suite_selection}"
    return 0
  fi
  TASK_PENDING=(); FREE_GPUS=("${GPUS[@]}")
  local index
  for index in "${!TASK_LINES[@]}"; do TASK_PENDING[$index]=1; done
  PID_GPU=(); PID_TASK=(); PID_FAMILY=(); PID_LOG=()
  RUNNING=0; QWEN_RUNNING=0; ANY_FAILED=0
  trap stop_all INT TERM
  printf '[QUEUE] tasks=%s GPUs=%s qwen_limit=%s snapshot=%s\n' \
    "${#TASK_LINES[@]}" "${GPUS[*]}" "${QWEN_LIMIT}" "${snapshot}"

  local eligible gpu done_pid status pid
  while :; do
    while (( ${#FREE_GPUS[@]} > 0 )); do
      if ! eligible="$(find_eligible_task)"; then break; fi
      gpu="${FREE_GPUS[0]}"
      FREE_GPUS=("${FREE_GPUS[@]:1}")
      launch_task "${size}" "${eligible}" "${gpu}"
    done
    if (( RUNNING == 0 )); then break; fi
    if (( BASH_VERSINFO[0] > 5 || BASH_VERSINFO[1] >= 1 )); then
      done_pid=""
      if wait -n -p done_pid; then status=0; else status=$?; fi
      finish_running_task "${done_pid}" "${status}"
    else
      # Bash 5.0 has wait -n but not -p. Once any child finishes, scan the
      # tracked PIDs and reap every process that is no longer alive. This can
      # free several GPUs at once and never waits for unrelated live tasks.
      if wait -n; then :; else :; fi
      for pid in "${!PID_GPU[@]}"; do
        if ! kill -0 "${pid}" 2>/dev/null; then
          if wait "${pid}"; then status=0; else status=$?; fi
          finish_running_task "${pid}" "${status}"
        fi
      done
    fi
  done
  trap - INT TERM
  if (( ANY_FAILED != 0 )); then
    printf '[FAILED] Queue drained with one or more failed tasks. Rerun the same command to resume.\n' >&2
    return 1
  fi
  printf '[DONE] Queue drained successfully: size=%s suite=%s\n' "${size}" "${suite_selection}"
}

print_paths() {
  local summary_dir=summary
  [[ "${LATER_CAPTION_PROFILE:-legacy}" == midasheng_rewardv2_v3 ]] && summary_dir=summary_midasheng_rewardv2_v3
  [[ "${LATER_CAPTION_PROFILE:-legacy}" == midasheng_rewardv3 ]] && summary_dir=summary_midasheng_rewardv3
  [[ "${LATER_CAPTION_PROFILE:-legacy}" == qwen25_v1_v2 ]] && summary_dir=summary_qwen25_v1_v2
  printf 'Evaluation root: %s\n' "${OUTPUT_ROOT}"
  printf 'Logs:            %s/logs\n' "${OUTPUT_ROOT}"
  printf 'Summary JSON:    %s/%s/results.json\n' "${OUTPUT_ROOT}" "${summary_dir}"
  printf 'Summary CSV:     %s/%s/results.csv\n' "${OUTPUT_ROOT}" "${summary_dir}"
  printf 'Summary Markdown:%s/%s/results.md\n' "${OUTPUT_ROOT}" "${summary_dir}"
}

command="${1:-}"
suite="${2:-all}"
case "${suite}" in all|paraspeechcaps|stylecap|emotiontalk) ;; *) usage >&2; exit 2 ;; esac
case "${command}" in
  inventory) "${QWEN_PY}" "${HELPER}" inventory ;;
  check) check_all ;;
  smoke-ready) run_dynamic_queue smoke "${suite}" ;;
  full-ready) run_dynamic_queue full "${suite}" ;;
  summarize) "${QWEN_PY}" "${HELPER}" summarize --output-root "${OUTPUT_ROOT}" ;;
  paths) print_paths ;;
  help|-h|--help|'') usage ;;
  *) printf 'Unknown command: %s\n' "${command}" >&2; usage >&2; exit 2 ;;
esac
