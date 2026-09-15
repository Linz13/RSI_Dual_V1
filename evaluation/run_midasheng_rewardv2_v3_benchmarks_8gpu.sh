#!/usr/bin/env bash
set -Eeuo pipefail
umask 000

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${ROOT}/run_later_caption_benchmarks_8gpu.sh"
export LATER_CAPTION_PROFILE=midasheng_rewardv2_v3
export LATER_CAPTION_EVAL_ROOT="${LATER_CAPTION_EVAL_ROOT:-${ROOT}/later_caption_eval_runs_20260902_run01}"
QWEN_PY="${QWEN_PY:-${ROOT}/../../miniconda3/envs/qwen3-captioner/bin/python}"
command="${1:-help}"

case "${command}" in
  inventory|paths|summarize|check)
    exec bash "${RUNNER}" "${command}"
    ;;
  run|watch) ;;
  *)
    printf 'Usage: bash %s {inventory|check|run|watch|summarize|paths}\n' "$0"
    printf 'run: smoke then full for a frozen snapshot of committed V2 r0-r19 and V3 r0-r9.\n'
    printf 'watch: repeat until all 30 checkpoints are committed and evaluated.\n'
    printf 'Completed results are reused. POLL_SECONDS defaults to 300.\n'
    [[ "${command}" == help ]] && exit 0
    exit 2
    ;;
esac

[[ "${POLL_SECONDS:-300}" =~ ^[1-9][0-9]*$ ]] || { printf 'Invalid POLL_SECONDS\n' >&2; exit 2; }
mkdir -p "${LATER_CAPTION_EVAL_ROOT}/inventory"
exec {WATCH_LOCK_FD}>"${LATER_CAPTION_EVAL_ROOT}/.midasheng_rewardv2_v3.lock"
flock -n "${WATCH_LOCK_FD}" || { printf 'A V2/V3 evaluator is already running.\n' >&2; exit 1; }
export LATER_CAPTION_INVENTORY_FILE
LATER_CAPTION_INVENTORY_FILE="$(mktemp "${LATER_CAPTION_EVAL_ROOT}/inventory/v2_v3_XXXXXXXX.json")"
trap 'rm -f -- "${LATER_CAPTION_INVENTORY_FILE}"' EXIT

while :; do
  "${QWEN_PY}" "${ROOT}/later_caption_eval.py" inventory --json >"${LATER_CAPTION_INVENTORY_FILE}"
  counts="$("${QWEN_PY}" - "${LATER_CAPTION_INVENTORY_FILE}" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    inventory = json.load(handle)
for candidate in inventory["candidates"]:
    if candidate["status"] == "invalid":
        print(f'{candidate["candidate_id"]}: {candidate["detail"]}', file=sys.stderr)
if inventory["counts"]["invalid"]:
    sys.exit(1)
print(inventory["counts"]["ready"], inventory["counts"]["pending"])
PY
  )"
  read -r ready pending <<<"${counts}"
  printf '[INVENTORY] committed=%s pending=%s\n' "${ready}" "${pending}"
  bash "${RUNNER}" smoke-ready all
  bash "${RUNNER}" full-ready all
  bash "${RUNNER}" summarize
  if [[ "${command}" == run ]] || (( pending == 0 )); then
    printf '[DONE] Snapshot evaluated; uncommitted checkpoints=%s\n' "${pending}"
    break
  fi
  printf '[WAIT] Checking for new commits in %s seconds.\n' "${POLL_SECONDS:-300}"
  sleep "${POLL_SECONDS:-300}"
done
