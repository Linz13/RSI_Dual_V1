#!/usr/bin/env bash
set -Eeuo pipefail
umask 000

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="${ROOT}/run_later_caption_benchmarks_8gpu.sh"
export LATER_CAPTION_PROFILE=qwen25_v1_v2
export LATER_CAPTION_EVAL_ROOT="${LATER_CAPTION_EVAL_ROOT:-${ROOT}/qwen25_v1_v2_caption_eval_runs_20260907_run01}"
QWEN_PY="${QWEN_PY:-${ROOT}/../../miniconda3/envs/qwen3-captioner/bin/python}"

case "${1:-help}" in
  inventory|check|summarize|paths) exec bash "${DRIVER}" "$1" ;;
  status|run) mode="$1" ;;
  help|-h|--help)
    printf 'Usage: bash %s {inventory|check|status|run|summarize|paths}\n' "$0"
    printf 'Evaluate committed Qwen2.5-Omni-3B V1/V2 caption_final checkpoints on three suites.\n'
    printf 'run freezes the current inventory, runs smoke then full, and verifies the snapshot.\n'
    printf 'Pending rounds are reported and skipped. Rerun to pick up new commits or resume.\n'
    printf 'Eight GPUs: EVAL_GPUS=0,1,2,3,4,5,6,7. Outputs: LATER_CAPTION_EVAL_ROOT.\n'
    exit 0 ;;
  *) printf 'Unknown command: %s\n' "$1" >&2; exit 2 ;;
esac

export LATER_CAPTION_INVENTORY_FILE
LATER_CAPTION_INVENTORY_FILE="$(mktemp)"
trap 'rm -f -- "${LATER_CAPTION_INVENTORY_FILE}"' EXIT
"${QWEN_PY}" "${ROOT}/later_caption_eval.py" inventory --json >"${LATER_CAPTION_INVENTORY_FILE}" || {
  "${QWEN_PY}" "${ROOT}/later_caption_eval.py" inventory
  exit 1
}
"${QWEN_PY}" - "${LATER_CAPTION_INVENTORY_FILE}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    inventory = json.load(handle)
print("Inventory:", inventory["counts"])
for c in inventory["candidates"]:
    if c["status"] != "ready":
        print(c["candidate_id"], c["status"], c["detail"])
if inventory["counts"]["ready"] == 0:
    raise SystemExit("No committed checkpoints are ready.")
PY

if [[ "${mode}" == status ]]; then
  for size in smoke full; do
    printf '\nIncomplete %s tasks (empty means none):\n' "${size}"
    "${QWEN_PY}" "${ROOT}/later_caption_eval.py" tasks \
      --output-root "${LATER_CAPTION_EVAL_ROOT}" --size "${size}" --suite all \
      --inventory-file "${LATER_CAPTION_INVENTORY_FILE}"
  done
  exit 0
fi

mkdir -p "${LATER_CAPTION_EVAL_ROOT}"
exec {RUN_LOCK_FD}>"${LATER_CAPTION_EVAL_ROOT}/.qwen25_v1_v2.lock"
flock -n "${RUN_LOCK_FD}" || { printf 'A Qwen2.5 evaluator is already running.\n' >&2; exit 1; }
bash "${DRIVER}" smoke-ready all
bash "${DRIVER}" full-ready all
"${QWEN_PY}" - "${ROOT}" "${LATER_CAPTION_INVENTORY_FILE}" "${LATER_CAPTION_EVAL_ROOT}" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import later_caption_eval as e
with open(sys.argv[2], encoding="utf-8") as handle:
    candidates = [e.Candidate(**c) for c in json.load(handle)["candidates"] if c["status"] == "ready"]
for c in candidates:
    for suite in e.BENCHMARKS:
        ok, detail = e.completion_status(Path(sys.argv[3]), suite, "full", c.candidate_id, c)
        if not ok:
            raise SystemExit(f"Incomplete: {c.candidate_id}/{suite}: {detail}")
print(f"Verified {len(candidates) * 3} full tasks for the committed snapshot.")
PY
bash "${DRIVER}" summarize
bash "${DRIVER}" paths
