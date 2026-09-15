#!/usr/bin/env bash
set -Eeuo pipefail
umask 000

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="${ROOT}/run_later_caption_benchmarks_8gpu.sh"
HELPER="${ROOT}/later_caption_eval.py"
export LATER_CAPTION_PROFILE=midasheng_rewardv3
export LATER_CAPTION_EVAL_ROOT="${LATER_CAPTION_EVAL_ROOT:-${ROOT}/later_caption_eval_runs_20260902_run01}"
QWEN_PY="${QWEN_PY:-${ROOT}/../../miniconda3/envs/qwen3-captioner/bin/python}"

case "${1:-help}" in
  inventory|paths|summarize)
    exec bash "${DRIVER}" "$1"
    ;;
  check|status|run) mode="$1" ;;
  help|-h|--help)
    printf 'Usage: bash %s {inventory|check|status|run|summarize|paths}\n' "$0"
    printf 'Evaluate RewardV3 r0-r9 caption_final on EmotionTalk, StyleCap, and ParaSpeechCaps.\n'
    printf 'run: smoke, then full, then verify all 30 full results. Resume matching outputs.\n'
    printf 'status: read-only checkpoint and smoke/full completion checks; no GPU needed.\n'
    printf 'check: validate the existing shared environments, inputs, and eight GPUs.\n'
    printf 'Overrides: EVAL_GPUS=0,1,2,3,4,5,6,7; LATER_CAPTION_EVAL_ROOT=/shared/output\n'
    exit 0
    ;;
  *) printf 'Unknown command: %s\n' "$1" >&2; exit 2 ;;
esac

[[ -x "${QWEN_PY}" ]] || { printf 'Missing Python: %s\n' "${QWEN_PY}" >&2; exit 1; }
export LATER_CAPTION_INVENTORY_FILE
LATER_CAPTION_INVENTORY_FILE="$(mktemp)"
trap 'rm -f -- "${LATER_CAPTION_INVENTORY_FILE}"' EXIT
"${QWEN_PY}" "${HELPER}" inventory --json >"${LATER_CAPTION_INVENTORY_FILE}"
"${QWEN_PY}" - "${LATER_CAPTION_INVENTORY_FILE}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    candidates = json.load(handle)["candidates"]
expected = {f"midasheng_rewardv3_r{number}" for number in range(10)}
if len(candidates) != 10 or {c["candidate_id"] for c in candidates} != expected:
    raise SystemExit("Expected exactly RewardV3 r0-r9; refusing a different evaluation scope.")
bad = [f'{c["candidate_id"]}: {c["status"]}: {c["detail"]}'
       for c in candidates if c["status"] != "ready"]
if bad:
    raise SystemExit("Uncommitted or invalid checkpoints:\n" + "\n".join(bad))
print("Validated all ten committed RewardV3 caption_final checkpoints.")
PY

show_status() {
  "${QWEN_PY}" - "${ROOT}" "${LATER_CAPTION_INVENTORY_FILE}" "${LATER_CAPTION_EVAL_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import later_caption_eval as evaluation

with open(sys.argv[2], encoding="utf-8") as handle:
    candidates = [evaluation.Candidate(**item) for item in json.load(handle)["candidates"]]
output_root = Path(sys.argv[3])
for size in ("smoke", "full"):
    missing = []
    for candidate in candidates:
        for suite in evaluation.BENCHMARKS:
            ok, detail = evaluation.completion_status(
                output_root, suite, size, candidate.candidate_id, candidate
            )
            if not ok:
                missing.append(f"  {suite}/{candidate.candidate_id}: {detail}")
    print(f"{size}: {30 - len(missing)}/30 complete")
    for line in missing:
        print(line)
PY
}

case "${mode}" in
  status) show_status ;;
  check)
    bash "${DRIVER}" check
    show_status
    ;;
  run)
    mkdir -p "${LATER_CAPTION_EVAL_ROOT}"
    exec {RUN_LOCK_FD}>"${LATER_CAPTION_EVAL_ROOT}/.midasheng_rewardv3.lock"
    flock -n "${RUN_LOCK_FD}" || { printf 'A RewardV3 evaluator is already running.\n' >&2; exit 1; }
    bash "${DRIVER}" smoke-ready all
    bash "${DRIVER}" full-ready all
    bash "${DRIVER}" summarize
    "${QWEN_PY}" - "${LATER_CAPTION_EVAL_ROOT}/summary_midasheng_rewardv3/results.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    entries = json.load(handle)["entries"]
expected = {(f"r{number}", suite) for number in range(10)
            for suite in ("emotiontalk", "stylecap", "paraspeechcaps")}
actual = {(e["label"], e["suite"]) for e in entries
          if e["trajectory"] == "midasheng_rewardv3" and e["state"] == "complete"}
if actual != expected:
    raise SystemExit(f"Incomplete RewardV3 full results: {sorted(expected - actual)}")
print("Verified: all 30 RewardV3 full benchmark results are complete.")
PY
    bash "${DRIVER}" paths
    ;;
esac
