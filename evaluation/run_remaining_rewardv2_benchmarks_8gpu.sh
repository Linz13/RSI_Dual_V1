#!/usr/bin/env bash
set -Eeuo pipefail

# Complete the three benchmark suites for MiDasheng RewardV2 r4-r9.
# Existing, identity-matched outputs are reused. The underlying scheduler takes
# a frozen checkpoint inventory and protects the shared output root with flock.

umask 000
BENCHMARK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="${BENCHMARK_ROOT}/run_later_caption_benchmarks_8gpu.sh"
HELPER="${BENCHMARK_ROOT}/later_caption_eval.py"
CLUSTER_ROOT="$(cd -- "${BENCHMARK_ROOT}/../.." && pwd)"

export LATER_CAPTION_EVAL_ROOT="${LATER_CAPTION_EVAL_ROOT:-${BENCHMARK_ROOT}/later_caption_eval_runs_20260902_run01}"
QWEN_PY="${QWEN_PY:-${CLUSTER_ROOT}/miniconda3/envs/qwen3-captioner/bin/python}"

usage() {
  cat <<'EOF'
Usage: bash run_remaining_rewardv2_benchmarks_8gpu.sh {check|run|status}

  check   Validate GPUs, environments, benchmark inputs, checkpoint identities,
          and confirm RewardV2 r0-r9 are all committed and ready.
  run     Resume missing RewardV2 r4-r9 smoke/full evaluations for all three
          benchmarks, regenerate summaries, and verify the final matrix.
  status  Show the currently incomplete smoke/full tasks without using GPUs.

Environment overrides:
  EVAL_GPUS=0,1,2,3,4,5,6,7
  QWEN_EVAL_CONCURRENCY=4
  LATER_CAPTION_EVAL_ROOT=/shared/path/to/later_caption_eval_runs_20260902_run01
  QWEN_PY=/path/to/qwen3-captioner/bin/python
EOF
}

require_file() {
  [[ -f "$1" ]] || { printf 'Missing file: %s\n' "$1" >&2; exit 1; }
}

make_inventory() {
  INVENTORY_FILE="$(mktemp)"
  "${QWEN_PY}" "${HELPER}" inventory --json >"${INVENTORY_FILE}"
}

validate_reward_inventory() {
  "${QWEN_PY}" - "${INVENTORY_FILE}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
reward = {
    item["candidate_id"]: item
    for item in payload.get("candidates", [])
    if item.get("trajectory") == "midasheng_rewardv2"
}
expected = {f"midasheng_rewardv2_r{number}" for number in range(10)}
missing = sorted(expected - set(reward))
not_ready = sorted(
    (candidate_id, reward[candidate_id].get("status"), reward[candidate_id].get("detail", ""))
    for candidate_id in expected & reward.keys()
    if reward[candidate_id].get("status") != "ready"
)
if missing or not_ready:
    if missing:
        print("Missing RewardV2 candidates: " + ", ".join(missing), file=sys.stderr)
    for candidate_id, status, detail in not_ready:
        print(f"RewardV2 candidate is not ready: {candidate_id}: {status}: {detail}", file=sys.stderr)
    raise SystemExit(1)
print("Validated RewardV2 checkpoints: r0-r9 are committed and ready.")
PY
}

task_file_for_size() {
  local size="$1" task_file="$2"
  "${QWEN_PY}" "${HELPER}" tasks \
    --output-root "${LATER_CAPTION_EVAL_ROOT}" \
    --size "${size}" --suite all --inventory-file "${INVENTORY_FILE}" \
    >"${task_file}"
}

validate_remaining_scope() {
  local size="$1" task_file
  task_file="$(mktemp)"
  task_file_for_size "${size}" "${task_file}"
  "${QWEN_PY}" - "${size}" "${task_file}" <<'PY'
import sys

size, path = sys.argv[1:]
rows = [line.rstrip("\n").split("\t") for line in open(path, encoding="utf-8") if line.strip()]
allowed = {f"midasheng_rewardv2_r{number}" for number in range(4, 10)}
unexpected = sorted({row[2] for row in rows if len(row) >= 3 and row[2] not in allowed})
if unexpected:
    print(
        f"Refusing to run: incomplete {size} tasks exist outside RewardV2 r4-r9: "
        + ", ".join(unexpected),
        file=sys.stderr,
    )
    raise SystemExit(1)
print(f"Incomplete {size} tasks in scope: {len(rows)}")
for row in rows:
    print(f"  {row[1]}/{row[2]}")
PY
  rm -f -- "${task_file}"
}

verify_final_summary() {
  local summary="${LATER_CAPTION_EVAL_ROOT}/summary/results.json"
  "${QWEN_PY}" - "${summary}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
entries = {
    (entry["label"], entry["suite"]): entry["state"]
    for entry in payload.get("entries", [])
    if entry.get("trajectory") == "midasheng_rewardv2"
}
labels = ["base", *(f"r{number}" for number in range(10))]
suites = ("emotiontalk", "paraspeechcaps", "stylecap")
incomplete = [
    f"{label}/{suite}={entries.get((label, suite), 'absent')}"
    for label in labels
    for suite in suites
    if entries.get((label, suite)) != "complete"
]
if incomplete:
    print("RewardV2 summary remains incomplete: " + ", ".join(incomplete), file=sys.stderr)
    raise SystemExit(1)
print("Verified final RewardV2 matrix: base and r0-r9 are complete on all three benchmarks.")
PY
}

require_file "${DRIVER}"
require_file "${HELPER}"
[[ -x "${QWEN_PY}" ]] || { printf 'Python is not executable: %s\n' "${QWEN_PY}" >&2; exit 1; }

mode="${1:-}"
case "${mode}" in
  check)
    bash "${DRIVER}" check
    make_inventory
    trap 'rm -f -- "${INVENTORY_FILE:-}"' EXIT
    validate_reward_inventory
    validate_remaining_scope smoke
    validate_remaining_scope full
    ;;
  status)
    make_inventory
    trap 'rm -f -- "${INVENTORY_FILE:-}"' EXIT
    validate_reward_inventory
    validate_remaining_scope smoke
    validate_remaining_scope full
    ;;
  run)
    make_inventory
    trap 'rm -f -- "${INVENTORY_FILE:-}"' EXIT
    validate_reward_inventory
    validate_remaining_scope smoke
    bash "${DRIVER}" smoke-ready all

    rm -f -- "${INVENTORY_FILE}"
    make_inventory
    validate_reward_inventory
    validate_remaining_scope full
    bash "${DRIVER}" full-ready all

    bash "${DRIVER}" summarize
    verify_final_summary
    printf 'Summary: %s/summary/results.md\n' "${LATER_CAPTION_EVAL_ROOT}"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
