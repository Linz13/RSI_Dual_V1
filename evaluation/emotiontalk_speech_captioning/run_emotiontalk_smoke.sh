#!/usr/bin/env bash
set -Eeuo pipefail

# Reproducible, resumable Qwen3-Omni base smoke for EmotionTalk.
# This entry point deliberately does not launch the full 1,929 x 4 inference.
umask 000

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAPTION_ROOT="$(cd "$ROOT/../.." && pwd)"
QWEN_PY="${QWEN_PY:-/data/L202500147/miniconda3/envs/qwen3-captioner/bin/python}"
METRICS_PY="${METRICS_PY:-/data/L202500147/miniconda3/envs/emotiontalk-metrics/bin/python}"
MODEL_PATH="${MODEL_PATH:-$CAPTION_ROOT/models/Qwen3-Omni-30B-A3B-Captioner}"
SMOKE_OUTPUT="${SMOKE_OUTPUT:-$ROOT/runs/smoke_qwen3_base_20260901_run04}"
METRICS_OUTPUT="${METRICS_OUTPUT:-$SMOKE_OUTPUT/metrics_standard_public}"
METRICS_CACHE="${METRICS_CACHE:-$ROOT/cache/aac_metrics}"
METRICS_TMP="${METRICS_TMP:-$ROOT/cache/aac_tmp}"
HF_CACHE="${HF_CACHE:-$ROOT/cache/huggingface}"
GPU="${GPU:-0}"

export AAC_METRICS_CACHE_PATH="$METRICS_CACHE"
export AAC_METRICS_TMP_PATH="$METRICS_TMP"
export HF_HOME="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE/datasets"
export TRANSFORMERS_CACHE="$HF_CACHE/transformers"
export SENTENCE_TRANSFORMERS_HOME="$HF_CACHE/sentence_transformers"
export XDG_CACHE_HOME="$ROOT/cache/xdg"
# The Xet CDN is not reachable on some compute nodes; force the regular
# Hugging Face HTTP range downloader, which is resumable in the same cache.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# Calling the metrics interpreter by absolute path does not activate its
# Conda environment, so expose the environment's Java runtime explicitly.
METRICS_BIN="$(dirname "$METRICS_PY")"
METRICS_PREFIX="$(dirname "$METRICS_BIN")"
export JAVA_HOME="${JAVA_HOME:-$METRICS_PREFIX}"
export PATH="$METRICS_BIN:$PATH"

mkdir -p "$SMOKE_OUTPUT" "$METRICS_OUTPUT" "$METRICS_CACHE" "$METRICS_TMP" \
  "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" \
  "$SENTENCE_TRANSFORMERS_HOME" "$XDG_CACHE_HOME"

usage() {
  cat <<'EOF'
Usage: bash run_emotiontalk_smoke.sh <command>

Commands:
  prepare-data   Extract official test WAVs and regenerate manifests/provenance.
  check          Full manifest/audio validation, unit tests, and metric imports.
  prepare-metrics Download AAC/FENSE/Chinese-BERTScore/MS-CLAP resources.
  smoke-local    Run one Qwen3-Omni base sample for all four tasks on one GPU.
  smoke-score    Score the four smoke predictions with all seven metrics.
  resume-check   Re-run smoke-local and verify completed events are unchanged.
  smoke          Run prepare-data, check, prepare-metrics, inference, scoring, resume check.
EOF
}

check_metrics_environment() {
  command -v java >/dev/null || { echo "Java runtime not found in metrics environment" >&2; return 1; }
  "$METRICS_PY" - <<'PY'
import aac_metrics, torch, torchaudio, torchvision
print(f"torch={torch.__version__}")
print(f"torchaudio={torchaudio.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"aac_metrics={getattr(aac_metrics, '__version__', 'unknown')}")
assert torch.__version__.startswith("2.6.0+cu124"), torch.__version__
assert torchaudio.__version__.startswith("2.6.0+cu124"), torchaudio.__version__
assert torchvision.__version__.startswith("0.21.0+cu124"), torchvision.__version__
PY
}

prepare_data() {
  "$QWEN_PY" "$ROOT/scripts/prepare_benchmark.py" --root "$ROOT"
  "$QWEN_PY" "$ROOT/validate_benchmark.py" --root "$ROOT"
}

prepare_metrics() {
  check_metrics_environment
  AAC_METRICS_DEVICE=cpu "$METRICS_PY" -m aac_metrics.download \
    --cache_path "$METRICS_CACHE" --tmp_path "$METRICS_TMP" \
    --bert_score false --verbose 1
  HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
    SENTENCE_TRANSFORMERS_HOME="$SENTENCE_TRANSFORMERS_HOME" \
    "$METRICS_PY" - <<'PY'
from aac_metrics.classes.bert_score_mrefs import BERTScoreMRefs
BERTScoreMRefs(model="google-bert/bert-base-chinese", device="cpu", verbose=1)
print("Chinese BERTScore model ready")
PY
}

check() {
  "$QWEN_PY" "$ROOT/validate_benchmark.py" --root "$ROOT"
  (cd "$ROOT" && "$QWEN_PY" -m pytest -q)
  check_metrics_environment
}

assert_smoke_predictions() {
  SMOKE_OUTPUT="$SMOKE_OUTPUT" "$QWEN_PY" - <<'PY'
import json, os
from pathlib import Path
path = Path(os.environ["SMOKE_OUTPUT"]) / "predictions.jsonl"
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
assert len(rows) == 4, f"expected 4 predictions, got {len(rows)}"
assert {(r.get("id"), r.get("task")) for r in rows}.__len__() == 4
assert {r.get("task") for r in rows} == {"speaker", "style", "emotion", "overall"}
for row in rows:
    assert isinstance(row.get("prediction"), str) and row["prediction"].strip(), row
    assert "\n" not in row["prediction"] and "\r" not in row["prediction"], row
    assert any("\u3400" <= char <= "\u9fff" for char in row["prediction"]), row
    assert not row.get("error"), row
print(f"validated {len(rows)}/4 non-empty smoke predictions")
PY
}

smoke_local() {
  test -d "$MODEL_PATH"
  CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$QWEN_PY" "$ROOT/run_inference.py" \
      --backend qwen3_omni --tasks all --max-samples 1 --gpu 0 \
      --model-path "$MODEL_PATH" --output-dir "$SMOKE_OUTPUT" \
      --max-new-tokens 128 --resume
  assert_smoke_predictions
}

smoke_score() {
  check_metrics_environment
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 AAC_METRICS_DEVICE=cuda_if_available "$METRICS_PY" "$ROOT/evaluate.py" \
    --predictions "$SMOKE_OUTPUT/predictions.jsonl" \
    --references "$ROOT/data/test_references.jsonl" \
    --profile standard_public --allow-partial --require-all-metrics \
    --device cuda_if_available --output-dir "$METRICS_OUTPUT"
  SMOKE_OUTPUT="$SMOKE_OUTPUT" "$QWEN_PY" - <<'PY'
import json, os
from pathlib import Path
report = json.loads((Path(os.environ["SMOKE_OUTPUT"]) / "metrics_standard_public/scores.json").read_text())
assert set(report["tasks"]) == {"speaker", "style", "emotion", "overall"}
for task, value in report["tasks"].items():
    metrics = {k: v for k, v in value["metrics"].items() if k != "aac_metrics_version"}
    assert len(metrics) == 7, (task, metrics)
    bad = {k: v for k, v in metrics.items() if v.get("status") != "ok"}
    assert not bad, (task, bad)
print("validated all four tasks x seven metrics: status=ok")
PY
}

resume_check() {
  test -f "$SMOKE_OUTPUT/events.jsonl"
  before="$(sha256sum "$SMOKE_OUTPUT/events.jsonl" | awk '{print $1}')"
  smoke_local
  after="$(sha256sum "$SMOKE_OUTPUT/events.jsonl" | awk '{print $1}')"
  test "$before" = "$after" || { echo "resume changed completed events" >&2; return 1; }
  echo "resume check passed: completed events unchanged"
}

command="${1:-}"
case "$command" in
  prepare-data) prepare_data ;;
  check) check ;;
  prepare-metrics) prepare_metrics ;;
  smoke-local) smoke_local ;;
  smoke-score) smoke_score ;;
  resume-check) resume_check ;;
  smoke) prepare_data; check; prepare_metrics; smoke_local; smoke_score; resume_check ;;
  *) usage; exit 2 ;;
esac
