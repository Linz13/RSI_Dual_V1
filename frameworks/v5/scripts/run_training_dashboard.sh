#!/usr/bin/env bash
# Foreground supervisor: Ctrl-C stops only this bridge and its TensorBoard child.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${1:-$ROOT/runs/midasheng_v5_run01}"
LABEL="${2:-v5}"
PORT="${3:-6006}"
MONITOR_ENV="${DUALISL_MONITOR_ENV:-$ROOT/.monitor-venv}"
if [[ ! "$LABEL" =~ ^[a-zA-Z0-9_-]+$ || ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1024 || PORT > 65535 )); then
  echo 'Usage: bash scripts/run_training_dashboard.sh [ABS_RUN_DIR] [unique_label] [port 1024..65535]' >&2
  exit 2
fi
if [[ ! -x "$MONITOR_ENV/bin/tensorboard" ]]; then
  echo 'Run bash scripts/setup_monitor_env.sh first.' >&2
  exit 1
fi
# Fail before starting the exporter if the port is occupied.
"$MONITOR_ENV/bin/python" - "$PORT" <<'PY'
import socket, sys
with socket.socket() as sock:
    sock.bind(('127.0.0.1', int(sys.argv[1])))
PY
children=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${children[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${children[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"$MONITOR_ENV/bin/python" "$ROOT/scripts/export_training_dashboard.py" \
  --run-dir "$SOURCE" --output-dir "$ROOT/monitoring/$LABEL" --watch 10 &
children+=("$!")
"$MONITOR_ENV/bin/tensorboard" --logdir "$ROOT/monitoring" --host 127.0.0.1 --port "$PORT" --load_fast=false &
children+=("$!")
echo "Open forwarded http://127.0.0.1:$PORT; watching $SOURCE"
wait -n "${children[@]}"
