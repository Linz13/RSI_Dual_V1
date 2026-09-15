#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MONITOR_ENV="${DUALISL_MONITOR_ENV:-$ROOT/.monitor-venv}"
PYTHON="${DUALISL_MONITOR_PYTHON:-/data/L202500147/miniconda3/envs/qwen3-captioner/bin/python}"
if [[ ! -e "$MONITOR_ENV" ]]; then
  "$PYTHON" -m venv "$MONITOR_ENV"
elif [[ ! -f "$MONITOR_ENV/pyvenv.cfg" ]]; then
  echo "Refusing non-venv target: $MONITOR_ENV" >&2
  exit 1
fi
if ! "$MONITOR_ENV/bin/python" -c 'import tensorboard; assert tensorboard.__version__ == "2.20.0"' 2>/dev/null; then
  "$MONITOR_ENV/bin/python" -m pip install --timeout 20 --retries 1 'tensorboard==2.20.0'
fi
"$MONITOR_ENV/bin/python" -m pip freeze > "$MONITOR_ENV/installed-requirements.txt"
echo "Monitor ready: $MONITOR_ENV/bin/tensorboard (training environments unchanged)"
