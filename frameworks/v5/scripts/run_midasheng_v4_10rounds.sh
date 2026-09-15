#!/usr/bin/env bash
# Fixed full-data, dual-base V4 entry. Training is foreground; dashboard is separate.
set -euo pipefail
umask 000
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-train}"
RUN_DIR="${2:-${DUALISL_RUN_DIR:-$PROJECT_ROOT/runs/midasheng_7b_reward_v4_10rounds_optimized_run01}}"
if [[ $# -gt 2 || "$RUN_DIR" != /* ]]; then
  echo "Usage: bash $0 [check|train|dashboard] [ABS_RUN_DIR]" >&2
  exit 2
fi
case "$MODE" in check|train|dashboard) ;; *) echo 'Expected check, train or dashboard' >&2; exit 2 ;; esac
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DUALISL_RUN_DIR="$RUN_DIR"
export DUALISL_ROUNDS=10
DRIVER_PYTHON="${DUALISL_DRIVER_PYTHON:-$PROJECT_ROOT/../../miniconda3/envs/qwen3-tts/bin/python}"
cd "$PROJECT_ROOT"
if [[ "$MODE" == dashboard ]]; then
  MONITOR_ENV="${DUALISL_MONITOR_ENV:-$PROJECT_ROOT/.monitor-venv}"
  if [[ ! -x "$MONITOR_ENV/bin/python" ]]; then
    echo 'First run: bash scripts/setup_monitor_env.sh' >&2
    exit 2
  fi
  LABEL=$("$MONITOR_ENV/bin/python" - "$RUN_DIR" <<'PY'
from pathlib import Path
import hashlib, sys
print('v4_10rounds_' + hashlib.sha256(str(Path(sys.argv[1]).resolve()).encode()).hexdigest()[:12])
PY
)
  exec bash scripts/run_training_dashboard.sh "$RUN_DIR" "$LABEL" "${DUALISL_DASHBOARD_PORT:-6006}"
fi
# Read-only CPU preflight. Never creates a run or loads a model.
"$DRIVER_PYTHON" - "$RUN_DIR" <<'PY'
import json, os, sys
from pathlib import Path
from dual_isl_train.config import load_config
from scripts.reward_v4_run_guard import PROJECT, run_contract, check_run_directory
config = load_config(PROJECT/'configs/train_8gpu_h100_midasheng_reward_v4.yaml')
contract = run_contract(config, 'train', 10)
root = Path(config['run']['output_dir'])
action = check_run_directory(root, contract)
if config['data'].get('max_records_per_role') is not None:
    raise ValueError('Ten-round entry requires full datasets, not a smoke limit')
counts = {}
for role, expected in [('paired',39),('audio_only',184),('caption_only',196)]:
    path = Path(config['data'][role+'_path']).resolve()
    if path != (PROJECT.parent/'training_data_v2'/f'{role}.jsonl').resolve():
        raise ValueError(f'Unexpected dataset override: {role}')
    count = sum(1 for line in path.open() if line.strip())
    if count != expected:
        raise ValueError(f'{role}: expected {expected} source records, got {count}')
    counts[role] = count
if config['training']['group_size'] != 4 or config['training']['warmstart']:
    raise ValueError('Expected group_size=4, warmstart=false')
generation = config['tts']['generation']
if generation.get('reference_replay_reuse') is not True or generation.get('rollout_schedule') != 'previous_round_lpt':
    raise ValueError('This entry requires both validated TTS optimizations')
for role in ('captioner','tts','critics'):
    if not os.access(config[role]['python'],os.X_OK):
        raise ValueError(f'Missing {role} interpreter')
for role,key in [('captioner','model_path'),('tts','model_path'),('tts','tokenizer_path'),('critics','whisper_model')]:
    if not Path(config[role][key]).exists():
        raise ValueError(f'Missing {role}.{key}')
latest = root/'latest.json'
if latest.exists() and json.loads(latest.read_text()).get('round',-1) >= 9:
    raise ValueError('This ten-round run is already complete; do not append by repeating the command')
print(json.dumps({'run_dir':str(root), 'action':action, 'rounds':10, 'world_size':8,
                  'start':'dual base for NEW runs; original state for matching resume',
                  'source_records':counts, 'cycle_sft_selection':config['reward']['cycle_sft_selection'],
                  'reference_replay_reuse':True,'rollout_schedule':generation['rollout_schedule']},indent=2),flush=True)
PY
if [[ "$MODE" == check ]]; then
  echo 'CPU preflight passed; no model loaded and no run created.'
  exit 0
fi
exec bash scripts/run_midasheng_reward_v4_h100_single_node.sh train 8 10
