#!/usr/bin/env bash
# Qwen2.5-Omni-3B V4, full data, dual-base ten rounds; no implicit smoke.
set -euo pipefail
umask 000
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-train}"
RUN_DIR="${2:-${DUALISL_RUN_DIR:-$PROJECT_ROOT/runs/qwen25_omni_3b_reward_v4_10rounds_optimized_run01}}"
if [[ $# -gt 2 || "$RUN_DIR" != /* ]]; then
  echo "Usage: bash $0 [check|train|dashboard] [ABS_RUN_DIR]" >&2
  exit 2
fi
case "$MODE" in check|train|dashboard) ;; *) echo 'Expected check, train or dashboard' >&2; exit 2 ;; esac
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DUALISL_RUN_DIR="$RUN_DIR"
export DUALISL_ROUNDS=10
export DUALISL_SHARED_WRITABLE=1
DRIVER_PYTHON="${DUALISL_DRIVER_PYTHON:-$PROJECT_ROOT/../../miniconda3/envs/qwen3-tts/bin/python}"
CAPTION_PYTHON="$PROJECT_ROOT/../../miniconda3/envs/qwen2_5-omni-3b-captioner/bin/python"
CONFIG="$PROJECT_ROOT/configs/train_8gpu_h100_qwen2_5_omni_3b_reward_v4.yaml"
cd "$PROJECT_ROOT"
if [[ "$MODE" == dashboard ]]; then
  MONITOR_ENV="${DUALISL_MONITOR_ENV:-$PROJECT_ROOT/.monitor-venv}"
  if [[ ! -x "$MONITOR_ENV/bin/python" ]]; then echo 'Run bash scripts/setup_monitor_env.sh first.' >&2; exit 2; fi
  LABEL=$("$MONITOR_ENV/bin/python" - "$RUN_DIR" <<'PY'
from pathlib import Path
import hashlib, sys
print('qwen25_v4_' + hashlib.sha256(str(Path(sys.argv[1]).resolve()).encode()).hexdigest()[:12])
PY
)
  exec bash scripts/run_training_dashboard.sh "$RUN_DIR" "$LABEL" "${DUALISL_DASHBOARD_PORT:-6006}"
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Guard before creating directories, loading a model or acquiring a GPU.
"$DRIVER_PYTHON" -m scripts.qwen25_reward_v4_run_guard --config "$CONFIG" --mode train --rounds 10
"$DRIVER_PYTHON" - "$CONFIG" <<'PY'
import json, os, sys
from pathlib import Path
from dual_isl_train.config import load_config
config = load_config(sys.argv[1])
for role in ('captioner','tts','critics'):
    if not os.access(config[role]['python'], os.X_OK):
        raise ValueError(f'Missing {role} interpreter')
for role,key in [('captioner','model_path'),('tts','model_path'),('tts','tokenizer_path'),('critics','whisper_model')]:
    if not Path(config[role][key]).exists():
        raise ValueError(f'Missing {role}.{key}')
provenance = Path(config['captioner']['model_path'])/'.download_provenance.json'
if not provenance.is_file():
    raise ValueError('Missing local Qwen model provenance')
print(json.dumps({'run_dir':config['run']['output_dir'], 'captioner':config['captioner']['model_path'],
                  'world_size':8,'rounds':10,'cycle_sft_selection':config['reward']['cycle_sft_selection'],
                  'tts_generation':config['tts']['generation']},indent=2),flush=True)
PY
# CPU imports only, including the configured attention extension; no model load.
"$CAPTION_PYTHON" - <<'PY'
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
from qwen_omni_utils import process_mm_info
import peft
import flash_attn
print('Qwen thinker/processor, PEFT, qwen_omni_utils and flash_attn imports passed.',flush=True)
PY
if [[ "$MODE" == check ]]; then
  echo 'CPU preflight passed; no run created and no model loaded.'
  exit 0
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then echo 'Set CUDA_VISIBLE_DEVICES to the eight allocated H100 GPUs.' >&2; exit 2; fi
"$DRIVER_PYTHON" - <<'PY'
import torch
if torch.cuda.device_count() != 8:
    raise ValueError('Exactly eight visible H100 GPUs required')
if any('H100' not in torch.cuda.get_device_name(i) for i in range(8)):
    raise ValueError('This entry targets a single node of eight H100 GPUs')
PY
mkdir -p "$RUN_DIR"
exec 9>"$RUN_DIR/.launcher.lock"
if ! flock -n 9; then echo 'Another launcher holds this run lock.' >&2; exit 2; fi
ACTION=$("$DRIVER_PYTHON" -m scripts.qwen25_reward_v4_run_guard --config "$CONFIG" --mode train --rounds 10 --reserve)
chmod a+rwx "$RUN_DIR"
echo "Qwen2.5-Omni-3B RewardV4 rounds=10 GPUs=8 action=$ACTION run=$RUN_DIR"
"$DRIVER_PYTHON" -m dual_isl_train "$ACTION" --config "$CONFIG"
"$DRIVER_PYTHON" scripts/verify_tts_ddp_sync.py --run-dir "$RUN_DIR" --expected-world-size 8 --require-memory-bounded-grpo
