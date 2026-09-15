#!/usr/bin/env bash
# Only installs the small orchestration dependencies if absent; no model packages are upgraded.
set -euo pipefail
V5_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$V5_ROOT${PYTHONPATH:+:$PYTHONPATH}"
V5_PY="${DUALISL_DRIVER_PYTHON:-$V5_ROOT/../../miniconda3/envs/dualisl-critic/bin/python}"
"$V5_PY" -m pip install 'PyYAML>=6' 'jsonschema>=4.20' 'requests>=2.31'
"$V5_PY" - "$V5_ROOT" <<'PY'
import json,subprocess,sys
from pathlib import Path
from dual_isl_train.config import load_config
c=load_config(Path(sys.argv[1])/'configs/v5_midasheng.yaml')
e=json.loads(Path(c['labeling']['config_path']).read_text())['backends']['experts']
paths={c[r]['python'] for r in ('captioner','tts','critics')}
paths.update((c['labeling']['asr']['python'],e['python'],e['rate_python'],e['voxlect_python']))
for py in sorted(paths):
    if subprocess.run([py,'-c','import yaml,jsonschema,requests'],capture_output=True).returncode:
        subprocess.run([py,'-m','pip','install','PyYAML>=6','jsonschema>=4.20','requests>=2.31'],check=True)
    subprocess.run([py,'-c','import yaml,jsonschema,requests'],check=True)
print('V5 runtime dependencies ready; no GPU execution.')
PY
