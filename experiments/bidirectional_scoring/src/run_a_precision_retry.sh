#!/usr/bin/env bash
# User-run A only. Reuse the completed B v2 result without launching Captioner.
set -eu
if [ "$#" -ne 1 ]; then
  echo '用法：bash run_a_precision_retry.sh GPU编号（例如 0）' >&2
  exit 2
fi
pilot_gpu="$1"
pilot_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$pilot_root"
umask 000
python3 -B pilot.py score-a --run runs/smoke01 --gpu "$pilot_gpu" --tag v3_fp32 --sdpa-backend math --score-dtype float32 --skip-report
python3 -B pilot.py report --run runs/smoke01 --tag v3_combined --a-tag v3_fp32 --b-tag v2
echo "A 评分与检查：$pilot_root/runs/smoke01/scores/a/v3_fp32/"
echo "A v3 + 已有 B v2 汇总：$pilot_root/runs/smoke01/reports/v3_combined/"
