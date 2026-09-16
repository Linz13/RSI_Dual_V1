#!/usr/bin/env bash
# User-run fixed pilot: A in FP32, B in BF16, both with math SDPA.
# No generation, training, sampling, or automatic expansion.
set -u
if [ "$#" -ne 1 ]; then
  echo '用法：bash run_pilot30.sh GPU编号（例如 bash run_pilot30.sh 0）' >&2
  exit 2
fi
pilot_gpu="$1"
pilot_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$pilot_root" || exit 1
umask 000
python3 -B pilot.py score-a --run runs/pilot30 --gpu "$pilot_gpu" --tag v1 --sdpa-backend math --score-dtype float32 --skip-report
pilot_a_status=$?
python3 -B pilot.py score-b --run runs/pilot30 --gpu "$pilot_gpu" --tag v1 --sdpa-backend math --skip-report
pilot_b_status=$?
python3 -B pilot.py report --run runs/pilot30 --tag v1
pilot_report_status=$?
echo "A exit=$pilot_a_status; B exit=$pilot_b_status; report exit=$pilot_report_status"
echo "日志：$pilot_root/runs/pilot30/logs/"
echo "评分：$pilot_root/runs/pilot30/scores/"
echo "汇总（同时保留失败、缺失和人工排除状态）：$pilot_root/runs/pilot30/reports/v1/"
if [ "$pilot_a_status" -ne 0 ] || [ "$pilot_b_status" -ne 0 ] || [ "$pilot_report_status" -ne 0 ]; then
  exit 1
fi
