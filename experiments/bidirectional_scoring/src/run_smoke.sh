#!/usr/bin/env bash
# User-run models only; no installation, training, or automatic pilot expansion.
set -u
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo '用法：bash run_smoke.sh GPU编号 [结果标签，默认 v2]（例如 bash run_smoke.sh 0）' >&2
  exit 2
fi
pilot_gpu="$1"
pilot_tag="${2:-v2}"
pilot_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$pilot_root" || exit 1
umask 000
python3 -B pilot.py score-a --run runs/smoke01 --gpu "$pilot_gpu" --tag "$pilot_tag" --sdpa-backend math
pilot_a_status=$?
python3 -B pilot.py score-b --run runs/smoke01 --gpu "$pilot_gpu" --tag "$pilot_tag" --sdpa-backend math
pilot_b_status=$?
echo "A exit=$pilot_a_status; B exit=$pilot_b_status"
echo "日志：$pilot_root/runs/smoke01/logs/"
echo "汇总（各方向成功后生成）：$pilot_root/runs/smoke01/reports/$pilot_tag/"
if [ "$pilot_a_status" -ne 0 ] || [ "$pilot_b_status" -ne 0 ]; then
  exit 1
fi
