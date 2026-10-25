#!/usr/bin/env bash
# Demo C - injected fault: VIO output frozen for 2 s (stale pose re-sent) on EuRoC MH_01,
# raw vs supervised bridge into PX4 SITL EKF2 (real IMU). SITL on localhost only.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd); TAG=$(date +%Y%m%d-%H%M%S)
RUNS=${DEMO_RUNS:-$HOME/work/demo-runs}/demo-c-$TAG; OUT=$REPO/benchmarks/results/gps_denied/demo-c-$TAG
for P in raw supervised; do
  "$HERE/_hil.sh" MH_01_easy inertial 20.5 90 $P freeze "$RUNS/MH_01-freeze-$P" --fault-at-s 60 --fault-length-s 2
done
cd "$HERE" && PYTHONPATH="$REPO/src" "$HOME/.venvs/px4/bin/python" demo_plot.py c \
  --sequence "$HOME/aegisland-data/euroc/MH_01_easy/mav0" --run "$RUNS/MH_01-freeze-raw" \
  --supervised-run "$RUNS/MH_01-freeze-supervised" --fault-at-s 60 --fault-length-s 2 --output "$OUT"
"$HERE/_finish.sh" "$OUT" "$RUNS/MH_01-freeze-raw" "$RUNS/MH_01-freeze-supervised"
