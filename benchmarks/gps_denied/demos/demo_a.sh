#!/usr/bin/env bash
# Demo A - localization: EuRoC MH_01 (cuVSLAM Stereo-Inertial) -> supervisor -> MAVLink
# ODOMETRY -> PX4 SITL EKF2 with the sequence's real IMU. SITL on localhost only.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd); TAG=$(date +%Y%m%d-%H%M%S)
RUNS=${DEMO_RUNS:-$HOME/work/demo-runs}/demo-a-$TAG; OUT=$REPO/benchmarks/results/gps_denied/demo-a-$TAG
"$HERE/_hil.sh" MH_01_easy inertial 20.5 155 supervised none "$RUNS/MH_01-supervised"
cd "$HERE" && PYTHONPATH="$REPO/src" "$HOME/.venvs/px4/bin/python" demo_plot.py a \
  --sequence "$HOME/aegisland-data/euroc/MH_01_easy/mav0" --run "$RUNS/MH_01-supervised" --output "$OUT"
"$HERE/_finish.sh" "$OUT" "$RUNS/MH_01-supervised"
