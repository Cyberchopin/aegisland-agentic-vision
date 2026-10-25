#!/usr/bin/env bash
# Demo B - gradual degradation: EuRoC MH_05 dark segment, raw vs supervised bridge into
# PX4 SITL EKF2 (real IMU). cuVSLAM never reports a failure. SITL on localhost only.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd); TAG=$(date +%Y%m%d-%H%M%S)
RUNS=${DEMO_RUNS:-$HOME/work/demo-runs}/demo-b-$TAG; OUT=$REPO/benchmarks/results/gps_denied/demo-b-$TAG
SERIES=$REPO/benchmarks/results/gps_denied/degradation-study-inertial-20261024-104830/series.json
for P in raw supervised; do "$HERE/_hil.sh" MH_05_difficult inertial 12.0 95 $P none "$RUNS/MH_05-$P"; done
cd "$HERE" && PYTHONPATH="$REPO/src" "$HOME/.venvs/px4/bin/python" demo_plot.py b \
  --sequence "$HOME/aegisland-data/euroc/MH_05_difficult/mav0" --run "$RUNS/MH_05-raw" \
  --supervised-run "$RUNS/MH_05-supervised" --series "$SERIES" --sequence-name MH_05_difficult --output "$OUT"
"$HERE/_finish.sh" "$OUT" "$RUNS/MH_05-raw" "$RUNS/MH_05-supervised"
