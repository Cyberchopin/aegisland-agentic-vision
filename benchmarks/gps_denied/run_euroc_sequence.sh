#!/usr/bin/env bash
# Extract one EuRoC Machine Hall sequence and run the full stage-1 benchmark on it.
#
#   benchmarks/gps_denied/run_euroc_sequence.sh MH_01_easy /path/to/machine_hall.zip [--smoke]
#
# Data goes to ~/aegisland-data/euroc/<SEQ> (never into git). Results go to NEW
# directories under benchmarks/results/gps_denied/ named <kind>-<SEQ>-<tag>.
# Stereo and stereo-inertial each run twice to check determinism.
set -euo pipefail

SEQ=$1
ZIP=$2
SMOKE=${3:-}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
DATA=$HOME/aegisland-data/euroc
EXAMPLES=$HOME/.cache/aegisland-dl/cuVSLAM-src-v17.0.1/examples/euroc
CUPY=$HOME/.venvs/cuvslam-17.0.1/bin/python
PY=$REPO/.venv/bin/python
RESULTS=$REPO/benchmarks/results/gps_denied
TAG=$(date +%Y%m%d-%H%M%S)
export LD_LIBRARY_PATH=$HOME/.venvs/cuvslam-17.0.1/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/lib/wsl/lib

MAV0=$DATA/$SEQ/mav0
if [ ! -d "$MAV0" ]; then
  mkdir -p "$DATA"
  INNER="machine_hall/$SEQ/$SEQ.zip"
  echo "[extract] $INNER"
  unzip -t "$ZIP" "$INNER" | tail -1
  unzip -o -j -q "$ZIP" "$INNER" -d "$DATA"
  unzip -tq "$DATA/$SEQ.zip"
  unzip -q "$DATA/$SEQ.zip" -d "$DATA/$SEQ"
  rm "$DATA/$SEQ.zip"
  cp "$EXAMPLES/sensor_cam0.yaml" "$MAV0/cam0/sensor_recalibrated.yaml"
  cp "$EXAMPLES/sensor_cam1.yaml" "$MAV0/cam1/sensor_recalibrated.yaml"
  cp "$EXAMPLES/sensor_imu0.yaml" "$MAV0/imu0/sensor_recalibrated.yaml"
fi
GT=$MAV0/state_groundtruth_estimate0/data.csv

run_cuvslam() {  # mode output [extra args]
  local mode=$1 out=$2; shift 2
  "$CUPY" "$REPO/benchmarks/gps_denied/run_cuvslam_euroc.py" --sequence "$MAV0" ${CUVSLAM_EXTRA:-} \
    --cuvslam-examples "$EXAMPLES" --mode "$mode" --output "$out" "$@"
  "$PY" "$REPO/benchmarks/gps_denied/evaluate_trajectory.py" --estimate "$out/trajectory_body.tum" \
    --groundtruth "$GT" --output "$out-eval" > /dev/null
  echo "[done] $out"
}

if [ "$SMOKE" = "--smoke" ]; then
  run_cuvslam stereo "$RESULTS/cuvslam-stereo-$SEQ-$TAG-smoke200" --max-frames 200
  exit 0
fi

for mode in stereo inertial; do
  for run in 1 2; do
    run_cuvslam "$mode" "$RESULTS/cuvslam-$mode-$SEQ-$TAG-run$run"
  done
done
"$PY" "$REPO/benchmarks/gps_denied/run_vdr_euroc.py" --sequence "$MAV0" \
  --output "$RESULTS/vdr-$SEQ-$TAG"
for mode in stereo inertial; do
  "$PY" "$REPO/benchmarks/gps_denied/compare_motion_direction.py" --sequence "$MAV0" \
    --vdr-run "$RESULTS/vdr-$SEQ-$TAG" --cuvslam-run "$RESULTS/cuvslam-$mode-$SEQ-$TAG-run1" \
    --output "$RESULTS/compare-vdr-cuvslam-$mode-$SEQ-$TAG" > /dev/null
done
"$PY" "$REPO/benchmarks/gps_denied/fault_replay.py" \
  --cuvslam-run "$RESULTS/cuvslam-inertial-$SEQ-$TAG-run1" --sequence "$MAV0" \
  --output "$RESULTS/fault-replay-$SEQ-$TAG" > /dev/null
echo "[all done] $SEQ $TAG"
