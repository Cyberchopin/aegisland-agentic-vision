#!/usr/bin/env bash
# Shared runner for the SITL demos: one HIL replay into PX4 SITL (localhost only).
# usage: _hil.sh <SEQ> <MODE> <START_S> <DURATION_S> <POLICY> <FAULT> <OUT> [extra args]
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
R=$REPO/benchmarks/results/gps_denied
SEQ=$1 MODE=$2 START=$3 DUR=$4 POLICY=$5 FAULT=$6 OUT=$7; shift 7
PX4_BUILD=${PX4_BUILD:-$HOME/work/px4/PX4-Autopilot-v1.17.0/build/px4_sitl_default}
RUN=${CUVSLAM_RUN:-$R/cuvslam-$MODE-$SEQ-20261024-102659-signals}
VDR=$(ls -d $R/vdr-$SEQ-2026* | tail -1)
cd "$REPO/benchmarks/gps_denied"
PYTHONPATH="$REPO/src" "$HOME/.venvs/px4/bin/python" sitl_hil_replay.py --px4-build "$PX4_BUILD" \
  --sequence "$HOME/aegisland-data/euroc/$SEQ/mav0" --cuvslam-run "$RUN" --vdr-run "$VDR" \
  --start-s "$START" --duration-s "$DUR" --policy "$POLICY" --fault "$FAULT" --speed "${SPEED:-4}" \
  --output "$OUT" "$@"
