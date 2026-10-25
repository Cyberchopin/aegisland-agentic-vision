"""
Evaluate a TUM-format trajectory against EuRoC ground truth.

Writes `metrics.json` and `SHA256SUMS` into a NEW output directory. SE(3) ATE is
reported in meters; Sim(3) ATE is included only as a scale-diagnostic and is
labelled as not metric. Run with the AegisLand environment:

  python benchmarks/gps_denied/evaluate_trajectory.py \
      --estimate RUN/trajectory_body.tum \
      --groundtruth SEQ/mav0/state_groundtruth_estimate0/data.csv \
      --output RUN-eval
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from aegisland.trajectory_eval import (
    absolute_trajectory_error,
    load_euroc_groundtruth,
    load_tum,
    relative_pose_error,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--estimate", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory")
    parser.add_argument("--max-time-difference-s", type=float, default=0.005)
    parser.add_argument("--rpe-delta-frames", type=int, default=20)
    args = parser.parse_args()

    estimate = load_tum(args.estimate)
    ground_truth = load_euroc_groundtruth(args.groundtruth)
    tolerance = args.max_time_difference_s
    metrics = {
        "estimate": str(args.estimate.resolve()),
        "estimate_sha256": hashlib.sha256(args.estimate.read_bytes()).hexdigest(),
        "groundtruth": str(args.groundtruth.resolve()),
        "groundtruth_sha256": hashlib.sha256(args.groundtruth.read_bytes()).hexdigest(),
        "estimate_poses": len(estimate.timestamps_s),
        "groundtruth_poses": len(ground_truth.timestamps_s),
        "max_time_difference_s": tolerance,
        "ate_se3": absolute_trajectory_error(
            estimate, ground_truth, max_time_difference_s=tolerance
        ),
        "ate_sim3_scale_diagnostic": absolute_trajectory_error(
            estimate, ground_truth, max_time_difference_s=tolerance, with_scale=True
        ),
        "rpe": relative_pose_error(
            estimate,
            ground_truth,
            delta_frames=args.rpe_delta_frames,
            max_time_difference_s=tolerance,
        ),
        "notes": (
            "Dataset replay accuracy only; does not imply field-camera or flight performance."
        ),
    }

    args.output.mkdir(parents=True, exist_ok=False)
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (args.output / "SHA256SUMS").write_text(
        f"{hashlib.sha256(metrics_path.read_bytes()).hexdigest()}  metrics.json\n",
        encoding="utf-8",
    )
    print(json.dumps({k: metrics[k] for k in ("ate_se3", "rpe")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
