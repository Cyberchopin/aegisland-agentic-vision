"""
Cross-sequence relocalization on EuRoC Machine Hall ("know the arena before the race").

All MH sequences share the Leica ground-truth world. A map built on MH_01 lives in
MH_01's cuVSLAM world. One fixed transform, map_from_gtworld, is estimated by an
SE(3) Umeyama fit of the MH_01 smoothed SLAM trajectory to ground truth.

  guess     map_from_rig guess for frame N of another sequence, computed from GT
            ("we know the take-off spot"), optionally perturbed (position m, yaw deg)
  evaluate  error of a relocalized run in the MAP frame: the trajectory is moved
            to the GT world with the fixed transform only; no per-run alignment.
            Also reports the usual per-run SE(3)-aligned ATE for comparison.

Run with the AegisLand environment.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from compare_motion_direction import poses_from_trajectory, transform

from aegisland.trajectory_eval import (
    absolute_trajectory_error,
    associate,
    load_euroc_groundtruth,
    load_tum,
    quaternion_to_matrix,
    umeyama_alignment,
)

MAX_DT_S = 0.005


def map_from_gtworld(map_run: Path, map_sequence: Path) -> np.ndarray:
    estimate = load_tum(map_run / "trajectory_slam_smoothed_body.tum")
    gt = load_euroc_groundtruth(map_sequence / "state_groundtruth_estimate0" / "data.csv")
    pairs = associate(estimate.timestamps_s, gt.timestamps_s, max_time_difference_s=MAX_DT_S)
    est = np.array([estimate.positions_m[i] for i, _ in pairs])
    ref = np.array([gt.positions_m[j] for _, j in pairs])
    rotation, translation, _ = umeyama_alignment(ref, est, with_scale=False)  # gt -> map
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = rotation, translation
    return out


def matrix_to_xyzw(rotation: np.ndarray) -> list[float]:
    w = math.sqrt(max(0.0, 1 + np.trace(rotation))) / 2
    if w > 1e-6:
        return [(rotation[2, 1] - rotation[1, 2]) / (4 * w), (rotation[0, 2] - rotation[2, 0]) / (4 * w),
                (rotation[1, 0] - rotation[0, 1]) / (4 * w), w]
    raise ValueError("rotation too close to 180 degrees for this simple conversion")


def guess(args) -> int:
    t_map_gt = map_from_gtworld(args.map_run, args.map_sequence)
    rig_from_imu = transform(args.sequence / "imu0" / "sensor_recalibrated.yaml")
    names = [line.split(",")[0] for line in (args.sequence / "cam0" / "data.csv").read_text()
             .splitlines() if line and not line.startswith("#")]
    stamp_s = int(names[args.frame]) / 1e9
    gt = load_euroc_groundtruth(args.sequence / "state_groundtruth_estimate0" / "data.csv")
    k = int(np.argmin(np.abs(gt.timestamps_s - stamp_s)))
    if abs(gt.timestamps_s[k] - stamp_s) > MAX_DT_S:
        raise ValueError("no ground truth at that frame; choose a later --frame")
    gt_from_body = np.eye(4)
    gt_from_body[:3, :3] = quaternion_to_matrix(gt.quaternions_xyzw[k])
    gt_from_body[:3, 3] = gt.positions_m[k]
    map_from_rig = t_map_gt @ gt_from_body @ np.linalg.inv(rig_from_imu)
    yaw = math.radians(args.perturb_yaw_deg)
    perturb = np.eye(4)
    perturb[:3, :3] = [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0],
                       [0, 0, 1]]
    perturb[:3, 3] = [args.perturb_m, 0.0, 0.0]
    map_from_rig = map_from_rig @ perturb
    out = {"translation": map_from_rig[:3, 3].tolist(),
           "rotation_xyzw": matrix_to_xyzw(map_from_rig[:3, :3]),
           "frame": args.frame, "timestamp_ns": int(names[args.frame]),
           "perturbation": {"position_m": args.perturb_m, "yaw_deg": args.perturb_yaw_deg},
           "map_from_gtworld": t_map_gt.tolist(),
           "source": "ground truth at the start frame ('known take-off spot'), perturbed"}
    args.output.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("translation", "perturbation")}))
    return 0


def evaluate(args) -> int:
    t_map_gt = np.array(json.loads(args.guess_json.read_text())["map_from_gtworld"])
    gt = load_euroc_groundtruth(args.sequence / "state_groundtruth_estimate0" / "data.csv")
    result = {}
    for name in ("trajectory_slam_body.tum", "trajectory_slam_smoothed_body.tum",
                 "trajectory_body.tum"):
        estimate = load_tum(args.run / name)
        pairs = associate(estimate.timestamps_s, gt.timestamps_s, max_time_difference_s=MAX_DT_S)
        poses = poses_from_trajectory(estimate)
        in_gt = np.array([(np.linalg.inv(t_map_gt) @ poses[i])[:3, 3] for i, _ in pairs])
        ref = np.array([gt.positions_m[j] for _, j in pairs])
        error = np.linalg.norm(in_gt - ref, axis=1)
        result[name] = {
            "map_frame_rmse_m": float(np.sqrt(np.mean(error ** 2))),
            "map_frame_max_m": float(error.max()),
            "map_frame_final_m": float(error[-1]),
            "se3_aligned": absolute_trajectory_error(estimate, gt, with_scale=False,
                                                     max_time_difference_s=MAX_DT_S),
            "note": "trajectory_body.tum is odometry in its own start frame; its map-frame "
                    "numbers are only meaningful after a successful relocalization reset "
                    "of the SLAM frame, not of odometry",
        }
    args.output.write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps({k: round(v["map_frame_rmse_m"], 3) for k, v in result.items()}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("guess")
    g.add_argument("--map-run", type=Path, required=True)
    g.add_argument("--map-sequence", type=Path, required=True)
    g.add_argument("--sequence", type=Path, required=True)
    g.add_argument("--frame", type=int, required=True)
    g.add_argument("--perturb-m", type=float, default=0.0)
    g.add_argument("--perturb-yaw-deg", type=float, default=0.0)
    g.add_argument("--output", type=Path, required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--run", type=Path, required=True)
    e.add_argument("--sequence", type=Path, required=True)
    e.add_argument("--guess-json", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return guess(args) if args.command == "guess" else evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
