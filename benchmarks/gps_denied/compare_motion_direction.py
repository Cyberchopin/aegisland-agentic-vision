"""
Scale-free comparison of VisualDeadReckoner vs ground truth and cuVSLAM on EuRoC.

VisualDeadReckoner (VDR) outputs normalized image motion, so it is NOT compared in
meters. Instead, for each consecutive cam0 frame pair:

  reference relative pose (GT or cuVSLAM, cam0 frame)
    -> predicted pixel motion of the scene point at the image center
       (fisheye cam0 model, for assumed depths: inf = rotation only, 2/5/10 m)
  VDR  -> pixel motion of the image center under its homography
  score: cosine direction agreement, magnitude ratio (informative only)

Also reports: VDR failure rate vs GT angular speed, cuVSLAM tracking gaps,
camera/IMU clock offset (|gyro| vs angular speed from trajectories), and the
cam0-IMU extrinsic chain (recalibrated vs original EuRoC calibration).

Run with the AegisLand environment; writes a NEW output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np

from aegisland.localization_diagnostics import (
    angular_speed_from_poses,
    direction_agreement,
    estimate_time_offset,
    failure_rate_by_bin,
    predicted_center_flow,
    vdr_image_flow,
)
from aegisland.trajectory_eval import (
    associate,
    load_euroc_groundtruth,
    load_tum,
    quaternion_to_matrix,
)

DEPTHS_M = (None, 2.0, 5.0, 10.0)
MINIMUM_REFERENCE_PX = 1.0
MAX_TIME_DIFFERENCE_S = 0.005
ANGULAR_SPEED_EDGES = (0.0, 0.25, 0.5, 1.0, 2.0, 100.0)


def yaml_list(path: Path, key: str) -> list[float]:
    """Read `key: [a, b, ...]` (possibly multi-line) from a flat EuRoC yaml file."""
    match = re.search(rf"\b{key}:\s*\[([^\]]*)\]", path.read_text(encoding="utf-8"), re.DOTALL)
    if match is None:
        raise ValueError(f"{path}: missing list '{key}'")
    return [float(value) for value in match.group(1).replace("\n", " ").split(",") if value.strip()]


def transform(path: Path) -> np.ndarray:
    return np.array(yaml_list(path, "data")).reshape(4, 4)


def poses_from_trajectory(trajectory) -> np.ndarray:
    poses = np.repeat(np.eye(4)[None], len(trajectory.timestamps_s), axis=0)
    for index, quaternion in enumerate(trajectory.quaternions_xyzw):
        poses[index, :3, :3] = quaternion_to_matrix(quaternion)
    poses[:, :3, 3] = trajectory.positions_m
    return poses


def rotation_angle_deg(rotation: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(rotation) - 1.0) / 2.0))))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_imu(path: Path) -> tuple[np.ndarray, np.ndarray]:
    times, gyro = [], []
    with path.open(encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if not row or row[0].startswith("#"):
                continue
            nanoseconds = int(row[0])
            times.append(nanoseconds // 1_000_000_000 + (nanoseconds % 1_000_000_000) * 1e-9)
            gyro.append([float(value) for value in row[1:4]])
    return np.array(times), np.array(gyro)


def clock_alignment(
    imu_csv: Path,
    camera_times: np.ndarray,
    gt_mid: np.ndarray,
    gt_speed: np.ndarray,
    cu_mid: np.ndarray,
    cu_speed: np.ndarray,
) -> dict[str, object]:
    """|gyro| on the IMU clock vs angular speed from trajectories on the camera/GT clocks."""
    imu_times, gyro = load_imu(imu_csv)
    gyro_speed = np.linalg.norm(gyro, axis=1)
    per_interval = np.diff(np.searchsorted(imu_times, camera_times))
    return {
        "convention": "positive offset = second stream lags the IMU",
        # Trajectory angular speed is an interval average; average |gyro| over the same
        # interval (camera period / GT period) to avoid point-sampling aliasing.
        "cuvslam_vs_imu": estimate_time_offset(
            imu_times, gyro_speed, cu_mid, cu_speed, search_s=0.1, step_s=0.0005,
            average_window_s=float(np.median(np.diff(camera_times))),
        ),
        "groundtruth_vs_imu": estimate_time_offset(
            imu_times, gyro_speed, gt_mid, gt_speed, search_s=0.1, step_s=0.0005,
            average_window_s=float(np.median(np.diff(gt_mid))),
        ),
        "cuvslam_vs_imu_point_sampled_biased": estimate_time_offset(
            imu_times, gyro_speed, cu_mid, cu_speed, search_s=0.1, step_s=0.0005
        ),
        "imu_samples_per_camera_interval": {
            "min": int(per_interval.min()),
            "max": int(per_interval.max()),
            "intervals_with_zero": int(np.sum(per_interval == 0)),
            "intervals_with_10": int(np.sum(per_interval == 10)),
            "intervals": int(per_interval.size),
        },
    }


def project_factory(cam0_yaml: Path):
    """Fisheye cam0 projection function and the image-center viewing ray."""
    focal_x, focal_y, center_x, center_y = yaml_list(cam0_yaml, "intrinsics")
    width, height = (int(value) for value in yaml_list(cam0_yaml, "resolution"))
    camera_matrix = np.array([[focal_x, 0, center_x], [0, focal_y, center_y], [0, 0, 1.0]])
    distortion = np.array(yaml_list(cam0_yaml, "distortion_coefficients"))

    def project(points: np.ndarray) -> np.ndarray:
        pixels, _ = cv2.fisheye.projectPoints(
            points.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), camera_matrix, distortion
        )
        return pixels.reshape(-1, 2)

    normalized = cv2.fisheye.undistortPoints(
        np.array([[[width / 2.0, height / 2.0]]]), camera_matrix, distortion
    ).reshape(2)
    return project, np.array([normalized[0], normalized[1], 1.0])


def seconds(nanoseconds: int) -> float:
    return nanoseconds // 1_000_000_000 + (nanoseconds % 1_000_000_000) * 1e-9


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sequence", type=Path, required=True, help="path to mav0/")
    parser.add_argument("--vdr-run", type=Path, required=True)
    parser.add_argument("--cuvslam-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory")
    args = parser.parse_args()
    sequence = args.sequence

    # Calibration: cuVSLAM recalibrated files are relative to cam0.
    cam0_yaml = sequence / "cam0" / "sensor_recalibrated.yaml"
    width, height = (int(value) for value in yaml_list(cam0_yaml, "resolution"))
    project, center_ray = project_factory(cam0_yaml)
    cam0_from_imu = transform(sequence / "imu0" / "sensor_recalibrated.yaml")
    body_from_cam0 = np.linalg.inv(cam0_from_imu)

    extrinsics = {"source": "imu0/sensor_recalibrated.yaml (cam0_from_imu), inverted"}
    original = sequence / "cam0" / "sensor.yaml"
    if original.is_file():
        # Original EuRoC: T_BS of cam0 = body(imu0)_from_cam0.
        original_body_from_cam0 = transform(original)
        difference = np.linalg.inv(original_body_from_cam0) @ body_from_cam0
        extrinsics.update(
            original_vs_recalibrated_rotation_deg=rotation_angle_deg(difference[:3, :3]),
            original_vs_recalibrated_translation_m=float(np.linalg.norm(difference[:3, 3])),
        )

    vdr = load_jsonl(args.vdr_run / "frames.jsonl")
    vdr_times = np.array([seconds(row["timestamp_ns"]) for row in vdr])
    ground_truth = load_euroc_groundtruth(
        sequence / "state_groundtruth_estimate0" / "data.csv"
    )
    gt_world_from_cam0 = poses_from_trajectory(ground_truth) @ body_from_cam0
    cuvslam = load_tum(args.cuvslam_run / "trajectory_body.tum")
    cu_world_from_cam0 = poses_from_trajectory(cuvslam) @ body_from_cam0

    gt_index = dict(associate(vdr_times, ground_truth.timestamps_s,
                              max_time_difference_s=MAX_TIME_DIFFERENCE_S))
    cu_index = dict(associate(vdr_times, cuvslam.timestamps_s, max_time_difference_s=1e-6))

    def reference_flow(poses: np.ndarray, index: dict[int, int], frame: int, depth):
        if frame - 1 not in index or frame not in index:
            return np.array([np.nan, np.nan])
        current_from_previous = np.linalg.inv(poses[index[frame]]) @ poses[index[frame - 1]]
        return predicted_center_flow(current_from_previous, project, center_ray, depth_m=depth)

    frames = range(1, len(vdr))
    vdr_flow = np.array(
        [
            vdr_image_flow(row["vx_normalized"], row["vy_normalized"], width, height)
            if row["success"]
            else [np.nan, np.nan]
            for row in vdr[1:]
        ],
        dtype=np.float64,
    )
    agreement = {}
    for depth in DEPTHS_M:
        label = "inf_rotation_only" if depth is None else f"{depth:g}m"
        gt_flow = np.array([reference_flow(gt_world_from_cam0, gt_index, f, depth) for f in frames])
        cu_flow = np.array([reference_flow(cu_world_from_cam0, cu_index, f, depth) for f in frames])
        agreement[label] = {
            "vdr_vs_groundtruth": direction_agreement(
                vdr_flow, gt_flow, minimum_reference_px=MINIMUM_REFERENCE_PX
            ),
            "cuvslam_vs_groundtruth": direction_agreement(
                cu_flow, gt_flow, minimum_reference_px=MINIMUM_REFERENCE_PX
            ),
            "vdr_vs_cuvslam": direction_agreement(
                vdr_flow, cu_flow, minimum_reference_px=MINIMUM_REFERENCE_PX
            ),
        }

    # Failure behavior vs ground-truth angular speed at each VDR frame.
    gt_mid, gt_speed = angular_speed_from_poses(ground_truth.timestamps_s, gt_world_from_cam0)
    speed_at_frame = np.interp(vdr_times[1:], gt_mid, gt_speed)
    vdr_valid = np.array([row["valid"] for row in vdr[1:]])
    vdr_success = np.array([row["success"] for row in vdr[1:]])
    cu_frames = load_jsonl(args.cuvslam_run / "frames.jsonl")
    cu_tracked = np.array([row["tracked"] for row in cu_frames])
    gaps, current = [], 0
    for tracked in cu_tracked:
        if not tracked:
            current += 1
        elif current:
            gaps.append(current)
            current = 0
    if current:
        gaps.append(current)

    # Clock alignment: |gyro| (IMU clock) vs angular speed from trajectories (camera clock).
    imu_csv = sequence / "imu0" / "data.csv"
    cu_mid, cu_speed = angular_speed_from_poses(cuvslam.timestamps_s, cu_world_from_cam0)
    clock: dict[str, object] = {"skipped": f"{imu_csv} not found"}
    if imu_csv.is_file():
        clock = clock_alignment(imu_csv, vdr_times, gt_mid, gt_speed, cu_mid, cu_speed)

    report = {
        "tool": "benchmarks/gps_denied/compare_motion_direction.py",
        "limits": [
            "VDR units are normalized image motion; no meter comparison is made.",
            "Predicted center flow needs a depth assumption; results are given per depth.",
            (
                "VDR fits one homography to the distorted fisheye image; this is part of "
                "what is being measured, not corrected for."
            ),
            "Agreement in direction says nothing about accumulated drift or position error.",
            "Single sequence; not representative of the competition camera or arena.",
        ],
        "inputs": {
            "sequence": str(sequence.resolve()),
            "vdr_run": str(args.vdr_run.resolve()),
            "vdr_frames_sha256": hashlib.sha256(
                (args.vdr_run / "frames.jsonl").read_bytes()
            ).hexdigest(),
            "cuvslam_run": str(args.cuvslam_run.resolve()),
            "cuvslam_trajectory_sha256": hashlib.sha256(
                (args.cuvslam_run / "trajectory_body.tum").read_bytes()
            ).hexdigest(),
        },
        "extrinsics": extrinsics,
        "center_ray_cam0": center_ray.tolist(),
        "frame_pairs": len(vdr) - 1,
        "direction_agreement": agreement,
        "failure_behavior": {
            "vdr_compensation_success_fraction": float(np.mean(vdr_success)),
            "vdr_valid_fraction": float(np.mean(vdr_valid)),
            "vdr_invalid_by_groundtruth_angular_speed_rad_s": failure_rate_by_bin(
                speed_at_frame, vdr_valid, edges=ANGULAR_SPEED_EDGES
            ),
            "cuvslam_tracked_fraction": float(np.mean(cu_tracked)),
            "cuvslam_untracked_gaps_frames": gaps,
        },
        "clock_alignment": clock,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "comparison.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "SHA256SUMS").write_text(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  comparison.json\n", encoding="utf-8"
    )
    summary = {
        label: {
            name: (values["cosine_median"], values["fraction_same_half_plane"], values["used"])
            for name, values in rows.items()
        }
        for label, rows in agreement.items()
    }
    print(json.dumps({"agreement(cos_median, same_half, used)": summary,
                      "failure": report["failure_behavior"]["vdr_valid_fraction"],
                      "clock": {k: v["offset_s"] for k, v in clock.items()
                                if k.endswith("_vs_imu")}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
