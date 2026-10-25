"""
Format check for aegisland.euroc_writer: re-write the first N frames of a real
EuRoC sequence through the writer (images, IMU, the sequence's own calibration)
so the existing tools can be run on both copies and compared.

The writer output must be consumable by run_cuvslam_euroc.py unchanged; with
identical pixels, IMU and calibration the trajectories should match.

Run with the AegisLand environment; writes a NEW directory.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from compare_motion_direction import transform, yaml_list

from aegisland.euroc_writer import CameraCalibration, EurocRecordingWriter, ImuNoise


def camera(path: Path) -> CameraCalibration:
    return CameraCalibration(
        tuple(int(v) for v in yaml_list(path, "resolution")),
        tuple(yaml_list(path, "intrinsics")), "fisheye",
        tuple(yaml_list(path, "distortion_coefficients")), 20.0)


def scalar(path: Path, key: str) -> float:
    for line in path.read_text().splitlines():
        if line.startswith(f"{key}:"):
            return float(line.split(":")[1].split("#")[0])
    raise ValueError(f"{path}: missing {key}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sequence", type=Path, required=True, help="mav0/ with *_recalibrated")
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True, help="new recording root")
    args = parser.parse_args()
    seq = args.sequence
    writer = EurocRecordingWriter(args.output)
    imu_yaml = seq / "imu0" / "sensor_recalibrated.yaml"
    writer.write_calibration(
        camera(seq / "cam0" / "sensor_recalibrated.yaml"),
        camera(seq / "cam1" / "sensor_recalibrated.yaml"),
        cam0_from_cam1=transform(seq / "cam1" / "sensor_recalibrated.yaml"),
        cam0_from_imu=transform(imu_yaml),
        imu=ImuNoise(*(scalar(imu_yaml, k) for k in (
            "gyroscope_noise_density", "gyroscope_random_walk", "accelerometer_noise_density",
            "accelerometer_random_walk", "rate_hz"))),
        source=f"rewritten from {seq}")
    names = {}
    for side in ("cam0", "cam1"):
        with (seq / side / "data.csv").open() as stream:
            names[side] = {int(r[0]): r[1] for r in csv.reader(stream) if r and r[0][0] != "#"}
    stamps = sorted(set(names["cam0"]) & set(names["cam1"]))[: args.frames]
    for stamp in stamps:
        left = cv2.imread(str(seq / "cam0" / "data" / names["cam0"][stamp]), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(seq / "cam1" / "data" / names["cam1"][stamp]), cv2.IMREAD_GRAYSCALE)
        writer.add_stereo(stamp, left, stamp, right)
    imu = np.loadtxt(seq / "imu0" / "data.csv", delimiter=",", comments="#", dtype=str)
    for row in imu:
        if int(row[0]) <= stamps[-1]:
            writer.add_imu(int(row[0]), [float(v) for v in row[1:4]], [float(v) for v in row[4:7]])
    print(writer.close({"source": str(seq), "frames": len(stamps)})["counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
