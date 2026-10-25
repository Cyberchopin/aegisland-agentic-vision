"""
Write a small synthetic EuRoC-format (ASL) stereo sequence with known metric motion.

Purpose: a pipeline smoke test (file layout, calibration parsing, GPU tracker,
frame conventions, units, evaluator) before a real dataset is available. A
textured plane rendered through an equidistant fisheye model is far easier than
real flight data, so a good score here says nothing about real-world accuracy.

Layout matches what cuVSLAM's EuRoC `dataset_utils.py` reads, with
`sensor_recalibrated.yaml` files (transforms relative to cam0, fisheye model).
Body frame = cam0 frame (imu0 extrinsic is identity), so ground truth is
world_from_cam0. Stereo-only: no IMU samples are generated.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 752, 480
FOCAL_PX = 460.0
BASELINE_M = 0.11
PLANE_DEPTH_M = 4.0
TEXTURE_PIXELS_PER_M = 200
TEXTURE_SIZE_M = 16.0
RATE_HZ = 20


def rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def matrix_to_quaternion_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    w = math.sqrt(max(0.0, 1.0 + np.trace(rotation))) / 2.0
    x = math.copysign(
        math.sqrt(max(0.0, 1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])) / 2,
        rotation[2, 1] - rotation[1, 2],
    )
    y = math.copysign(
        math.sqrt(max(0.0, 1 - rotation[0, 0] + rotation[1, 1] - rotation[2, 2])) / 2,
        rotation[0, 2] - rotation[2, 0],
    )
    z = math.copysign(
        math.sqrt(max(0.0, 1 - rotation[0, 0] - rotation[1, 1] + rotation[2, 2])) / 2,
        rotation[1, 0] - rotation[0, 1],
    )
    return w, x, y, z


def make_texture(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    size = int(TEXTURE_SIZE_M * TEXTURE_PIXELS_PER_M)
    texture = np.zeros((size, size), np.float32)
    for cell in (3, 9, 27, 81):
        noise = rng.random((size // cell + 1, size // cell + 1)).astype(np.float32)
        texture += cv2.resize(noise, (size, size), interpolation=cv2.INTER_CUBIC)[:size, :size]
    texture = cv2.GaussianBlur(texture, (3, 3), 0)
    texture -= texture.min()
    return np.uint8(255 * texture / max(float(texture.max()), 1e-6))


def camera_rays() -> np.ndarray:
    """Unit rays in camera frame for each pixel, equidistant fisheye with zero distortion."""
    u, v = np.meshgrid(np.arange(WIDTH, dtype=np.float64), np.arange(HEIGHT, dtype=np.float64))
    xd = (u - WIDTH / 2.0) / FOCAL_PX
    yd = (v - HEIGHT / 2.0) / FOCAL_PX
    theta = np.hypot(xd, yd)
    scale = np.where(theta > 1e-12, np.sin(theta) / np.maximum(theta, 1e-12), 1.0)
    return np.stack([xd * scale, yd * scale, np.cos(theta)], axis=-1)


def render(
    texture: np.ndarray, rays: np.ndarray, rotation: np.ndarray, origin: np.ndarray
) -> np.ndarray:
    world_rays = rays @ rotation.T
    distance = (PLANE_DEPTH_M - origin[2]) / np.maximum(world_rays[..., 2], 1e-6)
    points = origin + distance[..., None] * world_rays
    offset = TEXTURE_SIZE_M / 2.0
    map_x = ((points[..., 0] + offset) * TEXTURE_PIXELS_PER_M).astype(np.float32)
    map_y = ((points[..., 1] + offset) * TEXTURE_PIXELS_PER_M).astype(np.float32)
    return cv2.remap(texture, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def trajectory(frame: int) -> tuple[np.ndarray, np.ndarray]:
    """world_from_cam0 at a frame: smooth metric motion, meters and radians."""
    t = frame / RATE_HZ
    position = np.array([0.6 * math.sin(0.4 * t), 0.2 * math.sin(0.7 * t), 0.4 * math.sin(0.3 * t)])
    rotation = rotation_matrix(
        0.05 * math.sin(0.5 * t), 0.08 * math.sin(0.35 * t), 0.1 * math.sin(0.25 * t)
    )
    return rotation, position


def yaml_matrix(matrix: np.ndarray) -> str:
    values = ", ".join(f"{value:.9f}" for value in matrix.flatten())
    return f"T_BS:\n  cols: 4\n  rows: 4\n  data: [{values}]\n"


def camera_yaml(matrix: np.ndarray) -> str:
    return (
        "sensor_type: camera\ncomment: synthetic AegisLand smoke camera\n"
        + yaml_matrix(matrix)
        + f"rate_hz: {RATE_HZ}\nresolution: [{WIDTH}, {HEIGHT}]\n"
        + "camera_model: fisheye\n"
        + f"intrinsics: [{FOCAL_PX}, {FOCAL_PX}, {WIDTH / 2.0}, {HEIGHT / 2.0}]\n"
        + "distortion_model: fisheye\ndistortion_coefficients: [0.0, 0.0, 0.0, 0.0]\n"
    )


def write_sequence(output: Path, frames: int, seed: int) -> None:
    mav0 = output / "mav0"
    # Refuse to overwrite an existing sequence.
    mav0.mkdir(parents=True, exist_ok=False)
    for sub in ("cam0/data", "cam1/data", "imu0", "state_groundtruth_estimate0"):
        (mav0 / sub).mkdir(parents=True)

    cam1_from_rig = np.eye(4)
    cam1_from_rig[0, 3] = BASELINE_M
    (mav0 / "cam0/sensor_recalibrated.yaml").write_text(camera_yaml(np.eye(4)))
    (mav0 / "cam1/sensor_recalibrated.yaml").write_text(camera_yaml(cam1_from_rig))
    (mav0 / "imu0/sensor_recalibrated.yaml").write_text(
        "sensor_type: imu\ncomment: synthetic placeholder; no IMU samples\n"
        + yaml_matrix(np.eye(4))
        + "rate_hz: 200\ngyroscope_noise_density: 1.6968e-04\n"
        + "gyroscope_random_walk: 1.9393e-05\naccelerometer_noise_density: 2.0e-3\n"
        + "accelerometer_random_walk: 3.0e-3\n"
    )

    texture = make_texture(seed)
    rays = camera_rays()
    start_ns = 1_000_000_000_000
    period_ns = 1_000_000_000 // RATE_HZ
    camera_rows, truth_rows = [], []
    for frame in range(frames):
        timestamp_ns = start_ns + frame * period_ns
        rotation, position = trajectory(frame)
        right_origin = position + rotation @ np.array([BASELINE_M, 0.0, 0.0])
        name = f"{timestamp_ns}.png"
        cv2.imwrite(str(mav0 / "cam0/data" / name), render(texture, rays, rotation, position))
        cv2.imwrite(str(mav0 / "cam1/data" / name), render(texture, rays, rotation, right_origin))
        camera_rows.append(f"{timestamp_ns},{name}")
        w, x, y, z = matrix_to_quaternion_wxyz(rotation)
        truth_rows.append(
            f"{timestamp_ns},{position[0]:.9f},{position[1]:.9f},{position[2]:.9f},"
            f"{w:.9f},{x:.9f},{y:.9f},{z:.9f}"
        )

    header = "#timestamp [ns],filename\n"
    for camera in ("cam0", "cam1"):
        (mav0 / camera / "data.csv").write_text(header + "\n".join(camera_rows) + "\n")
    (mav0 / "state_groundtruth_estimate0/data.csv").write_text(
        "#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], "
        "q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []\n" + "\n".join(truth_rows) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, required=True, help="new directory")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    write_sequence(args.output, args.frames, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
