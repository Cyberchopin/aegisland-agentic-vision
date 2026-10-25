"""
Convert Kalibr results (camchain-imucam.yaml + imu.yaml, already parsed into
dicts) into the calibration objects used by aegisland.euroc_writer.

Kalibr conventions (from Kalibr's output format):
  cam0.T_cam_imu      maps IMU-frame points into cam0  -> cam0_from_imu
  cam1.T_cn_cnm1      maps cam0-frame points into cam1 -> cam1_from_cam0
  cam*.timeshift_cam_imu   t_imu = t_cam + shift  [s]
  camera_model pinhole + distortion_model radtan     -> "radtan" (k1, k2, p1, p2)
  camera_model pinhole + distortion_model equidistant -> "fisheye" (k1..k4)

The time shift is returned, not applied: cuVSLAM has no offset parameter, so a
recording's IMU stamps are corrected by `t_imu_corrected = t_imu - shift`
(see docs/dronehacks/camera-imu-calibration.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .euroc_writer import CameraCalibration, ImuNoise

MODELS = {("pinhole", "radtan"): "radtan", ("pinhole", "equidistant"): "fisheye"}


@dataclass(frozen=True)
class KalibrCalibration:
    cam0: CameraCalibration
    cam1: CameraCalibration
    cam0_from_cam1: np.ndarray
    cam0_from_imu: np.ndarray
    imu: ImuNoise
    timeshift_cam_imu_s: float


def _camera(entry: dict, rate_hz: float) -> CameraCalibration:
    key = (entry["camera_model"], entry["distortion_model"])
    if key not in MODELS:
        raise ValueError(f"unsupported Kalibr model {key}; use pinhole-radtan or pinhole-equi")
    return CameraCalibration(tuple(int(v) for v in entry["resolution"]),
                             tuple(float(v) for v in entry["intrinsics"]), MODELS[key],
                             tuple(float(v) for v in entry["distortion_coeffs"]), rate_hz)


def _matrix(rows) -> np.ndarray:
    matrix = np.array(rows, dtype=float)
    if matrix.shape != (4, 4) or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError("expected a 4x4 homogeneous transform")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5):
        raise ValueError("transform rotation is not orthonormal")
    return matrix


def from_kalibr(camchain: dict, imu_yaml: dict, *, camera_rate_hz: float) -> KalibrCalibration:
    for name in ("cam0", "cam1"):
        if name not in camchain:
            raise ValueError(f"camchain is missing {name}")
    imu_entry = imu_yaml.get("imu0", imu_yaml)
    cam1_from_cam0 = _matrix(camchain["cam1"]["T_cn_cnm1"])
    shift = float(camchain["cam0"].get("timeshift_cam_imu", 0.0))
    return KalibrCalibration(
        cam0=_camera(camchain["cam0"], camera_rate_hz),
        cam1=_camera(camchain["cam1"], camera_rate_hz),
        cam0_from_cam1=np.linalg.inv(cam1_from_cam0),
        cam0_from_imu=_matrix(camchain["cam0"]["T_cam_imu"]),
        imu=ImuNoise(
            float(imu_entry["gyroscope_noise_density"]), float(imu_entry["gyroscope_random_walk"]),
            float(imu_entry["accelerometer_noise_density"]),
            float(imu_entry["accelerometer_random_walk"]), float(imu_entry["update_rate"])),
        timeshift_cam_imu_s=shift,
    )
