"""
Write a sensor recording in the EuRoC/ASL layout used by the GPS-denied benchmarks.

    <root>/mav0/cam0/data.csv, cam0/data/<ts>.png      left, 8-bit gray
    <root>/mav0/cam1/...                               right
    <root>/mav0/imu0/data.csv                          ts, w[rad/s] x3, a[m/s^2] x3
    <root>/mav0/depth0/data.csv, depth0/data/<ts>.png  optional, 16-bit, millimetres
    <root>/mav0/{cam0,cam1,imu0}/sensor*.yaml          calibration (see write_calibration)
    <root>/recording.json, <root>/SHA256SUMS

Device-independent and dependency-light (numpy + OpenCV for PNG), so it is unit
tested without hardware; the DepthAI front end lives in
benchmarks/gps_denied/oak_record.py.

Timestamps are integer nanoseconds on ONE clock (the camera device clock for OAK).
Stereo frames are only written as pairs with identical timestamps (the left
stamp is used for both; the measured left/right difference is kept in
cam_pairs.csv). Out-of-order or duplicate samples are refused and counted,
never silently reordered.

cuVSLAM's EuRoC loader picks the lens model from the FILE NAME, not from the
yaml contents: `sensor_recalibrated.yaml` = fisheye/equidistant with extrinsics
relative to cam0; `sensor.yaml` = Brown (k1, k2, p1, p2) with T_BS relative to
the body (= IMU). write_calibration writes the file that matches the model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

IMU_HEADER = (
    "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
    "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n"
)
CAMERA_HEADER = "#timestamp [ns],filename\n"
MAX_PAIR_DIFFERENCE_NS = 1_000_000  # cuVSLAM: stereo timestamps within 1 ms


@dataclass(frozen=True)
class CameraCalibration:
    resolution: tuple[int, int]  # width, height
    intrinsics: tuple[float, float, float, float]  # fu, fv, cu, cv
    model: str  # "fisheye" (equidistant k1..k4) or "radtan" (k1, k2, p1, p2)
    distortion: tuple[float, ...]
    rate_hz: float


@dataclass(frozen=True)
class ImuNoise:
    gyroscope_noise_density: float
    gyroscope_random_walk: float
    accelerometer_noise_density: float
    accelerometer_random_walk: float
    rate_hz: float


@dataclass
class _Stream:
    rows: list[str] = field(default_factory=list)
    last_ns: int | None = None
    refused: int = 0


class EurocRecordingWriter:
    def __init__(self, root: Path, *, depth: bool = False):
        self.root = Path(root)
        self.mav0 = self.root / "mav0"
        # Never overwrite an existing recording.
        self.mav0.mkdir(parents=True, exist_ok=False)
        for name in ("cam0", "cam1", "imu0") + (("depth0",) if depth else ()):
            (self.mav0 / name).mkdir()
            if name != "imu0":
                (self.mav0 / name / "data").mkdir()
        self.depth = depth
        self.streams = {name: _Stream() for name in ("cam", "imu", "depth")}
        self.pairs: list[str] = []
        self.unpaired = 0
        self.shape: tuple[int, int] | None = None
        self.closed = False

    @staticmethod
    def _cv2():
        import cv2  # lazy: calibration-only users (kalibr import) need no OpenCV

        return cv2

    def _accept(self, stream: str, timestamp_ns: int) -> bool:
        state = self.streams[stream]
        if state.last_ns is not None and timestamp_ns <= state.last_ns:
            state.refused += 1
            return False
        state.last_ns = timestamp_ns
        return True

    def add_stereo(self, left_ns: int, left: np.ndarray, right_ns: int, right: np.ndarray) -> bool:
        """Write one stereo pair; False if refused (unpaired, out of order)."""
        if abs(left_ns - right_ns) > MAX_PAIR_DIFFERENCE_NS:
            self.unpaired += 1
            return False
        for image in (left, right):
            if image.dtype != np.uint8 or image.ndim != 2:
                raise ValueError("stereo images must be 2-D uint8 grayscale")
        if left.shape != right.shape or (self.shape is not None and left.shape != self.shape):
            raise ValueError("stereo image size changed")
        if not self._accept("cam", left_ns):
            return False
        self.shape = left.shape
        name = f"{left_ns}.png"
        for camera, image in (("cam0", left), ("cam1", right)):
            if not self._cv2().imwrite(str(self.mav0 / camera / "data" / name), image):
                raise OSError(f"failed to write {camera}/{name}")
        self.streams["cam"].rows.append(f"{left_ns},{name}\n")
        self.pairs.append(f"{left_ns},{right_ns},{right_ns - left_ns}\n")
        return True

    def add_depth(self, timestamp_ns: int, depth_mm: np.ndarray) -> bool:
        if not self.depth:
            raise ValueError("writer created without depth")
        if depth_mm.dtype != np.uint16 or depth_mm.ndim != 2:
            raise ValueError("depth must be 2-D uint16 millimetres")
        if not self._accept("depth", timestamp_ns):
            return False
        name = f"{timestamp_ns}.png"
        if not self._cv2().imwrite(str(self.mav0 / "depth0" / "data" / name), depth_mm):
            raise OSError(f"failed to write depth0/{name}")
        self.streams["depth"].rows.append(f"{timestamp_ns},{name}\n")
        return True

    def add_imu(self, timestamp_ns: int, gyro_rad_s, accel_m_s2) -> bool:
        values = [*map(float, gyro_rad_s), *map(float, accel_m_s2)]
        if len(values) != 6 or not np.all(np.isfinite(values)):
            raise ValueError("IMU sample needs 3 finite gyro and 3 finite accel values")
        if not self._accept("imu", timestamp_ns):
            return False
        # repr() = shortest exact round-trip, so no precision is lost.
        self.streams["imu"].rows.append(f"{timestamp_ns}," + ",".join(repr(v) for v in values)
                                        + "\n")
        return True

    def write_calibration(
        self,
        cam0: CameraCalibration,
        cam1: CameraCalibration,
        *,
        cam0_from_cam1: np.ndarray,
        cam0_from_imu: np.ndarray,
        imu: ImuNoise,
        source: str,
    ) -> list[str]:
        """Write the yaml files cuVSLAM's loader expects for this lens model."""
        return write_calibration(self.mav0, cam0, cam1, cam0_from_cam1=cam0_from_cam1,
                                 cam0_from_imu=cam0_from_imu, imu=imu, source=source)


    def close(self, metadata: dict) -> dict:
        """Write csv files, recording.json and SHA256SUMS; returns the manifest."""
        if self.closed:
            raise RuntimeError("writer already closed")
        for camera in ("cam0", "cam1"):
            (self.mav0 / camera / "data.csv").write_text(
                CAMERA_HEADER + "".join(self.streams["cam"].rows))
        (self.mav0 / "cam_pairs.csv").write_text(
            "#left_ns,right_ns,right_minus_left_ns\n" + "".join(self.pairs))
        (self.mav0 / "imu0" / "data.csv").write_text(IMU_HEADER + "".join(self.streams["imu"].rows))
        if self.depth:
            (self.mav0 / "depth0" / "data.csv").write_text(
                CAMERA_HEADER + "".join(self.streams["depth"].rows))
        manifest = {
            "format": "EuRoC/ASL (aegisland euroc_writer)",
            "clock": "single device clock, integer ns",
            "counts": {
                "stereo_pairs": len(self.streams["cam"].rows),
                "imu_samples": len(self.streams["imu"].rows),
                "depth_frames": len(self.streams["depth"].rows),
                "unpaired_stereo_refused": self.unpaired,
                "out_of_order_refused": {k: v.refused for k, v in self.streams.items()},
            },
            **metadata,
        }
        (self.root / "recording.json").write_text(json.dumps(manifest, indent=1) + "\n")
        sums = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS" and "/data/" not in path.as_posix():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                sums.append(f"{digest}  {path.relative_to(self.root).as_posix()}\n")
        (self.root / "SHA256SUMS").write_text("".join(sums))
        self.closed = True
        return manifest


def write_calibration(
    mav0: Path,
    cam0: CameraCalibration,
    cam1: CameraCalibration,
    *,
    cam0_from_cam1: np.ndarray,
    cam0_from_imu: np.ndarray,
    imu: ImuNoise,
    source: str,
) -> list[str]:
    """Write cam0/cam1/imu0 yaml files into an existing mav0/ (see module docstring)."""
    if cam0.model != cam1.model:
        raise ValueError("both cameras must use the same lens model")
    for calibration in (cam0, cam1):
        if len(calibration.distortion) != 4:
            # cuVSLAM's EuRoC loader reads exactly k1..k4 (fisheye) or k1, k2, p1, p2.
            raise ValueError("distortion must have 4 coefficients for this loader")
    for matrix in (cam0_from_cam1, cam0_from_imu):
        if matrix.shape != (4, 4) or not np.allclose(matrix[3], [0, 0, 0, 1]):
            raise ValueError("extrinsics must be 4x4 homogeneous transforms")
    if cam0.model == "fisheye":
        # Recalibrated convention: every T_BS is cam0_from_sensor.
        name, frames = "sensor_recalibrated.yaml", {
            "cam0": np.eye(4), "cam1": cam0_from_cam1, "imu0": cam0_from_imu}
    elif cam0.model == "radtan":
        # Original EuRoC convention: T_BS is body(=IMU)_from_sensor.
        imu_from_cam0 = np.linalg.inv(cam0_from_imu)
        name, frames = "sensor.yaml", {
            "cam0": imu_from_cam0, "cam1": imu_from_cam0 @ cam0_from_cam1, "imu0": np.eye(4)}
    else:
        raise ValueError(f"unsupported lens model {cam0.model!r}")
    written = []
    for sensor, calibration in (("cam0", cam0), ("cam1", cam1)):
        text = _camera_yaml(calibration, frames[sensor], source)
        (mav0 / sensor / name).write_text(text)
        written.append(f"{sensor}/{name}")
    (mav0 / "imu0" / name).write_text(_imu_yaml(imu, frames["imu0"], source))
    written.append(f"imu0/{name}")
    return written


def merge_imu(gyro: np.ndarray, accel: np.ndarray) -> np.ndarray:
    """Combine separately stamped gyro and accel reports into EuRoC rows.

    gyro, accel: (N, 4) arrays [t_ns, x, y, z]. Accel is linearly interpolated
    onto the gyro timestamps; gyro samples outside the accel time span are dropped
    (no extrapolation). Returns (M, 7) [t_ns, wx, wy, wz, ax, ay, az].
    """
    gyro = gyro[np.argsort(gyro[:, 0], kind="stable")]
    accel = accel[np.argsort(accel[:, 0], kind="stable")]
    if len(accel) < 2 or len(gyro) == 0:
        return np.empty((0, 7))
    inside = (gyro[:, 0] >= accel[0, 0]) & (gyro[:, 0] <= accel[-1, 0])
    gyro = gyro[inside]
    merged = np.empty((len(gyro), 7))
    merged[:, :4] = gyro
    for k in range(3):
        merged[:, 4 + k] = np.interp(gyro[:, 0], accel[:, 0], accel[:, 1 + k])
    return merged


def _matrix_yaml(matrix: np.ndarray) -> str:
    values = ", ".join(f"{v:.9g}" for v in matrix.reshape(-1))
    return f"T_BS:\n  cols: 4\n  rows: 4\n  data: [{values}]\n"


def _camera_yaml(calibration: CameraCalibration, t_bs: np.ndarray, source: str) -> str:
    model = "fisheye" if calibration.model == "fisheye" else "pinhole"
    distortion = "fisheye" if calibration.model == "fisheye" else "radial-tangential"
    return (
        "sensor_type: camera\n"
        f"comment: \"{source}\"\n"
        + _matrix_yaml(t_bs)
        + f"rate_hz: {calibration.rate_hz:g}\n"
        f"resolution: [{calibration.resolution[0]}, {calibration.resolution[1]}]\n"
        f"camera_model: {model}\n"
        f"intrinsics: [{', '.join(f'{v:.9g}' for v in calibration.intrinsics)}]\n"
        f"distortion_model: {distortion}\n"
        f"distortion_coefficients: [{', '.join(f'{v:.9g}' for v in calibration.distortion)}]\n"
    )


def _imu_yaml(imu: ImuNoise, t_bs: np.ndarray, source: str) -> str:
    return (
        "sensor_type: imu\n"
        f"comment: \"{source}\"\n"
        + _matrix_yaml(t_bs)
        + f"rate_hz: {imu.rate_hz:g}\n"
        f"gyroscope_noise_density: {imu.gyroscope_noise_density:.6g}\n"
        f"gyroscope_random_walk: {imu.gyroscope_random_walk:.6g}\n"
        f"accelerometer_noise_density: {imu.accelerometer_noise_density:.6g}\n"
        f"accelerometer_random_walk: {imu.accelerometer_random_walk:.6g}\n"
    )
