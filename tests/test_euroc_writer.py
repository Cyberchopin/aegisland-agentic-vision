import json
import re

import cv2
import numpy as np
import pytest

from aegisland.euroc_writer import (
    CameraCalibration,
    EurocRecordingWriter,
    ImuNoise,
    merge_imu,
)

IMU = ImuNoise(1.7e-4, 1.9e-5, 2e-3, 3e-3, 200.0)


def camera(model: str) -> CameraCalibration:
    return CameraCalibration((64, 48), (40.0, 40.0, 32.0, 24.0), model, (0.01, -0.02, 0.0, 0.0),
                             20.0)


def yaml_matrix(text: str) -> np.ndarray:
    values = re.search(r"data: \[([^\]]*)\]", text).group(1)
    return np.array([float(v) for v in values.split(",")]).reshape(4, 4)


def image(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (48, 64), dtype=np.uint8)


def test_stereo_imu_and_depth_round_trip_in_euroc_layout(tmp_path) -> None:
    writer = EurocRecordingWriter(tmp_path / "rec", depth=True)
    assert writer.add_stereo(1_000, image(1), 1_400, image(2))
    assert writer.add_stereo(51_000, image(3), 51_000, image(4))
    assert not writer.add_stereo(51_000, image(5), 51_000, image(6))  # duplicate stamp
    assert not writer.add_stereo(90_000, image(7), 2_000_000, image(8))  # > 1 ms apart
    assert writer.add_imu(500, (0.1, 0.2, 0.3), (0.0, 0.0, 9.8))
    assert not writer.add_imu(400, (0, 0, 0), (0, 0, 0))  # out of order
    assert writer.add_depth(1_000, np.full((48, 64), 1234, np.uint16))
    manifest = writer.close({"device": "synthetic"})

    mav0 = tmp_path / "rec" / "mav0"
    rows = (mav0 / "cam0" / "data.csv").read_text().splitlines()
    assert rows == ["#timestamp [ns],filename", "1000,1000.png", "51000,51000.png"]
    assert (mav0 / "cam1" / "data.csv").read_text() == (mav0 / "cam0" / "data.csv").read_text()
    assert np.array_equal(cv2.imread(str(mav0 / "cam1" / "data" / "1000.png"), 0), image(2))
    assert (mav0 / "imu0" / "data.csv").read_text().splitlines()[1] == "500,0.1,0.2,0.3,0.0,0.0,9.8"
    depth = cv2.imread(str(mav0 / "depth0" / "data" / "1000.png"), cv2.IMREAD_UNCHANGED)
    assert depth.dtype == np.uint16 and int(depth[0, 0]) == 1234
    assert (mav0 / "cam_pairs.csv").read_text().splitlines()[1] == "1000,1400,400"
    assert manifest["counts"]["unpaired_stereo_refused"] == 1
    assert manifest["counts"]["out_of_order_refused"] == {"cam": 1, "imu": 1, "depth": 0}
    assert json.loads((tmp_path / "rec" / "recording.json").read_text())["device"] == "synthetic"
    assert "mav0/imu0/data.csv" in (tmp_path / "rec" / "SHA256SUMS").read_text()


def test_writer_refuses_existing_directory_and_bad_images(tmp_path) -> None:
    EurocRecordingWriter(tmp_path / "rec")
    with pytest.raises(FileExistsError):
        EurocRecordingWriter(tmp_path / "rec")
    writer = EurocRecordingWriter(tmp_path / "rec2")
    with pytest.raises(ValueError, match="uint8"):
        writer.add_stereo(1, image(1).astype(np.float32), 1, image(2))
    with pytest.raises(ValueError, match="without depth"):
        writer.add_depth(1, np.zeros((2, 2), np.uint16))


def test_fisheye_calibration_uses_recalibrated_cam0_relative_files(tmp_path) -> None:
    writer = EurocRecordingWriter(tmp_path / "rec")
    cam0_from_cam1 = np.eye(4)
    cam0_from_cam1[0, 3] = 0.075
    cam0_from_imu = np.eye(4)
    cam0_from_imu[:3, 3] = (0.01, 0.02, 0.03)
    files = writer.write_calibration(camera("fisheye"), camera("fisheye"),
                                     cam0_from_cam1=cam0_from_cam1, cam0_from_imu=cam0_from_imu,
                                     imu=IMU, source="test")
    assert files == ["cam0/sensor_recalibrated.yaml", "cam1/sensor_recalibrated.yaml",
                     "imu0/sensor_recalibrated.yaml"]
    mav0 = tmp_path / "rec" / "mav0"
    assert np.allclose(yaml_matrix((mav0 / "cam1" / "sensor_recalibrated.yaml").read_text()),
                       cam0_from_cam1)
    imu_text = (mav0 / "imu0" / "sensor_recalibrated.yaml").read_text()
    assert np.allclose(yaml_matrix(imu_text), cam0_from_imu)
    assert "gyroscope_noise_density: 0.00017" in imu_text


def test_radtan_calibration_uses_body_relative_original_convention(tmp_path) -> None:
    writer = EurocRecordingWriter(tmp_path / "rec")
    cam0_from_imu = np.eye(4)
    cam0_from_imu[:3, 3] = (0.01, 0.0, 0.0)
    writer.write_calibration(camera("radtan"), camera("radtan"), cam0_from_cam1=np.eye(4),
                             cam0_from_imu=cam0_from_imu, imu=IMU, source="test")
    mav0 = tmp_path / "rec" / "mav0"
    assert not (mav0 / "cam0" / "sensor_recalibrated.yaml").exists()
    cam0 = (mav0 / "cam0" / "sensor.yaml").read_text()
    assert np.allclose(yaml_matrix(cam0), np.linalg.inv(cam0_from_imu))
    assert "distortion_model: radial-tangential" in cam0
    assert np.allclose(yaml_matrix((mav0 / "imu0" / "sensor.yaml").read_text()), np.eye(4))


def test_calibration_rejects_mixed_models_and_wrong_coefficient_count(tmp_path) -> None:
    writer = EurocRecordingWriter(tmp_path / "rec")
    with pytest.raises(ValueError, match="same lens model"):
        writer.write_calibration(camera("fisheye"), camera("radtan"), cam0_from_cam1=np.eye(4),
                                 cam0_from_imu=np.eye(4), imu=IMU, source="t")
    long = CameraCalibration((64, 48), (1, 1, 1, 1), "radtan", tuple(range(8)), 20.0)
    with pytest.raises(ValueError, match="4 coefficients"):
        writer.write_calibration(long, long, cam0_from_cam1=np.eye(4), cam0_from_imu=np.eye(4),
                                 imu=IMU, source="t")


def test_merge_imu_interpolates_accel_onto_gyro_stamps_without_extrapolation() -> None:
    gyro = np.array([[0, 1, 1, 1], [5, 2, 2, 2], [10, 3, 3, 3], [20, 4, 4, 4]], float)
    accel = np.array([[2, 0, 0, 0], [12, 10, 0, -10]], float)
    merged = merge_imu(gyro, accel)
    assert merged[:, 0].tolist() == [5, 10]
    assert merged[0, 4:].tolist() == pytest.approx([3.0, 0.0, -3.0])
    assert merged[1, 1:4].tolist() == [3, 3, 3]
