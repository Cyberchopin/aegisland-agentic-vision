import numpy as np
import pytest

from aegisland.kalibr_import import from_kalibr

CAMCHAIN = {
    "cam0": {
        "camera_model": "pinhole", "distortion_model": "equidistant",
        "intrinsics": [450.0, 451.0, 320.0, 200.0], "resolution": [640, 400],
        "distortion_coeffs": [0.01, -0.002, 0.0003, -0.00004],
        "T_cam_imu": [[0, -1, 0, 0.01], [1, 0, 0, 0.02], [0, 0, 1, 0.0], [0, 0, 0, 1]],
        "timeshift_cam_imu": 0.0042,
    },
    "cam1": {
        "camera_model": "pinhole", "distortion_model": "equidistant",
        "intrinsics": [449.0, 450.0, 321.0, 201.0], "resolution": [640, 400],
        "distortion_coeffs": [0.0, 0.0, 0.0, 0.0],
        "T_cn_cnm1": [[1, 0, 0, -0.075], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    },
}
IMU = {"imu0": {"gyroscope_noise_density": 1e-4, "gyroscope_random_walk": 2e-5,
                "accelerometer_noise_density": 1e-3, "accelerometer_random_walk": 3e-4,
                "update_rate": 200.0}}


def test_kalibr_conventions_map_to_cam0_relative_transforms() -> None:
    calibration = from_kalibr(CAMCHAIN, IMU, camera_rate_hz=20.0)
    # T_cn_cnm1 is cam1_from_cam0; cam1 sits +7.5 cm to the right of cam0.
    assert calibration.cam0_from_cam1[:3, 3] == pytest.approx([0.075, 0, 0])
    assert np.allclose(calibration.cam0_from_imu, np.array(CAMCHAIN["cam0"]["T_cam_imu"]))
    assert calibration.cam0.model == "fisheye"
    assert calibration.cam0.intrinsics == (450.0, 451.0, 320.0, 200.0)
    assert calibration.imu.rate_hz == 200.0
    assert calibration.timeshift_cam_imu_s == pytest.approx(0.0042)


def test_unsupported_models_and_bad_transforms_are_refused() -> None:
    bad_model = {**CAMCHAIN, "cam0": {**CAMCHAIN["cam0"], "camera_model": "omni"}}
    with pytest.raises(ValueError, match="unsupported Kalibr model"):
        from_kalibr(bad_model, IMU, camera_rate_hz=20.0)
    skew = {**CAMCHAIN, "cam0": {**CAMCHAIN["cam0"],
                                 "T_cam_imu": [[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                                               [0, 0, 0, 1]]}}
    with pytest.raises(ValueError, match="orthonormal"):
        from_kalibr(skew, IMU, camera_rate_hz=20.0)
    with pytest.raises(ValueError, match="missing cam1"):
        from_kalibr({"cam0": CAMCHAIN["cam0"]}, IMU, camera_rate_hz=20.0)
