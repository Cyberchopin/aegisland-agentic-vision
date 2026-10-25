import math

import numpy as np
import pytest

from aegisland.localization_diagnostics import (
    angular_speed_from_poses,
    direction_agreement,
    estimate_time_offset,
    failure_rate_by_bin,
    predicted_center_flow,
    vdr_image_flow,
)

FOCAL = 400.0
WIDTH, HEIGHT = 640, 480


def pinhole(points: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            FOCAL * points[:, 0] / points[:, 2] + WIDTH / 2,
            FOCAL * points[:, 1] / points[:, 2] + HEIGHT / 2,
        ],
        axis=1,
    )


def _pose(rotation: np.ndarray | None = None, translation=(0.0, 0.0, 0.0)) -> np.ndarray:
    pose = np.eye(4)
    if rotation is not None:
        pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def _rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


CENTER_RAY = np.array([0.0, 0.0, 1.0])


def test_camera_translation_right_moves_scene_left_by_f_t_over_z() -> None:
    # Camera moves +0.1 m along its x axis: world_from_current = translate(+0.1).
    # current_from_previous therefore translates points by -0.1 m.
    current_from_previous = np.linalg.inv(_pose(translation=(0.1, 0.0, 0.0)))

    flow = predicted_center_flow(
        current_from_previous, pinhole, CENTER_RAY, depth_m=4.0
    )

    np.testing.assert_allclose(flow, [-FOCAL * 0.1 / 4.0, 0.0], atol=1e-9)


def test_translation_has_no_flow_at_infinite_depth_but_rotation_does() -> None:
    translation_only = np.linalg.inv(_pose(translation=(0.5, 0.0, 0.0)))
    assert np.allclose(
        predicted_center_flow(
            translation_only, pinhole, CENTER_RAY, depth_m=None
        ),
        0.0,
    )

    rotation_only = np.linalg.inv(_pose(rotation=_rotation_y(0.05)))
    flow = predicted_center_flow(
        rotation_only, pinhole, CENTER_RAY, depth_m=None
    )
    assert flow[0] == pytest.approx(-FOCAL * math.tan(0.05))
    assert flow[1] == pytest.approx(0.0, abs=1e-9)


def test_vdr_image_flow_inverts_normalized_velocity() -> None:
    flow = vdr_image_flow(np.array([-0.025]), np.array([0.01]), WIDTH, HEIGHT)
    np.testing.assert_allclose(flow, [[16.0, -4.8]])


def test_direction_agreement_ignores_small_reference_motion() -> None:
    reference = np.array([[10.0, 0.0], [0.0, 10.0], [0.1, 0.0], [-10.0, 0.0]])
    estimate = np.array([[5.0, 0.0], [0.0, -3.0], [-9.0, 0.0], [-20.0, 0.0]])

    result = direction_agreement(estimate, reference, minimum_reference_px=1.0)

    assert result["used"] == 3
    assert result["skipped_small_reference"] == 1
    assert result["cosine_median"] == pytest.approx(1.0)
    assert result["fraction_same_half_plane"] == pytest.approx(2 / 3)
    assert result["magnitude_ratio_median"] == pytest.approx(0.5)


def test_direction_agreement_skips_missing_estimates() -> None:
    reference = np.array([[10.0, 0.0], [10.0, 0.0]])
    estimate = np.array([[np.nan, np.nan], [3.0, 0.0]])

    result = direction_agreement(estimate, reference, minimum_reference_px=1.0)

    assert result["used"] == 1
    assert result["skipped_missing_estimate"] == 1


def test_direction_agreement_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        direction_agreement(np.zeros((2, 2)), np.zeros((3, 2)), minimum_reference_px=1.0)


def test_estimate_time_offset_recovers_known_lag() -> None:
    times = np.arange(0.0, 20.0, 0.005)
    signal = np.abs(np.sin(1.3 * times)) + 0.5 * np.abs(np.sin(3.1 * times + 0.4))
    lag = 0.030
    # The delayed stream reports at time t what the reference had at t - lag.
    delayed_times = np.arange(0.0, 20.0, 0.05)
    delayed = np.interp(delayed_times - lag, times, signal)

    result = estimate_time_offset(
        times, signal, delayed_times, delayed, search_s=0.1, step_s=0.0025
    )

    assert result["offset_s"] == pytest.approx(lag, abs=0.003)
    assert result["correlation"] > 0.99


def test_angular_speed_from_poses_constant_yaw_rate() -> None:
    rate = 0.5
    times = np.arange(0.0, 2.0, 0.05)
    poses = np.stack([_pose(rotation=_rotation_y(rate * t)) for t in times])

    mid_times, speeds = angular_speed_from_poses(times, poses)

    assert len(mid_times) == len(times) - 1
    np.testing.assert_allclose(speeds, rate, rtol=1e-6)


def test_failure_rate_by_bin_counts_invalid_frames() -> None:
    values = np.array([0.1, 0.2, 1.5, 1.6, 3.0])
    valid = np.array([True, True, False, True, False])

    rows = failure_rate_by_bin(values, valid, edges=[0.0, 1.0, 2.0, 10.0])

    assert [row["frames"] for row in rows] == [2, 2, 1]
    assert [row["invalid_fraction"] for row in rows] == [0.0, 0.5, 1.0]


def test_interval_averaging_removes_aliasing_bias_against_frame_rate_stream() -> None:
    # 200 Hz rate with 60 Hz vibration vs a 20 Hz stream that reports the
    # interval-average rate (what a finite difference of poses gives). True offset is 0.
    rng = np.random.default_rng(1)
    fast_t = np.arange(0.0, 60.0, 0.005)
    fast = (
        np.abs(np.sin(0.9 * fast_t))
        + 0.4 * np.abs(np.sin(2.3 * fast_t + 1.0))
        + 0.5 * np.sin(2 * np.pi * 60.0 * fast_t)
        + rng.normal(0.0, 0.05, fast_t.size)
    )
    frame_t = np.arange(0.0, 60.0, 0.05)
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (fast[1:] + fast[:-1]) * 0.005)])
    interval_mean = np.diff(np.interp(frame_t, fast_t, cumulative)) / 0.05
    mid = 0.5 * (frame_t[1:] + frame_t[:-1])

    point = estimate_time_offset(
        fast_t, fast, mid, interval_mean, search_s=0.04, step_s=0.0025
    )
    averaged = estimate_time_offset(
        fast_t, fast, mid, interval_mean, search_s=0.04, step_s=0.0025, average_window_s=0.05
    )

    assert abs(point["offset_s"]) == pytest.approx(0.0025)  # biased by one grid step
    assert abs(averaged["offset_s"]) < 1e-9
    assert averaged["correlation"] > point["correlation"]
    assert averaged["average_window_s"] == 0.05
