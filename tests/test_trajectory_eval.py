import math

import numpy as np
import pytest

from aegisland.trajectory_eval import (
    Trajectory,
    absolute_trajectory_error,
    associate,
    load_euroc_groundtruth,
    load_tum,
    relative_pose_error,
    umeyama_alignment,
    write_tum,
)


def _rotation_z(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _quaternion_z(angle_rad: float) -> np.ndarray:
    # xyzw order.
    return np.array([0.0, 0.0, math.sin(angle_rad / 2), math.cos(angle_rad / 2)])


def _helix(count: int = 50) -> Trajectory:
    t = np.arange(count, dtype=np.float64) * 0.05
    positions = np.stack([np.cos(t), np.sin(t), 0.1 * t], axis=1)
    quaternions = np.stack([_quaternion_z(angle) for angle in t])
    return Trajectory(timestamps_s=t, positions_m=positions, quaternions_xyzw=quaternions)


def test_umeyama_recovers_rigid_transform_without_scale() -> None:
    source = _helix().positions_m
    rotation = _rotation_z(0.7)
    translation = np.array([1.0, -2.0, 0.5])
    target = source @ rotation.T + translation

    estimated_rotation, estimated_translation, scale = umeyama_alignment(
        source, target, with_scale=False
    )

    assert scale == 1.0
    np.testing.assert_allclose(estimated_rotation, rotation, atol=1e-9)
    np.testing.assert_allclose(estimated_translation, translation, atol=1e-9)


def test_umeyama_recovers_scale_when_requested() -> None:
    source = _helix().positions_m
    target = 2.5 * source

    _, _, scale = umeyama_alignment(source, target, with_scale=True)

    assert scale == pytest.approx(2.5)


def test_ate_is_zero_for_rigidly_transformed_copy() -> None:
    ground_truth = _helix()
    moved = Trajectory(
        timestamps_s=ground_truth.timestamps_s,
        positions_m=ground_truth.positions_m @ _rotation_z(1.1).T + np.array([3.0, 0.0, 1.0]),
        quaternions_xyzw=ground_truth.quaternions_xyzw,
    )

    result = absolute_trajectory_error(moved, ground_truth, max_time_difference_s=0.001)

    assert result["pairs"] == 50
    assert result["unit"] == "m"
    assert result["rmse"] == pytest.approx(0.0, abs=1e-9)


def test_ate_rejects_scale_alignment_label_as_metric() -> None:
    ground_truth = _helix()
    scaled = Trajectory(
        timestamps_s=ground_truth.timestamps_s,
        positions_m=0.5 * ground_truth.positions_m,
        quaternions_xyzw=ground_truth.quaternions_xyzw,
    )

    rigid = absolute_trajectory_error(scaled, ground_truth, max_time_difference_s=0.001)
    similarity = absolute_trajectory_error(
        scaled, ground_truth, max_time_difference_s=0.001, with_scale=True
    )

    # Without scale correction a half-scale trajectory has real metric error.
    assert rigid["rmse"] > 0.1
    assert similarity["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert similarity["alignment"] == "sim3"
    assert similarity["unit"] == "m (after scale correction; not a metric-scale result)"
    assert similarity["scale"] == pytest.approx(2.0)


def test_ate_measures_known_constant_offset_error() -> None:
    ground_truth = _helix()
    noisy_positions = ground_truth.positions_m.copy()
    # Alternate +/- 0.1 m in z: the mean is removed by alignment, residual is 0.1 m.
    noisy_positions[:, 2] += np.where(np.arange(50) % 2 == 0, 0.1, -0.1)
    noisy = Trajectory(
        timestamps_s=ground_truth.timestamps_s,
        positions_m=noisy_positions,
        quaternions_xyzw=ground_truth.quaternions_xyzw,
    )

    result = absolute_trajectory_error(noisy, ground_truth, max_time_difference_s=0.001)

    assert result["rmse"] == pytest.approx(0.1, abs=5e-3)


def test_associate_drops_pairs_outside_time_tolerance() -> None:
    reference = np.array([0.0, 0.1, 0.2, 0.3])
    query = np.array([0.001, 0.15, 0.299, 0.5])

    pairs = associate(query, reference, max_time_difference_s=0.005)

    assert pairs == [(0, 0), (2, 3)]


def test_associate_rejects_unsorted_timestamps() -> None:
    with pytest.raises(ValueError, match="sorted"):
        associate(np.array([0.2, 0.1]), np.array([0.0, 0.1]), max_time_difference_s=0.01)


def test_ate_requires_minimum_number_of_pairs() -> None:
    ground_truth = _helix()
    far_future = Trajectory(
        timestamps_s=ground_truth.timestamps_s + 100.0,
        positions_m=ground_truth.positions_m,
        quaternions_xyzw=ground_truth.quaternions_xyzw,
    )

    with pytest.raises(ValueError, match="pairs"):
        absolute_trajectory_error(far_future, ground_truth, max_time_difference_s=0.01)


def test_rpe_is_zero_for_rigidly_transformed_copy_and_detects_drift() -> None:
    ground_truth = _helix()
    world_rotation = _rotation_z(0.4)
    rotated_quaternions = []
    for angle in ground_truth.timestamps_s:
        rotated_quaternions.append(_quaternion_z(angle + 0.4))
    moved = Trajectory(
        timestamps_s=ground_truth.timestamps_s,
        positions_m=ground_truth.positions_m @ world_rotation.T + 5.0,
        quaternions_xyzw=np.stack(rotated_quaternions),
    )

    exact = relative_pose_error(moved, ground_truth, delta_frames=5, max_time_difference_s=0.001)
    assert exact["translation_rmse_m"] == pytest.approx(0.0, abs=1e-9)
    assert exact["rotation_rmse_deg"] == pytest.approx(0.0, abs=1e-6)

    drifting = Trajectory(
        timestamps_s=ground_truth.timestamps_s,
        positions_m=ground_truth.positions_m * 1.1,
        quaternions_xyzw=ground_truth.quaternions_xyzw,
    )
    drift = relative_pose_error(
        drifting, ground_truth, delta_frames=5, max_time_difference_s=0.001
    )
    assert drift["translation_rmse_m"] > 0.0
    assert drift["pairs"] == 45


def test_tum_round_trip_and_euroc_groundtruth_loading(tmp_path) -> None:
    trajectory = _helix(5)
    path = tmp_path / "trajectory.tum"
    write_tum(path, trajectory)

    loaded = load_tum(path)
    np.testing.assert_allclose(loaded.timestamps_s, trajectory.timestamps_s)
    np.testing.assert_allclose(loaded.positions_m, trajectory.positions_m)
    np.testing.assert_allclose(loaded.quaternions_xyzw, trajectory.quaternions_xyzw)

    euroc = tmp_path / "data.csv"
    euroc.write_text(
        "#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], "
        "q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []\n"
        "1403636579758555392,4.688,-1.786,0.783,0.534,-0.153,-0.827,-0.082\n"
        "1403636579763555584,4.688,-1.786,0.787,0.534,-0.153,-0.827,-0.082\n"
    )
    ground_truth = load_euroc_groundtruth(euroc)
    assert ground_truth.timestamps_s[0] == pytest.approx(1403636579.758555392)
    np.testing.assert_allclose(ground_truth.positions_m[1], [4.688, -1.786, 0.787])
    # Converted from wxyz to xyzw and normalized.
    expected = np.array([-0.153, -0.827, -0.082, 0.534])
    np.testing.assert_allclose(
        ground_truth.quaternions_xyzw[0], expected / np.linalg.norm(expected)
    )


def test_trajectory_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        Trajectory(
            timestamps_s=np.array([0.0, 1.0]),
            positions_m=np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]]),
            quaternions_xyzw=np.array([[0.0, 0.0, 0.0, 1.0]] * 2),
        )
