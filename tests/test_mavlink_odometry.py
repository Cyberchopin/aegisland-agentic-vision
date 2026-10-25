import math

import numpy as np
import pytest

from aegisland.localization_message import SCHEMA, LocalizationMessage, TrackingState
from aegisland.mavlink_odometry import (
    MAV_ESTIMATOR_TYPE_VIO,
    MAV_FRAME_BODY_FRD,
    MAV_FRAME_LOCAL_FRD,
    UNTRUSTED_QUALITY,
    dataset_to_frd,
    to_odometry_fields,
)


def message(**overrides) -> LocalizationMessage:
    fields = {
        "schema": SCHEMA,
        "source": "test",
        "sequence": 3,
        "frame_id": "odom_frd",
        "child_frame_id": "body_frd",
        "clock_domain": "host_monotonic",
        "capture_time_ns": 1_000_000_000,
        "publish_time_ns": 1_010_000_000,
        "position_m": (1.0, 2.0, -0.5),
        "orientation_xyzw": (0.0, 0.0, math.sin(math.radians(15)), math.cos(math.radians(15))),
        "velocity_mps": None,
        "position_variance_m2": None,
        "quality": 80,
        "valid": True,
        "tracking_state": TrackingState.TRACKING,
        "reset_counter": 0,
    }
    fields.update(overrides)
    return LocalizationMessage(**fields)


def yaw_deg(q_wxyz) -> float:
    w, x, y, z = q_wxyz
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def test_frd_message_maps_to_local_frd_vio_with_wxyz_quaternion() -> None:
    fields = to_odometry_fields(message(), time_usec=1_000_000)
    assert fields["frame_id"] == MAV_FRAME_LOCAL_FRD
    assert fields["child_frame_id"] == MAV_FRAME_BODY_FRD
    assert fields["estimator_type"] == MAV_ESTIMATOR_TYPE_VIO
    assert (fields["x"], fields["y"], fields["z"]) == (1.0, 2.0, -0.5)
    assert yaw_deg(fields["q"]) == pytest.approx(30.0)
    assert fields["quality"] == 80


def test_unknown_variance_uses_nan_first_element_and_known_variance_fills_diagonal() -> None:
    unknown = to_odometry_fields(message(), time_usec=0)["pose_covariance"]
    assert math.isnan(unknown[0])
    known = to_odometry_fields(message(position_variance_m2=(0.1, 0.2, 0.3)), time_usec=0)
    covariance = known["pose_covariance"]
    assert (covariance[0], covariance[6], covariance[11]) == (0.1, 0.2, 0.3)
    assert math.isnan(covariance[15])


def test_verdicts_mark_quality_or_inflate_variance() -> None:
    untrusted = to_odometry_fields(message(), time_usec=0, verdict="untrusted")
    assert untrusted["quality"] == UNTRUSTED_QUALITY
    degraded = to_odometry_fields(
        message(position_variance_m2=(0.01, 2.0, 0.01)), time_usec=0, verdict="degraded"
    )
    assert degraded["quality"] == 80
    assert degraded["pose_covariance"][0] == 1.0
    assert degraded["pose_covariance"][6] == 2.0
    with pytest.raises(ValueError, match="unknown verdict"):
        to_odometry_fields(message(), time_usec=0, verdict="maybe")


def test_lost_tracking_is_sent_as_failed_quality() -> None:
    lost = message(valid=False, quality=-1, tracking_state=TrackingState.LOST)
    assert to_odometry_fields(lost, time_usec=0)["quality"] == -1


def test_dataset_frames_are_refused_until_converted() -> None:
    raw = message(frame_id="odom_dataset", child_frame_id="body_dataset")
    with pytest.raises(ValueError, match="not sendable"):
        to_odometry_fields(raw, time_usec=0)


def test_dataset_to_frd_flips_z_up_world_and_preserves_yaw_sign_convention() -> None:
    # Dataset world: x forward, y left, z up (FLU-like). Body = FLU too.
    flu_to_frd = np.diag([1.0, -1.0, -1.0])
    yaw_left = math.radians(20)  # +20 deg about z-up = turning left
    raw = message(
        frame_id="odom_dataset",
        child_frame_id="body_dataset",
        position_m=(1.0, 2.0, 3.0),
        orientation_xyzw=(0.0, 0.0, math.sin(yaw_left / 2), math.cos(yaw_left / 2)),
        position_variance_m2=(0.1, 0.2, 0.3),
    )
    converted = dataset_to_frd(
        raw, frd_from_dataset_world=flu_to_frd, dataset_body_from_frd_body=flu_to_frd
    )
    assert converted.frame_id == "odom_frd"
    assert converted.position_m == pytest.approx((1.0, -2.0, -3.0))
    assert converted.position_variance_m2 == pytest.approx((0.1, 0.2, 0.3))
    fields = to_odometry_fields(converted, time_usec=0)
    # Turning left is a negative yaw in FRD/NED.
    assert yaw_deg(fields["q"]) == pytest.approx(-20.0)


def test_dataset_to_frd_rejects_reflections() -> None:
    raw = message(frame_id="odom_dataset", child_frame_id="body_dataset")
    with pytest.raises(ValueError, match="proper rotations"):
        dataset_to_frd(
            raw,
            frd_from_dataset_world=np.diag([1.0, 1.0, -1.0]),
            dataset_body_from_frd_body=np.eye(3),
        )
