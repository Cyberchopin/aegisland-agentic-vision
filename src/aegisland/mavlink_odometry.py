"""
Convert `aegisland.localization.v1` messages to MAVLink ODOMETRY (#331) fields.

Pure functions only: this module builds a dict of field values and never opens
a link. It is used by the PX4 SITL replay tool; sending to real hardware is out
of scope (CONTROL OUTPUT = DISABLED applies to everything AegisLand emits).

Frame policy (checked against PX4 v1.17.0 mavlink_receiver.cpp / EKF2.cpp):
- world odom_ned -> MAV_FRAME_LOCAL_NED, odom_frd -> MAV_FRAME_LOCAL_FRD.
  PX4 converts ENU/FLU *positions* but copies the quaternion unchanged, so we
  never send ENU/FLU: conversion happens here, explicitly.
- body body_frd -> child MAV_FRAME_BODY_FRD (only used for velocity in PX4).
- `*_dataset` frames are offline-only and must be converted first
  (see `dataset_to_frd`).
- estimator_type = MAV_ESTIMATOR_TYPE_VIO (routes to vehicle_visual_odometry).
- unknown covariance: first element NaN (MAVLink convention); PX4 then uses
  EKF2_EVP_NOISE. Known variance can only RAISE the noise above that floor
  (PX4 takes fmaxf(param^2, message variance)).

Trust marking (see docs/dronehacks/sitl-ekf2-ev-validation.md for evidence):
- "untrusted" -> quality 0 (requires EKF2_EV_QMIN >= 1 on the vehicle; with the
  default QMIN = 0 PX4 fuses any quality, including -1)
- "degraded"  -> keep quality, inflate position variance
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .localization_message import LocalizationMessage, TrackingState
from .trajectory_eval import quaternion_to_matrix

MAV_FRAME_LOCAL_NED = 1
MAV_FRAME_BODY_FRD = 12
MAV_FRAME_LOCAL_FRD = 20
MAV_ESTIMATOR_TYPE_VIO = 3

WORLD_FRAME_IDS = {"odom_ned": MAV_FRAME_LOCAL_NED, "odom_frd": MAV_FRAME_LOCAL_FRD}
BODY_FRAME_IDS = {"body_frd": MAV_FRAME_BODY_FRD}

UNTRUSTED_QUALITY = 0
DEGRADED_VARIANCE_M2 = 1.0


@dataclass(frozen=True)
class TrustMarking:
    """How a supervisor verdict changes the outgoing message."""

    quality: int | None = None  # None = keep the tracker's value
    min_position_variance_m2: float | None = None


TRUST_POLICY = {
    "ok": TrustMarking(),
    "degraded": TrustMarking(min_position_variance_m2=DEGRADED_VARIANCE_M2),
    "untrusted": TrustMarking(quality=UNTRUSTED_QUALITY),
}


def _matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = [
            0.25 * s,
            (rotation[2, 1] - rotation[1, 2]) / s,
            (rotation[0, 2] - rotation[2, 0]) / s,
            (rotation[1, 0] - rotation[0, 1]) / s,
        ]
    else:
        i = int(np.argmax(np.diag(rotation)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(1.0 + rotation[i, i] - rotation[j, j] - rotation[k, k])
        q = [0.0] * 4
        q[0] = (rotation[k, j] - rotation[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (rotation[j, i] + rotation[i, j]) / s
        q[1 + k] = (rotation[k, i] + rotation[i, k]) / s
    quaternion = np.array(q)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion if quaternion[0] >= 0 else -quaternion


def dataset_to_frd(
    message: LocalizationMessage,
    *,
    frd_from_dataset_world: np.ndarray,
    dataset_body_from_frd_body: np.ndarray,
) -> LocalizationMessage:
    """Re-express an odom_dataset/body_dataset message in odom_frd/body_frd.

    frd_from_dataset_world: 3x3 rotation, dataset world axes -> FRD world axes
    dataset_body_from_frd_body: 3x3 rotation, FRD body axes -> dataset body axes
    Both must be proper rotations; translation offsets (lever arms) are not applied.
    """
    if message.frame_id != "odom_dataset" or message.child_frame_id != "body_dataset":
        raise ValueError("dataset_to_frd expects odom_dataset / body_dataset")
    for matrix in (frd_from_dataset_world, dataset_body_from_frd_body):
        if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-6) or np.linalg.det(matrix) < 0:
            raise ValueError("frame changes must be proper rotations")
    world_from_body = quaternion_to_matrix(np.array(message.orientation_xyzw))
    rotation = frd_from_dataset_world @ world_from_body @ dataset_body_from_frd_body
    w, x, y, z = _matrix_to_quaternion_wxyz(rotation)
    position = frd_from_dataset_world @ np.array(message.position_m)
    velocity = None
    if message.velocity_mps is not None:
        velocity = tuple(float(v) for v in frd_from_dataset_world @ np.array(message.velocity_mps))
    variance = None
    if message.position_variance_m2 is not None:
        # Diagonal covariance under rotation: keep the trace-preserving diagonal.
        covariance = np.diag(message.position_variance_m2)
        variance = tuple(
            float(v)
            for v in np.diag(frd_from_dataset_world @ covariance @ frd_from_dataset_world.T)
        )
    fields = {
        name: getattr(message, name) for name in LocalizationMessage.__dataclass_fields__
    }
    fields.update(
        frame_id="odom_frd",
        child_frame_id="body_frd",
        position_m=tuple(float(v) for v in position),
        orientation_xyzw=(float(x), float(y), float(z), float(w)),
        velocity_mps=velocity,
        position_variance_m2=variance,
    )
    return LocalizationMessage(**fields)


def apply_trust(message_quality: int, variance, verdict: str) -> tuple[int, tuple | None]:
    try:
        marking = TRUST_POLICY[verdict]
    except KeyError as error:
        raise ValueError(f"unknown verdict {verdict!r}; expected {sorted(TRUST_POLICY)}") from error
    quality = message_quality if marking.quality is None else marking.quality
    if marking.min_position_variance_m2 is not None:
        floor = marking.min_position_variance_m2
        variance = tuple(max(floor, v) for v in (variance or (floor, floor, floor)))
    return quality, variance


def to_odometry_fields(
    message: LocalizationMessage,
    *,
    time_usec: int,
    verdict: str = "ok",
) -> dict[str, object]:
    """MAVLink ODOMETRY field values for one message (FRD/NED frames only).

    `time_usec` is the capture time on the clock PX4 knows about: the companion
    clock when TIMESYNC is running (PX4 maps it), otherwise PX4 ignores it and
    stamps the arrival time instead.
    """
    if message.frame_id not in WORLD_FRAME_IDS:
        raise ValueError(
            f"frame_id {message.frame_id!r} not sendable; convert to {sorted(WORLD_FRAME_IDS)}"
        )
    if message.child_frame_id not in BODY_FRAME_IDS:
        raise ValueError(f"child_frame_id {message.child_frame_id!r} not sendable")
    nan = float("nan")
    lost = (
        not message.valid
        or message.quality == -1
        or message.tracking_state is TrackingState.LOST
    )
    quality, variance = apply_trust(-1 if lost else message.quality, message.position_variance_m2,
                                    verdict)
    pose_covariance = [nan] + [0.0] * 20
    if variance is not None:
        pose_covariance = [0.0] * 21
        # Upper-right triangle, row-major: (0,0)=0, (1,1)=6, (2,2)=11; rotation unknown.
        pose_covariance[0], pose_covariance[6], pose_covariance[11] = variance
        pose_covariance[15] = pose_covariance[18] = pose_covariance[20] = nan
    x, y, z, w = message.orientation_xyzw
    velocity = message.velocity_mps or (nan, nan, nan)
    return {
        "time_usec": int(time_usec),
        "frame_id": WORLD_FRAME_IDS[message.frame_id],
        "child_frame_id": BODY_FRAME_IDS[message.child_frame_id],
        "x": float(message.position_m[0]),
        "y": float(message.position_m[1]),
        "z": float(message.position_m[2]),
        "q": [float(w), float(x), float(y), float(z)],
        "vx": float(velocity[0]),
        "vy": float(velocity[1]),
        "vz": float(velocity[2]),
        "rollspeed": nan,
        "pitchspeed": nan,
        "yawspeed": nan,
        "pose_covariance": pose_covariance,
        "velocity_covariance": [nan] + [0.0] * 20,
        "reset_counter": int(message.reset_counter) % 256,
        "estimator_type": MAV_ESTIMATOR_TYPE_VIO,
        "quality": int(quality),
    }
