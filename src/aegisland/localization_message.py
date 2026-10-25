"""
Stage-3 draft: a self-describing localization message for shadow supervision.

The message carries everything a consumer needs to interpret a pose without
guessing: frames, units (meters, radians via unit quaternion, m/s, m^2),
capture time vs publish time, the clock domain of both, validity, a tracking
state and a reset counter. It mirrors what PX4 EKF2 needs from MAVLink
ODOMETRY (FRD frames, quality -1..100, reset_counter) so a later bridge is a
pure conversion, but nothing in this module sends anything anywhere.

`quality` is an ordinal health score, not a calibrated probability.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from enum import StrEnum

SCHEMA = "aegisland.localization.v1"
# *_dataset frames are the recording's own conventions (e.g. EuRoC imu0 body,
# tracker-defined world); valid for offline replay only, never for PX4.
WORLD_FRAMES = frozenset({"odom_frd", "odom_ned", "odom_dataset"})
BODY_FRAMES = frozenset({"body_frd", "body_dataset"})
CLOCK_DOMAINS = frozenset({"host_monotonic", "device_monotonic", "px4_boot", "dataset"})
QUATERNION_TOLERANCE = 1e-3


class TrackingState(StrEnum):
    INITIALIZING = "initializing"
    TRACKING = "tracking"
    LOST = "lost"
    RELOCALIZED = "relocalized"


@dataclass(frozen=True)
class LocalizationMessage:
    schema: str
    source: str
    sequence: int
    frame_id: str
    child_frame_id: str
    clock_domain: str
    capture_time_ns: int
    publish_time_ns: int
    position_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    velocity_mps: tuple[float, float, float] | None
    position_variance_m2: tuple[float, float, float] | None
    quality: int
    valid: bool
    tracking_state: TrackingState
    reset_counter: int


def _finite(values) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def validate(message: LocalizationMessage) -> list[str]:
    """Return every contract violation; an empty list means structurally valid."""
    errors = []
    if message.schema != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    if message.frame_id not in WORLD_FRAMES:
        errors.append(f"frame_id must be one of {sorted(WORLD_FRAMES)}")
    if message.child_frame_id not in BODY_FRAMES:
        errors.append(f"child_frame_id must be one of {sorted(BODY_FRAMES)}")
    if message.clock_domain not in CLOCK_DOMAINS:
        errors.append(f"clock_domain must be one of {sorted(CLOCK_DOMAINS)}")
    if message.sequence < 0 or message.reset_counter < 0:
        errors.append("sequence and reset_counter must be non-negative")
    if message.capture_time_ns < 0:
        errors.append("capture_time_ns must be non-negative")
    if message.publish_time_ns < message.capture_time_ns:
        errors.append("publish_time_ns must not precede capture_time_ns")
    if len(message.position_m) != 3 or not _finite(message.position_m):
        errors.append("position_m must be 3 finite values")
    quaternion = message.orientation_xyzw
    if len(quaternion) != 4 or not _finite(quaternion):
        errors.append("orientation_xyzw must be 4 finite values")
    elif abs(math.sqrt(sum(value * value for value in quaternion)) - 1.0) > QUATERNION_TOLERANCE:
        errors.append("orientation_xyzw must be a unit quaternion")
    if message.velocity_mps is not None and (
        len(message.velocity_mps) != 3 or not _finite(message.velocity_mps)
    ):
        errors.append("velocity_mps must be None or 3 finite values")
    variance = message.position_variance_m2
    if variance is not None and (
        len(variance) != 3 or not _finite(variance) or min(variance) < 0
    ):
        errors.append("position_variance_m2 must be None or 3 finite non-negative values")
    if not -1 <= message.quality <= 100:
        errors.append("quality must be in [-1, 100]")
    if message.valid and message.quality == -1:
        errors.append("valid message cannot carry quality -1 (failed)")
    if message.valid and message.tracking_state in (
        TrackingState.LOST,
        TrackingState.INITIALIZING,
    ):
        errors.append("valid message cannot have tracking_state lost/initializing")
    return errors


def to_dict(message: LocalizationMessage) -> dict:
    payload = asdict(message)
    payload["tracking_state"] = message.tracking_state.value
    return payload


def from_dict(payload: dict) -> LocalizationMessage:
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"unsupported schema {payload.get('schema')!r}; expected {SCHEMA}")
    names = [field.name for field in fields(LocalizationMessage)]
    missing = [name for name in names if name not in payload]
    if missing:
        raise ValueError(f"missing fields: {', '.join(missing)}")
    values = {name: payload[name] for name in names}
    for name in ("position_m", "orientation_xyzw", "velocity_mps", "position_variance_m2"):
        if values[name] is not None:
            values[name] = tuple(float(value) for value in values[name])
    values["tracking_state"] = TrackingState(values["tracking_state"])
    return LocalizationMessage(**values)
