import math

import pytest

from aegisland.localization_message import (
    SCHEMA,
    LocalizationMessage,
    TrackingState,
    from_dict,
    to_dict,
    validate,
)


def message(**overrides) -> LocalizationMessage:
    fields = {
        "schema": SCHEMA,
        "source": "cuvslam-17.0.1",
        "sequence": 7,
        "frame_id": "odom_frd",
        "child_frame_id": "body_frd",
        "clock_domain": "host_monotonic",
        "capture_time_ns": 1_000_000_000,
        "publish_time_ns": 1_004_000_000,
        "position_m": (1.0, 2.0, -0.5),
        "orientation_xyzw": (0.0, 0.0, 0.0, 1.0),
        "velocity_mps": (0.1, 0.0, 0.0),
        "position_variance_m2": (0.01, 0.01, 0.02),
        "quality": 80,
        "valid": True,
        "tracking_state": TrackingState.TRACKING,
        "reset_counter": 0,
    }
    fields.update(overrides)
    return LocalizationMessage(**fields)


def test_well_formed_message_has_no_errors_and_round_trips() -> None:
    original = message()

    assert validate(original) == []
    assert from_dict(to_dict(original)) == original


def test_publish_before_capture_is_rejected() -> None:
    errors = validate(message(publish_time_ns=999_000_000))

    assert any("publish_time_ns" in error for error in errors)


def test_non_unit_quaternion_and_non_finite_position_are_rejected() -> None:
    errors = validate(
        message(orientation_xyzw=(0.0, 0.0, 0.0, 2.0), position_m=(math.nan, 0.0, 0.0))
    )

    assert any("orientation" in error for error in errors)
    assert any("position_m" in error for error in errors)


def test_valid_flag_must_agree_with_failed_quality_and_lost_state() -> None:
    assert any("quality" in error for error in validate(message(quality=-1)))
    assert any(
        "tracking_state" in error
        for error in validate(message(tracking_state=TrackingState.LOST))
    )
    assert validate(
        message(quality=-1, valid=False, tracking_state=TrackingState.LOST)
    ) == []


def test_unknown_frames_and_negative_variance_are_rejected() -> None:
    errors = validate(message(frame_id="map_enu", position_variance_m2=(0.01, -1.0, 0.0)))

    assert any("frame_id" in error for error in errors)
    assert any("variance" in error for error in errors)


def test_from_dict_rejects_wrong_schema_and_missing_fields() -> None:
    payload = to_dict(message())
    with pytest.raises(ValueError, match="schema"):
        from_dict({**payload, "schema": "aegisland.localization.v0"})

    payload.pop("capture_time_ns")
    with pytest.raises(ValueError, match="capture_time_ns"):
        from_dict(payload)


def test_quality_out_of_range_is_rejected() -> None:
    assert any("quality" in error for error in validate(message(quality=101)))
