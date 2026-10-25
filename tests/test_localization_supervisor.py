import math

import numpy as np

from aegisland.localization_message import SCHEMA, LocalizationMessage, TrackingState
from aegisland.localization_supervisor import (
    LocalizationShadowSupervisor,
    SupervisorConfig,
)

PERIOD_NS = 50_000_000


def msg(index: int, *, x: float | None = None, capture_ns: int | None = None, **overrides):
    capture = index * PERIOD_NS if capture_ns is None else capture_ns
    fields = {
        "schema": SCHEMA,
        "source": "test",
        "sequence": index,
        "frame_id": "odom_frd",
        "child_frame_id": "body_frd",
        "clock_domain": "dataset",
        "capture_time_ns": capture,
        "publish_time_ns": capture + 5_000_000,
        "position_m": (0.05 * index if x is None else x, 0.0, 0.0),
        "orientation_xyzw": (0.0, 0.0, 0.0, 1.0),
        "velocity_mps": None,
        "position_variance_m2": None,
        "quality": 90,
        "valid": True,
        "tracking_state": TrackingState.TRACKING,
        "reset_counter": 0,
    }
    fields.update(overrides)
    return LocalizationMessage(**fields)


def supervisor() -> LocalizationShadowSupervisor:
    return LocalizationShadowSupervisor(SupervisorConfig(expected_period_ns=PERIOD_NS))


def run(stream):
    sup = supervisor()
    return [sup.assess(message, now_ns=message.publish_time_ns) for message in stream]


def test_nominal_stream_is_ok_and_never_commands() -> None:
    results = run([msg(i) for i in range(10)])

    assert {r.status for r in results} == {"ok"}
    assert all(r.control_output == "DISABLED" and r.command_executed is False for r in results)


def test_dropped_frames_are_reported_from_sequence_gap() -> None:
    stream = [msg(i) for i in range(5)] + [msg(i) for i in range(8, 12)]

    results = run(stream)

    assert results[5].status == "degraded"
    assert "dropped_frames" in results[5].reasons


def test_stale_message_is_a_fault() -> None:
    sup = supervisor()
    late = msg(1)

    result = sup.assess(late, now_ns=late.capture_time_ns + 300_000_000)

    assert result.status == "fault"
    assert "stale" in result.reasons


def test_camera_outage_detected_by_tick_without_messages() -> None:
    sup = supervisor()
    for i in range(3):
        sup.assess(msg(i), now_ns=msg(i).publish_time_ns)

    quiet = sup.tick(now_ns=2 * PERIOD_NS + 500_000_000)

    assert quiet.status == "fault"
    assert "no_messages" in quiet.reasons


def test_time_going_backwards_is_a_fault() -> None:
    stream = [msg(0), msg(1), msg(2, capture_ns=PERIOD_NS // 2)]

    results = run(stream)

    assert results[2].status == "fault"
    assert "capture_time_not_increasing" in results[2].reasons


def test_pose_jump_without_reset_is_a_fault_but_with_reset_is_relocalization() -> None:
    jump = [msg(0), msg(1), msg(2, x=3.0)]
    assert "pose_jump" in run(jump)[2].reasons

    relocalized = [
        msg(0),
        msg(1),
        msg(2, x=3.0, reset_counter=1, tracking_state=TrackingState.RELOCALIZED),
    ]
    result = run(relocalized)[2]
    assert "pose_jump" not in result.reasons
    assert "relocalized" in result.reasons
    assert result.status == "degraded"


def test_lost_tracking_and_invalid_contract_are_faults() -> None:
    lost = msg(1, valid=False, quality=-1, tracking_state=TrackingState.LOST)
    assert "tracking_lost" in run([msg(0), lost])[1].reasons

    broken = msg(1, orientation_xyzw=(0.0, 0.0, 0.0, 3.0))
    result = run([msg(0), broken])[1]
    assert result.status == "fault"
    assert any(reason.startswith("contract:") for reason in result.reasons)


def _yawing_stream_with_imu(imu_offset_s: float):
    sup = LocalizationShadowSupervisor(
        SupervisorConfig(expected_period_ns=PERIOD_NS, imu_check_window_s=6.0)
    )
    rate = lambda t: 0.8 * math.sin(1.7 * t) + 0.5 * math.sin(4.1 * t + 0.3)
    yaw = 0.0
    results = []
    imu_t = 0.0
    for index in range(160):
        t = index * PERIOD_NS / 1e9
        while imu_t <= t:
            # IMU stamped late by imu_offset_s: at stamp s it reports the rate at s - offset.
            sup.add_imu(int(imu_t * 1e9), (0.0, 0.0, rate(imu_t - imu_offset_s)))
            imu_t += 0.005
        if index:
            yaw += rate(t - PERIOD_NS / 2e9) * PERIOD_NS / 1e9
        quaternion = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
        message = msg(index, x=0.0, orientation_xyzw=quaternion)
        results.append(sup.assess(message, now_ns=message.publish_time_ns))
    return results


def test_imu_camera_time_offset_is_detected_only_when_injected() -> None:
    aligned = _yawing_stream_with_imu(0.0)
    shifted = _yawing_stream_with_imu(0.06)

    assert not any("imu_time_offset" in r.reasons for r in aligned)
    flagged = [r for r in shifted if "imu_time_offset" in r.reasons]
    assert flagged
    assert np.isclose(flagged[-1].imu_offset_s, 0.06, atol=0.01)


def test_imu_offset_flag_clears_when_estimate_becomes_unavailable() -> None:
    sup = LocalizationShadowSupervisor(
        SupervisorConfig(expected_period_ns=PERIOD_NS, imu_check_window_s=6.0)
    )
    rate = lambda t: 0.8 * math.sin(1.7 * t) + 0.5 * math.sin(4.1 * t + 0.3)
    yaw, imu_t, results = 0.0, 0.0, []
    for index in range(400):
        t = index * PERIOD_NS / 1e9
        moving = index < 160  # then hover: rotation stops, offset is unobservable
        while imu_t <= t:
            gyro_z = rate(imu_t - 0.06) if moving else 0.0
            sup.add_imu(int(imu_t * 1e9), (0.0, 0.0, gyro_z))
            imu_t += 0.005
        if index and moving:
            yaw += rate(t - PERIOD_NS / 2e9) * PERIOD_NS / 1e9
        quaternion = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
        message = msg(index, x=0.0, orientation_xyzw=quaternion)
        results.append(sup.assess(message, now_ns=message.publish_time_ns))

    assert any("imu_time_offset" in r.reasons for r in results[:160])
    assert "imu_time_offset" not in results[-1].reasons
    assert results[-1].imu_offset_s is None


def yawing(index: int, rate_rad_s: float, **overrides):
    half = 0.5 * rate_rad_s * index * PERIOD_NS / 1e9
    return msg(index, orientation_xyzw=(0.0, 0.0, math.sin(half), math.cos(half)), **overrides)


def replay_with_gyro(count: int, pose_rate: float, gyro_rate) -> list:
    sup = supervisor()
    results = []
    for i in range(count):
        for k in range(10):  # 200 Hz gyro inside each 50 ms frame interval
            t_ns = (i - 1) * PERIOD_NS + (k + 1) * PERIOD_NS // 10
            sup.add_imu(t_ns, (0.0, 0.0, gyro_rate(i)))
        message = yawing(i, pose_rate)
        results.append(sup.assess(message, now_ns=message.publish_time_ns))
    return results


def test_rotation_consistency_tolerates_constant_gyro_bias() -> None:
    results = replay_with_gyro(120, 0.5, lambda i: 0.5 + 0.08)  # 0.08 rad/s bias

    late = results[40:]
    assert all("rotation_inconsistent" not in r.reasons for r in late)
    assert late[-1].rotation_residual_rad_s < 0.005


def test_rotation_inconsistency_is_degraded_not_fault() -> None:
    # Gyro agrees for 60 frames, then the poses keep turning while the gyro says 0.3 rad/s less.
    results = replay_with_gyro(120, 0.5, lambda i: 0.5 if i < 60 else 0.2)

    assert all("rotation_inconsistent" not in r.reasons for r in results[30:60])
    flagged = [r for r in results[60:] if "rotation_inconsistent" in r.reasons]
    assert flagged
    assert {r.status for r in flagged} == {"degraded"}


def test_low_texture_uses_smoothed_feature_matches_and_can_be_disabled() -> None:
    sup = supervisor()
    statuses = []
    for i in range(30):
        message = msg(i)
        matches = 900 if i < 15 else 300
        statuses.append(sup.assess(message, now_ns=message.publish_time_ns,
                                   feature_matches=matches))
    assert "low_texture" not in statuses[14].reasons
    assert "low_texture" in statuses[-1].reasons
    assert statuses[-1].status == "degraded"

    disabled = LocalizationShadowSupervisor(
        SupervisorConfig(expected_period_ns=PERIOD_NS, min_feature_matches=None)
    )
    result = disabled.assess(msg(0), now_ns=msg(0).publish_time_ns, feature_matches=0)
    assert "low_texture" not in result.reasons


def test_rotation_check_waits_for_a_gyro_bias_estimate() -> None:
    # A large constant bias must not be reported before the bias history exists.
    results = replay_with_gyro(15, 0.5, lambda i: 0.5 + 0.08)

    assert all("rotation_inconsistent" not in r.reasons for r in results)
    assert results[-1].rotation_residual_rad_s is None
