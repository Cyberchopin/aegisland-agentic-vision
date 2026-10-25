from aegisland.localization_faults import (
    Event,
    FaultSpec,
    build_events,
    score_detection,
)
from aegisland.localization_message import SCHEMA, LocalizationMessage, TrackingState

PERIOD_NS = 50_000_000


def nominal(count: int = 40) -> list[LocalizationMessage]:
    return [
        LocalizationMessage(
            schema=SCHEMA,
            source="test",
            sequence=i,
            frame_id="odom_dataset",
            child_frame_id="body_dataset",
            clock_domain="dataset",
            capture_time_ns=i * PERIOD_NS,
            publish_time_ns=i * PERIOD_NS + 2_000_000,
            position_m=(0.01 * i, 0.0, 0.0),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            velocity_mps=None,
            position_variance_m2=None,
            quality=90,
            valid=True,
            tracking_state=TrackingState.TRACKING,
            reset_counter=0,
        )
        for i in range(count)
    ]


def messages(events: list[Event]) -> list[LocalizationMessage]:
    return [event.message for event in events if event.kind == "message"]


def test_drop_removes_only_frames_inside_window() -> None:
    events, window = build_events(
        nominal(), [], FaultSpec("dropped_frames", start=10, length=5), PERIOD_NS
    )

    sequences = [m.sequence for m in messages(events)]
    assert sequences == list(range(10)) + list(range(15, 40))
    assert window == (10 * PERIOD_NS, 15 * PERIOD_NS)


def test_outage_removes_frames_and_adds_ticks() -> None:
    events, _ = build_events(
        nominal(), [], FaultSpec("camera_outage", start=10, length=20), PERIOD_NS
    )

    assert len(messages(events)) == 20
    assert sum(event.kind == "tick" for event in events) >= 20


def test_pose_jump_offsets_all_later_positions_without_reset() -> None:
    events, _ = build_events(
        nominal(), [], FaultSpec("pose_jump", start=20, magnitude=1.0), PERIOD_NS
    )

    out = messages(events)
    assert out[19].position_m[0] == 0.19
    assert out[20].position_m[0] == 1.2
    assert all(m.reset_counter == 0 for m in out)


def test_stale_delivery_delays_publish_time_inside_window() -> None:
    events, _ = build_events(
        nominal(),
        [],
        FaultSpec("stale_delivery", start=5, length=3, magnitude=0.3),
        PERIOD_NS,
    )

    delayed = [m for m in messages(events) if m.publish_time_ns - m.capture_time_ns > 1e8]
    assert [m.sequence for m in delayed] == [5, 6, 7]
    times = [event.time_ns for event in events]
    assert times == sorted(times)


def test_out_of_order_swaps_capture_order() -> None:
    events, _ = build_events(
        nominal(), [], FaultSpec("out_of_order", start=12), PERIOD_NS
    )

    sequences = [m.sequence for m in messages(events)]
    assert sequences[12:14] == [13, 12]


def test_imu_offset_shifts_imu_stamps() -> None:
    imu = [(i * 5_000_000, (0.0, 0.0, 0.1)) for i in range(10)]

    events, window = build_events(
        nominal(), imu, FaultSpec("imu_time_offset", magnitude=0.05), PERIOD_NS
    )

    stamps = sorted(event.time_ns for event in events if event.kind == "imu")
    assert stamps[0] == 50_000_000
    assert window == (0, 39 * PERIOD_NS + 2_000_000)


def test_score_detection_reports_latency_and_false_positives() -> None:
    assessments = [
        (0, "ok", ()),
        (100, "fault", ("stale",)),
        (500, "fault", ("pose_jump",)),
        (700, "ok", ()),
    ]

    score = score_detection(assessments, window=(400, 600), expected={"pose_jump"})

    assert score["detected"] is True
    assert score["latency_ns"] == 100
    assert score["false_alarms_outside_window"] == 1
