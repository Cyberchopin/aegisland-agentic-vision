from aegisland.reliability_benchmark import (
    evaluate_reliability,
    run_cascading_failure,
)


def test_cascading_failure_moves_through_capability_modes() -> None:
    rows = run_cascading_failure()

    assert rows[19].navigation_mode == "gps_primary"

    assert (
        rows[20].navigation_mode
        == "visual_inertial_fallback"
    )

    assert (
        rows[32].navigation_mode
        == "visual_fallback"
    )

    assert rows[40].action == "hold_and_scan"


def test_combined_position_sensor_failure_never_continues_mission() -> None:
    rows = run_cascading_failure()

    unsafe = [
        row
        for row in rows
        if (
            not row.gps_available
            and row.camera_fault_active
            and row.action == "continue_mission"
        )
    ]

    assert unsafe == []


def test_out_of_order_imu_is_time_synchronized() -> None:
    rows = run_cascading_failure()

    healthy_ooo = [
        row
        for row in rows
        if (
            row.imu_transport_state
            == "out_of_order_healthy"
        )
    ]

    assert healthy_ooo

    assert all(
        row.imu_sync_valid
        for row in healthy_ooo
    )

    assert all(
        row.imu_sync_method == "interpolated"
        for row in healthy_ooo
    )


def test_reliability_metrics_report_safe_failure_response() -> None:
    rows = run_cascading_failure()
    metrics = evaluate_reliability(rows)

    assert metrics.unsafe_continuation_frames == 0

    assert (
        metrics.camera_perception_detection_latency_frames
        == 0
    )

    assert metrics.safe_hold_latency_frames == 0
