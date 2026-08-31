from aegisland.imu_fault_timeline import (
    run_imu_fault_timeline,
)


def test_out_of_order_imu_packets_are_compensated() -> None:
    rows = run_imu_fault_timeline()

    row = rows[25]

    assert not row.gps_available
    assert row.out_of_order_insertions > 0

    assert row.imu_sync_valid
    assert row.imu_sync_method == "interpolated"

    assert (
        row.navigation_mode
        == "visual_inertial_fallback"
    )


def test_stale_imu_degrades_to_visual_only_navigation() -> None:
    rows = run_imu_fault_timeline()

    row = rows[32]

    assert row.imu_transport_state == "stale"
    assert not row.imu_sync_valid
    assert row.imu_health_state == "failed"

    assert row.visual_health_state == "healthy"
    assert row.navigation_mode == "visual_fallback"

    # Losing IMU alone should not unnecessarily stop the mission
    # while visual localization remains healthy.
    assert row.action == "continue_mission"


def test_imu_recovery_is_hysteretic() -> None:
    rows = run_imu_fault_timeline()

    first = rows[38]
    second = rows[39]
    third = rows[40]

    assert first.imu_sync_valid
    assert first.imu_health_state == "degraded"

    assert second.imu_health_state == "degraded"
    assert third.imu_health_state == "healthy"


def test_out_of_order_counter_continues_after_recovery() -> None:
    rows = run_imu_fault_timeline()

    before_recovery = rows[37].out_of_order_insertions
    after_recovery = rows[45].out_of_order_insertions

    assert after_recovery > before_recovery
