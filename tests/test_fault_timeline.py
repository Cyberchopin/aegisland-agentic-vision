from aegisland.fault_timeline import (
    run_fault_timeline,
)


def test_fault_timeline_covers_complete_scenario() -> None:
    rows = run_fault_timeline()

    assert len(rows) == 80
    assert rows[0].frame == 0
    assert rows[-1].frame == 79


def test_gps_loss_precedes_camera_fault() -> None:
    rows = run_fault_timeline()

    assert rows[19].gps_available
    assert not rows[20].gps_available

    assert not rows[39].camera_fault_active
    assert rows[40].camera_fault_active
    assert rows[55].camera_fault_active
    assert not rows[56].camera_fault_active


def test_timeline_records_navigation_observability() -> None:
    rows = run_fault_timeline()

    row = rows[30]

    assert row.navigation_mode
    assert row.visual_health_state
    assert row.imu_health_state
    assert row.action
