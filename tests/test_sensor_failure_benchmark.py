from aegisland.sensor_failure_benchmark import (
    run_recovery_sequence,
    run_static_cases,
)


def test_all_sensor_failure_matrix_cases_enter_expected_mode() -> None:
    cases = run_static_cases()

    assert cases
    assert all(case.passed for case in cases)


def test_loss_of_gps_enters_visual_inertial_fallback() -> None:
    cases = {
        case.case: case
        for case in run_static_cases()
    }

    result = cases[
        "gps_lost_visual_imu_healthy"
    ]

    assert (
        result.actual_mode
        == "visual_inertial_fallback"
    )


def test_loss_of_all_position_sources_degrades_safely() -> None:
    cases = {
        case.case: case
        for case in run_static_cases()
    }

    result = cases[
        "gps_lost_camera_bad_imu_only"
    ]

    assert result.actual_mode == "degraded"
    assert result.action == "hold_and_scan"


def test_gps_authority_returns_progressively_after_recovery() -> None:
    sequence = run_recovery_sequence()

    first_recovered = sequence[5]
    later_recovered = sequence[-1]

    assert first_recovered["navigation_mode"] == "gps_primary"
    assert later_recovered["navigation_mode"] == "gps_primary"

    assert (
        later_recovered["gps_weight"]
        > first_recovered["gps_weight"]
    )
