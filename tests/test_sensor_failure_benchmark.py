from aegisland.sensor_failure_benchmark import (
    run_recovery_sequence,
    run_static_cases,
)


def test_all_sensor_failure_matrix_cases_enter_expected_mode() -> None:
    cases = run_static_cases()

    assert cases
    assert all(
        case.passed
        for case in cases
    )


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

    assert (
        result.actual_mode
        == "degraded"
    )
    assert (
        result.action
        == "hold_and_scan"
    )


def test_gps_authority_recovers_conservatively_then_transfers() -> None:
    sequence = run_recovery_sequence()

    recovery_start = next(
        index
        for index in range(1, len(sequence))
        if (
            sequence[index]["gps_available"]
            and not sequence[index - 1]["gps_available"]
        )
    )

    first_physical_recovery = (
        sequence[recovery_start]
    )

    first_authorized = next(
        row
        for row in sequence[recovery_start:]
        if (
            row["navigation_mode"]
            == "gps_primary"
        )
    )

    later_recovered = sequence[-1]

    # Physical GPS return does not immediately regain authority.
    assert (
        first_physical_recovery[
            "navigation_mode"
        ]
        == "visual_inertial_fallback"
    )

    assert (
        first_authorized[
            "navigation_mode"
        ]
        == "gps_primary"
    )

    assert (
        later_recovered[
            "navigation_mode"
        ]
        == "gps_primary"
    )

    # Once authority is restored, smooth fusion progressively
    # transfers weight back toward GPS.
    assert (
        later_recovered["gps_weight"]
        >
        first_authorized["gps_weight"]
    )
