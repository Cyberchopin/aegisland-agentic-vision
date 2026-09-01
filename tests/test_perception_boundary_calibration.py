from aegisland.perception_boundary_calibration import (
    FINE_SEVERITIES,
    calibrate,
    endpoint_confusion,
)
from aegisland.perception_severity_benchmark import (
    FAULT_FAMILIES,
    run,
)


def test_fine_grid_has_one_percent_resolution() -> None:
    assert len(FINE_SEVERITIES) == 101
    assert FINE_SEVERITIES[0] == 0.0
    assert FINE_SEVERITIES[-1] == 1.0


def test_boundary_calibration_is_deterministic() -> None:
    severities = (
        0.0,
        0.25,
        0.50,
        0.75,
        1.0,
    )

    first = run(
        seeds=(42,),
        severities=severities,
    )

    second = run(
        seeds=(42,),
        severities=severities,
    )

    assert calibrate(first) == calibrate(second)


def test_clean_state_has_no_authority_intervention() -> None:
    results = run(
        seeds=(42, 43),
        severities=(0.0,),
    )

    assert all(
        result.authority == "full"
        for result in results
    )


def test_endpoint_faults_always_trigger_intervention() -> None:
    results = run(
        seeds=(42, 43),
        severities=(1.0,),
    )

    assert all(
        result.authority != "full"
        for result in results
    )


def test_endpoint_confusion_covers_all_fault_families() -> None:
    results = run(
        seeds=(42,),
        severities=(1.0,),
    )

    matrix = endpoint_confusion(results)

    injected = {
        family
        for family, _ in matrix
    }

    assert injected == set(
        FAULT_FAMILIES
    )
