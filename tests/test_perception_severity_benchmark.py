from aegisland.perception_severity_benchmark import (
    FAULT_FAMILIES,
    run,
)


def test_severity_sweep_is_deterministic() -> None:
    first = run(
        seeds=(42,),
        severities=(0.0, 0.5, 1.0),
    )

    second = run(
        seeds=(42,),
        severities=(0.0, 0.5, 1.0),
    )

    assert first == second


def test_clean_frames_do_not_trigger_intervention() -> None:
    results = run(
        seeds=(42, 43, 44),
        severities=(0.0,),
    )

    assert len(results) == (
        3 * len(FAULT_FAMILIES)
    )

    for result in results:
        assert result.observed_failure == "healthy"
        assert result.authority == "full"
        assert result.localization_trusted


def test_maximum_fault_severity_intervenes() -> None:
    results = run(
        seeds=(42, 43, 44),
        severities=(1.0,),
    )

    for result in results:
        assert result.authority != "full"


def test_full_sweep_has_expected_shape() -> None:
    results = run()

    assert len(results) == (
        5
        * len(FAULT_FAMILIES)
        * 11
    )


def test_full_sweep_has_no_clean_false_authority_revokes() -> None:
    results = run()

    clean_results = [
        result
        for result in results
        if result.severity == 0.0
    ]

    assert clean_results

    assert all(
        result.authority == "full"
        for result in clean_results
    )
