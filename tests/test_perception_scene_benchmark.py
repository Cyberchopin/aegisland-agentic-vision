from aegisland.perception_scene_benchmark import (
    HEALTHY_SCENES,
    run,
)
from aegisland.perception_scenes import probe


def test_scene_baseline_probe_passes() -> None:
    results = probe()

    assert results
    assert all(result.passed for result in results)


def test_scene_distribution_contains_six_healthy_scenes() -> None:
    assert len(HEALTHY_SCENES) == 6


def test_clean_cross_scene_inputs_keep_full_authority() -> None:
    results = run(
        severities=(0.0,),
    )

    assert results

    assert all(
        result.observed_failure == "healthy"
        for result in results
    )

    assert all(
        result.authority == "full"
        for result in results
    )


def test_endpoint_faults_intervene_across_scenes() -> None:
    results = run(
        severities=(1.0,),
    )

    assert results

    assert all(
        result.authority != "full"
        for result in results
    )


def test_authority_does_not_reenter_on_coarse_sweep() -> None:
    results = run(
        severities=(
            0.0,
            0.25,
            0.50,
            0.75,
            1.0,
        ),
    )

    ranks = {
        "full": 0,
        "reduced": 1,
        "revoked": 2,
    }

    groups = {}

    for result in results:
        key = (
            result.scene,
            result.fault_family,
        )

        groups.setdefault(
            key,
            [],
        ).append(result)

    for rows in groups.values():
        rows.sort(
            key=lambda row: row.severity
        )

        previous = -1

        for row in rows:
            current = ranks[row.authority]

            assert current >= previous

            previous = current
