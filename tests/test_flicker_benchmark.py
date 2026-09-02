from aegisland.flicker_benchmark import (
    run_benchmark,
)


def test_health_does_not_reenter_healthy_during_flicker() -> None:
    metrics = run_benchmark()

    assert metrics.premature_trust_grants > 0
    assert metrics.premature_healthy_grants == 0


def test_capability_policy_blocks_raw_reentry_during_flicker() -> None:
    metrics = run_benchmark()

    assert (
        metrics.raw_continue_frames_during_unstable_recovery
        == 0
    )

    assert (
        metrics.final_continue_frames_during_unstable_recovery
        == 0
    )


def test_navigation_authority_waits_for_healthy_sensor_state() -> None:
    metrics = run_benchmark()

    assert (
        metrics.raw_recovery_frame
        >=
        metrics.stable_health_frame
    )


def test_stabilizer_delays_control_reentry_after_raw_recovery() -> None:
    metrics = run_benchmark()

    assert (
        metrics.final_recovery_frame
        >
        metrics.raw_recovery_frame
    )


def test_recovery_pipeline_is_ordered() -> None:
    metrics = run_benchmark()

    assert (
        metrics.stable_physical_recovery_frame
        <
        metrics.stable_trust_frame
        <=
        metrics.stable_health_frame
        <=
        metrics.raw_recovery_frame
        <=
        metrics.final_recovery_frame
    )
