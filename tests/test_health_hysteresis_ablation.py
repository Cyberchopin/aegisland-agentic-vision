from aegisland.health_hysteresis_ablation import (
    run_ablation,
)


def test_hysteresis_blocks_premature_healthy_reentry() -> None:
    result = run_ablation()

    assert (
        result.no_hysteresis.premature_healthy_grants
        > 0
    )

    assert (
        result.aegisland_hysteresis
        .premature_healthy_grants
        == 0
    )


def test_hysteresis_reduces_unstable_healthy_frames() -> None:
    result = run_ablation()

    assert (
        result.aegisland_hysteresis
        .unstable_healthy_frames
        <
        result.no_hysteresis.unstable_healthy_frames
    )


def test_hysteresis_recovers_more_conservatively() -> None:
    result = run_ablation()

    assert (
        result.aegisland_hysteresis
        .stable_health_latency_frames
        >
        result.no_hysteresis
        .stable_health_latency_frames
    )
