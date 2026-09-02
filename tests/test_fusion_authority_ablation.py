from aegisland.fusion_authority_ablation import (
    run_ablation,
)


def test_capability_policy_blocks_premature_authority() -> None:
    result = run_ablation()

    assert (
        result.confidence_only
        .premature_authority_frames
        > 0
    )

    assert (
        result.capability_aware
        .premature_authority_frames
        == 0
    )


def test_capability_policy_blocks_premature_raw_continue() -> None:
    result = run_ablation()

    assert (
        result.confidence_only
        .premature_raw_continue_frames
        > 0
    )

    assert (
        result.capability_aware
        .premature_raw_continue_frames
        == 0
    )


def test_capability_policy_reduces_planner_flapping() -> None:
    result = run_ablation()

    assert (
        result.capability_aware
        .raw_action_transitions
        <
        result.confidence_only
        .raw_action_transitions
    )


def test_capability_policy_waits_for_health_recovery() -> None:
    result = run_ablation()

    assert (
        result.capability_aware
        .stable_raw_recovery_frame
        >=
        result.capability_aware
        .stable_health_frame
    )


def test_capability_policy_trades_availability_for_authority_safety() -> None:
    result = run_ablation()

    assert (
        result.capability_aware
        .stable_raw_recovery_frame
        >
        result.confidence_only
        .stable_raw_recovery_frame
    )
