from aegisland.domain import Action, Decision, SafetyLevel
from aegisland.execution import ExecutionSafetyGuard, ExecutionState


def decision(action: Action = Action.CONTINUE_MISSION) -> Decision:
    return Decision(
        action=action,
        safety_level=SafetyLevel.NOMINAL,
        risk_score=0.1,
        requires_human_approval=False,
        reasons=("test",),
        evidence_id="execution-test",
    )


def test_execution_guard_starts_healthy() -> None:
    guard = ExecutionSafetyGuard()

    result = guard.observe_command_status("completed")

    assert result.state == ExecutionState.HEALTHY
    assert not result.requires_human_attention


def test_timeout_latches_fail_closed() -> None:
    guard = ExecutionSafetyGuard()

    result = guard.observe_command_status("timeout")

    assert result.state == ExecutionState.FAIL_CLOSED
    assert result.requires_human_attention


def test_fail_closed_blocks_future_normal_autonomy() -> None:
    guard = ExecutionSafetyGuard()

    guard.observe_command_status("failed")

    guarded = guard.guard_decision(
        decision(Action.CONTINUE_MISSION)
    )

    assert guarded.action == Action.REQUEST_HUMAN_APPROVAL
    assert guarded.safety_level == SafetyLevel.CRITICAL
    assert guarded.requires_human_approval


def test_fail_closed_cannot_reset_without_operator_ack() -> None:
    guard = ExecutionSafetyGuard()
    guard.observe_command_status("timeout")

    recovered = guard.try_reset(
        operator_acknowledged=False,
        channel_healthy=True,
    )

    assert not recovered
    assert guard.state == ExecutionState.FAIL_CLOSED


def test_fail_closed_cannot_reset_with_unhealthy_channel() -> None:
    guard = ExecutionSafetyGuard()
    guard.observe_command_status("failed")

    recovered = guard.try_reset(
        operator_acknowledged=True,
        channel_healthy=False,
    )

    assert not recovered
    assert guard.state == ExecutionState.FAIL_CLOSED


def test_fail_closed_resets_only_after_full_handshake() -> None:
    guard = ExecutionSafetyGuard()
    guard.observe_command_status("timeout")

    recovered = guard.try_reset(
        operator_acknowledged=True,
        channel_healthy=True,
    )

    assert recovered
    assert guard.state == ExecutionState.HEALTHY
