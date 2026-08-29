from aegisland.domain import Action, Decision, SafetyLevel
from aegisland.stability import ActionStabilizer


def decision(
    action: Action,
    level: SafetyLevel = SafetyLevel.HIGH,
) -> Decision:
    return Decision(
        action=action,
        safety_level=level,
        risk_score=0.5,
        requires_human_approval=False,
        reasons=("test",),
        evidence_id="stability-test",
    )


def test_safety_escalation_is_immediate() -> None:
    stabilizer = ActionStabilizer(latch_frames=3)

    stabilizer.stabilize(
        decision(Action.LAND)
    )

    result = stabilizer.stabilize(
        decision(
            Action.EMERGENCY_RECOVERY,
            SafetyLevel.CRITICAL,
        )
    )

    assert result.action == Action.EMERGENCY_RECOVERY


def test_single_safe_frame_does_not_immediately_cancel_emergency() -> None:
    stabilizer = ActionStabilizer(latch_frames=3)

    stabilizer.stabilize(
        decision(
            Action.EMERGENCY_RECOVERY,
            SafetyLevel.CRITICAL,
        )
    )

    result = stabilizer.stabilize(
        decision(
            Action.LAND,
            SafetyLevel.HIGH,
        )
    )

    assert result.action == Action.EMERGENCY_RECOVERY


def test_action_can_deescalate_after_latch_expires() -> None:
    stabilizer = ActionStabilizer(latch_frames=2)

    stabilizer.stabilize(
        decision(
            Action.EMERGENCY_RECOVERY,
            SafetyLevel.CRITICAL,
        )
    )

    stabilizer.stabilize(decision(Action.LAND))
    stabilizer.stabilize(decision(Action.LAND))
    result = stabilizer.stabilize(decision(Action.LAND))

    assert result.action == Action.LAND
