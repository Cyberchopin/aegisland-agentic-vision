from aegisland.domain import (
    Action,
    Decision,
    SafetyLevel,
    Telemetry,
    VisionEvidence,
    ZoneCandidate,
)
from aegisland.recovery import Maneuver, RecoveryPlanner


def evidence() -> VisionEvidence:
    candidate = ZoneCandidate(
        "Z2-1",
        (0, 0, 10, 10),
        0.84,
        0.02,
        0.01,
        0.05,
        0.02,
        0.9,
        True,
    )

    return VisionEvidence(
        evidence_id="recovery-test",
        frame_index=0,
        confidence=0.9,
        obstacle_risk=0.91,
        motion_risk=0.2,
        candidates=(candidate,),
    )


def emergency_decision() -> Decision:
    return Decision(
        action=Action.EMERGENCY_RECOVERY,
        safety_level=SafetyLevel.CRITICAL,
        risk_score=0.9,
        requires_human_approval=False,
        reasons=("compound emergency",),
        evidence_id="recovery-test",
        target_zone_id="Z2-1",
    )


def test_recovery_plan_contains_safe_emergency_sequence() -> None:
    planner = RecoveryPlanner()

    plan = planner.plan(
        emergency_decision(),
        Telemetry(
            battery_percent=2,
            altitude_m=10,
            horizontal_speed_mps=2.0,
        ),
        evidence(),
    )

    assert plan is not None

    maneuvers = [step.maneuver for step in plan.steps]

    assert maneuvers == [
        Maneuver.BRAKE,
        Maneuver.EVADE,
        Maneuver.ALIGN_SAFE_ZONE,
        Maneuver.DESCEND,
        Maneuver.LAND,
    ]


def test_recovery_skips_brake_when_already_slow() -> None:
    planner = RecoveryPlanner()

    plan = planner.plan(
        emergency_decision(),
        Telemetry(
            battery_percent=2,
            altitude_m=10,
            horizontal_speed_mps=0.2,
        ),
        evidence(),
    )

    assert plan is not None
    assert plan.first_step is not None
    assert plan.first_step.maneuver == Maneuver.EVADE


def test_non_recovery_decision_has_no_recovery_plan() -> None:
    decision = Decision(
        action=Action.CONTINUE_MISSION,
        safety_level=SafetyLevel.NOMINAL,
        risk_score=0.1,
        requires_human_approval=False,
        reasons=("safe",),
        evidence_id="test",
    )

    plan = RecoveryPlanner().plan(
        decision,
        Telemetry(80, 10),
        evidence(),
    )

    assert plan is None
