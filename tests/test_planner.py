from aegisland.domain import Action, SafetyLevel, Telemetry, VisionEvidence, ZoneCandidate
from aegisland.planner import SafetyPlanner


def zone(score: float = 0.82, *, safe: bool = True) -> ZoneCandidate:
    return ZoneCandidate(
        zone_id="Z2-1",
        bbox_xywh=(0, 0, 100, 100),
        score=score,
        edge_density=0.04,
        motion_occupancy=0.02,
        texture_risk=0.08,
        appearance_occupancy=0.02,
        clearance=0.9,
        safe=safe,
    )


def evidence(
    *,
    confidence: float = 0.85,
    obstacle: float = 0.08,
    motion: float = 0.03,
    candidate: ZoneCandidate | None = None,
) -> VisionEvidence:
    return VisionEvidence(
        evidence_id="ev-test",
        frame_index=1,
        confidence=confidence,
        obstacle_risk=obstacle,
        motion_risk=motion,
        candidates=(candidate or zone(),),
    )


def telemetry(battery: float, *, gps: bool = True) -> Telemetry:
    return Telemetry(
        battery_percent=battery,
        altitude_m=12,
        gps_available=gps,
        home_link_available=gps,
    )


def test_open_cv_result_changes_the_action() -> None:
    planner = SafetyPlanner()
    safe = planner.decide(telemetry(6), evidence(candidate=zone(0.84, safe=True)))
    unsafe = planner.decide(telemetry(6), evidence(candidate=zone(0.43, safe=False)))

    assert safe.action == Action.LAND
    assert safe.target_zone_id == "Z2-1"
    assert unsafe.action == Action.REQUEST_HUMAN_APPROVAL
    assert unsafe.requires_human_approval


def test_collision_risk_has_priority_over_battery_plan() -> None:
    decision = SafetyPlanner().decide(
        telemetry(12),
        evidence(
            obstacle=0.91,
            motion=0.20,
            candidate=zone(0.84, safe=True),
        ),
    )

    assert decision.action == Action.EVADE_AND_HOLD
    assert decision.safety_level == SafetyLevel.CRITICAL


def test_critical_battery_uses_least_risk_zone_without_waiting() -> None:
    decision = SafetyPlanner().decide(
        telemetry(2.5), evidence(candidate=zone(0.50, safe=False))
    )
    assert decision.action == Action.EMERGENCY_LAND
    assert decision.target_zone_id == "Z2-1"
    assert not decision.requires_human_approval


def test_low_confidence_is_fail_closed() -> None:
    decision = SafetyPlanner().decide(telemetry(40), evidence(confidence=0.2))
    assert decision.action == Action.HOLD_AND_SCAN
    assert decision.safety_level == SafetyLevel.HIGH


def test_low_battery_returns_home_only_when_navigation_is_available() -> None:
    planner = SafetyPlanner()
    rth = planner.decide(telemetry(11, gps=True), evidence())
    no_gps = planner.decide(telemetry(11, gps=False), evidence())
    assert rth.action == Action.RETURN_HOME
    assert no_gps.action == Action.REQUEST_HUMAN_APPROVAL


def test_collision_and_critical_battery_needs_emergency_recovery() -> None:
    decision = SafetyPlanner().decide(
        telemetry(2),
        evidence(
            obstacle=0.91,
            motion=0.20,
            candidate=zone(0.84, safe=True),
        ),
    )

    assert decision.safety_level == SafetyLevel.CRITICAL
    assert decision.action == Action.EMERGENCY_RECOVERY
    assert decision.target_zone_id == "Z2-1"


def test_high_temporal_risk_triggers_preemptive_evasion() -> None:
    evidence = VisionEvidence(
        evidence_id="temporal-danger",
        frame_index=10,
        confidence=0.95,
        obstacle_risk=0.05,
        motion_risk=0.10,
        temporal_risk=0.8,
    )

    telemetry = Telemetry(
        battery_percent=80,
        altitude_m=10,
        gps_available=True,
        home_link_available=True,
    )

    decision = SafetyPlanner().decide(
        telemetry,
        evidence,
    )

    assert decision.action == Action.EVADE_AND_HOLD


def test_temporal_risk_plus_critical_battery_triggers_recovery() -> None:
    evidence = VisionEvidence(
        evidence_id="temporal-compound",
        frame_index=11,
        confidence=0.95,
        obstacle_risk=0.05,
        motion_risk=0.10,
        temporal_risk=0.8,
    )

    telemetry = Telemetry(
        battery_percent=2,
        altitude_m=10,
        gps_available=True,
        home_link_available=True,
    )

    decision = SafetyPlanner().decide(
        telemetry,
        evidence,
    )

    assert decision.action == Action.EMERGENCY_RECOVERY
