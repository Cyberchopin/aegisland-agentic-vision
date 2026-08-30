from aegisland.agent import AegisLandAgent
from aegisland.domain import Action, Telemetry, VisionEvidence, ZoneCandidate
from aegisland.planner import SafetyPlanner
from aegisland.trace import MemoryTraceStore


def candidate() -> ZoneCandidate:
    return ZoneCandidate(
        "Z2-1", (0, 0, 10, 10), 0.8, 0.02, 0.01, 0.05, 0.02, 0.9, True
    )


class FakePerception:
    def __init__(self) -> None:
        self.calls: list[bool] = []

    def observe(self, frame, frame_index, *, active_perception=False):
        self.calls.append(active_perception)
        confidence = 0.75 if active_perception else 0.40
        evidence = VisionEvidence(
            evidence_id="retry" if active_perception else "first",
            frame_index=frame_index,
            confidence=confidence,
            obstacle_risk=0.05,
            motion_risk=0.02,
            candidates=(candidate(),),
            active_perception_used=active_perception,
        )
        return evidence, frame

    def enhance_for_active_perception(self, frame):
        return frame


def test_uncertainty_triggers_a_second_opencv_tool_call() -> None:
    perception = FakePerception()
    traces = MemoryTraceStore()
    agent = AegisLandAgent(perception, SafetyPlanner(), traces)
    telemetry = Telemetry(25, 10)

    event, _ = agent.step(object(), telemetry, 0)

    assert perception.calls == [False, True]
    assert event.evidence.evidence_id == "retry"
    assert event.evidence.active_perception_used
    assert len(traces.events) == 1
    assert traces.events[0].command["hardware_command_sent"] is False


class CriticalRecoveryPerception:
    def observe(self, frame, frame_index, *, active_perception=False):
        evidence = VisionEvidence(
            evidence_id="critical-recovery",
            frame_index=frame_index,
            confidence=0.90,
            obstacle_risk=0.91,
            motion_risk=0.20,
            candidates=(candidate(),),
            active_perception_used=False,
        )
        return evidence, frame

    def enhance_for_active_perception(self, frame):
        return frame


def test_critical_battery_and_collision_flows_through_agent() -> None:
    perception = CriticalRecoveryPerception()
    traces = MemoryTraceStore()

    agent = AegisLandAgent(
        perception,
        SafetyPlanner(),
        traces,
    )

    telemetry = Telemetry(
        battery_percent=2,
        altitude_m=10,
    )

    event, _ = agent.step(
        object(),
        telemetry,
        0,
    )

    assert event.decision.action == Action.EMERGENCY_RECOVERY
    assert event.decision.safety_level.value == "critical"
    assert event.decision.target_zone_id == "Z2-1"
    assert event.command["hardware_command_sent"] is False

    maneuvers = [
        step["maneuver"]
        for step in event.command["recovery_plan"]
    ]

    assert maneuvers == [
        "evade",
        "align_safe_zone",
        "descend",
        "land",
    ]

    assert len(traces.events) == 1


class SequencedPerception:
    def __init__(self) -> None:
        self.index = 0

    def observe(self, frame, frame_index, *, active_perception=False):
        obstacle = 0.91 if self.index == 0 else 0.05

        evidence = VisionEvidence(
            evidence_id=f"sequence-{self.index}",
            frame_index=frame_index,
            confidence=0.90,
            obstacle_risk=obstacle,
            motion_risk=0.02,
            candidates=(candidate(),),
        )

        self.index += 1
        return evidence, frame

    def enhance_for_active_perception(self, frame):
        return frame


def test_emergency_action_does_not_flap_after_one_safe_frame() -> None:
    perception = SequencedPerception()
    traces = MemoryTraceStore()

    agent = AegisLandAgent(
        perception,
        SafetyPlanner(),
        traces,
    )

    telemetry = Telemetry(
        battery_percent=2,
        altitude_m=10,
    )

    first, _ = agent.step(object(), telemetry, 0)
    second, _ = agent.step(object(), telemetry, 1)

    assert first.raw_decision is not None
    assert second.raw_decision is not None

    assert first.raw_decision.action == Action.EMERGENCY_RECOVERY
    assert first.decision.action == Action.EMERGENCY_RECOVERY

    assert second.raw_decision.action == Action.EMERGENCY_LAND

    # Stabilizer prevents one clear frame from immediately cancelling
    # the critical recovery state.
    assert second.decision.action == Action.EMERGENCY_RECOVERY


class ApprovalPerception:
    def observe(self, frame, frame_index, *, active_perception=False):
        evidence = VisionEvidence(
            evidence_id="approval-integration",
            frame_index=frame_index,
            confidence=0.90,
            obstacle_risk=0.05,
            motion_risk=0.02,
            candidates=(candidate(),),
        )
        return evidence, frame

    def enhance_for_active_perception(self, frame):
        return frame


def test_human_review_creates_pending_approval_request() -> None:
    traces = MemoryTraceStore()

    agent = AegisLandAgent(
        ApprovalPerception(),
        SafetyPlanner(),
        traces,
    )

    event, _ = agent.step(
        object(),
        Telemetry(
            battery_percent=11,
            altitude_m=10,
            gps_available=False,
            home_link_available=False,
        ),
        0,
    )

    assert event.decision.action == Action.REQUEST_HUMAN_APPROVAL
    assert event.command["approval_id"]
    assert event.command["approval_status"] == "pending"


def test_pending_human_approval_blocks_command_execution() -> None:
    traces = MemoryTraceStore()

    agent = AegisLandAgent(
        ApprovalPerception(),
        SafetyPlanner(),
        traces,
    )

    event, _ = agent.step(
        object(),
        Telemetry(
            battery_percent=11,
            altitude_m=10,
            gps_available=False,
            home_link_available=False,
        ),
        0,
    )

    assert event.decision.action == Action.REQUEST_HUMAN_APPROVAL
    assert event.command["approval_status"] == "pending"
    assert event.command["status"] == "awaiting_human_approval"
    assert event.command["hardware_command_sent"] is False
    assert event.command["command_status"] == "planned"


def test_approved_action_can_dispatch() -> None:
    traces = MemoryTraceStore()

    agent = AegisLandAgent(
        ApprovalPerception(),
        SafetyPlanner(),
        traces,
    )

    first, _ = agent.step(
        object(),
        Telemetry(
            battery_percent=11,
            altitude_m=10,
            gps_available=False,
            home_link_available=False,
        ),
        0,
    )

    approval_id = first.command["approval_id"]

    assert first.command["approval_status"] == "pending"
    assert first.command["command_status"] == "planned"

    assert agent.approve_action(approval_id)

    second, _ = agent.step(
        object(),
        Telemetry(
            battery_percent=11,
            altitude_m=10,
            gps_available=False,
            home_link_available=False,
        ),
        1,
    )

    assert second.command["approval_status"] == "approved"
    assert second.command["command_status"] == "completed"


def test_rejected_action_remains_blocked() -> None:
    traces = MemoryTraceStore()

    agent = AegisLandAgent(
        ApprovalPerception(),
        SafetyPlanner(),
        traces,
    )

    first, _ = agent.step(
        object(),
        Telemetry(
            battery_percent=11,
            altitude_m=10,
            gps_available=False,
            home_link_available=False,
        ),
        0,
    )

    approval_id = first.command["approval_id"]

    assert agent.reject_action(approval_id)

    second, _ = agent.step(
        object(),
        Telemetry(
            battery_percent=11,
            altitude_m=10,
            gps_available=False,
            home_link_available=False,
        ),
        1,
    )

    assert second.command["approval_status"] == "rejected"
    assert second.command["status"] == "approval_rejected"
    assert second.command["command_status"] == "planned"


def test_expired_action_remains_blocked() -> None:
    traces = MemoryTraceStore()

    agent = AegisLandAgent(
        ApprovalPerception(),
        SafetyPlanner(),
        traces,
    )

    first, _ = agent.step(
        object(),
        Telemetry(
            battery_percent=11,
            altitude_m=10,
            gps_available=False,
            home_link_available=False,
        ),
        0,
    )

    approval_id = first.command["approval_id"]

    assert agent.expire_action(approval_id)

    second, _ = agent.step(
        object(),
        Telemetry(
            battery_percent=11,
            altitude_m=10,
            gps_available=False,
            home_link_available=False,
        ),
        1,
    )

    assert second.command["approval_status"] == "expired"
    assert second.command["status"] == "approval_expired"
    assert second.command["command_status"] == "planned"


class ApproachingMotionPerception:
    def __init__(self) -> None:
        self.positions = iter(
            [
                (40.0, 100.0),
                (60.0, 100.0),
                (80.0, 100.0),
                (100.0, 100.0),
                (120.0, 100.0),
            ]
        )

    def observe(self, frame, frame_index, active_perception=False):
        from aegisland.domain import VisionEvidence, ZoneCandidate

        object_center = next(self.positions)

        zone = ZoneCandidate(
            zone_id="TARGET",
            bbox_xywh=(160, 80, 80, 40),
            score=0.9,
            edge_density=0.01,
            motion_occupancy=0.0,
            texture_risk=0.05,
            appearance_occupancy=0.0,
            clearance=0.9,
            safe=True,
        )

        evidence = VisionEvidence(
            evidence_id=f"approach-{frame_index}",
            frame_index=frame_index,
            confidence=0.95,
            obstacle_risk=0.0,
            motion_risk=0.1,
            candidates=(zone,),
            motion_object_center=object_center,
        )

        return evidence, frame

    def enhance_for_active_perception(self, frame):
        return frame


def test_agent_computes_temporal_risk_for_approaching_object() -> None:
    traces = MemoryTraceStore()

    agent = AegisLandAgent(
        ApproachingMotionPerception(),
        SafetyPlanner(),
        traces,
    )

    last_event = None

    for frame_index in range(5):
        last_event, _ = agent.step(
            object(),
            Telemetry(
                battery_percent=80,
                altitude_m=10,
                gps_available=True,
                home_link_available=True,
            ),
            frame_index,
        )

    assert last_event is not None
    assert last_event.evidence.temporal_risk > 0.0
    assert last_event.evidence.ttc_frames is not None
