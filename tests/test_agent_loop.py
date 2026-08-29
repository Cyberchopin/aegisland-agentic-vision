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
    assert len(traces.events) == 1
