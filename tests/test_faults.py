from aegisland.agent import AegisLandAgent
from aegisland.domain import (
    Action,
    Telemetry,
    VisionEvidence,
    ZoneCandidate,
)
from aegisland.faults import FaultInjectingCommandAdapter, FaultMode
from aegisland.planner import SafetyPlanner
from aegisland.trace import MemoryTraceStore


def _critical_candidate() -> ZoneCandidate:
    return ZoneCandidate(
        zone_id="Z2-1",
        bbox_xywh=(0, 0, 10, 10),
        score=0.8,
        edge_density=0.02,
        motion_occupancy=0.01,
        texture_risk=0.05,
        appearance_occupancy=0.02,
        clearance=0.9,
        safe=True,
    )


class CriticalRecoveryPerception:
    def observe(
        self,
        frame,
        frame_index,
        *,
        active_perception=False,
    ):
        evidence = VisionEvidence(
            evidence_id="critical-recovery",
            frame_index=frame_index,
            confidence=0.90,
            obstacle_risk=0.91,
            motion_risk=0.20,
            candidates=(_critical_candidate(),),
            active_perception_used=active_perception,
        )
        return evidence, frame

    def enhance_for_active_perception(self, frame):
        return frame



def make_agent(mode: FaultMode) -> AegisLandAgent:
    return AegisLandAgent(
        CriticalRecoveryPerception(),
        SafetyPlanner(),
        MemoryTraceStore(),
        command_adapter=FaultInjectingCommandAdapter(mode=mode),
    )


def test_ack_timeout_is_recorded() -> None:
    agent = make_agent(FaultMode.ACK_TIMEOUT)

    event, _ = agent.step(
        object(),
        Telemetry(
            battery_percent=2,
            altitude_m=10,
        ),
        0,
    )

    assert event.decision.action == Action.EMERGENCY_RECOVERY
    assert event.command["status"] == "timeout"
    assert event.command["command_status"] == "timeout"
    assert event.command["hardware_command_sent"] is False


def test_command_failure_is_recorded_separately() -> None:
    agent = make_agent(FaultMode.COMMAND_FAILURE)

    event, _ = agent.step(
        object(),
        Telemetry(
            battery_percent=2,
            altitude_m=10,
        ),
        0,
    )

    assert event.decision.action == Action.EMERGENCY_RECOVERY
    assert event.command["status"] == "failed"
    assert event.command["command_status"] == "failed"
    assert event.command["command_error"] == "simulated command failure"


def test_timeout_blocks_normal_autonomy_on_next_frame() -> None:
    traces = MemoryTraceStore()

    adapter = FaultInjectingCommandAdapter(
        mode=FaultMode.ACK_TIMEOUT,
    )

    agent = AegisLandAgent(
        CriticalRecoveryPerception(),
        SafetyPlanner(),
        traces,
        command_adapter=adapter,
    )

    first, _ = agent.step(
        object(),
        Telemetry(
            battery_percent=2,
            altitude_m=10,
        ),
        0,
    )

    assert first.command["execution_state"] == "fail_closed"

    # Restore the simulated command channel, but the execution guard
    # must remain latched until an explicit recovery/reset mechanism exists.
    adapter.mode = FaultMode.NONE

    second, _ = agent.step(
        object(),
        Telemetry(
            battery_percent=80,
            altitude_m=10,
        ),
        1,
    )

    assert second.decision.action == Action.REQUEST_HUMAN_APPROVAL
    assert second.decision.requires_human_approval
    assert second.command["execution_state"] == "fail_closed"


def test_operator_handshake_restores_autonomy_after_fail_closed() -> None:
    traces = MemoryTraceStore()

    adapter = FaultInjectingCommandAdapter(
        mode=FaultMode.ACK_TIMEOUT,
    )

    agent = AegisLandAgent(
        CriticalRecoveryPerception(),
        SafetyPlanner(),
        traces,
        command_adapter=adapter,
    )

    first, _ = agent.step(
        object(),
        Telemetry(
            battery_percent=2,
            altitude_m=10,
        ),
        0,
    )

    assert first.command["execution_state"] == "fail_closed"

    adapter.mode = FaultMode.NONE

    # Channel recovery alone must not restore autonomy.
    blocked, _ = agent.step(
        object(),
        Telemetry(
            battery_percent=80,
            altitude_m=10,
        ),
        1,
    )

    assert blocked.decision.action == Action.REQUEST_HUMAN_APPROVAL

    # Explicit operator + health handshake.
    recovered = agent.reset_execution_guard(
        operator_acknowledged=True,
        channel_healthy=True,
    )

    assert recovered

    third, _ = agent.step(
        object(),
        Telemetry(
            battery_percent=80,
            altitude_m=10,
        ),
        2,
    )

    assert third.command["execution_state"] == "healthy"
