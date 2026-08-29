from aegisland.agent import AegisLandAgent
from aegisland.domain import Action
from aegisland.perception import OpenCVLandingPerception
from aegisland.planner import SafetyPlanner
from aegisland.simulator import SCENARIOS, generate
from aegisland.trace import MemoryTraceStore


def test_critical_battery_collision_triggers_emergency_recovery() -> None:
    perception = OpenCVLandingPerception()
    traces = MemoryTraceStore()

    agent = AegisLandAgent(
        perception,
        SafetyPlanner(),
        traces,
    )

    scenario = SCENARIOS["critical_battery_collision"]

    actions = []

    for frame_index, (frame, telemetry) in enumerate(generate(scenario)):
        event, _ = agent.step(
            frame,
            telemetry,
            frame_index,
        )

        actions.append(event.decision.action)

    assert Action.EMERGENCY_RECOVERY in actions

    recovery_events = [
        event
        for event in traces.events
        if event.decision.action == Action.EMERGENCY_RECOVERY
    ]

    assert recovery_events

    command = recovery_events[0].command

    assert command["hardware_command_sent"] is False
    assert "recovery_plan" in command

    maneuvers = [
        step["maneuver"]
        for step in command["recovery_plan"]
    ]

    assert maneuvers == [
        "evade",
        "align_safe_zone",
        "descend",
        "land",
    ]
