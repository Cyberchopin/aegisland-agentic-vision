from aegisland.commands import CommandRuntime, CommandStatus
from aegisland.domain import Action, Decision, SafetyLevel


def decision() -> Decision:
    return Decision(
        action=Action.EMERGENCY_RECOVERY,
        safety_level=SafetyLevel.CRITICAL,
        risk_score=0.9,
        requires_human_approval=False,
        reasons=("compound emergency",),
        evidence_id="frame-42",
        target_zone_id="Z1-2",
    )


def test_command_lifecycle_reaches_completed() -> None:
    runtime = CommandRuntime()

    command = runtime.prepare(decision())
    assert command.status == CommandStatus.PLANNED

    command = runtime.dispatch(command)
    assert command.status == CommandStatus.DISPATCHED

    command = runtime.acknowledge(command)
    assert command.status == CommandStatus.ACKNOWLEDGED

    command = runtime.complete(command)
    assert command.status == CommandStatus.COMPLETED


def test_same_decision_generates_same_command_id() -> None:
    runtime = CommandRuntime()

    first = runtime.prepare(decision())
    second = runtime.prepare(decision())

    assert first.command_id == second.command_id


def test_command_can_fail_closed_on_timeout() -> None:
    runtime = CommandRuntime()

    command = runtime.prepare(decision())
    command = runtime.dispatch(command)
    command = runtime.timeout(command)

    assert command.status == CommandStatus.TIMEOUT
    assert command.error == "command acknowledgement timeout"
