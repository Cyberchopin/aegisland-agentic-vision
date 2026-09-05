from aegisland.domain import Action
from aegisland.px4.shadow_policy import ShadowAuthority
from aegisland.px4.supervisor import Px4ShadowSupervisor
from aegisland.px4.watchdog import LinkHealth
from aegisland.sensor_health import SensorHealthState


def establish_healthy(
    supervisor: Px4ShadowSupervisor,
) -> None:
    # Both telemetry and estimator need sustained recovery.
    for _ in range(3):
        decision = supervisor.step(
            telemetry_health=LinkHealth.HEALTHY,
            global_position_valid=True,
        )

    assert (
        decision.authority
        == ShadowAuthority.GRANTED
    )


def test_telemetry_degradation_revokes_immediately() -> None:
    supervisor = Px4ShadowSupervisor(
        telemetry_recovery_samples=3,
    )

    establish_healthy(supervisor)

    decision = supervisor.step(
        telemetry_health=LinkHealth.DEGRADED,
        global_position_valid=True,
    )

    assert (
        decision.telemetry_state
        == SensorHealthState.DEGRADED
    )
    assert (
        decision.authority
        == ShadowAuthority.REVOKED
    )
    assert decision.action == Action.HOLD_AND_SCAN


def test_failed_telemetry_revokes_immediately() -> None:
    supervisor = Px4ShadowSupervisor(
        telemetry_recovery_samples=3,
    )

    establish_healthy(supervisor)

    decision = supervisor.step(
        telemetry_health=LinkHealth.FAILED,
        global_position_valid=True,
    )

    assert (
        decision.telemetry_state
        == SensorHealthState.FAILED
    )
    assert (
        decision.authority
        == ShadowAuthority.REVOKED
    )


def test_telemetry_recovery_requires_three_samples() -> None:
    supervisor = Px4ShadowSupervisor(
        telemetry_recovery_samples=3,
    )

    establish_healthy(supervisor)

    # Hard link failure.
    supervisor.step(
        telemetry_health=LinkHealth.FAILED,
        global_position_valid=True,
    )

    first = supervisor.step(
        telemetry_health=LinkHealth.HEALTHY,
        global_position_valid=True,
    )

    second = supervisor.step(
        telemetry_health=LinkHealth.HEALTHY,
        global_position_valid=True,
    )

    third = supervisor.step(
        telemetry_health=LinkHealth.HEALTHY,
        global_position_valid=True,
    )

    assert (
        first.telemetry_state
        == SensorHealthState.DEGRADED
    )
    assert first.telemetry_recovery_streak == 1
    assert (
        first.authority
        == ShadowAuthority.REVOKED
    )

    assert (
        second.telemetry_state
        == SensorHealthState.DEGRADED
    )
    assert second.telemetry_recovery_streak == 2
    assert (
        second.authority
        == ShadowAuthority.REVOKED
    )

    assert (
        third.telemetry_state
        == SensorHealthState.HEALTHY
    )
    assert (
        third.authority
        == ShadowAuthority.GRANTED
    )


def test_cached_valid_position_cannot_override_stale_telemetry() -> None:
    supervisor = Px4ShadowSupervisor()

    establish_healthy(supervisor)

    decision = supervisor.step(
        telemetry_health=LinkHealth.FAILED,
        global_position_valid=True,
    )

    assert (
        decision.global_position_state
        == SensorHealthState.HEALTHY
    )
    assert (
        decision.telemetry_state
        == SensorHealthState.FAILED
    )
    assert (
        decision.authority
        == ShadowAuthority.REVOKED
    )
