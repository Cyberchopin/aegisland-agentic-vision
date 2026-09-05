from aegisland.domain import Action
from aegisland.px4.shadow_policy import (
    ShadowAuthority,
    ShadowNavigationMode,
    decide_shadow_action,
)
from aegisland.px4.watchdog import LinkHealth


def test_healthy_link_and_gps_continue() -> None:
    decision = decide_shadow_action(
        link_health=LinkHealth.HEALTHY,
        gps_available=True,
    )

    assert (
        decision.navigation_mode
        == ShadowNavigationMode.GPS_PRIMARY
    )
    assert (
        decision.authority
        == ShadowAuthority.GRANTED
    )
    assert (
        decision.action
        == Action.CONTINUE_MISSION
    )
    assert not decision.executed


def test_gps_loss_revokes_authority() -> None:
    decision = decide_shadow_action(
        link_health=LinkHealth.HEALTHY,
        gps_available=False,
    )

    assert (
        decision.navigation_mode
        == ShadowNavigationMode.DEGRADED
    )
    assert (
        decision.authority
        == ShadowAuthority.REVOKED
    )
    assert (
        decision.action
        == Action.HOLD_AND_SCAN
    )
    assert not decision.executed


def test_failed_link_holds() -> None:
    decision = decide_shadow_action(
        link_health=LinkHealth.FAILED,
        gps_available=True,
    )

    assert (
        decision.authority
        == ShadowAuthority.REVOKED
    )
    assert (
        decision.action
        == Action.HOLD_AND_SCAN
    )
