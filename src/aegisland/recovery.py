from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .domain import Action, Decision, Telemetry, VisionEvidence


class Maneuver(StrEnum):
    BRAKE = "brake"
    EVADE = "evade"
    ALIGN_SAFE_ZONE = "align_safe_zone"
    DESCEND = "descend"
    LAND = "land"


@dataclass(frozen=True, slots=True)
class RecoveryStep:
    maneuver: Maneuver
    reason: str
    target_zone_id: str | None = None


@dataclass(frozen=True, slots=True)
class RecoveryPlan:
    steps: tuple[RecoveryStep, ...]
    emergency: bool = True

    @property
    def first_step(self) -> RecoveryStep | None:
        return self.steps[0] if self.steps else None


class RecoveryPlanner:
    """Translate a high-level emergency decision into a safe simulated maneuver plan."""

    def plan(
        self,
        decision: Decision,
        telemetry: Telemetry,
        evidence: VisionEvidence,
    ) -> RecoveryPlan | None:
        if decision.action != Action.EMERGENCY_RECOVERY:
            return None

        steps: list[RecoveryStep] = []

        if telemetry.horizontal_speed_mps > 0.5:
            steps.append(
                RecoveryStep(
                    maneuver=Maneuver.BRAKE,
                    reason="Reduce horizontal kinetic energy before evasive motion.",
                )
            )

        steps.append(
            RecoveryStep(
                maneuver=Maneuver.EVADE,
                reason="Create immediate separation from the detected collision hazard.",
            )
        )

        if decision.target_zone_id is not None:
            steps.append(
                RecoveryStep(
                    maneuver=Maneuver.ALIGN_SAFE_ZONE,
                    reason="Re-orient toward the best currently observed landing zone.",
                    target_zone_id=decision.target_zone_id,
                )
            )

        steps.append(
            RecoveryStep(
                maneuver=Maneuver.DESCEND,
                reason="Critical battery requires immediate loss of altitude.",
                target_zone_id=decision.target_zone_id,
            )
        )

        steps.append(
            RecoveryStep(
                maneuver=Maneuver.LAND,
                reason="Terminate flight as soon as the contingency zone is reachable.",
                target_zone_id=decision.target_zone_id,
            )
        )

        return RecoveryPlan(steps=tuple(steps))
