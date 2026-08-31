from __future__ import annotations

from dataclasses import dataclass

from .domain import Action, Decision, SafetyLevel

ACTION_PRIORITY = {
    Action.CONTINUE_MISSION: 0,
    Action.RETURN_HOME: 1,
    Action.HOLD_AND_SCAN: 2,
    Action.REQUEST_HUMAN_APPROVAL: 3,
    Action.LAND: 4,
    Action.EVADE_AND_HOLD: 5,
    Action.EMERGENCY_LAND: 6,
    Action.EMERGENCY_RECOVERY: 7,
}


@dataclass(slots=True)
class StabilizerState:
    latched_action: Action | None = None
    remaining_frames: int = 0


class ActionStabilizer:
    """Prevent rapid action oscillation while allowing immediate safety escalation."""

    def __init__(self, *, latch_frames: int = 3) -> None:
        self.latch_frames = latch_frames
        self.state = StabilizerState()

    def stabilize(self, decision: Decision) -> Decision:
        current = self.state.latched_action

        if current is None:
            self._latch(decision.action)
            return decision

        incoming_priority = ACTION_PRIORITY[decision.action]
        current_priority = ACTION_PRIORITY[current]

        # Safety escalation must never be delayed.
        if incoming_priority > current_priority:
            self._latch(decision.action)
            return decision

        # Same action refreshes the latch.
        if decision.action == current:
            self._latch(current)
            return decision

        # Prevent immediate de-escalation caused by one noisy frame.
        if self.state.remaining_frames > 0:
            self.state.remaining_frames -= 1

            return Decision(
                action=current,
                safety_level=max(
                    decision.safety_level,
                    SafetyLevel.HIGH,
                    key=lambda level: {
                        SafetyLevel.NOMINAL: 0,
                        SafetyLevel.CAUTION: 1,
                        SafetyLevel.HIGH: 2,
                        SafetyLevel.CRITICAL: 3,
                    }[level],
                ),
                risk_score=decision.risk_score,
                requires_human_approval=decision.requires_human_approval,
                reasons=(
                    *decision.reasons,
                    f"Action {current.value} remains latched to prevent rapid control oscillation.",
                ),
                evidence_id=decision.evidence_id,
                target_zone_id=decision.target_zone_id,
            )

        self._latch(decision.action)
        return decision

    def _latch(self, action: Action) -> None:
        self.state.latched_action = action
        self.state.remaining_frames = self.latch_frames
