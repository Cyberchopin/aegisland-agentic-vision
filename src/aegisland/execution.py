from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .domain import Action, Decision, SafetyLevel


class ExecutionState(StrEnum):
    HEALTHY = "healthy"
    FAIL_CLOSED = "fail_closed"


@dataclass(frozen=True, slots=True)
class ExecutionAssessment:
    state: ExecutionState
    requires_human_attention: bool
    reason: str


class ExecutionSafetyGuard:
    """Latch execution failures and block normal mission continuation."""

    def __init__(self) -> None:
        self.state = ExecutionState.HEALTHY

    def observe_command_status(self, status: str) -> ExecutionAssessment:
        if status in {"timeout", "failed"}:
            self.state = ExecutionState.FAIL_CLOSED

            return ExecutionAssessment(
                state=self.state,
                requires_human_attention=True,
                reason=(
                    "Command execution could not be verified; "
                    "the autonomy runtime entered fail-closed mode."
                ),
            )

        return ExecutionAssessment(
            state=self.state,
            requires_human_attention=self.state == ExecutionState.FAIL_CLOSED,
            reason=(
                "Execution channel is healthy."
                if self.state == ExecutionState.HEALTHY
                else "Execution guard remains latched fail-closed."
            ),
        )

    def try_reset(
        self,
        *,
        operator_acknowledged: bool,
        channel_healthy: bool,
    ) -> bool:
        """Recover autonomy only after both human acknowledgement and health verification."""

        if self.state != ExecutionState.FAIL_CLOSED:
            return True

        if not operator_acknowledged:
            return False

        if not channel_healthy:
            return False

        self.state = ExecutionState.HEALTHY
        return True

    def guard_decision(self, decision: Decision) -> Decision:
        if self.state != ExecutionState.FAIL_CLOSED:
            return decision

        return Decision(
            action=Action.REQUEST_HUMAN_APPROVAL,
            safety_level=SafetyLevel.CRITICAL,
            risk_score=max(decision.risk_score, 0.95),
            requires_human_approval=True,
            reasons=(
                *decision.reasons,
                "Normal autonomous action is blocked because command execution is unverified.",
            ),
            evidence_id=decision.evidence_id,
            target_zone_id=decision.target_zone_id,
        )
