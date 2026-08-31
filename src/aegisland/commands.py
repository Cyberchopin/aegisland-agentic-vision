from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

from .domain import Decision


class CommandStatus(StrEnum):
    PLANNED = "planned"
    DISPATCHED = "dispatched"
    ACKNOWLEDGED = "acknowledged"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CommandEnvelope:
    command_id: str
    action: str
    target_zone_id: str | None
    status: CommandStatus
    attempt: int = 1
    error: str | None = None


class CommandRuntime:
    """Simulation-safe command lifecycle with idempotency and timeout handling."""

    def __init__(self, *, timeout_s: float = 1.0) -> None:
        self.timeout_s = timeout_s
        self._seen: dict[str, CommandEnvelope] = {}

    def prepare(self, decision: Decision) -> CommandEnvelope:
        key = (
            f"{decision.evidence_id}:"
            f"{decision.action.value}:"
            f"{decision.target_zone_id}"
        )

        command_id = hashlib.sha1(key.encode()).hexdigest()[:12]

        existing = self._seen.get(command_id)
        if existing is not None:
            return existing

        command = CommandEnvelope(
            command_id=command_id,
            action=decision.action.value,
            target_zone_id=decision.target_zone_id,
            status=CommandStatus.PLANNED,
        )

        self._seen[command_id] = command
        return command

    def dispatch(self, command: CommandEnvelope) -> CommandEnvelope:
        updated = CommandEnvelope(
            command_id=command.command_id,
            action=command.action,
            target_zone_id=command.target_zone_id,
            status=CommandStatus.DISPATCHED,
            attempt=command.attempt,
        )

        self._seen[command.command_id] = updated
        return updated

    def acknowledge(self, command: CommandEnvelope) -> CommandEnvelope:
        updated = CommandEnvelope(
            command_id=command.command_id,
            action=command.action,
            target_zone_id=command.target_zone_id,
            status=CommandStatus.ACKNOWLEDGED,
            attempt=command.attempt,
        )

        self._seen[command.command_id] = updated
        return updated

    def complete(self, command: CommandEnvelope) -> CommandEnvelope:
        updated = CommandEnvelope(
            command_id=command.command_id,
            action=command.action,
            target_zone_id=command.target_zone_id,
            status=CommandStatus.COMPLETED,
            attempt=command.attempt,
        )

        self._seen[command.command_id] = updated
        return updated

    def fail(self, command: CommandEnvelope, error: str) -> CommandEnvelope:
        updated = CommandEnvelope(
            command_id=command.command_id,
            action=command.action,
            target_zone_id=command.target_zone_id,
            status=CommandStatus.FAILED,
            attempt=command.attempt,
            error=error,
        )

        self._seen[command.command_id] = updated
        return updated

    def timeout(self, command: CommandEnvelope) -> CommandEnvelope:
        updated = CommandEnvelope(
            command_id=command.command_id,
            action=command.action,
            target_zone_id=command.target_zone_id,
            status=CommandStatus.TIMEOUT,
            attempt=command.attempt,
            error="command acknowledgement timeout",
        )

        self._seen[command.command_id] = updated
        return updated
