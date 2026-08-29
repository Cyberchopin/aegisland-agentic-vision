from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .domain import Decision


class FaultMode(StrEnum):
    NONE = "none"
    ACK_TIMEOUT = "ack_timeout"
    COMMAND_FAILURE = "command_failure"


@dataclass(slots=True)
class FaultInjectingCommandAdapter:
    mode: FaultMode = FaultMode.NONE

    def execute(self, decision: Decision) -> dict[str, object]:
        if self.mode == FaultMode.ACK_TIMEOUT:
            return {
                "adapter": "simulation",
                "action": decision.action.value,
                "target_zone_id": decision.target_zone_id,
                "status": "timeout",
                "hardware_command_sent": False,
            }

        if self.mode == FaultMode.COMMAND_FAILURE:
            return {
                "adapter": "simulation",
                "action": decision.action.value,
                "target_zone_id": decision.target_zone_id,
                "status": "failed",
                "hardware_command_sent": False,
            }

        return {
            "adapter": "simulation",
            "action": decision.action.value,
            "target_zone_id": decision.target_zone_id,
            "status": "simulated",
            "hardware_command_sent": False,
        }
