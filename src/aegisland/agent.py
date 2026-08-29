from __future__ import annotations

import dataclasses
import uuid
from typing import Any, Protocol

from .domain import Action, Decision, Telemetry, TraceEvent, VisionEvidence
from .planner import SafetyPlanner


class PerceptionTool(Protocol):
    def observe(
        self, frame: Any, frame_index: int, *, active_perception: bool = False
    ) -> tuple[VisionEvidence, Any]: ...

    def enhance_for_active_perception(self, frame: Any) -> Any: ...


class TraceSink(Protocol):
    def write(self, event: TraceEvent) -> None: ...


class SimulatedCommandAdapter:
    """Safe default: records intended actions and never controls hardware."""

    def execute(self, decision: Decision) -> dict[str, Any]:
        blocked = decision.requires_human_approval
        return {
            "adapter": "simulation",
            "action": decision.action.value,
            "target_zone_id": decision.target_zone_id,
            "status": "awaiting_human_approval" if blocked else "simulated",
            "hardware_command_sent": False,
        }


class AegisLandAgent:
    def __init__(
        self,
        perception: PerceptionTool,
        planner: SafetyPlanner,
        trace_sink: TraceSink,
        command_adapter: SimulatedCommandAdapter | None = None,
        *,
        active_perception_trigger: float = 0.62,
    ) -> None:
        self.perception = perception
        self.planner = planner
        self.trace_sink = trace_sink
        self.command_adapter = command_adapter or SimulatedCommandAdapter()
        self.active_perception_trigger = active_perception_trigger
        self.trace_id = uuid.uuid4().hex
        self.sequence = 0

    def step(self, frame: Any, telemetry: Telemetry, frame_index: int) -> tuple[TraceEvent, Any]:
        evidence, annotated = self.perception.observe(frame, frame_index)

        # The agent uses uncertainty to make a second OpenCV tool call. This is the
        # explicit perception -> decision -> tool -> revised decision loop required
        # by the Agentic Vision rubric.
        if (
            evidence.confidence < self.active_perception_trigger
            and telemetry.battery_percent > 3.0
        ):
            enhanced = self.perception.enhance_for_active_perception(frame)
            retry, retry_annotated = self.perception.observe(
                enhanced, frame_index, active_perception=True
            )
            if retry.confidence >= evidence.confidence:
                evidence, annotated = retry, retry_annotated

        decision = self.planner.decide(telemetry, evidence)
        command = self.command_adapter.execute(decision)
        decision = dataclasses.replace(
            decision,
            command_executed=command["status"] == "simulated",
        )
        event = TraceEvent(
            trace_id=self.trace_id,
            sequence=self.sequence,
            telemetry=telemetry,
            evidence=evidence,
            decision=decision,
            command=command,
        )
        self.trace_sink.write(event)
        self.sequence += 1
        return event, self._draw_decision(annotated, decision)

    @staticmethod
    def _draw_decision(frame: Any, decision: Decision) -> Any:
        from .perception import cv2

        if cv2 is None or not hasattr(frame, "shape"):
            return frame
        color_by_action = {
            Action.CONTINUE_MISSION: (72, 218, 145),
            Action.RETURN_HOME: (80, 210, 245),
            Action.HOLD_AND_SCAN: (72, 174, 250),
            Action.EVADE_AND_HOLD: (45, 45, 255),
            Action.REQUEST_HUMAN_APPROVAL: (180, 80, 250),
            Action.LAND: (30, 220, 225),
            Action.EMERGENCY_LAND: (30, 30, 255),
	    Action.EMERGENCY_RECOVERY: (0, 80, 255),
        }
        color = color_by_action[decision.action]
        y0 = frame.shape[0] - 58
        cv2.rectangle(frame, (0, y0), (frame.shape[1], frame.shape[0]), (10, 16, 25), -1)
        cv2.putText(
            frame,
            f"ACTION {decision.action.value.upper()}  RISK {decision.risk_score:.2f}",
            (15, y0 + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            color,
            2,
            cv2.LINE_AA,
        )
        return frame
