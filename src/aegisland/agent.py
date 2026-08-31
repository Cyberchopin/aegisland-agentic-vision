from __future__ import annotations

import dataclasses
import uuid
from typing import Any, Protocol

from .approval import ApprovalManager
from .commands import CommandRuntime
from .domain import Action, Decision, Telemetry, TraceEvent, VisionEvidence
from .execution import ExecutionSafetyGuard
from .fusion import DynamicConfidenceFusion
from .planner import SafetyPlanner
from .recovery import RecoveryPlanner
from .sensor_health import SensorHealthMonitor, SensorHealthState
from .sensor_sync import SensorSynchronizer
from .stability import ActionStabilizer
from .targeting import LandingTargetManager
from .temporal import TemporalRiskAssessor


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
        recovery_planner: RecoveryPlanner | None = None,
        target_manager: LandingTargetManager | None = None,
        action_stabilizer: ActionStabilizer | None = None,
        command_runtime: CommandRuntime | None = None,
        execution_guard: ExecutionSafetyGuard | None = None,
        approval_manager: ApprovalManager | None = None,
        temporal_risk_assessor: TemporalRiskAssessor | None = None,
        confidence_fusion: DynamicConfidenceFusion | None = None,
        sensor_synchronizer: SensorSynchronizer | None = None,
        sensor_health_monitor: SensorHealthMonitor | None = None,
        *,
        active_perception_trigger: float = 0.62,
    ) -> None:
        self.perception = perception
        self.planner = planner
        self.trace_sink = trace_sink
        self.command_adapter = command_adapter or SimulatedCommandAdapter()
        self.recovery_planner = recovery_planner or RecoveryPlanner()
        self.target_manager = target_manager or LandingTargetManager()
        self.action_stabilizer = action_stabilizer or ActionStabilizer()
        self.command_runtime = command_runtime or CommandRuntime()
        self.execution_guard = execution_guard or ExecutionSafetyGuard()
        self.approval_manager = approval_manager or ApprovalManager()
        self.temporal_risk_assessor = (
            temporal_risk_assessor or TemporalRiskAssessor()
        )
        self.confidence_fusion = (
            confidence_fusion or DynamicConfidenceFusion()
        )
        self.sensor_synchronizer = (
            sensor_synchronizer or SensorSynchronizer()
        )
        self.sensor_health_monitor = (
            sensor_health_monitor or SensorHealthMonitor()
        )
        self.active_perception_trigger = active_perception_trigger
        self.trace_id = uuid.uuid4().hex
        self.sequence = 0

    def ingest_imu(
        self,
        *,
        timestamp_s: float,
        yaw_rad: float,
        yaw_rate_rad_s: float,
        ax: float,
        ay: float,
    ) -> None:
        self.sensor_synchronizer.add_imu(
            timestamp_s=timestamp_s,
            yaw_rad=yaw_rad,
            yaw_rate_rad_s=yaw_rate_rad_s,
            ax=ax,
            ay=ay,
        )

    def approve_action(self, approval_id: str) -> bool:
        request = self.approval_manager.get(approval_id)

        if request is None:
            return False

        self.approval_manager.approve(request)
        return True

    def reject_action(self, approval_id: str) -> bool:
        request = self.approval_manager.get(approval_id)

        if request is None:
            return False

        self.approval_manager.reject(request)
        return True

    def expire_action(self, approval_id: str) -> bool:
        request = self.approval_manager.get(approval_id)

        if request is None:
            return False

        self.approval_manager.expire(request)
        return True

    def reset_execution_guard(
        self,
        *,
        operator_acknowledged: bool,
        channel_healthy: bool,
    ) -> bool:
        return self.execution_guard.try_reset(
            operator_acknowledged=operator_acknowledged,
            channel_healthy=channel_healthy,
        )

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

        sensor_timestamp_s = (
            telemetry.timestamp_s
            if telemetry.timestamp_s > 0.0
            else frame_index / 30.0
        )

        if evidence.visual_localization_valid:
            self.sensor_synchronizer.add_visual(
                timestamp_s=sensor_timestamp_s,
                x=evidence.visual_relative_x,
                y=evidence.visual_relative_y,
                confidence=evidence.visual_localization_confidence,
            )

        synchronized = self.sensor_synchronizer.snapshot(
            sensor_timestamp_s
        )

        if synchronized.imu_valid:
            if synchronized.imu_sync_method.value == "exact":
                imu_confidence = 0.95
            elif synchronized.imu_sync_method.value == "interpolated":
                imu_confidence = 0.90
            elif synchronized.imu_sync_method.value == "extrapolated":
                imu_confidence = 0.75
            else:
                imu_confidence = 0.0
        else:
            imu_confidence = 0.0

        gps_health = self.sensor_health_monitor.assess(
            "gps",
            confidence=(
                1.0 if telemetry.gps_available else 0.0
            ),
            valid=telemetry.gps_available,
        )

        visual_health = self.sensor_health_monitor.assess(
            "visual",
            confidence=evidence.visual_localization_confidence,
            valid=evidence.visual_localization_valid,
        )

        imu_health = self.sensor_health_monitor.assess(
            "imu",
            confidence=imu_confidence,
            valid=synchronized.imu_valid,
        )

        fusion = self.confidence_fusion.fuse(
            gps_confidence=gps_health.effective_confidence,
            visual_confidence=visual_health.effective_confidence,
            visual_valid=(
                visual_health.state
                != SensorHealthState.FAILED
            ),
            imu_confidence=imu_health.effective_confidence,
            imu_valid=(
                imu_health.state
                != SensorHealthState.FAILED
            ),
        )

        evidence = dataclasses.replace(
            evidence,
            navigation_mode=fusion.mode.value,
            fused_navigation_confidence=(
                fusion.fused_confidence
            ),
            navigation_gps_weight=fusion.gps_weight,
            navigation_visual_weight=fusion.visual_weight,
            navigation_imu_weight=fusion.imu_weight,
            healthy_navigation_sources=(
                fusion.healthy_sources
            ),
            degraded_navigation_sources=(
                fusion.degraded_sources
            ),
            imu_sync_valid=synchronized.imu_valid,
            imu_sync_method=synchronized.imu_sync_method.value,
            imu_confidence=imu_confidence,

            gps_health_state=gps_health.state.value,
            visual_health_state=visual_health.state.value,
            imu_health_state=imu_health.state.value,

            gps_effective_confidence=(
                gps_health.effective_confidence
            ),
            visual_effective_confidence=(
                visual_health.effective_confidence
            ),
            imu_effective_confidence=(
                imu_health.effective_confidence
            ),
        )

        target = self.target_manager.select(evidence)

        if target is not None:
            target_x, target_y, target_w, target_h = target.bbox_xywh

            target_center = (
                target_x + target_w / 2.0,
                target_y + target_h / 2.0,
            )

            temporal = self.temporal_risk_assessor.observe(
                object_center=evidence.motion_object_center,
                target_center=target_center,
            )

            ttc_frames = None

            if temporal.ttc is not None:
                candidate_ttc = temporal.ttc.ttc_frames

                if candidate_ttc != float("inf"):
                    ttc_frames = round(candidate_ttc, 3)

            evidence = dataclasses.replace(
                evidence,
                temporal_risk=round(temporal.risk, 4),
                ttc_frames=ttc_frames,
            )

        raw_decision = self.planner.decide(
            telemetry,
            evidence,
            target_zone=target,
        )

        decision = self.action_stabilizer.stabilize(raw_decision)
        decision = self.execution_guard.guard_decision(decision)

        approval = self.approval_manager.request(decision)

        if approval is not None and approval.status.value == "requested":
            approval = self.approval_manager.mark_pending(approval)

        recovery_plan = self.recovery_planner.plan(
            decision,
            telemetry,
            evidence,
        )

        command_envelope = self.command_runtime.prepare(decision)

        approval_status = approval.status.value if approval is not None else None

        if approval_status in {"pending", "requested"}:
            command = {
                "adapter": "simulation",
                "action": decision.action.value,
                "target_zone_id": decision.target_zone_id,
                "status": "awaiting_human_approval",
                "hardware_command_sent": False,
            }

        elif approval_status in {"rejected", "expired"}:
            command = {
                "adapter": "simulation",
                "action": decision.action.value,
                "target_zone_id": decision.target_zone_id,
                "status": f"approval_{approval_status}",
                "hardware_command_sent": False,
            }

        else:
            command_envelope = self.command_runtime.dispatch(command_envelope)

            execution_decision = (
                dataclasses.replace(
                    decision,
                    requires_human_approval=False,
                )
                if approval_status == "approved"
                else decision
            )

            command = self.command_adapter.execute(execution_decision)

        adapter_status = command.get("status")

        if adapter_status in {
            "awaiting_human_approval",
            "approval_rejected",
            "approval_expired",
        }:
            pass
        elif adapter_status == "timeout":
            command_envelope = self.command_runtime.timeout(command_envelope)
        elif adapter_status == "failed":
            command_envelope = self.command_runtime.fail(
                command_envelope,
                "simulated command failure",
            )
        else:
            command_envelope = self.command_runtime.acknowledge(command_envelope)
            command_envelope = self.command_runtime.complete(command_envelope)

        command["command_id"] = command_envelope.command_id
        command["command_status"] = command_envelope.status.value

        if approval is not None:
            command["approval_id"] = approval.approval_id
            command["approval_status"] = approval.status.value

        execution = self.execution_guard.observe_command_status(
            command_envelope.status.value
        )

        command["execution_state"] = execution.state.value
        command["requires_human_attention"] = execution.requires_human_attention
        command["execution_reason"] = execution.reason

        if command_envelope.error is not None:
            command["command_error"] = command_envelope.error

        if recovery_plan is not None:
            command["recovery_plan"] = [
                {
                    "maneuver": step.maneuver.value,
                    "reason": step.reason,
                    "target_zone_id": step.target_zone_id,
                }
                for step in recovery_plan.steps
            ]
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
            raw_decision=raw_decision,
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
