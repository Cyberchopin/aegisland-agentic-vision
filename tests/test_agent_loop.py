from aegisland.agent import AegisLandAgent
from aegisland.domain import Action, SafetyLevel, Telemetry, VisionEvidence, ZoneCandidate
from aegisland.planner import SafetyPlanner
from aegisland.trace import MemoryTraceStore


def candidate() -> ZoneCandidate:
    return ZoneCandidate(
        "Z2-1", (0, 0, 10, 10), 0.8, 0.02, 0.01, 0.05, 0.02, 0.9, True
    )


class FakePerception:
    def __init__(self) -> None:
        self.calls: list[bool] = []

    def observe(self, frame, frame_index, *, active_perception=False):
        self.calls.append(active_perception)
        confidence = 0.75 if active_perception else 0.40
        evidence = VisionEvidence(
            evidence_id="retry" if active_perception else "first",
            frame_index=frame_index,
            confidence=confidence,
            obstacle_risk=0.05,
            motion_risk=0.02,
            candidates=(candidate(),),
            active_perception_used=active_perception,
        )
        return evidence, frame

    def enhance_for_active_perception(self, frame):
        return frame


def test_uncertainty_triggers_a_second_opencv_tool_call() -> None:
    perception = FakePerception()
    traces = MemoryTraceStore()
    agent = AegisLandAgent(perception, SafetyPlanner(), traces)
    telemetry = Telemetry(25, 10)

    event, _ = agent.step(object(), telemetry, 0)

    assert perception.calls == [False, True]
    assert event.evidence.evidence_id == "retry"
    assert event.evidence.active_perception_used
    assert len(traces.events) == 1
    assert traces.events[0].command["hardware_command_sent"] is False


class CriticalRecoveryPerception:
    def observe(self, frame, frame_index, *, active_perception=False):
        evidence = VisionEvidence(
            evidence_id="critical-recovery",
            frame_index=frame_index,
            confidence=0.90,
            obstacle_risk=0.91,
            motion_risk=0.20,
            candidates=(candidate(),),
            active_perception_used=False,
        )
        return evidence, frame

    def enhance_for_active_perception(self, frame):
        return frame


def test_critical_battery_and_collision_flows_through_agent() -> None:
    perception = CriticalRecoveryPerception()
    traces = MemoryTraceStore()

    agent = AegisLandAgent(
        perception,
        SafetyPlanner(),
        traces,
    )

    telemetry = Telemetry(
        battery_percent=2,
        altitude_m=10,
    )

    event, _ = agent.step(
        object(),
        telemetry,
        0,
    )

    assert event.decision.action == Action.EMERGENCY_RECOVERY
    assert event.decision.safety_level.value == "critical"
    assert event.decision.target_zone_id == "Z2-1"
    assert event.command["hardware_command_sent"] is False

    maneuvers = [
        step["maneuver"]
        for step in event.command["recovery_plan"]
    ]

    assert maneuvers == [
        "evade",
        "align_safe_zone",
        "descend",
        "land",
    ]

    assert len(traces.events) == 1


class SequencedPerception:
    def __init__(self) -> None:
        self.index = 0

    def observe(self, frame, frame_index, *, active_perception=False):
        obstacle = 0.91 if self.index == 0 else 0.05

        evidence = VisionEvidence(
            evidence_id=f"sequence-{self.index}",
            frame_index=frame_index,
            confidence=0.90,
            obstacle_risk=obstacle,
            motion_risk=0.02,
            candidates=(candidate(),),
        )

        self.index += 1
        return evidence, frame

    def enhance_for_active_perception(self, frame):
        return frame


def test_emergency_action_does_not_flap_after_one_safe_frame() -> None:
    perception = SequencedPerception()
    traces = MemoryTraceStore()

    agent = AegisLandAgent(
        perception,
        SafetyPlanner(),
        traces,
    )

    telemetry = Telemetry(
        battery_percent=2,
        altitude_m=10,
    )

    first, _ = agent.step(object(), telemetry, 0)
    second, _ = agent.step(object(), telemetry, 1)

    assert first.raw_decision is not None
    assert second.raw_decision is not None

    assert first.raw_decision.action == Action.EMERGENCY_RECOVERY
    assert first.decision.action == Action.EMERGENCY_RECOVERY

    assert second.raw_decision.action == Action.EMERGENCY_LAND

    # Stabilizer prevents one clear frame from immediately cancelling
    # the critical recovery state.
    assert second.decision.action == Action.EMERGENCY_RECOVERY


class ApprovalPerception:
    def observe(self, frame, frame_index, *, active_perception=False):
        evidence = VisionEvidence(
            evidence_id="approval-integration",
            frame_index=frame_index,
            confidence=0.90,
            obstacle_risk=0.05,
            motion_risk=0.02,
            candidates=(candidate(),),
        )
        return evidence, frame

    def enhance_for_active_perception(self, frame):
        return frame


def test_human_review_creates_pending_approval_request() -> None:
    traces = MemoryTraceStore()

    agent = AegisLandAgent(
        ApprovalPerception(),
        SafetyPlanner(),
        traces,
    )

    event, _ = agent.step(
        object(),
        Telemetry(
            battery_percent=11,
            altitude_m=10,
            gps_available=False,
            home_link_available=False,
        ),
        0,
    )

    assert event.decision.action == Action.REQUEST_HUMAN_APPROVAL
    assert event.command["approval_id"]
    assert event.command["approval_status"] == "pending"


def test_pending_human_approval_blocks_command_execution() -> None:
    traces = MemoryTraceStore()

    agent = AegisLandAgent(
        ApprovalPerception(),
        SafetyPlanner(),
        traces,
    )

    event, _ = agent.step(
        object(),
        Telemetry(
            battery_percent=11,
            altitude_m=10,
            gps_available=False,
            home_link_available=False,
        ),
        0,
    )

    assert event.decision.action == Action.REQUEST_HUMAN_APPROVAL
    assert event.command["approval_status"] == "pending"
    assert event.command["status"] == "awaiting_human_approval"
    assert event.command["hardware_command_sent"] is False
    assert event.command["command_status"] == "planned"


def test_approved_action_can_dispatch() -> None:
    traces = MemoryTraceStore()

    agent = AegisLandAgent(
        ApprovalPerception(),
        SafetyPlanner(),
        traces,
    )

    first, _ = agent.step(
        object(),
        Telemetry(
            battery_percent=11,
            altitude_m=10,
            gps_available=False,
            home_link_available=False,
        ),
        0,
    )

    approval_id = first.command["approval_id"]

    assert first.command["approval_status"] == "pending"
    assert first.command["command_status"] == "planned"

    assert agent.approve_action(approval_id)

    second, _ = agent.step(
        object(),
        Telemetry(
            battery_percent=11,
            altitude_m=10,
            gps_available=False,
            home_link_available=False,
        ),
        1,
    )

    assert second.command["approval_status"] == "approved"
    assert second.command["command_status"] == "completed"


def test_rejected_action_remains_blocked() -> None:
    traces = MemoryTraceStore()

    agent = AegisLandAgent(
        ApprovalPerception(),
        SafetyPlanner(),
        traces,
    )

    first, _ = agent.step(
        object(),
        Telemetry(
            battery_percent=11,
            altitude_m=10,
            gps_available=False,
            home_link_available=False,
        ),
        0,
    )

    approval_id = first.command["approval_id"]

    assert agent.reject_action(approval_id)

    second, _ = agent.step(
        object(),
        Telemetry(
            battery_percent=11,
            altitude_m=10,
            gps_available=False,
            home_link_available=False,
        ),
        1,
    )

    assert second.command["approval_status"] == "rejected"
    assert second.command["status"] == "approval_rejected"
    assert second.command["command_status"] == "planned"


def test_expired_action_remains_blocked() -> None:
    traces = MemoryTraceStore()

    agent = AegisLandAgent(
        ApprovalPerception(),
        SafetyPlanner(),
        traces,
    )

    first, _ = agent.step(
        object(),
        Telemetry(
            battery_percent=11,
            altitude_m=10,
            gps_available=False,
            home_link_available=False,
        ),
        0,
    )

    approval_id = first.command["approval_id"]

    assert agent.expire_action(approval_id)

    second, _ = agent.step(
        object(),
        Telemetry(
            battery_percent=11,
            altitude_m=10,
            gps_available=False,
            home_link_available=False,
        ),
        1,
    )

    assert second.command["approval_status"] == "expired"
    assert second.command["status"] == "approval_expired"
    assert second.command["command_status"] == "planned"


class ApproachingMotionPerception:
    def __init__(self) -> None:
        self.positions = iter(
            [
                (40.0, 100.0),
                (60.0, 100.0),
                (80.0, 100.0),
                (100.0, 100.0),
                (120.0, 100.0),
            ]
        )

    def observe(self, frame, frame_index, active_perception=False):
        from aegisland.domain import VisionEvidence, ZoneCandidate

        object_center = next(self.positions)

        zone = ZoneCandidate(
            zone_id="TARGET",
            bbox_xywh=(160, 80, 80, 40),
            score=0.9,
            edge_density=0.01,
            motion_occupancy=0.0,
            texture_risk=0.05,
            appearance_occupancy=0.0,
            clearance=0.9,
            safe=True,
        )

        evidence = VisionEvidence(
            evidence_id=f"approach-{frame_index}",
            frame_index=frame_index,
            confidence=0.95,
            obstacle_risk=0.0,
            motion_risk=0.1,
            candidates=(zone,),
            motion_object_center=object_center,
        )

        return evidence, frame

    def enhance_for_active_perception(self, frame):
        return frame


def test_agent_computes_temporal_risk_for_approaching_object() -> None:
    traces = MemoryTraceStore()

    agent = AegisLandAgent(
        ApproachingMotionPerception(),
        SafetyPlanner(),
        traces,
    )

    last_event = None

    for frame_index in range(5):
        last_event, _ = agent.step(
            object(),
            Telemetry(
                battery_percent=80,
                altitude_m=10,
                gps_available=True,
                home_link_available=True,
            ),
            frame_index,
        )

    assert last_event is not None
    assert last_event.evidence.temporal_risk > 0.0
    assert last_event.evidence.ttc_frames is not None


class VisualFallbackPerception:
    def observe(
        self,
        frame,
        frame_index,
        active_perception=False,
    ):
        from aegisland.domain import VisionEvidence

        evidence = VisionEvidence(
            evidence_id=f"visual-fallback-{frame_index}",
            frame_index=frame_index,
            confidence=0.95,
            obstacle_risk=0.0,
            motion_risk=0.0,
            visual_localization_valid=True,
            visual_localization_confidence=0.82,
        )

        return evidence, frame

    def enhance_for_active_perception(self, frame):
        return frame


def test_agent_uses_visual_navigation_when_gps_is_lost() -> None:
    traces = MemoryTraceStore()

    agent = AegisLandAgent(
        VisualFallbackPerception(),
        SafetyPlanner(),
        traces,
    )

    event, _ = agent.step(
        object(),
        Telemetry(
            battery_percent=80,
            altitude_m=10,
            gps_available=False,
            home_link_available=True,
        ),
        0,
    )

    assert event.evidence.navigation_mode == "visual_fallback"

    # The visual sensor is still in health-monitor recovery on the
    # first observation, so fusion consumes degraded effective
    # confidence rather than the raw 0.82 measurement.
    assert event.evidence.visual_localization_confidence == 0.82
    assert event.evidence.visual_health_state == "degraded"
    assert event.evidence.visual_effective_confidence == 0.574
    assert (
        event.evidence.fused_navigation_confidence
        == event.evidence.visual_effective_confidence
    )

    assert "visual" in event.evidence.healthy_navigation_sources
    assert "gps" in event.evidence.degraded_navigation_sources

    assert event.decision.action == Action.CONTINUE_MISSION
    assert event.decision.safety_level == SafetyLevel.CAUTION


def test_gps_loss_uses_time_aligned_visual_inertial_fallback() -> None:
    traces = MemoryTraceStore()

    agent = AegisLandAgent(
        VisualFallbackPerception(),
        SafetyPlanner(),
        traces,
    )

    agent.ingest_imu(
        timestamp_s=0.995,
        yaw_rad=0.10,
        yaw_rate_rad_s=0.02,
        ax=0.10,
        ay=0.00,
    )

    agent.ingest_imu(
        timestamp_s=1.005,
        yaw_rad=0.12,
        yaw_rate_rad_s=0.02,
        ax=0.12,
        ay=0.00,
    )

    event, _ = agent.step(
        object(),
        Telemetry(
            battery_percent=80,
            altitude_m=10,
            gps_available=False,
            home_link_available=True,
            timestamp_s=1.0,
        ),
        30,
    )

    assert event.evidence.imu_sync_valid
    assert event.evidence.imu_sync_method == "interpolated"
    assert event.evidence.imu_confidence == 0.90

    assert (
        event.evidence.navigation_mode
        == "visual_inertial_fallback"
    )

    assert "visual" in event.evidence.healthy_navigation_sources
    assert "imu" in event.evidence.healthy_navigation_sources
    assert "gps" in event.evidence.degraded_navigation_sources

    assert event.decision.action == Action.CONTINUE_MISSION
    assert event.decision.safety_level == SafetyLevel.CAUTION


class FailedVisualPerception:
    def observe(
        self,
        frame,
        frame_index,
        active_perception=False,
    ):
        from aegisland.domain import VisionEvidence

        return (
            VisionEvidence(
                evidence_id=f"failed-visual-{frame_index}",
                frame_index=frame_index,
                confidence=0.95,
                obstacle_risk=0.0,
                motion_risk=0.0,
                visual_localization_valid=False,
                visual_localization_confidence=0.05,
            ),
            frame,
        )

    def enhance_for_active_perception(self, frame):
        return frame


def test_sensor_health_filters_visual_and_imu_before_fusion() -> None:
    agent = AegisLandAgent(
        VisualFallbackPerception(),
        SafetyPlanner(),
        MemoryTraceStore(),
    )

    agent.ingest_imu(
        timestamp_s=0.995,
        yaw_rad=0.1,
        yaw_rate_rad_s=0.02,
        ax=0.0,
        ay=0.0,
    )
    agent.ingest_imu(
        timestamp_s=1.005,
        yaw_rad=0.12,
        yaw_rate_rad_s=0.02,
        ax=0.0,
        ay=0.0,
    )

    event, _ = agent.step(
        object(),
        Telemetry(
            battery_percent=80,
            altitude_m=10,
            gps_available=False,
            home_link_available=True,
            timestamp_s=1.0,
        ),
        30,
    )

    assert event.evidence.gps_health_state == "failed"
    assert event.evidence.visual_health_state == "degraded"
    assert event.evidence.imu_health_state == "degraded"

    assert (
        event.evidence.navigation_mode
        == "visual_inertial_fallback"
    )


def test_failed_camera_with_gps_loss_enters_safe_degraded_mode() -> None:
    agent = AegisLandAgent(
        FailedVisualPerception(),
        SafetyPlanner(),
        MemoryTraceStore(),
    )

    agent.ingest_imu(
        timestamp_s=1.0,
        yaw_rad=0.1,
        yaw_rate_rad_s=0.01,
        ax=0.0,
        ay=0.0,
    )

    event, _ = agent.step(
        object(),
        Telemetry(
            battery_percent=80,
            altitude_m=10,
            gps_available=False,
            home_link_available=True,
            timestamp_s=1.0,
        ),
        30,
    )

    assert event.evidence.visual_health_state == "failed"
    assert event.evidence.navigation_mode == "degraded"
    assert event.decision.action == Action.HOLD_AND_SCAN


def test_visual_sensor_requires_sustained_recovery_before_healthy() -> None:
    agent = AegisLandAgent(
        VisualFallbackPerception(),
        SafetyPlanner(),
        MemoryTraceStore(),
    )

    states = []

    for frame_index in range(3):
        timestamp = 2.0 + frame_index / 30.0

        event, _ = agent.step(
            object(),
            Telemetry(
                battery_percent=80,
                altitude_m=10,
                gps_available=True,
                home_link_available=True,
                timestamp_s=timestamp,
            ),
            frame_index,
        )

        states.append(
            event.evidence.visual_health_state
        )

    assert states[0] == "degraded"
    assert states[1] == "degraded"
    assert states[2] == "healthy"
