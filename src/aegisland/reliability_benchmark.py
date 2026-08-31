from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from .agent import AegisLandAgent
from .perception import OpenCVLandingPerception
from .planner import SafetyPlanner
from .simulated_sensors import ingest_out_of_order_imu
from .simulator import SCENARIOS, generate
from .trace import MemoryTraceStore

IMU_STALE_START = 32
IMU_STALE_END = 45


@dataclass(frozen=True, slots=True)
class CascadingFaultRow:
    frame: int
    timestamp_s: float

    gps_available: bool
    camera_fault_active: bool
    imu_transport_state: str

    vision_confidence: float
    camera_match_count: int
    camera_inlier_ratio: float

    visual_health_state: str
    visual_effective_confidence: float

    imu_sync_valid: bool
    imu_sync_method: str
    imu_health_state: str
    imu_effective_confidence: float

    out_of_order_insertions: int

    fused_navigation_confidence: float
    gps_weight: float
    visual_weight: float
    imu_weight: float

    navigation_mode: str
    raw_action: str
    action: str


@dataclass(frozen=True, slots=True)
class ReliabilityMetrics:
    frames: int

    imu_stale_detection_latency_frames: int | None
    camera_perception_detection_latency_frames: int | None
    safe_hold_latency_frames: int | None
    navigation_degradation_latency_frames: int | None

    imu_health_recovery_latency_frames: int | None
    visual_health_recovery_latency_frames: int | None
    mission_resume_latency_frames: int | None

    unsafe_continuation_frames: int

    out_of_order_frames: int
    compensated_out_of_order_frames: int
    out_of_order_sync_success_rate: float


def _first_frame(
    rows: list[CascadingFaultRow],
    *,
    start: int,
    predicate,
) -> int | None:
    for row in rows:
        if row.frame >= start and predicate(row):
            return row.frame

    return None


def _latency(
    event_frame: int | None,
    start_frame: int,
) -> int | None:
    if event_frame is None:
        return None

    return event_frame - start_frame


def run_cascading_failure() -> list[CascadingFaultRow]:
    scenario = SCENARIOS[
        "gps_denied_camera_failure"
    ]

    agent = AegisLandAgent(
        OpenCVLandingPerception(),
        SafetyPlanner(),
        MemoryTraceStore(),
    )

    rows: list[CascadingFaultRow] = []

    for frame_index, (frame, telemetry) in enumerate(
        generate(scenario)
    ):
        timestamp_s = telemetry.timestamp_s

        if IMU_STALE_START <= frame_index < IMU_STALE_END:
            imu_transport_state = "stale"

            # No fresh IMU packet is injected.
            # TimeSyncBuffer must eventually reject stale data.
        else:
            if frame_index >= IMU_STALE_END:
                imu_transport_state = "out_of_order_recovery"
            else:
                imu_transport_state = "out_of_order_healthy"

            ingest_out_of_order_imu(
                agent,
                timestamp_s=timestamp_s,
                frame_index=frame_index,
            )

        event, _ = agent.step(
            frame,
            telemetry,
            frame_index,
        )

        evidence = event.evidence

        camera_fault_active = (
            scenario.camera_fault_start_frame is not None
            and frame_index
            >= scenario.camera_fault_start_frame
            and (
                scenario.camera_fault_end_frame is None
                or frame_index
                < scenario.camera_fault_end_frame
            )
        )

        rows.append(
            CascadingFaultRow(
                frame=frame_index,
                timestamp_s=timestamp_s,

                gps_available=telemetry.gps_available,
                camera_fault_active=camera_fault_active,
                imu_transport_state=imu_transport_state,

                vision_confidence=evidence.confidence,
                camera_match_count=evidence.camera_match_count,
                camera_inlier_ratio=evidence.camera_inlier_ratio,

                visual_health_state=evidence.visual_health_state,
                visual_effective_confidence=(
                    evidence.visual_effective_confidence
                ),

                imu_sync_valid=evidence.imu_sync_valid,
                imu_sync_method=evidence.imu_sync_method,
                imu_health_state=evidence.imu_health_state,
                imu_effective_confidence=(
                    evidence.imu_effective_confidence
                ),

                out_of_order_insertions=(
                    agent.sensor_synchronizer
                    .imu
                    .out_of_order_insertions
                ),

                fused_navigation_confidence=(
                    evidence.fused_navigation_confidence
                ),
                gps_weight=evidence.navigation_gps_weight,
                visual_weight=evidence.navigation_visual_weight,
                imu_weight=evidence.navigation_imu_weight,

                navigation_mode=evidence.navigation_mode,

                raw_action=(
                    event.raw_decision.action.value
                    if event.raw_decision is not None
                    else event.decision.action.value
                ),

                action=event.decision.action.value,
            )
        )

    return rows


def evaluate_reliability(
    rows: list[CascadingFaultRow],
) -> ReliabilityMetrics:
    scenario = SCENARIOS[
        "gps_denied_camera_failure"
    ]

    camera_start = scenario.camera_fault_start_frame
    camera_end = scenario.camera_fault_end_frame

    if camera_start is None or camera_end is None:
        raise RuntimeError(
            "Camera fault window is required."
        )

    imu_failed = _first_frame(
        rows,
        start=IMU_STALE_START,
        predicate=lambda row: (
            row.imu_health_state == "failed"
        ),
    )

    camera_detected = _first_frame(
        rows,
        start=camera_start,
        predicate=lambda row: (
            row.camera_match_count == 0
        ),
    )

    safe_hold = _first_frame(
        rows,
        start=camera_start,
        predicate=lambda row: (
            row.action == "hold_and_scan"
        ),
    )

    nav_degraded = _first_frame(
        rows,
        start=camera_start,
        predicate=lambda row: (
            row.navigation_mode == "degraded"
        ),
    )

    imu_recovered = _first_frame(
        rows,
        start=IMU_STALE_END,
        predicate=lambda row: (
            row.imu_health_state == "healthy"
        ),
    )

    visual_recovered = _first_frame(
        rows,
        start=camera_end,
        predicate=lambda row: (
            row.visual_health_state == "healthy"
        ),
    )

    mission_resumed = _first_frame(
        rows,
        start=camera_end,
        predicate=lambda row: (
            row.action == "continue_mission"
        ),
    )

    unsafe_continuation_frames = sum(
        1
        for row in rows
        if (
            not row.gps_available
            and row.camera_fault_active
            and row.action == "continue_mission"
        )
    )

    out_of_order_rows = [
        row
        for row in rows
        if row.imu_transport_state
        != "stale"
    ]

    compensated_rows = [
        row
        for row in out_of_order_rows
        if (
            row.imu_sync_valid
            and row.imu_sync_method
            == "interpolated"
        )
    ]

    success_rate = (
        len(compensated_rows)
        / len(out_of_order_rows)
        if out_of_order_rows
        else 0.0
    )

    return ReliabilityMetrics(
        frames=len(rows),

        imu_stale_detection_latency_frames=_latency(
            imu_failed,
            IMU_STALE_START,
        ),

        camera_perception_detection_latency_frames=_latency(
            camera_detected,
            camera_start,
        ),

        safe_hold_latency_frames=_latency(
            safe_hold,
            camera_start,
        ),

        navigation_degradation_latency_frames=_latency(
            nav_degraded,
            camera_start,
        ),

        imu_health_recovery_latency_frames=_latency(
            imu_recovered,
            IMU_STALE_END,
        ),

        visual_health_recovery_latency_frames=_latency(
            visual_recovered,
            camera_end,
        ),

        mission_resume_latency_frames=_latency(
            mission_resumed,
            camera_end,
        ),

        unsafe_continuation_frames=(
            unsafe_continuation_frames
        ),

        out_of_order_frames=len(
            out_of_order_rows
        ),

        compensated_out_of_order_frames=len(
            compensated_rows
        ),

        out_of_order_sync_success_rate=round(
            success_rate,
            4,
        ),
    )


def _write_rows(
    rows: list[CascadingFaultRow],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                CascadingFaultRow.__dataclass_fields__
            ),
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(asdict(row))


def _write_metrics(
    metrics: ReliabilityMetrics,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = asdict(metrics)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(data),
        )

        writer.writeheader()
        writer.writerow(data)


def main() -> None:
    rows = run_cascading_failure()
    metrics = evaluate_reliability(rows)

    timeline_path = Path(
        "runs/evaluation/"
        "cascading_sensor_failure_timeline.csv"
    )

    summary_path = Path(
        "runs/evaluation/"
        "cascading_sensor_failure_summary.csv"
    )

    _write_rows(
        rows,
        timeline_path,
    )

    _write_metrics(
        metrics,
        summary_path,
    )

    print("CASCADING SENSOR FAILURE")
    print("=" * 95)

    for row in rows:
        interesting = (
            row.frame % 10 == 0
            or 18 <= row.frame <= 22
            or 30 <= row.frame <= 48
            or 54 <= row.frame <= 62
        )

        if not interesting:
            continue

        print(
            f"{row.frame:3d} "
            f"gps={row.gps_available!s:5} "
            f"cam={row.camera_fault_active!s:5} "
            f"imu={row.imu_health_state:9} "
            f"vis={row.visual_health_state:9} "
            f"mode={row.navigation_mode:28} "
            f"action={row.action}"
        )

    print()
    print("RELIABILITY METRICS")
    print("=" * 95)

    for key, value in asdict(metrics).items():
        print(
            f"{key:42} {value}"
        )

    print()
    print(f"Timeline: {timeline_path}")
    print(f"Summary:  {summary_path}")


if __name__ == "__main__":
    main()
