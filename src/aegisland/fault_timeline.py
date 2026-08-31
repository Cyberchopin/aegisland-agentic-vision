from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from .agent import AegisLandAgent
from .perception import OpenCVLandingPerception
from .planner import SafetyPlanner
from .simulator import SCENARIOS, generate
from .trace import MemoryTraceStore


@dataclass(frozen=True, slots=True)
class FaultTimelineRow:
    frame: int
    timestamp_s: float

    gps_available: bool
    camera_fault_active: bool
    vision_confidence: float

    camera_compensation_used: bool
    camera_match_count: int
    camera_inlier_count: int
    camera_inlier_ratio: float

    visual_localization_valid: bool
    visual_raw_confidence: float
    visual_health_state: str
    visual_effective_confidence: float

    imu_sync_valid: bool
    imu_health_state: str
    imu_effective_confidence: float

    fused_navigation_confidence: float

    gps_weight: float
    visual_weight: float
    imu_weight: float

    navigation_mode: str
    raw_action: str
    action: str
    safety_level: str
    risk_score: float


def run_fault_timeline(
    scenario_name: str = "gps_denied_camera_failure",
) -> list[FaultTimelineRow]:
    scenario = SCENARIOS[scenario_name]

    agent = AegisLandAgent(
        OpenCVLandingPerception(),
        SafetyPlanner(),
        MemoryTraceStore(),
    )

    rows: list[FaultTimelineRow] = []

    for frame_index, (frame, telemetry) in enumerate(
        generate(scenario)
    ):
        timestamp_s = telemetry.timestamp_s

        # Deterministic simulated IMU stream.
        #
        # It is intentionally simple: this benchmark evaluates
        # synchronization, health and failover behavior rather than
        # claiming physical IMU realism.
        agent.ingest_imu(
            timestamp_s=timestamp_s,
            yaw_rad=0.002 * frame_index,
            yaw_rate_rad_s=0.03,
            ax=0.02,
            ay=0.0,
        )

        event, _ = agent.step(
            frame,
            telemetry,
            frame_index,
        )

        evidence = event.evidence

        fault_active = (
            scenario.camera_fault_start_frame is not None
            and frame_index >= scenario.camera_fault_start_frame
            and (
                scenario.camera_fault_end_frame is None
                or frame_index < scenario.camera_fault_end_frame
            )
        )

        rows.append(
            FaultTimelineRow(
                frame=frame_index,
                timestamp_s=timestamp_s,

                gps_available=telemetry.gps_available,
                camera_fault_active=fault_active,
                vision_confidence=evidence.confidence,

                camera_compensation_used=(
                    evidence.camera_compensation_used
                ),
                camera_match_count=(
                    evidence.camera_match_count
                ),
                camera_inlier_count=(
                    evidence.camera_inlier_count
                ),
                camera_inlier_ratio=(
                    evidence.camera_inlier_ratio
                ),

                visual_localization_valid=(
                    evidence.visual_localization_valid
                ),
                visual_raw_confidence=(
                    evidence.visual_localization_confidence
                ),
                visual_health_state=(
                    evidence.visual_health_state
                ),
                visual_effective_confidence=(
                    evidence.visual_effective_confidence
                ),

                imu_sync_valid=evidence.imu_sync_valid,
                imu_health_state=evidence.imu_health_state,
                imu_effective_confidence=(
                    evidence.imu_effective_confidence
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
                safety_level=event.decision.safety_level.value,
                risk_score=event.decision.risk_score,
            )
        )

    return rows


def write_timeline(
    rows: list[FaultTimelineRow],
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
                FaultTimelineRow.__dataclass_fields__
            ),
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenario",
        default="gps_denied_camera_failure",
    )

    parser.add_argument(
        "--output",
        default=(
            "runs/evaluation/"
            "gps_denied_camera_failure_timeline.csv"
        ),
    )

    args = parser.parse_args()

    rows = run_fault_timeline(
        args.scenario
    )

    write_timeline(
        rows,
        Path(args.output),
    )

    print(
        "frame gps fault "
        "matches inlier visual-health "
        "fused mode action"
    )
    print("-" * 118)

    for row in rows:
        # Print transition windows plus every tenth frame.
        interesting = (
            row.frame % 10 == 0
            or 18 <= row.frame <= 22
            or 38 <= row.frame <= 43
            or 54 <= row.frame <= 61
        )

        if not interesting:
            continue

        print(
            f"{row.frame:3d} "
            f"{row.gps_available!s:5} "
            f"{row.camera_fault_active!s:5} "
            f"{row.camera_match_count:7d} "
            f"{row.camera_inlier_ratio:6.2f} "
            f"{row.visual_health_state:9} "
            f"{row.fused_navigation_confidence:5.2f} "
            f"{row.navigation_mode:28} "
            f"{row.action}"
        )

    print()
    print(f"Timeline written to: {args.output}")


if __name__ == "__main__":
    main()
