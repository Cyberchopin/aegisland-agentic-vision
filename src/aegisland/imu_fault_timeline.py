from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .agent import AegisLandAgent
from .perception import OpenCVLandingPerception
from .planner import SafetyPlanner
from .simulated_sensors import ingest_out_of_order_imu
from .simulator import SCENARIOS, generate
from .trace import MemoryTraceStore


@dataclass(frozen=True, slots=True)
class IMUFaultTimelineRow:
    frame: int
    timestamp_s: float
    gps_available: bool
    imu_transport_state: str

    imu_sync_valid: bool
    imu_sync_method: str
    imu_health_state: str
    imu_raw_confidence: float
    imu_effective_confidence: float

    visual_health_state: str
    visual_effective_confidence: float

    out_of_order_insertions: int

    fused_navigation_confidence: float
    gps_weight: float
    visual_weight: float
    imu_weight: float

    navigation_mode: str
    raw_action: str
    action: str


def run_imu_fault_timeline() -> list[IMUFaultTimelineRow]:
    agent = AegisLandAgent(
        OpenCVLandingPerception(),
        SafetyPlanner(),
        MemoryTraceStore(),
    )

    rows: list[IMUFaultTimelineRow] = []

    scenario = SCENARIOS["nominal"]

    for frame_index, (frame, telemetry) in enumerate(
        generate(scenario)
    ):
        timestamp_s = telemetry.timestamp_s

        gps_available = frame_index < 20

        telemetry = replace(
            telemetry,
            gps_available=gps_available,
            home_link_available=gps_available,
        )

        if 30 <= frame_index < 38:
            transport_state = "stale"
            # Deliberately inject NO current IMU packet.
        elif frame_index >= 38:
            transport_state = "out_of_order_recovery"

            ingest_out_of_order_imu(
                agent,
                timestamp_s=timestamp_s,
                frame_index=frame_index,
            )
        else:
            transport_state = "out_of_order_healthy"

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

        rows.append(
            IMUFaultTimelineRow(
                frame=frame_index,
                timestamp_s=timestamp_s,
                gps_available=gps_available,
                imu_transport_state=transport_state,

                imu_sync_valid=evidence.imu_sync_valid,
                imu_sync_method=evidence.imu_sync_method,
                imu_health_state=evidence.imu_health_state,
                imu_raw_confidence=evidence.imu_confidence,
                imu_effective_confidence=(
                    evidence.imu_effective_confidence
                ),

                visual_health_state=(
                    evidence.visual_health_state
                ),
                visual_effective_confidence=(
                    evidence.visual_effective_confidence
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


def write_timeline(
    rows: list[IMUFaultTimelineRow],
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
                IMUFaultTimelineRow.__dataclass_fields__
            ),
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    rows = run_imu_fault_timeline()

    path = Path(
        "runs/evaluation/"
        "gps_denied_imu_transport_timeline.csv"
    )

    write_timeline(
        rows,
        path,
    )

    print(
        "frame gps transport              sync "
        "imu-health OOO mode                         action"
    )
    print("-" * 110)

    for row in rows:
        interesting = (
            row.frame % 10 == 0
            or 18 <= row.frame <= 22
            or 28 <= row.frame <= 42
        )

        if not interesting:
            continue

        print(
            f"{row.frame:3d} "
            f"{row.gps_available!s:5} "
            f"{row.imu_transport_state:22} "
            f"{row.imu_sync_method:12} "
            f"{row.imu_health_state:9} "
            f"{row.out_of_order_insertions:3d} "
            f"{row.navigation_mode:28} "
            f"{row.action}"
        )

    print()
    print(f"Timeline written to: {path}")


if __name__ == "__main__":
    main()
