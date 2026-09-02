from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .agent import AegisLandAgent
from .domain import Telemetry, VisionEvidence
from .planner import SafetyPlanner
from .trace import MemoryTraceStore


class SyntheticNavigationPerception:
    def __init__(
        self,
        *,
        visual_valid: bool,
        visual_confidence: float,
    ) -> None:
        self.visual_valid = visual_valid
        self.visual_confidence = visual_confidence

    def observe(
        self,
        frame,
        frame_index,
        active_perception=False,
    ):
        evidence = VisionEvidence(
            evidence_id=f"sensor-matrix-{frame_index}",
            frame_index=frame_index,
            confidence=0.95,
            obstacle_risk=0.0,
            motion_risk=0.0,
            visual_localization_valid=self.visual_valid,
            visual_localization_confidence=self.visual_confidence,
            visual_relative_x=0.01 * frame_index,
            visual_relative_y=0.0,
        )

        return evidence, frame

    def enhance_for_active_perception(self, frame):
        return frame


@dataclass(frozen=True, slots=True)
class FailureCaseResult:
    case: str
    expected_mode: str
    actual_mode: str
    action: str
    safety_level: str
    fused_confidence: float
    gps_weight: float
    visual_weight: float
    imu_weight: float
    imu_sync_method: str
    passed: bool


def build_agent(
    *,
    visual_valid: bool,
    visual_confidence: float,
) -> AegisLandAgent:
    return AegisLandAgent(
        SyntheticNavigationPerception(
            visual_valid=visual_valid,
            visual_confidence=visual_confidence,
        ),
        SafetyPlanner(),
        MemoryTraceStore(),
    )


def run_case(
    *,
    name: str,
    expected_mode: str,
    gps_available: bool,
    visual_valid: bool,
    visual_confidence: float,
    imu: bool,
) -> FailureCaseResult:
    agent = build_agent(
        visual_valid=visual_valid,
        visual_confidence=visual_confidence,
    )

    event = None

    # Static matrix cases represent confirmed steady-state
    # sensor health, not the transient startup/recovery window.
    for warmup_index in range(3):
        timestamp = (
            1.0
            + warmup_index / 30.0
        )

        if imu:
            agent.ingest_imu(
                timestamp_s=timestamp - 0.005,
                yaw_rad=0.10,
                yaw_rate_rad_s=0.02,
                ax=0.10,
                ay=0.00,
            )

            agent.ingest_imu(
                timestamp_s=timestamp + 0.005,
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
                gps_available=gps_available,
                home_link_available=True,
                timestamp_s=timestamp,
            ),
            30 + warmup_index,
        )

    assert event is not None

    evidence = event.evidence

    return FailureCaseResult(
        case=name,
        expected_mode=expected_mode,
        actual_mode=evidence.navigation_mode,
        action=event.decision.action.value,
        safety_level=(
            event.decision.safety_level.value
        ),
        fused_confidence=(
            evidence.fused_navigation_confidence
        ),
        gps_weight=(
            evidence.navigation_gps_weight
        ),
        visual_weight=(
            evidence.navigation_visual_weight
        ),
        imu_weight=(
            evidence.navigation_imu_weight
        ),
        imu_sync_method=evidence.imu_sync_method,
        passed=(
            evidence.navigation_mode
            == expected_mode
        ),
    )


def run_static_cases() -> list[FailureCaseResult]:
    return [
        run_case(
            name="gps_healthy",
            expected_mode="gps_primary",
            gps_available=True,
            visual_valid=True,
            visual_confidence=0.80,
            imu=True,
        ),
        run_case(
            name="gps_lost_visual_imu_healthy",
            expected_mode="visual_inertial_fallback",
            gps_available=False,
            visual_valid=True,
            visual_confidence=0.82,
            imu=True,
        ),
        run_case(
            name="gps_lost_visual_only",
            expected_mode="visual_fallback",
            gps_available=False,
            visual_valid=True,
            visual_confidence=0.82,
            imu=False,
        ),
        run_case(
            name="gps_lost_camera_bad_imu_only",
            expected_mode="degraded",
            gps_available=False,
            visual_valid=False,
            visual_confidence=0.10,
            imu=True,
        ),
    ]


def run_recovery_sequence() -> list[dict[str, object]]:
    agent = build_agent(
        visual_valid=True,
        visual_confidence=0.85,
    )

    rows: list[dict[str, object]] = []

    for frame_index, gps_available in enumerate(
        [
            True,
            True,
            True,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
        ]
    ):
        timestamp = 2.0 + frame_index / 30.0

        agent.ingest_imu(
            timestamp_s=timestamp,
            yaw_rad=0.10,
            yaw_rate_rad_s=0.01,
            ax=0.0,
            ay=0.0,
        )

        event, _ = agent.step(
            object(),
            Telemetry(
                battery_percent=80,
                altitude_m=10,
                gps_available=gps_available,
                home_link_available=True,
                timestamp_s=timestamp,
            ),
            frame_index,
        )

        rows.append(
            {
                "frame": frame_index,
                "gps_available": gps_available,
                "navigation_mode": event.evidence.navigation_mode,
                "fused_confidence": (
                    event.evidence.fused_navigation_confidence
                ),
                "gps_weight": event.evidence.navigation_gps_weight,
                "visual_weight": event.evidence.navigation_visual_weight,
                "imu_weight": event.evidence.navigation_imu_weight,
                "action": event.decision.action.value,
            }
        )

    return rows


def main() -> None:
    output_dir = Path("runs/evaluation")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cases = run_static_cases()

    static_path = (
        output_dir
        / "sensor_failure_matrix.csv"
    )

    with static_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                FailureCaseResult.__dataclass_fields__
            ),
        )

        writer.writeheader()

        for case in cases:
            writer.writerow(
                {
                    field: getattr(case, field)
                    for field
                    in FailureCaseResult.__dataclass_fields__
                }
            )

    recovery = run_recovery_sequence()

    recovery_path = (
        output_dir
        / "sensor_recovery_sequence.csv"
    )

    with recovery_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(recovery[0]),
        )
        writer.writeheader()
        writer.writerows(recovery)

    print("SENSOR FAILURE MATRIX")
    print("=" * 84)

    for case in cases:
        marker = "PASS" if case.passed else "FAIL"

        print(
            f"{marker:4} "
            f"{case.case:34} "
            f"{case.actual_mode:28} "
            f"action={case.action}"
        )

    print()
    print("RECOVERY / HANDOVER SEQUENCE")
    print("=" * 84)

    for row in recovery:
        print(
            f"frame={row['frame']} "
            f"gps={row['gps_available']!s:5} "
            f"mode={row['navigation_mode']:28} "
            f"Wgps={row['gps_weight']:.3f} "
            f"Wvis={row['visual_weight']:.3f} "
            f"Wimu={row['imu_weight']:.3f}"
        )

    print()
    print(f"Written: {static_path}")
    print(f"Written: {recovery_path}")


if __name__ == "__main__":
    main()
