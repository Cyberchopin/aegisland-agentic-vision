from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

from .fault_timeline import FaultTimelineRow, run_fault_timeline


@dataclass(frozen=True, slots=True)
class HysteresisMetrics:
    recovery_samples: int
    premature_healthy_grants: int
    health_state_transitions: int
    unstable_healthy_frames: int
    stable_health_frame: int
    stable_health_latency_frames: int


@dataclass(frozen=True, slots=True)
class HysteresisAblation:
    no_hysteresis: HysteresisMetrics
    aegisland_hysteresis: HysteresisMetrics


def _stable_physical_recovery(
    rows: list[FaultTimelineRow],
) -> int:
    fault_frames = [
        row.frame
        for row in rows
        if row.camera_fault_active
    ]

    last_fault = max(fault_frames)

    return next(
        row.frame
        for row in rows
        if row.frame > last_fault
    )


def _first_recovery_attempt(
    rows: list[FaultTimelineRow],
) -> int:
    fault_seen = False

    for row in rows:
        if row.camera_fault_active:
            fault_seen = True
            continue

        if fault_seen:
            return row.frame

    raise RuntimeError("Recovery attempt not found.")


def _metrics(
    *,
    recovery_samples: int,
) -> HysteresisMetrics:
    rows = run_fault_timeline(
        "gps_denied_camera_flicker",
        sensor_health_recovery_samples=recovery_samples,
    )

    stable_physical = _stable_physical_recovery(rows)
    first_attempt = _first_recovery_attempt(rows)

    last_fault = max(
        row.frame
        for row in rows
        if row.camera_fault_active
    )

    recovery_window = [
        row
        for row in rows
        if first_attempt <= row.frame
    ]

    premature_healthy = sum(
        previous.visual_health_state != "healthy"
        and current.visual_health_state == "healthy"
        and first_attempt <= current.frame <= last_fault
        for previous, current in pairwise(rows)
    )

    unstable_healthy_frames = sum(
        row.visual_health_state == "healthy"
        and first_attempt <= row.frame <= last_fault
        for row in rows
    )

    health_transitions = sum(
        previous.visual_health_state
        != current.visual_health_state
        for previous, current in pairwise(
            recovery_window
        )
    )

    stable_health = next(
        row.frame
        for row in rows
        if row.frame >= stable_physical
        and row.visual_health_state == "healthy"
    )

    return HysteresisMetrics(
        recovery_samples=recovery_samples,
        premature_healthy_grants=premature_healthy,
        health_state_transitions=health_transitions,
        unstable_healthy_frames=unstable_healthy_frames,
        stable_health_frame=stable_health,
        stable_health_latency_frames=(
            stable_health - stable_physical
        ),
    )


def run_ablation() -> HysteresisAblation:
    return HysteresisAblation(
        no_hysteresis=_metrics(
            recovery_samples=1
        ),
        aegisland_hysteresis=_metrics(
            recovery_samples=3
        ),
    )


def main() -> None:
    result = run_ablation()

    print()
    print("SENSOR HEALTH HYSTERESIS ABLATION")
    print("=" * 82)

    print(
        f"{'metric':38}"
        f"{'samples=1':>18}"
        f"{'samples=3':>18}"
    )

    print("-" * 82)

    fields = (
        "premature_healthy_grants",
        "unstable_healthy_frames",
        "health_state_transitions",
        "stable_health_frame",
        "stable_health_latency_frames",
    )

    for field in fields:
        print(
            f"{field:38}"
            f"{getattr(result.no_hysteresis, field):>18}"
            f"{getattr(result.aegisland_hysteresis, field):>18}"
        )

    print("=" * 82)


if __name__ == "__main__":
    main()
