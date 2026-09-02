from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

from .fault_timeline import FaultTimelineRow, run_fault_timeline

CONTINUE = "continue_mission"


@dataclass(frozen=True, slots=True)
class FlickerMetrics:
    fault_start_frame: int
    first_recovery_attempt_frame: int
    last_flicker_fault_frame: int
    stable_physical_recovery_frame: int

    premature_trust_grants: int
    premature_healthy_grants: int

    raw_continue_frames_during_unstable_recovery: int
    final_continue_frames_during_unstable_recovery: int

    raw_action_transitions: int
    final_action_transitions: int

    stable_trust_frame: int
    stable_health_frame: int
    raw_recovery_frame: int
    final_recovery_frame: int

    trust_recovery_latency_frames: int
    health_recovery_latency_frames: int
    raw_recovery_latency_frames: int
    final_recovery_latency_frames: int


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


def _grant_events(
    rows: list[FaultTimelineRow],
    *,
    start: int,
    end: int,
    field: str,
    target: object,
) -> int:
    return sum(
        getattr(previous, field) != target
        and getattr(current, field) == target
        and start <= current.frame <= end
        for previous, current in pairwise(rows)
    )


def _transition_count(
    values: list[str],
) -> int:
    return sum(
        previous != current
        for previous, current in pairwise(values)
    )


def run_benchmark() -> FlickerMetrics:
    rows = run_fault_timeline(
        "gps_denied_camera_flicker"
    )

    fault_frames = [
        row.frame
        for row in rows
        if row.camera_fault_active
    ]

    fault_start = min(fault_frames)
    last_fault = max(fault_frames)

    first_recovery = _first_recovery_attempt(
        rows
    )

    stable_physical = next(
        row.frame
        for row in rows
        if row.frame > last_fault
    )

    unstable_rows = [
        row
        for row in rows
        if first_recovery <= row.frame <= last_fault
    ]

    premature_trust = _grant_events(
        rows,
        start=first_recovery,
        end=last_fault,
        field="visual_localization_trusted",
        target=True,
    )

    premature_healthy = _grant_events(
        rows,
        start=first_recovery,
        end=last_fault,
        field="visual_health_state",
        target="healthy",
    )

    raw_unstable_continue = sum(
        row.raw_action == CONTINUE
        for row in unstable_rows
    )

    final_unstable_continue = sum(
        row.action == CONTINUE
        for row in unstable_rows
    )

    stable_trust = next(
        row.frame
        for row in rows
        if row.frame >= stable_physical
        and row.visual_localization_trusted
    )

    stable_health = next(
        row.frame
        for row in rows
        if row.frame >= stable_physical
        and row.visual_health_state == "healthy"
    )

    raw_recovery = next(
        row.frame
        for row in rows
        if row.frame >= stable_physical
        and row.raw_action == CONTINUE
    )

    final_recovery = next(
        row.frame
        for row in rows
        if row.frame >= stable_physical
        and row.action == CONTINUE
    )

    recovery_rows = [
        row
        for row in rows
        if first_recovery
        <= row.frame
        <= final_recovery
    ]

    raw_transitions = _transition_count(
        [
            row.raw_action
            for row in recovery_rows
        ]
    )

    final_transitions = _transition_count(
        [
            row.action
            for row in recovery_rows
        ]
    )

    return FlickerMetrics(
        fault_start_frame=fault_start,
        first_recovery_attempt_frame=first_recovery,
        last_flicker_fault_frame=last_fault,
        stable_physical_recovery_frame=stable_physical,
        premature_trust_grants=premature_trust,
        premature_healthy_grants=premature_healthy,
        raw_continue_frames_during_unstable_recovery=(
            raw_unstable_continue
        ),
        final_continue_frames_during_unstable_recovery=(
            final_unstable_continue
        ),
        raw_action_transitions=raw_transitions,
        final_action_transitions=final_transitions,
        stable_trust_frame=stable_trust,
        stable_health_frame=stable_health,
        raw_recovery_frame=raw_recovery,
        final_recovery_frame=final_recovery,
        trust_recovery_latency_frames=(
            stable_trust - stable_physical
        ),
        health_recovery_latency_frames=(
            stable_health - stable_physical
        ),
        raw_recovery_latency_frames=(
            raw_recovery - stable_physical
        ),
        final_recovery_latency_frames=(
            final_recovery - stable_physical
        ),
    )


def main() -> None:
    metrics = run_benchmark()

    print()
    print("INTERMITTENT CAMERA FLICKER BENCHMARK")
    print("=" * 76)

    for name in (
        "fault_start_frame",
        "first_recovery_attempt_frame",
        "last_flicker_fault_frame",
        "stable_physical_recovery_frame",
        "premature_trust_grants",
        "premature_healthy_grants",
        "raw_continue_frames_during_unstable_recovery",
        "final_continue_frames_during_unstable_recovery",
        "raw_action_transitions",
        "final_action_transitions",
        "stable_trust_frame",
        "stable_health_frame",
        "raw_recovery_frame",
        "final_recovery_frame",
        "trust_recovery_latency_frames",
        "health_recovery_latency_frames",
        "raw_recovery_latency_frames",
        "final_recovery_latency_frames",
    ):
        print(
            f"{name:48}"
            f"{getattr(metrics, name)}"
        )

    print("=" * 76)


if __name__ == "__main__":
    main()
