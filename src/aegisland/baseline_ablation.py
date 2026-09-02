from __future__ import annotations

from dataclasses import dataclass

from .fault_timeline import FaultTimelineRow, run_fault_timeline


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    frame: int
    fault: bool
    raw_valid: bool
    raw_confidence: float
    baseline_action: str
    aegisland_action: str


@dataclass(frozen=True, slots=True)
class Metrics:
    fault_start: int
    physical_recovery: int

    baseline_safe_frame: int
    aegisland_safe_frame: int

    baseline_time_to_safe: int
    aegisland_time_to_safe: int

    baseline_unsafe_continuation: int
    aegisland_unsafe_continuation: int

    baseline_recovery_frame: int
    aegisland_recovery_frame: int

    baseline_recovery_latency: int
    aegisland_recovery_latency: int


def baseline_action(
    row: FaultTimelineRow,
    *,
    visual_gate: float = 0.55,
) -> str:
    if row.gps_available:
        return "continue_mission"

    visual_usable = (
        row.visual_localization_valid
        and row.visual_raw_confidence >= visual_gate
    )

    if visual_usable:
        return "continue_mission"

    return "hold_and_scan"


def build_comparison(
    timeline: list[FaultTimelineRow],
) -> list[ComparisonRow]:
    return [
        ComparisonRow(
            frame=row.frame,
            fault=row.camera_fault_active,
            raw_valid=row.visual_localization_valid,
            raw_confidence=row.visual_raw_confidence,
            baseline_action=baseline_action(row),
            aegisland_action=row.action,
        )
        for row in timeline
    ]


def _fault_window(
    rows: list[ComparisonRow],
) -> tuple[int, int]:
    fault_start = next(
        row.frame
        for row in rows
        if row.fault
    )

    fault_seen = False

    for row in rows:
        if row.fault:
            fault_seen = True
            continue

        if fault_seen:
            return fault_start, row.frame

    raise RuntimeError("Fault recovery frame not found.")


def _first_non_continue(
    rows: list[ComparisonRow],
    *,
    start: int,
    field: str,
) -> int:
    return next(
        row.frame
        for row in rows
        if row.frame >= start
        and getattr(row, field) != "continue_mission"
    )


def _first_continue(
    rows: list[ComparisonRow],
    *,
    start: int,
    field: str,
) -> int:
    return next(
        row.frame
        for row in rows
        if row.frame >= start
        and getattr(row, field) == "continue_mission"
    )


def summarize(
    rows: list[ComparisonRow],
) -> Metrics:
    fault_start, physical_recovery = _fault_window(rows)

    baseline_safe = _first_non_continue(
        rows,
        start=fault_start,
        field="baseline_action",
    )

    aegisland_safe = _first_non_continue(
        rows,
        start=fault_start,
        field="aegisland_action",
    )

    baseline_recovery = _first_continue(
        rows,
        start=physical_recovery,
        field="baseline_action",
    )

    aegisland_recovery = _first_continue(
        rows,
        start=physical_recovery,
        field="aegisland_action",
    )

    baseline_unsafe = sum(
        row.fault
        and row.baseline_action == "continue_mission"
        for row in rows
    )

    aegisland_unsafe = sum(
        row.fault
        and row.aegisland_action == "continue_mission"
        for row in rows
    )

    return Metrics(
        fault_start=fault_start,
        physical_recovery=physical_recovery,
        baseline_safe_frame=baseline_safe,
        aegisland_safe_frame=aegisland_safe,
        baseline_time_to_safe=baseline_safe - fault_start,
        aegisland_time_to_safe=aegisland_safe - fault_start,
        baseline_unsafe_continuation=baseline_unsafe,
        aegisland_unsafe_continuation=aegisland_unsafe,
        baseline_recovery_frame=baseline_recovery,
        aegisland_recovery_frame=aegisland_recovery,
        baseline_recovery_latency=(
            baseline_recovery - physical_recovery
        ),
        aegisland_recovery_latency=(
            aegisland_recovery - physical_recovery
        ),
    )


def main() -> None:
    timeline = run_fault_timeline(
        "gps_denied_camera_failure"
    )

    rows = build_comparison(timeline)
    metrics = summarize(rows)

    print()
    print("BASELINE VS AEGISLAND ABLATION")
    print("=" * 72)

    print(
        f"fault_start_frame="
        f"{metrics.fault_start}"
    )

    print(
        f"physical_recovery_frame="
        f"{metrics.physical_recovery}"
    )

    print("-" * 72)

    print(
        f"{'metric':38}"
        f"{'baseline':>14}"
        f"{'aegisland':>14}"
    )

    print("-" * 72)

    values = (
        (
            "safe_state_frame",
            metrics.baseline_safe_frame,
            metrics.aegisland_safe_frame,
        ),
        (
            "time_to_safe_state_frames",
            metrics.baseline_time_to_safe,
            metrics.aegisland_time_to_safe,
        ),
        (
            "unsafe_continuation_frames",
            metrics.baseline_unsafe_continuation,
            metrics.aegisland_unsafe_continuation,
        ),
        (
            "recovery_frame",
            metrics.baseline_recovery_frame,
            metrics.aegisland_recovery_frame,
        ),
        (
            "recovery_latency_frames",
            metrics.baseline_recovery_latency,
            metrics.aegisland_recovery_latency,
        ),
    )

    for name, baseline, aegisland in values:
        print(
            f"{name:38}"
            f"{baseline:>14}"
            f"{aegisland:>14}"
        )

    print("=" * 72)

    print()
    print("FAULT WINDOW")
    print("-" * 92)

    print(
        "frame fault raw_valid raw_conf "
        "baseline             aegisland"
    )

    for row in rows:
        if 38 <= row.frame <= 61:
            print(
                f"{row.frame:3d} "
                f"{row.fault!s:5} "
                f"{row.raw_valid!s:9} "
                f"{row.raw_confidence:8.3f} "
                f"{row.baseline_action:20} "
                f"{row.aegisland_action}"
            )


if __name__ == "__main__":
    main()
