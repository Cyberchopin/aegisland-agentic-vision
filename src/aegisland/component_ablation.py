from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

from .fault_timeline import FaultTimelineRow, run_fault_timeline

CONTINUE = "continue_mission"
HOLD = "hold_and_scan"


@dataclass(frozen=True, slots=True)
class VariantMetrics:
    name: str
    safe_state_frame: int
    time_to_safe_state_frames: int
    unsafe_continuation_frames: int
    recovery_frame: int
    recovery_latency_frames: int
    false_interventions_before_fault: int
    action_transitions: int


@dataclass(frozen=True, slots=True)
class MatrixResult:
    fault_start_frame: int
    physical_recovery_frame: int
    variants: tuple[VariantMetrics, ...]


def raw_baseline_action(
    row: FaultTimelineRow,
    *,
    visual_gate: float = 0.55,
) -> str:
    if row.gps_available:
        return CONTINUE

    if (
        row.visual_localization_valid
        and row.visual_raw_confidence >= visual_gate
    ):
        return CONTINUE

    return HOLD


def semantic_trust_action(
    row: FaultTimelineRow,
    *,
    visual_gate: float = 0.55,
) -> str:
    if row.gps_available:
        return CONTINUE

    if (
        row.visual_localization_trusted
        and row.visual_trust_confidence >= visual_gate
    ):
        return CONTINUE

    return HOLD


def _fault_window(
    rows: list[FaultTimelineRow],
) -> tuple[int, int]:
    fault_start = next(
        row.frame
        for row in rows
        if row.camera_fault_active
    )

    seen_fault = False

    for row in rows:
        if row.camera_fault_active:
            seen_fault = True
            continue

        if seen_fault:
            return fault_start, row.frame

    raise RuntimeError("Camera recovery frame not found.")


def _metrics(
    *,
    name: str,
    rows: list[FaultTimelineRow],
    actions: list[str],
    fault_start: int,
    physical_recovery: int,
) -> VariantMetrics:
    safe_frame = next(
        row.frame
        for row, action in zip(rows, actions)
        if row.frame >= fault_start
        and action != CONTINUE
    )

    recovery_frame = next(
        row.frame
        for row, action in zip(rows, actions)
        if row.frame >= physical_recovery
        and action == CONTINUE
    )

    unsafe = sum(
        row.camera_fault_active
        and action == CONTINUE
        for row, action in zip(rows, actions)
    )

    false_interventions = sum(
        row.frame < fault_start
        and action != CONTINUE
        for row, action in zip(rows, actions)
    )

    transitions = sum(
        previous != current
        for previous, current in pairwise(actions)
    )

    return VariantMetrics(
        name=name,
        safe_state_frame=safe_frame,
        time_to_safe_state_frames=(
            safe_frame - fault_start
        ),
        unsafe_continuation_frames=unsafe,
        recovery_frame=recovery_frame,
        recovery_latency_frames=(
            recovery_frame - physical_recovery
        ),
        false_interventions_before_fault=(
            false_interventions
        ),
        action_transitions=transitions,
    )


def run_matrix(
    rows: list[FaultTimelineRow] | None = None,
) -> MatrixResult:
    if rows is None:
        rows = run_fault_timeline(
            "gps_denied_camera_failure"
        )

    fault_start, physical_recovery = (
        _fault_window(rows)
    )

    raw_actions = [
        raw_baseline_action(row)
        for row in rows
    ]

    trust_actions = [
        semantic_trust_action(row)
        for row in rows
    ]

    pre_stabilizer_actions = [
        row.raw_action
        for row in rows
    ]

    full_actions = [
        row.action
        for row in rows
    ]

    variants = (
        _metrics(
            name="raw_baseline",
            rows=rows,
            actions=raw_actions,
            fault_start=fault_start,
            physical_recovery=physical_recovery,
        ),
        _metrics(
            name="semantic_trust",
            rows=rows,
            actions=trust_actions,
            fault_start=fault_start,
            physical_recovery=physical_recovery,
        ),
        _metrics(
            name="pre_stabilizer_stack",
            rows=rows,
            actions=pre_stabilizer_actions,
            fault_start=fault_start,
            physical_recovery=physical_recovery,
        ),
        _metrics(
            name="full_aegisland",
            rows=rows,
            actions=full_actions,
            fault_start=fault_start,
            physical_recovery=physical_recovery,
        ),
    )

    return MatrixResult(
        fault_start_frame=fault_start,
        physical_recovery_frame=(
            physical_recovery
        ),
        variants=variants,
    )


def main() -> None:
    result = run_matrix()

    print()
    print("AEGISLAND COMPONENT ABLATION MATRIX")
    print("=" * 108)

    print(
        f"fault_start_frame="
        f"{result.fault_start_frame}"
    )

    print(
        f"physical_recovery_frame="
        f"{result.physical_recovery_frame}"
    )

    print("-" * 108)

    print(
        f"{'variant':24}"
        f"{'safe':>8}"
        f"{'safe_lat':>11}"
        f"{'unsafe':>10}"
        f"{'recover':>10}"
        f"{'rec_lat':>10}"
        f"{'false_int':>12}"
        f"{'transitions':>13}"
    )

    print("-" * 108)

    for variant in result.variants:
        print(
            f"{variant.name:24}"
            f"{variant.safe_state_frame:>8}"
            f"{variant.time_to_safe_state_frames:>11}"
            f"{variant.unsafe_continuation_frames:>10}"
            f"{variant.recovery_frame:>10}"
            f"{variant.recovery_latency_frames:>10}"
            f"{variant.false_interventions_before_fault:>12}"
            f"{variant.action_transitions:>13}"
        )

    print("=" * 108)

    print()
    print("TRUST TRANSITION WINDOW")
    print("-" * 112)

    print(
        "frame fault failure_type          "
        "raw_conf authority trusted trust_conf "
        "raw_action            final_action"
    )

    rows = run_fault_timeline(
        "gps_denied_camera_failure"
    )

    for row in rows:
        if 38 <= row.frame <= 61:
            print(
                f"{row.frame:3d} "
                f"{row.camera_fault_active!s:5} "
                f"{row.perception_failure_type:21} "
                f"{row.visual_raw_confidence:8.3f} "
                f"{row.visual_localization_authority:9} "
                f"{row.visual_localization_trusted!s:7} "
                f"{row.visual_trust_confidence:10.3f} "
                f"{row.raw_action:21} "
                f"{row.action}"
            )


if __name__ == "__main__":
    main()
