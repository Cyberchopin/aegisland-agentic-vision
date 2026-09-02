from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

from .fault_timeline import FaultTimelineRow, run_fault_timeline
from .fusion import (
    DynamicConfidenceFusion,
    FusionResult,
)

CONTINUE = "continue_mission"

NAVIGATION_AUTHORITY_MODES = {
    "visual_fallback",
    "visual_inertial_fallback",
}


class ConfidenceOnlyFusion(DynamicConfidenceFusion):
    """
    Counterfactual baseline.

    Uses the same current fusion implementation and thresholds, but
    deliberately ignores SensorHealth state when determining authority.

    This isolates the contribution of health-state-aware authority
    without changing confidence gates or weight-transition policy.
    """

    def fuse(
        self,
        *,
        gps_confidence: float,
        visual_confidence: float,
        visual_valid: bool,
        imu_confidence: float = 0.0,
        imu_valid: bool = False,
        gps_health_state: str = "healthy",
        visual_health_state: str = "healthy",
        imu_health_state: str = "healthy",
    ) -> FusionResult:
        del (
            gps_health_state,
            visual_health_state,
            imu_health_state,
        )

        return super().fuse(
            gps_confidence=gps_confidence,
            visual_confidence=visual_confidence,
            visual_valid=visual_valid,
            imu_confidence=imu_confidence,
            imu_valid=imu_valid,
            gps_health_state="healthy",
            visual_health_state="healthy",
            imu_health_state="healthy",
        )


@dataclass(frozen=True, slots=True)
class AuthorityMetrics:
    premature_authority_frames: int
    premature_raw_continue_frames: int
    raw_action_transitions: int

    stable_health_frame: int
    stable_raw_recovery_frame: int
    stable_final_recovery_frame: int

    raw_recovery_latency_frames: int
    final_recovery_latency_frames: int


@dataclass(frozen=True, slots=True)
class AuthorityAblation:
    confidence_only: AuthorityMetrics
    capability_aware: AuthorityMetrics


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


def _stable_continue_frame(
    rows: list[FaultTimelineRow],
    *,
    field: str,
    start: int,
) -> int:
    for index, row in enumerate(rows):
        if row.frame < start:
            continue

        if getattr(row, field) != CONTINUE:
            continue

        if all(
            getattr(candidate, field) == CONTINUE
            for candidate in rows[index:]
        ):
            return row.frame

    raise RuntimeError(
        f"Stable {field} recovery not found."
    )


def _metrics(
    rows: list[FaultTimelineRow],
) -> AuthorityMetrics:
    first_recovery = _first_recovery_attempt(rows)

    last_fault = max(
        row.frame
        for row in rows
        if row.camera_fault_active
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

    premature_authority = sum(
        row.navigation_mode
        in NAVIGATION_AUTHORITY_MODES
        for row in unstable_rows
    )

    premature_continue = sum(
        row.raw_action == CONTINUE
        for row in unstable_rows
    )

    recovery_rows = [
        row
        for row in rows
        if row.frame >= first_recovery
    ]

    raw_transitions = sum(
        previous.raw_action
        != current.raw_action
        for previous, current in pairwise(
            recovery_rows
        )
    )

    stable_health = next(
        row.frame
        for row in rows
        if (
            row.frame >= stable_physical
            and row.visual_health_state == "healthy"
        )
    )

    stable_raw = _stable_continue_frame(
        rows,
        field="raw_action",
        start=stable_physical,
    )

    stable_final = _stable_continue_frame(
        rows,
        field="action",
        start=stable_physical,
    )

    return AuthorityMetrics(
        premature_authority_frames=(
            premature_authority
        ),
        premature_raw_continue_frames=(
            premature_continue
        ),
        raw_action_transitions=raw_transitions,
        stable_health_frame=stable_health,
        stable_raw_recovery_frame=stable_raw,
        stable_final_recovery_frame=stable_final,
        raw_recovery_latency_frames=(
            stable_raw - stable_physical
        ),
        final_recovery_latency_frames=(
            stable_final - stable_physical
        ),
    )


def run_ablation() -> AuthorityAblation:
    confidence_only_rows = run_fault_timeline(
        "gps_denied_camera_flicker",
        confidence_fusion=ConfidenceOnlyFusion(),
    )

    capability_rows = run_fault_timeline(
        "gps_denied_camera_flicker",
        confidence_fusion=DynamicConfidenceFusion(),
    )

    return AuthorityAblation(
        confidence_only=_metrics(
            confidence_only_rows
        ),
        capability_aware=_metrics(
            capability_rows
        ),
    )


def main() -> None:
    result = run_ablation()

    print()
    print(
        "CAPABILITY-AWARE FUSION AUTHORITY ABLATION"
    )
    print("=" * 88)

    print(
        f"{'metric':42}"
        f"{'confidence-only':>22}"
        f"{'capability-aware':>22}"
    )

    print("-" * 88)

    fields = (
        "premature_authority_frames",
        "premature_raw_continue_frames",
        "raw_action_transitions",
        "stable_health_frame",
        "stable_raw_recovery_frame",
        "stable_final_recovery_frame",
        "raw_recovery_latency_frames",
        "final_recovery_latency_frames",
    )

    for field in fields:
        print(
            f"{field:42}"
            f"{getattr(result.confidence_only, field):>22}"
            f"{getattr(result.capability_aware, field):>22}"
        )

    print("=" * 88)


if __name__ == "__main__":
    main()
