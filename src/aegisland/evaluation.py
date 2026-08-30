from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from .agent import AegisLandAgent
from .domain import Action, SafetyLevel
from .perception import OpenCVLandingPerception
from .planner import SafetyPlanner
from .simulator import SCENARIOS, generate
from .trace import MemoryTraceStore


@dataclass(frozen=True, slots=True)
class ScenarioMetrics:
    scenario: str
    frames: int
    action_changes: int
    emergency_recovery_frames: int
    human_approval_frames: int
    unsafe_landing_frames: int
    safe_landing_frames: int
    forced_emergency_landing_frames: int
    avoidable_unsafe_landing_frames: int
    critical_frames: int
    first_critical_frame: int | None
    first_temporal_warning_frame: int | None
    max_risk_score: float
    max_motion_risk: float
    max_temporal_risk: float
    mean_processing_ms: float


def evaluate_scenario(
    name: str,
    *,
    camera_compensation: bool = True,
) -> ScenarioMetrics:
    scenario = SCENARIOS[name]

    perception = OpenCVLandingPerception()
    perception.motion_compensator.enabled = camera_compensation

    agent = AegisLandAgent(
        perception,
        SafetyPlanner(),
        MemoryTraceStore(),
    )

    events = []

    for frame_index, (frame, telemetry) in enumerate(generate(scenario)):
        event, _ = agent.step(
            frame,
            telemetry,
            frame_index,
        )
        events.append(event)

    action_changes = 0
    previous_action = None

    emergency_recovery_frames = 0
    human_approval_frames = 0
    unsafe_landing_frames = 0
    safe_landing_frames = 0
    forced_emergency_landing_frames = 0
    avoidable_unsafe_landing_frames = 0
    critical_frames = 0

    first_critical_frame = None
    first_temporal_warning_frame = None

    for event in events:
        action = event.decision.action

        if previous_action is not None and action != previous_action:
            action_changes += 1

        previous_action = action

        if action == Action.EMERGENCY_RECOVERY:
            emergency_recovery_frames += 1

        if event.decision.requires_human_approval:
            human_approval_frames += 1

        if event.decision.safety_level == SafetyLevel.CRITICAL:
            critical_frames += 1

            if first_critical_frame is None:
                first_critical_frame = event.evidence.frame_index

        if (
            first_temporal_warning_frame is None
            and event.evidence.temporal_risk >= 0.8
        ):
            first_temporal_warning_frame = event.evidence.frame_index

        if action in {
            Action.LAND,
            Action.EMERGENCY_LAND,
        }:
            target = next(
                (
                    candidate
                    for candidate in event.evidence.candidates
                    if candidate.zone_id
                    == event.decision.target_zone_id
                ),
                None,
            )

            if target is not None:
                if target.safe:
                    safe_landing_frames += 1

                else:
                    unsafe_landing_frames += 1

                    safe_zone_exists = any(
                        candidate.safe
                        for candidate in event.evidence.candidates
                    )

                    if (
                        action == Action.EMERGENCY_LAND
                        and not safe_zone_exists
                    ):
                        forced_emergency_landing_frames += 1
                    else:
                        avoidable_unsafe_landing_frames += 1

    return ScenarioMetrics(
        scenario=name,
        frames=len(events),
        action_changes=action_changes,
        emergency_recovery_frames=emergency_recovery_frames,
        human_approval_frames=human_approval_frames,
        unsafe_landing_frames=unsafe_landing_frames,
        safe_landing_frames=safe_landing_frames,
        forced_emergency_landing_frames=forced_emergency_landing_frames,
        avoidable_unsafe_landing_frames=avoidable_unsafe_landing_frames,
        critical_frames=critical_frames,
        first_critical_frame=first_critical_frame,
        first_temporal_warning_frame=first_temporal_warning_frame,
        max_risk_score=max(
            (event.decision.risk_score for event in events),
            default=0.0,
        ),
        max_motion_risk=max(
            (event.evidence.motion_risk for event in events),
            default=0.0,
        ),
        max_temporal_risk=max(
            (event.evidence.temporal_risk for event in events),
            default=0.0,
        ),
        mean_processing_ms=mean(
            event.evidence.processing_ms for event in events
        )
        if events
        else 0.0,
    )


def evaluate_all() -> list[ScenarioMetrics]:
    return [
        evaluate_scenario(name)
        for name in sorted(SCENARIOS)
    ]


def write_csv(
    metrics: list[ScenarioMetrics],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = [asdict(item) for item in metrics]

    if not rows:
        return

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        default="runs/evaluation/benchmark.csv",
    )

    args = parser.parse_args()

    metrics = evaluate_all()

    write_csv(
        metrics,
        Path(args.output),
    )

    for item in metrics:
        print(
            f"{item.scenario:30} "
            f"frames={item.frames:3d} "
            f"changes={item.action_changes:2d} "
            f"critical={item.critical_frames:3d} "
            f"temporal={item.max_temporal_risk:.2f} "
            f"unsafe_land={item.unsafe_landing_frames}"
        )

    print()
    print(f"Benchmark written to: {args.output}")


if __name__ == "__main__":
    main()
