from __future__ import annotations

import csv
from pathlib import Path

from .evaluation import evaluate_scenario
from .simulator import SCENARIOS


def main() -> None:
    output = Path(
        "runs/evaluation/camera_motion_ablation.csv"
    )
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    print("CAMERA MOTION COMPENSATION ABLATION")
    print("=" * 78)

    for name in sorted(SCENARIOS):
        without = evaluate_scenario(
            name,
            camera_compensation=False,
        )

        with_compensation = evaluate_scenario(
            name,
            camera_compensation=True,
        )

        motion_reduction_pct = 0.0

        if without.max_motion_risk > 1e-9:
            motion_reduction_pct = (
                1.0
                - with_compensation.max_motion_risk
                / without.max_motion_risk
            ) * 100.0

        row = {
            "scenario": name,
            "motion_risk_without": round(
                without.max_motion_risk,
                4,
            ),
            "motion_risk_with": round(
                with_compensation.max_motion_risk,
                4,
            ),
            "motion_reduction_pct": round(
                motion_reduction_pct,
                2,
            ),
            "action_changes_without": (
                without.action_changes
            ),
            "action_changes_with": (
                with_compensation.action_changes
            ),
            "avoidable_unsafe_without": (
                without.avoidable_unsafe_landing_frames
            ),
            "avoidable_unsafe_with": (
                with_compensation.avoidable_unsafe_landing_frames
            ),
            "latency_without_ms": round(
                without.mean_processing_ms,
                2,
            ),
            "latency_with_ms": round(
                with_compensation.mean_processing_ms,
                2,
            ),
        }

        rows.append(row)

        print(
            f"{name:30} "
            f"motion {row['motion_risk_without']:.3f}"
            f" -> {row['motion_risk_with']:.3f} | "
            f"reduction={row['motion_reduction_pct']:6.2f}% | "
            f"changes {row['action_changes_without']}"
            f" -> {row['action_changes_with']}"
        )

    with output.open(
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

    print()
    print(f"Ablation written to: {output}")


if __name__ == "__main__":
    main()
