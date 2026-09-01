from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from .perception_severity_benchmark import (
    DEFAULT_SEEDS,
    FAULT_FAMILIES,
    SeveritySweepResult,
    run,
)

FINE_SEVERITIES = tuple(
    round(index / 100.0, 2)
    for index in range(101)
)

EXPECTED_ENDPOINT_LABEL = {
    "overexposure": "overexposed",
    "underexposure": "underexposed",
    "occlusion": "occlusion_suspected",
    "blur": "blurred",
    "texture_degenerate": "texture_degenerate",
    "geometry_unstable": "geometry_unstable",
}


@dataclass(frozen=True, slots=True)
class BoundarySummary:
    fault_family: str

    detection_median: float | None
    detection_min: float | None
    detection_max: float | None

    authority_median: float | None
    authority_min: float | None
    authority_max: float | None


def _first_transition(
    rows: list[SeveritySweepResult],
    *,
    authority: bool,
) -> float | None:
    for row in sorted(
        rows,
        key=lambda item: item.severity,
    ):
        if authority:
            triggered = row.authority != "full"
        else:
            triggered = (
                row.observed_failure != "healthy"
            )

        if triggered:
            return row.severity

    return None


def calibrate(
    results: list[SeveritySweepResult],
) -> list[BoundarySummary]:
    summaries: list[BoundarySummary] = []

    seeds = sorted(
        {
            result.seed
            for result in results
        }
    )

    for family in FAULT_FAMILIES:
        detection_values: list[float] = []
        authority_values: list[float] = []

        for seed in seeds:
            rows = [
                result
                for result in results
                if result.seed == seed
                and result.fault_family == family
            ]

            detection = _first_transition(
                rows,
                authority=False,
            )

            authority = _first_transition(
                rows,
                authority=True,
            )

            if detection is not None:
                detection_values.append(detection)

            if authority is not None:
                authority_values.append(authority)

        summaries.append(
            BoundarySummary(
                fault_family=family,
                detection_median=(
                    median(detection_values)
                    if detection_values
                    else None
                ),
                detection_min=(
                    min(detection_values)
                    if detection_values
                    else None
                ),
                detection_max=(
                    max(detection_values)
                    if detection_values
                    else None
                ),
                authority_median=(
                    median(authority_values)
                    if authority_values
                    else None
                ),
                authority_min=(
                    min(authority_values)
                    if authority_values
                    else None
                ),
                authority_max=(
                    max(authority_values)
                    if authority_values
                    else None
                ),
            )
        )

    return summaries


def endpoint_confusion(
    results: list[SeveritySweepResult],
) -> Counter[tuple[str, str]]:
    matrix: Counter[tuple[str, str]] = Counter()

    for result in results:
        if result.severity != 1.0:
            continue

        matrix[
            (
                result.fault_family,
                result.observed_failure,
            )
        ] += 1

    return matrix


def _format(
    value: float | None,
) -> str:
    if value is None:
        return "none"

    return f"{value:.2f}"


def print_report(
    results: list[SeveritySweepResult],
    summaries: list[BoundarySummary],
) -> None:
    print()
    print("FINE PERCEPTION BOUNDARY CALIBRATION")
    print("=" * 104)

    clean = [
        result
        for result in results
        if result.severity == 0.0
    ]

    clean_false_diagnosis = sum(
        result.observed_failure != "healthy"
        for result in clean
    )

    clean_false_authority = sum(
        result.authority != "full"
        for result in clean
    )

    print(
        "evaluations="
        f"{len(results)}"
    )

    print(
        "clean_false_diagnosis_rate="
        f"{clean_false_diagnosis / len(clean):.4f}"
    )

    print(
        "clean_false_authority_rate="
        f"{clean_false_authority / len(clean):.4f}"
    )

    print("-" * 104)

    for summary in summaries:
        print(
            f"{summary.fault_family:20} "
            f"detect median={_format(summary.detection_median):5} "
            f"range=[{_format(summary.detection_min)}, "
            f"{_format(summary.detection_max)}] "
            f"authority median={_format(summary.authority_median):5} "
            f"range=[{_format(summary.authority_min)}, "
            f"{_format(summary.authority_max)}]"
        )

    print()
    print("ENDPOINT SEMANTIC CONFUSION")
    print("=" * 104)

    matrix = endpoint_confusion(results)

    total_correct = 0
    total = 0

    for family in FAULT_FAMILIES:
        expected = EXPECTED_ENDPOINT_LABEL[family]

        observed = Counter()

        for (
            injected_family,
            observed_label,
        ), count in matrix.items():
            if injected_family == family:
                observed[observed_label] += count

        correct = observed.get(
            expected,
            0,
        )

        family_total = sum(
            observed.values()
        )

        total_correct += correct
        total += family_total

        observed_text = ", ".join(
            f"{label}:{count}"
            for label, count in sorted(
                observed.items()
            )
        )

        print(
            f"{family:20} "
            f"expected={expected:22} "
            f"observed={observed_text}"
        )

    print("-" * 104)

    print(
        "endpoint_semantic_accuracy="
        f"{total_correct / total:.4f}"
    )

    endpoint_rows = [
        result
        for result in results
        if result.severity == 1.0
    ]

    authority_correct = sum(
        result.authority != "full"
        for result in endpoint_rows
    )

    print(
        "endpoint_authority_intervention_rate="
        f"{authority_correct / len(endpoint_rows):.4f}"
    )


def write_boundary_csv(
    summaries: list[BoundarySummary],
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
        writer = csv.writer(handle)

        writer.writerow(
            (
                "fault_family",
                "detection_median",
                "detection_min",
                "detection_max",
                "authority_median",
                "authority_min",
                "authority_max",
            )
        )

        for row in summaries:
            writer.writerow(
                (
                    row.fault_family,
                    row.detection_median,
                    row.detection_min,
                    row.detection_max,
                    row.authority_median,
                    row.authority_min,
                    row.authority_max,
                )
            )


def write_confusion_csv(
    results: list[SeveritySweepResult],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix = endpoint_confusion(results)

    observed_labels = sorted(
        {
            observed
            for _, observed in matrix
        }
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            (
                "injected_fault",
                *observed_labels,
            )
        )

        for family in FAULT_FAMILIES:
            writer.writerow(
                (
                    family,
                    *(
                        matrix.get(
                            (
                                family,
                                label,
                            ),
                            0,
                        )
                        for label in observed_labels
                    ),
                )
            )


def main() -> None:
    results = run(
        seeds=DEFAULT_SEEDS,
        severities=FINE_SEVERITIES,
    )

    summaries = calibrate(results)

    print_report(
        results,
        summaries,
    )

    boundary_path = Path(
        "runs/evaluation/"
        "perception_boundary_calibration.csv"
    )

    confusion_path = Path(
        "runs/evaluation/"
        "perception_endpoint_confusion.csv"
    )

    write_boundary_csv(
        summaries,
        boundary_path,
    )

    write_confusion_csv(
        results,
        confusion_path,
    )

    print()
    print(f"Boundary CSV:  {boundary_path}")
    print(f"Confusion CSV: {confusion_path}")


if __name__ == "__main__":
    main()
