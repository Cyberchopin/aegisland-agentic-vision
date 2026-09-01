from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from .perception_quality import PerceptionQualityMonitor
from .perception_trust import (
    PerceptionAuthority,
    PerceptionTrustGate,
)
from .sensor_faults import (
    CameraFault,
    CameraFaultInjector,
)

DEFAULT_SEEDS = (
    42,
    43,
    44,
    45,
    46,
)

DEFAULT_SEVERITIES = tuple(
    round(index / 10.0, 1)
    for index in range(11)
)

FAULT_FAMILIES = (
    "overexposure",
    "underexposure",
    "occlusion",
    "blur",
    "texture_degenerate",
    "geometry_unstable",
)


@dataclass(frozen=True, slots=True)
class SeveritySweepResult:
    seed: int
    fault_family: str
    severity: float

    observed_failure: str
    authority: str

    quality_score: float
    localization_trusted: bool
    effective_confidence: float


def _textured_frame(
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    gray = rng.integers(
        25,
        230,
        size=(240, 320),
        dtype=np.uint8,
    )

    return cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR,
    )


def _texture_degenerate_frame(
    frame: np.ndarray,
    severity: float,
) -> np.ndarray:
    blank = np.full_like(
        frame,
        128,
    )

    return cv2.addWeighted(
        frame,
        1.0 - severity,
        blank,
        severity,
        0.0,
    )


def _camera_fault(
    family: str,
) -> CameraFault:
    mapping = {
        "overexposure": CameraFault.OVEREXPOSURE,
        "underexposure": CameraFault.UNDEREXPOSURE,
        "occlusion": CameraFault.OCCLUSION,
        "blur": CameraFault.BLUR,
    }

    return mapping[family]


def _diagnose(
    frame: np.ndarray,
    *,
    has_reference: bool,
    match_count: int,
    inlier_ratio: float,
    compensation_used: bool,
):
    monitor = PerceptionQualityMonitor()

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    return monitor.analyze(
        gray,
        has_reference=has_reference,
        match_count=match_count,
        inlier_ratio=inlier_ratio,
        compensation_used=compensation_used,
    )


def _geometry_metrics(
    severity: float,
) -> tuple[int, float]:
    match_count = round(
        80.0 * (1.0 - severity)
    )

    inlier_ratio = (
        0.90 * (1.0 - severity)
    )

    return (
        match_count,
        inlier_ratio,
    )


def run(
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    severities: tuple[float, ...] = DEFAULT_SEVERITIES,
) -> list[SeveritySweepResult]:
    injector = CameraFaultInjector()
    trust_gate = PerceptionTrustGate()

    results: list[SeveritySweepResult] = []

    for seed in seeds:
        normal = _textured_frame(seed)

        for family in FAULT_FAMILIES:
            for severity in severities:
                if family == "texture_degenerate":
                    frame = _texture_degenerate_frame(
                        normal,
                        severity,
                    )

                    has_reference = False
                    match_count = 0
                    inlier_ratio = 0.0
                    compensation_used = False

                elif family == "geometry_unstable":
                    frame = normal.copy()

                    (
                        match_count,
                        inlier_ratio,
                    ) = _geometry_metrics(
                        severity
                    )

                    has_reference = True
                    compensation_used = False

                else:
                    injected = injector.apply(
                        normal,
                        _camera_fault(family),
                        severity=severity,
                    )

                    frame = injected.frame

                    has_reference = False
                    match_count = 0
                    inlier_ratio = 0.0
                    compensation_used = False

                quality = _diagnose(
                    frame,
                    has_reference=has_reference,
                    match_count=match_count,
                    inlier_ratio=inlier_ratio,
                    compensation_used=compensation_used,
                )

                trust = trust_gate.evaluate(
                    failure_type=(
                        quality.failure_type.value
                    ),
                    quality_score=(
                        quality.quality_score
                    ),
                    localization_confidence=0.90,
                    localization_valid=True,
                )

                results.append(
                    SeveritySweepResult(
                        seed=seed,
                        fault_family=family,
                        severity=severity,
                        observed_failure=(
                            quality.failure_type.value
                        ),
                        authority=(
                            trust.authority.value
                        ),
                        quality_score=(
                            quality.quality_score
                        ),
                        localization_trusted=(
                            trust.localization_trusted
                        ),
                        effective_confidence=(
                            trust.effective_confidence
                        ),
                    )
                )

    return results


def _first_threshold(
    rows: list[SeveritySweepResult],
    *,
    authority: bool,
) -> float | None:
    ordered = sorted(
        rows,
        key=lambda row: row.severity,
    )

    for row in ordered:
        if authority:
            triggered = (
                row.authority
                != PerceptionAuthority.FULL.value
            )
        else:
            triggered = (
                row.observed_failure
                != "healthy"
            )

        if triggered:
            return row.severity

    return None


def summarize(
    results: list[SeveritySweepResult],
) -> None:
    print()
    print("PERCEPTION FAULT SEVERITY SWEEP")
    print("=" * 100)

    clean_rows = [
        row
        for row in results
        if row.severity == 0.0
    ]

    false_diagnoses = sum(
        row.observed_failure != "healthy"
        for row in clean_rows
    )

    false_authority_interventions = sum(
        row.authority
        != PerceptionAuthority.FULL.value
        for row in clean_rows
    )

    print(
        "clean_false_diagnosis_rate="
        f"{false_diagnoses / len(clean_rows):.4f}"
    )

    print(
        "clean_false_authority_rate="
        f"{false_authority_interventions / len(clean_rows):.4f}"
    )

    print("-" * 100)

    seeds = sorted(
        {
            row.seed
            for row in results
        }
    )

    for family in FAULT_FAMILIES:
        detection_thresholds: list[float] = []
        authority_thresholds: list[float] = []

        for seed in seeds:
            rows = [
                row
                for row in results
                if row.seed == seed
                and row.fault_family == family
            ]

            detection = _first_threshold(
                rows,
                authority=False,
            )

            authority = _first_threshold(
                rows,
                authority=True,
            )

            if detection is not None:
                detection_thresholds.append(
                    detection
                )

            if authority is not None:
                authority_thresholds.append(
                    authority
                )

        detection_text = (
            f"{median(detection_thresholds):.2f}"
            if detection_thresholds
            else "none"
        )

        authority_text = (
            f"{median(authority_thresholds):.2f}"
            if authority_thresholds
            else "none"
        )

        endpoint_rows = [
            row
            for row in results
            if row.fault_family == family
            and row.severity == 1.0
        ]

        endpoint_failures = sorted(
            {
                row.observed_failure
                for row in endpoint_rows
            }
        )

        endpoint_authorities = sorted(
            {
                row.authority
                for row in endpoint_rows
            }
        )

        print(
            f"{family:20} "
            f"detect@~{detection_text:5} "
            f"authority@~{authority_text:5} "
            f"endpoint={','.join(endpoint_failures):22} "
            f"authority={','.join(endpoint_authorities)}"
        )


def write_csv(
    results: list[SeveritySweepResult],
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
                "seed",
                "fault_family",
                "severity",
                "observed_failure",
                "authority",
                "quality_score",
                "localization_trusted",
                "effective_confidence",
            )
        )

        for row in results:
            writer.writerow(
                (
                    row.seed,
                    row.fault_family,
                    row.severity,
                    row.observed_failure,
                    row.authority,
                    row.quality_score,
                    row.localization_trusted,
                    row.effective_confidence,
                )
            )


def main() -> None:
    results = run()

    summarize(results)

    path = Path(
        "runs/evaluation/"
        "perception_fault_severity_sweep.csv"
    )

    write_csv(
        results,
        path,
    )

    print()
    print(f"CSV: {path}")


if __name__ == "__main__":
    main()
