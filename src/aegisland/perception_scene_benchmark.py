from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from .perception_quality import PerceptionQualityMonitor
from .perception_scenes import build_scenes
from .perception_severity_benchmark import FAULT_FAMILIES
from .perception_trust import PerceptionTrustGate
from .sensor_faults import CameraFault, CameraFaultInjector

SEVERITIES = tuple(
    round(index / 100.0, 2)
    for index in range(101)
)

HEALTHY_SCENES = tuple(
    scene
    for scene in build_scenes()
    if scene.expected_failure == "healthy"
)


@dataclass(frozen=True, slots=True)
class SceneFaultResult:
    scene: str
    fault_family: str
    severity: float
    observed_failure: str
    authority: str
    quality_score: float


def _camera_fault(
    family: str,
) -> CameraFault:
    return {
        "overexposure": CameraFault.OVEREXPOSURE,
        "underexposure": CameraFault.UNDEREXPOSURE,
        "occlusion": CameraFault.OCCLUSION,
        "blur": CameraFault.BLUR,
    }[family]


def _texture_degenerate(
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


def _geometry_metrics(
    severity: float,
) -> tuple[int, float]:
    return (
        round(80.0 * (1.0 - severity)),
        0.90 * (1.0 - severity),
    )


def _evaluate(
    *,
    frame: np.ndarray,
    family: str,
    severity: float,
    injector: CameraFaultInjector,
    monitor: PerceptionQualityMonitor,
    gate: PerceptionTrustGate,
) -> tuple[str, str, float]:
    has_reference = False
    match_count = 0
    inlier_ratio = 0.0
    compensation_used = False

    if family == "texture_degenerate":
        observed_frame = _texture_degenerate(
            frame,
            severity,
        )

    elif family == "geometry_unstable":
        observed_frame = frame.copy()

        (
            match_count,
            inlier_ratio,
        ) = _geometry_metrics(severity)

        has_reference = True

    else:
        observed_frame = injector.apply(
            frame,
            _camera_fault(family),
            severity=severity,
        ).frame

    gray = cv2.cvtColor(
        observed_frame,
        cv2.COLOR_BGR2GRAY,
    )

    quality = monitor.analyze(
        gray,
        has_reference=has_reference,
        match_count=match_count,
        inlier_ratio=inlier_ratio,
        compensation_used=compensation_used,
    )

    trust = gate.evaluate(
        failure_type=quality.failure_type.value,
        quality_score=quality.quality_score,
        localization_confidence=0.90,
        localization_valid=True,
    )

    return (
        quality.failure_type.value,
        trust.authority.value,
        quality.quality_score,
    )


def run(
    *,
    severities: tuple[float, ...] = SEVERITIES,
) -> list[SceneFaultResult]:
    injector = CameraFaultInjector()
    monitor = PerceptionQualityMonitor()
    gate = PerceptionTrustGate()

    results: list[SceneFaultResult] = []

    for scene in HEALTHY_SCENES:
        for family in FAULT_FAMILIES:
            for severity in severities:
                (
                    failure,
                    authority,
                    quality,
                ) = _evaluate(
                    frame=scene.frame,
                    family=family,
                    severity=severity,
                    injector=injector,
                    monitor=monitor,
                    gate=gate,
                )

                results.append(
                    SceneFaultResult(
                        scene=scene.name,
                        fault_family=family,
                        severity=severity,
                        observed_failure=failure,
                        authority=authority,
                        quality_score=quality,
                    )
                )

    return results


def _first_transition(
    rows: list[SceneFaultResult],
    *,
    authority: bool,
) -> float | None:
    for row in sorted(
        rows,
        key=lambda item: item.severity,
    ):
        triggered = (
            row.authority != "full"
            if authority
            else row.observed_failure != "healthy"
        )

        if triggered:
            return row.severity

    return None


def _authority_rank(
    authority: str,
) -> int:
    return {
        "full": 0,
        "reduced": 1,
        "revoked": 2,
    }[authority]


def _authority_reentry_violations(
    rows: list[SceneFaultResult],
) -> int:
    ordered = sorted(
        rows,
        key=lambda item: item.severity,
    )

    violations = 0
    previous_rank = -1

    for row in ordered:
        rank = _authority_rank(
            row.authority
        )

        if rank < previous_rank:
            violations += 1

        previous_rank = max(
            previous_rank,
            rank,
        )

    return violations


def print_report(
    results: list[SceneFaultResult],
) -> None:
    print()
    print("CROSS-SCENE PERCEPTION ROBUSTNESS")
    print("=" * 112)

    clean = [
        row
        for row in results
        if row.severity == 0.0
    ]

    false_diagnosis = sum(
        row.observed_failure != "healthy"
        for row in clean
    )

    false_authority = sum(
        row.authority != "full"
        for row in clean
    )

    print(f"healthy_scenes={len(HEALTHY_SCENES)}")
    print(f"evaluations={len(results)}")

    print(
        "clean_false_diagnosis_rate="
        f"{false_diagnosis / len(clean):.4f}"
    )

    print(
        "clean_false_authority_rate="
        f"{false_authority / len(clean):.4f}"
    )

    print("-" * 112)

    total_reentry = 0

    for family in FAULT_FAMILIES:
        detection: list[float] = []
        authority: list[float] = []

        for scene in HEALTHY_SCENES:
            rows = [
                row
                for row in results
                if row.scene == scene.name
                and row.fault_family == family
            ]

            detected = _first_transition(
                rows,
                authority=False,
            )

            intervened = _first_transition(
                rows,
                authority=True,
            )

            if detected is not None:
                detection.append(detected)

            if intervened is not None:
                authority.append(intervened)

            total_reentry += (
                _authority_reentry_violations(
                    rows
                )
            )

        print(
            f"{family:20} "
            f"detect median={median(detection):.2f} "
            f"range=[{min(detection):.2f}, {max(detection):.2f}] "
            f"authority median={median(authority):.2f} "
            f"range=[{min(authority):.2f}, {max(authority):.2f}]"
        )

    print("-" * 112)

    print(
        "authority_reentry_violations="
        f"{total_reentry}"
    )

    endpoint = [
        row
        for row in results
        if row.severity == 1.0
    ]

    endpoint_intervention = sum(
        row.authority != "full"
        for row in endpoint
    )

    print(
        "endpoint_authority_intervention_rate="
        f"{endpoint_intervention / len(endpoint):.4f}"
    )

    print()
    print("ENDPOINT LABEL DISTRIBUTION")
    print("=" * 112)

    for family in FAULT_FAMILIES:
        labels = Counter(
            row.observed_failure
            for row in endpoint
            if row.fault_family == family
        )

        text = ", ".join(
            f"{label}:{count}"
            for label, count in sorted(
                labels.items()
            )
        )

        print(
            f"{family:20} {text}"
        )


def write_csv(
    results: list[SceneFaultResult],
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
                "scene",
                "fault_family",
                "severity",
                "observed_failure",
                "authority",
                "quality_score",
            )
        )

        for row in results:
            writer.writerow(
                (
                    row.scene,
                    row.fault_family,
                    row.severity,
                    row.observed_failure,
                    row.authority,
                    row.quality_score,
                )
            )


def main() -> None:
    results = run()

    print_report(results)

    path = Path(
        "runs/evaluation/"
        "perception_cross_scene_robustness.csv"
    )

    write_csv(
        results,
        path,
    )

    print()
    print(f"CSV: {path}")


if __name__ == "__main__":
    main()
