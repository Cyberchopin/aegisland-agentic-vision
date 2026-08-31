from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .perception_quality import (
    PerceptionFailureType,
    PerceptionQualityMonitor,
)
from .perception_trust import (
    PerceptionAuthority,
    PerceptionTrustGate,
)
from .sensor_faults import (
    CameraFault,
    CameraFaultInjector,
)


@dataclass(frozen=True, slots=True)
class FaultCaseResult:
    case: str
    expected_failure: str
    observed_failure: str
    expected_authority: str
    observed_authority: str
    quality_score: float
    passed: bool


def _textured_frame() -> np.ndarray:
    rng = np.random.default_rng(42)

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


def _diagnose(
    frame: np.ndarray,
    *,
    has_reference: bool = False,
    match_count: int = 0,
    inlier_ratio: float = 0.0,
    compensation_used: bool = False,
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


def run() -> list[FaultCaseResult]:
    injector = CameraFaultInjector()
    trust_gate = PerceptionTrustGate()

    normal = _textured_frame()

    cases = []

    def evaluate_case(
        *,
        name: str,
        frame: np.ndarray,
        expected_failure: PerceptionFailureType,
        expected_authority: PerceptionAuthority,
        has_reference: bool = False,
        match_count: int = 0,
        inlier_ratio: float = 0.0,
        compensation_used: bool = False,
    ) -> None:
        quality = _diagnose(
            frame,
            has_reference=has_reference,
            match_count=match_count,
            inlier_ratio=inlier_ratio,
            compensation_used=compensation_used,
        )

        trust = trust_gate.evaluate(
            failure_type=quality.failure_type.value,
            quality_score=quality.quality_score,
            localization_confidence=0.9,
            localization_valid=True,
        )

        cases.append(
            FaultCaseResult(
                case=name,
                expected_failure=expected_failure.value,
                observed_failure=quality.failure_type.value,
                expected_authority=expected_authority.value,
                observed_authority=trust.authority.value,
                quality_score=quality.quality_score,
                passed=(
                    quality.failure_type == expected_failure
                    and trust.authority == expected_authority
                ),
            )
        )

    evaluate_case(
        name="normal",
        frame=normal,
        expected_failure=PerceptionFailureType.HEALTHY,
        expected_authority=PerceptionAuthority.FULL,
    )

    evaluate_case(
        name="overexposure",
        frame=injector.apply(
            normal,
            CameraFault.OVEREXPOSURE,
            severity=1.0,
        ).frame,
        expected_failure=PerceptionFailureType.OVEREXPOSED,
        expected_authority=PerceptionAuthority.REVOKED,
    )

    evaluate_case(
        name="underexposure",
        frame=injector.apply(
            normal,
            CameraFault.UNDEREXPOSURE,
            severity=1.0,
        ).frame,
        expected_failure=PerceptionFailureType.UNDEREXPOSED,
        expected_authority=PerceptionAuthority.REVOKED,
    )

    evaluate_case(
        name="occlusion",
        frame=injector.apply(
            normal,
            CameraFault.OCCLUSION,
            severity=0.75,
        ).frame,
        expected_failure=(
            PerceptionFailureType.OCCLUSION_SUSPECTED
        ),
        expected_authority=PerceptionAuthority.REVOKED,
    )

    evaluate_case(
        name="blur",
        frame=injector.apply(
            normal,
            CameraFault.BLUR,
            severity=1.0,
        ).frame,
        expected_failure=PerceptionFailureType.BLURRED,
        expected_authority=PerceptionAuthority.REDUCED,
    )

    blank = np.full_like(
        normal,
        128,
    )

    evaluate_case(
        name="texture_degenerate",
        frame=blank,
        expected_failure=(
            PerceptionFailureType.TEXTURE_DEGENERATE
        ),
        expected_authority=PerceptionAuthority.REVOKED,
    )

    evaluate_case(
        name="geometry_unstable",
        frame=normal,
        expected_failure=(
            PerceptionFailureType.GEOMETRY_UNSTABLE
        ),
        expected_authority=PerceptionAuthority.REVOKED,
        has_reference=True,
        match_count=0,
        inlier_ratio=0.0,
        compensation_used=False,
    )

    return cases


def main() -> None:
    results = run()

    print()
    print("PERCEPTION FAULT DIAGNOSIS MATRIX")
    print("=" * 100)

    for result in results:
        status = "PASS" if result.passed else "FAIL"

        print(
            f"{status:4} "
            f"{result.case:20} "
            f"expected={result.expected_failure:22} "
            f"observed={result.observed_failure:22} "
            f"authority={result.observed_authority:8} "
            f"Q={result.quality_score:.4f}"
        )

    print("=" * 100)

    passed = sum(
        result.passed
        for result in results
    )

    total = len(results)

    print(
        f"accuracy={passed / total:.3f} "
        f"({passed}/{total})"
    )

    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
