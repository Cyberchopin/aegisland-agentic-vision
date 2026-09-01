from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .perception_quality import PerceptionQualityMonitor
from .perception_trust import PerceptionTrustGate


@dataclass(frozen=True, slots=True)
class SceneCase:
    name: str
    frame: np.ndarray
    expected_failure: str
    expected_authority: str


@dataclass(frozen=True, slots=True)
class SceneProbeResult:
    name: str
    expected_failure: str
    observed_failure: str
    expected_authority: str
    observed_authority: str
    quality_score: float
    entropy_bits: float
    sharpness: float
    feature_count: int
    passed: bool


def _random_texture() -> np.ndarray:
    rng = np.random.default_rng(42)

    gray = rng.integers(
        30,
        225,
        size=(240, 320),
        dtype=np.uint8,
    )

    return cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR,
    )


def _checkerboard() -> np.ndarray:
    height = 240
    width = 320
    tile = 20

    yy, xx = np.indices(
        (height, width)
    )

    board = (
        ((xx // tile) + (yy // tile))
        % 2
    )

    gray = np.where(
        board == 0,
        55,
        205,
    ).astype(np.uint8)

    return cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR,
    )


def _edge_grid() -> np.ndarray:
    frame = np.full(
        (240, 320, 3),
        125,
        dtype=np.uint8,
    )

    for x in range(20, 320, 40):
        cv2.line(
            frame,
            (x, 0),
            (x, 239),
            (45, 45, 45),
            3,
        )

    for y in range(20, 240, 40):
        cv2.line(
            frame,
            (0, y),
            (319, y),
            (215, 215, 215),
            3,
        )

    return frame


def _gradient_shapes() -> np.ndarray:
    gradient = np.tile(
        np.linspace(
            45,
            215,
            320,
            dtype=np.uint8,
        ),
        (240, 1),
    )

    frame = cv2.cvtColor(
        gradient,
        cv2.COLOR_GRAY2BGR,
    )

    cv2.rectangle(
        frame,
        (35, 35),
        (115, 105),
        (70, 70, 70),
        4,
    )

    cv2.circle(
        frame,
        (225, 75),
        38,
        (225, 225, 225),
        4,
    )

    cv2.line(
        frame,
        (30, 190),
        (290, 135),
        (65, 65, 65),
        5,
    )

    return frame


def _landing_pad() -> np.ndarray:
    frame = np.full(
        (240, 320, 3),
        115,
        dtype=np.uint8,
    )

    center = (
        160,
        120,
    )

    cv2.circle(
        frame,
        center,
        78,
        (210, 210, 210),
        5,
    )

    cv2.circle(
        frame,
        center,
        42,
        (55, 55, 55),
        5,
    )

    cv2.line(
        frame,
        (95, 120),
        (225, 120),
        (220, 220, 220),
        5,
    )

    cv2.line(
        frame,
        (160, 55),
        (160, 185),
        (220, 220, 220),
        5,
    )

    return frame


def _facade() -> np.ndarray:
    frame = np.full(
        (240, 320, 3),
        155,
        dtype=np.uint8,
    )

    for y in range(25, 220, 55):
        for x in range(25, 300, 55):
            cv2.rectangle(
                frame,
                (x, y),
                (x + 30, y + 28),
                (55, 55, 55),
                -1,
            )

            cv2.rectangle(
                frame,
                (x, y),
                (x + 30, y + 28),
                (220, 220, 220),
                2,
            )

    return frame


def _low_texture_wall() -> np.ndarray:
    return np.full(
        (240, 320, 3),
        128,
        dtype=np.uint8,
    )


def build_scenes() -> tuple[SceneCase, ...]:
    return (
        SceneCase(
            "random_texture",
            _random_texture(),
            "healthy",
            "full",
        ),
        SceneCase(
            "checkerboard",
            _checkerboard(),
            "healthy",
            "full",
        ),
        SceneCase(
            "edge_grid",
            _edge_grid(),
            "healthy",
            "full",
        ),
        SceneCase(
            "gradient_shapes",
            _gradient_shapes(),
            "healthy",
            "full",
        ),
        SceneCase(
            "landing_pad",
            _landing_pad(),
            "healthy",
            "full",
        ),
        SceneCase(
            "facade",
            _facade(),
            "healthy",
            "full",
        ),
        SceneCase(
            "low_texture_wall",
            _low_texture_wall(),
            "texture_degenerate",
            "revoked",
        ),
    )


def probe() -> list[SceneProbeResult]:
    monitor = PerceptionQualityMonitor()
    gate = PerceptionTrustGate()

    results: list[SceneProbeResult] = []

    for scene in build_scenes():
        gray = cv2.cvtColor(
            scene.frame,
            cv2.COLOR_BGR2GRAY,
        )

        quality = monitor.analyze(
            gray,
            has_reference=False,
            match_count=0,
            inlier_ratio=0.0,
            compensation_used=False,
        )

        trust = gate.evaluate(
            failure_type=quality.failure_type.value,
            quality_score=quality.quality_score,
            localization_confidence=0.90,
            localization_valid=True,
        )

        passed = (
            quality.failure_type.value
            == scene.expected_failure
            and trust.authority.value
            == scene.expected_authority
        )

        results.append(
            SceneProbeResult(
                name=scene.name,
                expected_failure=(
                    scene.expected_failure
                ),
                observed_failure=(
                    quality.failure_type.value
                ),
                expected_authority=(
                    scene.expected_authority
                ),
                observed_authority=(
                    trust.authority.value
                ),
                quality_score=(
                    quality.quality_score
                ),
                entropy_bits=(
                    quality.entropy_bits
                ),
                sharpness=(
                    quality.sharpness
                ),
                feature_count=(
                    quality.feature_count
                ),
                passed=passed,
            )
        )

    return results


def main() -> None:
    results = probe()

    print()
    print("PERCEPTION SCENE BASELINE PROBE")
    print("=" * 118)

    for result in results:
        status = (
            "PASS"
            if result.passed
            else "FAIL"
        )

        print(
            f"{status:4} "
            f"{result.name:20} "
            f"failure={result.observed_failure:20} "
            f"authority={result.observed_authority:8} "
            f"Q={result.quality_score:.4f} "
            f"H={result.entropy_bits:.3f} "
            f"sharp={result.sharpness:9.2f} "
            f"features={result.feature_count:3}"
        )

    failures = [
        result
        for result in results
        if not result.passed
    ]

    print("=" * 118)
    print(
        f"baseline_pass_rate="
        f"{(len(results) - len(failures)) / len(results):.4f}"
    )

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
