from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class NavigationMode(StrEnum):
    GPS_PRIMARY = "gps_primary"
    VISUAL_INERTIAL_FALLBACK = "visual_inertial_fallback"
    VISUAL_FALLBACK = "visual_fallback"
    DEGRADED = "degraded"


@dataclass(frozen=True, slots=True)
class FusionResult:
    mode: NavigationMode
    fused_confidence: float
    gps_confidence: float
    visual_confidence: float
    imu_confidence: float
    gps_weight: float
    visual_weight: float
    imu_weight: float
    healthy_sources: tuple[str, ...]
    degraded_sources: tuple[str, ...]


class DynamicConfidenceFusion:
    """
    Confidence-aware navigation sensor fusion.

    Current prototype sources:
        - GPS
        - visual localization

    Future sources:
        - IMU
        - depth camera
        - LiDAR / range sensing

    This module selects a navigation mode and provides graceful
    degradation when one sensing source becomes unreliable.
    """

    def __init__(
        self,
        *,
        gps_weight: float = 0.65,
        visual_weight: float = 0.35,
        imu_weight: float = 0.0,
        gps_gate: float = 0.50,
        visual_gate: float = 0.55,
        imu_gate: float = 0.55,
        transition_rate: float = 0.25,
    ) -> None:
        self.gps_weight = gps_weight
        self.visual_weight = visual_weight
        self.imu_weight = imu_weight

        self.gps_gate = gps_gate
        self.visual_gate = visual_gate
        self.imu_gate = imu_gate

        self.transition_rate = max(
            0.0,
            min(1.0, transition_rate),
        )

        self._current_weights = (
            gps_weight,
            visual_weight,
            imu_weight,
        )
        self._initialized = False

    def fuse(
        self,
        *,
        gps_confidence: float,
        visual_confidence: float,
        visual_valid: bool,
        imu_confidence: float = 0.0,
        imu_valid: bool = False,
    ) -> FusionResult:
        gps_confidence = self._clamp(gps_confidence)
        visual_confidence = self._clamp(visual_confidence)
        imu_confidence = self._clamp(imu_confidence)

        gps_healthy = gps_confidence >= self.gps_gate

        visual_healthy = (
            visual_valid
            and visual_confidence >= self.visual_gate
        )

        imu_healthy = (
            imu_valid
            and imu_confidence >= self.imu_gate
        )

        healthy_sources: list[str] = []
        degraded_sources: list[str] = []

        for name, healthy in (
            ("gps", gps_healthy),
            ("visual", visual_healthy),
            ("imu", imu_healthy),
        ):
            if healthy:
                healthy_sources.append(name)
            else:
                degraded_sources.append(name)

        if gps_healthy:
            mode = NavigationMode.GPS_PRIMARY

            target_weights = (
                self.gps_weight,
                self.visual_weight if visual_healthy else 0.0,
                self.imu_weight if imu_healthy else 0.0,
            )

        elif visual_healthy and imu_healthy:
            mode = NavigationMode.VISUAL_INERTIAL_FALLBACK

            target_weights = (
                0.0,
                0.60,
                0.40,
            )

        elif visual_healthy:
            mode = NavigationMode.VISUAL_FALLBACK

            target_weights = (
                0.0,
                1.0,
                0.0,
            )

        else:
            mode = NavigationMode.DEGRADED

            target_weights = (
                0.0,
                0.0,
                0.0,
            )

        target_total = sum(target_weights)

        if target_total > 0.0:
            target_weights = tuple(
                weight / target_total
                for weight in target_weights
            )

        if not self._initialized:
            self._current_weights = target_weights
            self._initialized = True

        else:
            self._current_weights = tuple(
                current
                + self.transition_rate
                * (target - current)
                for current, target in zip(
                    self._current_weights,
                    target_weights,
                )
            )

        current_total = sum(self._current_weights)

        if current_total > 1e-9:
            effective_weights = tuple(
                weight / current_total
                for weight in self._current_weights
            )
        else:
            effective_weights = (
                0.0,
                0.0,
                0.0,
            )

        gps_effective, visual_effective, imu_effective = (
            effective_weights
        )

        fused_confidence = (
            gps_effective * gps_confidence
            + visual_effective * visual_confidence
            + imu_effective * imu_confidence
        )

        return FusionResult(
            mode=mode,
            fused_confidence=round(
                fused_confidence,
                4,
            ),
            gps_confidence=gps_confidence,
            visual_confidence=visual_confidence,
            imu_confidence=imu_confidence,
            gps_weight=round(gps_effective, 4),
            visual_weight=round(visual_effective, 4),
            imu_weight=round(imu_effective, 4),
            healthy_sources=tuple(healthy_sources),
            degraded_sources=tuple(degraded_sources),
        )

    @staticmethod
    def _clamp(value: float) -> float:
        return max(
            0.0,
            min(1.0, float(value)),
        )
