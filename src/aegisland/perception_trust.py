from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .perception_quality import PerceptionFailureType


class PerceptionAuthority(StrEnum):
    FULL = "full"
    REDUCED = "reduced"
    REVOKED = "revoked"


@dataclass(frozen=True, slots=True)
class PerceptionTrustResult:
    authority: PerceptionAuthority
    localization_trusted: bool

    raw_confidence: float
    effective_confidence: float

    failure_type: str
    quality_score: float
    reason: str


class PerceptionTrustGate:
    """
    Convert semantic perception diagnosis into capability-specific trust.

    Important:
        This layer does NOT choose flight actions.

        It decides whether visual LOCALIZATION is allowed to contribute
        to the navigation solution.

    Fail fast:
        Severe semantic failures revoke localization authority immediately.

    Recover conservatively:
        Recovery hysteresis remains the responsibility of
        SensorHealthMonitor downstream.
    """

    HARD_LOCALIZATION_FAILURES = frozenset(
        {
            PerceptionFailureType.OVEREXPOSED,
            PerceptionFailureType.UNDEREXPOSED,
            PerceptionFailureType.TEXTURE_DEGENERATE,
            PerceptionFailureType.OCCLUSION_SUSPECTED,
            PerceptionFailureType.GEOMETRY_UNSTABLE,
        }
    )

    def __init__(
        self,
        *,
        blur_confidence_scale: float = 0.70,
    ) -> None:
        if not 0.0 <= blur_confidence_scale <= 1.0:
            raise ValueError(
                "blur_confidence_scale must be in [0, 1]"
            )

        self.blur_confidence_scale = blur_confidence_scale

    def evaluate(
        self,
        *,
        failure_type: str,
        quality_score: float,
        localization_confidence: float,
        localization_valid: bool,
    ) -> PerceptionTrustResult:
        raw_confidence = self._clamp(
            localization_confidence
        )

        quality_score = self._clamp(
            quality_score
        )

        # Estimator itself already says the localization is invalid.
        if not localization_valid:
            return PerceptionTrustResult(
                authority=PerceptionAuthority.REVOKED,
                localization_trusted=False,
                raw_confidence=raw_confidence,
                effective_confidence=0.0,
                failure_type=failure_type,
                quality_score=quality_score,
                reason=(
                    "Visual localization estimator reported "
                    "an invalid state."
                ),
            )

        try:
            semantic_failure = PerceptionFailureType(
                failure_type
            )
        except ValueError:
            # Backwards-compatible path for synthetic tests or perception
            # tools that do not yet provide semantic self-diagnosis.
            return PerceptionTrustResult(
                authority=PerceptionAuthority.FULL,
                localization_trusted=True,
                raw_confidence=raw_confidence,
                effective_confidence=raw_confidence,
                failure_type=failure_type,
                quality_score=quality_score,
                reason=(
                    "No recognized semantic diagnosis; "
                    "retaining legacy localization confidence."
                ),
            )

        if (
            semantic_failure
            in self.HARD_LOCALIZATION_FAILURES
        ):
            return PerceptionTrustResult(
                authority=PerceptionAuthority.REVOKED,
                localization_trusted=False,
                raw_confidence=raw_confidence,
                effective_confidence=0.0,
                failure_type=semantic_failure.value,
                quality_score=quality_score,
                reason=(
                    "Semantic perception failure invalidates "
                    "visual localization authority."
                ),
            )

        if semantic_failure == PerceptionFailureType.BLURRED:
            effective = (
                raw_confidence
                * self.blur_confidence_scale
            )

            return PerceptionTrustResult(
                authority=PerceptionAuthority.REDUCED,
                localization_trusted=True,
                raw_confidence=raw_confidence,
                effective_confidence=round(
                    effective,
                    4,
                ),
                failure_type=semantic_failure.value,
                quality_score=quality_score,
                reason=(
                    "Blur reduces localization authority "
                    "without fully revoking it."
                ),
            )

        return PerceptionTrustResult(
            authority=PerceptionAuthority.FULL,
            localization_trusted=True,
            raw_confidence=raw_confidence,
            effective_confidence=raw_confidence,
            failure_type=semantic_failure.value,
            quality_score=quality_score,
            reason=(
                "Semantic perception diagnosis supports "
                "full visual localization authority."
            ),
        )

    @staticmethod
    def _clamp(value: float) -> float:
        return max(
            0.0,
            min(1.0, float(value)),
        )
