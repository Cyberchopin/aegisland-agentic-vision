import pytest

from aegisland.perception_trust import (
    PerceptionAuthority,
    PerceptionTrustGate,
)


def evaluate(
    failure_type: str,
    *,
    confidence: float = 0.9,
    quality: float = 0.8,
    valid: bool = True,
):
    return PerceptionTrustGate().evaluate(
        failure_type=failure_type,
        quality_score=quality,
        localization_confidence=confidence,
        localization_valid=valid,
    )


def test_healthy_perception_receives_full_authority() -> None:
    result = evaluate("healthy")

    assert result.authority == PerceptionAuthority.FULL
    assert result.localization_trusted
    assert result.effective_confidence == 0.9


@pytest.mark.parametrize(
    "failure_type",
    [
        "overexposed",
        "underexposed",
        "texture_degenerate",
        "occlusion_suspected",
        "geometry_unstable",
    ],
)
def test_hard_semantic_failures_revoke_localization(
    failure_type: str,
) -> None:
    result = evaluate(
        failure_type,
        quality=0.1,
    )

    assert (
        result.authority
        == PerceptionAuthority.REVOKED
    )
    assert not result.localization_trusted
    assert result.effective_confidence == 0.0


def test_blur_reduces_but_does_not_revoke_authority() -> None:
    result = evaluate(
        "blurred",
        confidence=0.9,
        quality=0.4,
    )

    assert (
        result.authority
        == PerceptionAuthority.REDUCED
    )
    assert result.localization_trusted
    assert result.effective_confidence == pytest.approx(
        0.63
    )


def test_invalid_estimator_always_revokes_authority() -> None:
    result = evaluate(
        "healthy",
        valid=False,
    )

    assert (
        result.authority
        == PerceptionAuthority.REVOKED
    )
    assert not result.localization_trusted
    assert result.effective_confidence == 0.0


def test_unknown_diagnosis_preserves_legacy_behavior() -> None:
    result = evaluate(
        "unknown",
        confidence=0.82,
    )

    assert result.authority == PerceptionAuthority.FULL
    assert result.localization_trusted
    assert result.effective_confidence == 0.82


from aegisland.agent import AegisLandAgent
from aegisland.domain import Telemetry, VisionEvidence
from aegisland.planner import SafetyPlanner
from aegisland.trace import MemoryTraceStore


class TrustScenarioPerception:
    def __init__(
        self,
        failure_type: str,
        quality_score: float,
    ) -> None:
        self.failure_type = failure_type
        self.quality_score = quality_score

    def observe(
        self,
        frame,
        frame_index,
        *,
        active_perception=False,
    ):
        return (
            VisionEvidence(
                evidence_id="trust-scenario",
                frame_index=frame_index,
                confidence=0.90,
                obstacle_risk=0.0,
                motion_risk=0.0,
                visual_localization_valid=True,
                visual_localization_confidence=0.90,
                perception_failure_type=(
                    self.failure_type
                ),
                perception_quality_score=(
                    self.quality_score
                ),
            ),
            frame,
        )

    def enhance_for_active_perception(
        self,
        frame,
    ):
        return frame


def test_healthy_visual_localization_remains_available_without_gps() -> None:
    agent = AegisLandAgent(
        TrustScenarioPerception(
            "healthy",
            0.8,
        ),
        SafetyPlanner(),
        MemoryTraceStore(),
    )

    event = None

    for frame_index in range(3):
        event, _ = agent.step(
            object(),
            Telemetry(
                battery_percent=80,
                altitude_m=10,
                gps_available=False,
            ),
            frame_index,
        )

    assert event is not None

    assert (
        event.evidence.visual_localization_authority
        == "full"
    )

    assert (
        event.evidence.visual_localization_trusted
    )

    assert (
        event.evidence.visual_health_state
        == "healthy"
    )

    assert (
        event.evidence.navigation_mode
        == "visual_fallback"
    )


def test_semantic_failure_revokes_visual_authority_same_frame() -> None:
    agent = AegisLandAgent(
        TrustScenarioPerception(
            "overexposed",
            0.03,
        ),
        SafetyPlanner(),
        MemoryTraceStore(),
    )

    event, _ = agent.step(
        object(),
        Telemetry(
            battery_percent=80,
            altitude_m=10,
            gps_available=False,
        ),
        0,
    )

    assert (
        event.evidence.visual_localization_authority
        == "revoked"
    )

    assert not event.evidence.visual_localization_trusted
    assert event.evidence.visual_trust_confidence == 0.0

    assert (
        event.evidence.visual_health_state
        == "failed"
    )

    assert (
        event.evidence.navigation_mode
        == "degraded"
    )
