from aegisland.fusion import (
    DynamicConfidenceFusion,
    NavigationMode,
)


def test_gps_is_primary_when_healthy() -> None:
    fusion = DynamicConfidenceFusion()

    result = fusion.fuse(
        gps_confidence=1.0,
        visual_confidence=0.8,
        visual_valid=True,
    )

    assert result.mode == NavigationMode.GPS_PRIMARY
    assert "gps" in result.healthy_sources
    assert result.fused_confidence > 0.8


def test_visual_takes_over_when_gps_fails() -> None:
    fusion = DynamicConfidenceFusion()

    result = fusion.fuse(
        gps_confidence=0.0,
        visual_confidence=0.82,
        visual_valid=True,
    )

    assert result.mode == NavigationMode.VISUAL_FALLBACK
    assert result.fused_confidence == 0.82
    assert "gps" in result.degraded_sources
    assert "visual" in result.healthy_sources


def test_system_degrades_when_both_sources_are_unreliable() -> None:
    fusion = DynamicConfidenceFusion()

    result = fusion.fuse(
        gps_confidence=0.0,
        visual_confidence=0.20,
        visual_valid=False,
    )

    assert result.mode == NavigationMode.DEGRADED
    assert result.fused_confidence == 0.0
    assert "gps" in result.degraded_sources
    assert "visual" in result.degraded_sources


def test_invalid_confidence_values_are_clamped() -> None:
    fusion = DynamicConfidenceFusion()

    result = fusion.fuse(
        gps_confidence=1.5,
        visual_confidence=-0.5,
        visual_valid=True,
    )

    assert result.gps_confidence == 1.0
    assert result.visual_confidence == 0.0
    assert result.mode == NavigationMode.GPS_PRIMARY


def test_visual_inertial_fallback_when_gps_is_lost() -> None:
    fusion = DynamicConfidenceFusion()

    result = fusion.fuse(
        gps_confidence=0.0,
        visual_confidence=0.82,
        visual_valid=True,
        imu_confidence=0.95,
        imu_valid=True,
    )

    assert result.mode == NavigationMode.VISUAL_INERTIAL_FALLBACK
    assert "visual" in result.healthy_sources
    assert "imu" in result.healthy_sources
    assert "gps" in result.degraded_sources

    assert result.visual_weight > 0.0
    assert result.imu_weight > 0.0


def test_sensor_weights_transition_smoothly_after_gps_failure() -> None:
    fusion = DynamicConfidenceFusion(
        transition_rate=0.25,
    )

    healthy = fusion.fuse(
        gps_confidence=1.0,
        visual_confidence=0.80,
        visual_valid=True,
        imu_confidence=0.90,
        imu_valid=True,
    )

    degraded = fusion.fuse(
        gps_confidence=0.0,
        visual_confidence=0.80,
        visual_valid=True,
        imu_confidence=0.90,
        imu_valid=True,
    )

    assert healthy.mode == NavigationMode.GPS_PRIMARY
    assert degraded.mode == NavigationMode.VISUAL_INERTIAL_FALLBACK

    # GPS authority falls, but does not jump instantly to zero.
    assert 0.0 < degraded.gps_weight < healthy.gps_weight

    # Visual/inertial authority increases progressively.
    assert degraded.visual_weight >= healthy.visual_weight
    assert degraded.imu_weight >= healthy.imu_weight


def test_repeated_fusion_ticks_continue_fading_failed_gps_weight() -> None:
    fusion = DynamicConfidenceFusion(
        transition_rate=0.25,
    )

    fusion.fuse(
        gps_confidence=1.0,
        visual_confidence=0.8,
        visual_valid=True,
        imu_confidence=0.9,
        imu_valid=True,
    )

    first_fallback = fusion.fuse(
        gps_confidence=0.0,
        visual_confidence=0.8,
        visual_valid=True,
        imu_confidence=0.9,
        imu_valid=True,
    )

    later = first_fallback

    for _ in range(5):
        later = fusion.fuse(
            gps_confidence=0.0,
            visual_confidence=0.8,
            visual_valid=True,
            imu_confidence=0.9,
            imu_valid=True,
        )

    assert later.gps_weight < first_fallback.gps_weight
    assert later.imu_weight > first_fallback.imu_weight


def test_degraded_visual_cannot_establish_gps_denied_fallback() -> None:
    fusion = DynamicConfidenceFusion()

    result = fusion.fuse(
        gps_confidence=0.0,
        visual_confidence=0.70,
        visual_valid=True,
        imu_confidence=0.95,
        imu_valid=True,
        visual_health_state="degraded",
        imu_health_state="healthy",
    )

    assert result.mode == NavigationMode.DEGRADED
    assert "visual" in result.degraded_sources
    assert result.visual_weight == 0.0
    assert result.imu_weight == 0.0
    assert result.fused_confidence == 0.0


def test_degraded_visual_can_assist_healthy_gps_primary() -> None:
    fusion = DynamicConfidenceFusion()

    result = fusion.fuse(
        gps_confidence=1.0,
        visual_confidence=0.70,
        visual_valid=True,
        imu_confidence=0.95,
        imu_valid=True,
        gps_health_state="healthy",
        visual_health_state="degraded",
        imu_health_state="healthy",
    )

    assert result.mode == NavigationMode.GPS_PRIMARY
    assert "visual" in result.degraded_sources
    assert result.visual_weight > 0.0


def test_degraded_mode_clears_stale_authority_weights() -> None:
    fusion = DynamicConfidenceFusion()

    healthy = fusion.fuse(
        gps_confidence=0.0,
        visual_confidence=1.0,
        visual_valid=True,
        imu_confidence=0.95,
        imu_valid=True,
    )

    degraded = fusion.fuse(
        gps_confidence=0.0,
        visual_confidence=0.0,
        visual_valid=False,
        imu_confidence=0.95,
        imu_valid=True,
        visual_health_state="failed",
        imu_health_state="healthy",
    )

    assert (
        healthy.mode
        == NavigationMode.VISUAL_INERTIAL_FALLBACK
    )
    assert degraded.mode == NavigationMode.DEGRADED
    assert degraded.gps_weight == 0.0
    assert degraded.visual_weight == 0.0
    assert degraded.imu_weight == 0.0
    assert degraded.fused_confidence == 0.0
