from math import isfinite

from aegisland.temporal import TemporalRiskAssessor


def test_temporal_risk_increases_for_object_approaching_target() -> None:
    assessor = TemporalRiskAssessor(
        measurement_noise=1.0,
    )

    target = (200.0, 100.0)

    observations = [
        (100.0, 100.0),
        (120.0, 100.0),
        (140.0, 100.0),
        (160.0, 100.0),
        (180.0, 100.0),
    ]

    result = None

    for observation in observations:
        result = assessor.observe(
            object_center=observation,
            target_center=target,
        )

    assert result is not None
    assert result.track is not None
    assert result.ttc is not None

    assert result.track.vx > 0.0
    assert result.ttc.approaching
    assert isfinite(result.ttc.ttc_frames)
    assert result.risk > 0.0


def test_object_moving_away_has_low_temporal_risk() -> None:
    assessor = TemporalRiskAssessor(
        measurement_noise=1.0,
    )

    target = (200.0, 100.0)

    observations = [
        (160.0, 100.0),
        (140.0, 100.0),
        (120.0, 100.0),
        (100.0, 100.0),
    ]

    result = None

    for observation in observations:
        result = assessor.observe(
            object_center=observation,
            target_center=target,
        )

    assert result is not None
    assert result.ttc is not None

    assert result.track.vx < 0.0
    assert not result.ttc.approaching
    assert result.risk == 0.0


def test_tracker_predicts_through_one_missing_observation() -> None:
    assessor = TemporalRiskAssessor(
        measurement_noise=1.0,
    )

    target = (300.0, 100.0)

    assessor.observe(
        object_center=(100.0, 100.0),
        target_center=target,
    )

    assessor.observe(
        object_center=(120.0, 100.0),
        target_center=target,
    )

    assessor.observe(
        object_center=(140.0, 100.0),
        target_center=target,
    )

    result = assessor.observe(
        object_center=None,
        target_center=target,
    )

    assert not result.observed
    assert result.track is not None
    assert result.track.x > 140.0


def test_no_observation_before_initialization_has_zero_risk() -> None:
    assessor = TemporalRiskAssessor()

    result = assessor.observe(
        object_center=None,
        target_center=(200.0, 200.0),
    )

    assert result.track is None
    assert result.ttc is None
    assert result.risk == 0.0
