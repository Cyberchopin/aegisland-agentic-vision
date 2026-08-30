from math import isinf

from aegisland.ttc import estimate_ttc


def test_object_moving_toward_target_has_finite_ttc() -> None:
    result = estimate_ttc(
        object_x=0.0,
        object_y=0.0,
        velocity_x=10.0,
        velocity_y=0.0,
        target_x=100.0,
        target_y=0.0,
    )

    assert result.approaching
    assert abs(result.closing_speed_px_per_frame - 10.0) < 1e-6
    assert abs(result.ttc_frames - 10.0) < 1e-6
    assert result.risk == 0.5


def test_object_moving_away_has_infinite_ttc() -> None:
    result = estimate_ttc(
        object_x=50.0,
        object_y=0.0,
        velocity_x=-10.0,
        velocity_y=0.0,
        target_x=100.0,
        target_y=0.0,
    )

    assert not result.approaching
    assert isinf(result.ttc_frames)
    assert result.risk == 0.0


def test_object_crossing_sideways_is_not_closing() -> None:
    result = estimate_ttc(
        object_x=0.0,
        object_y=0.0,
        velocity_x=0.0,
        velocity_y=10.0,
        target_x=100.0,
        target_y=0.0,
    )

    assert not result.approaching
    assert isinf(result.ttc_frames)
    assert result.risk == 0.0


def test_imminent_collision_is_critical() -> None:
    result = estimate_ttc(
        object_x=80.0,
        object_y=0.0,
        velocity_x=10.0,
        velocity_y=0.0,
        target_x=100.0,
        target_y=0.0,
    )

    assert result.approaching
    assert result.ttc_frames == 2.0
    assert result.risk == 1.0


def test_stationary_object_has_no_temporal_collision() -> None:
    result = estimate_ttc(
        object_x=20.0,
        object_y=20.0,
        velocity_x=0.0,
        velocity_y=0.0,
        target_x=100.0,
        target_y=100.0,
    )

    assert not result.approaching
    assert isinf(result.ttc_frames)
    assert result.risk == 0.0
