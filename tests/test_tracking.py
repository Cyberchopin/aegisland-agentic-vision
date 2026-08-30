from aegisland.tracking import KalmanTracker2D


def test_tracker_estimates_forward_motion() -> None:
    tracker = KalmanTracker2D(
        dt=1.0,
        measurement_noise=1.0,
    )

    observations = [
        (100.0, 100.0),
        (110.0, 100.0),
        (120.0, 100.0),
        (130.0, 100.0),
        (140.0, 100.0),
    ]

    estimate = None

    for x, y in observations:
        estimate = tracker.update(x, y)

    assert estimate is not None

    assert estimate.vx > 5.0
    assert abs(estimate.vy) < 2.0

    assert estimate.predicted_x > estimate.x


def test_tracker_smooths_noisy_measurements() -> None:
    tracker = KalmanTracker2D(
        dt=1.0,
        measurement_noise=6.0,
    )

    observations = [
        (100.0, 100.0),
        (111.0, 99.0),
        (119.0, 102.0),
        (131.0, 98.0),
        (139.0, 101.0),
    ]

    estimate = None

    for x, y in observations:
        estimate = tracker.update(x, y)

    assert estimate is not None

    assert estimate.vx > 5.0
    assert abs(estimate.vy) < 3.0


def test_tracker_can_predict_missing_measurement() -> None:
    tracker = KalmanTracker2D()

    tracker.update(100.0, 100.0)
    tracker.update(110.0, 100.0)
    tracker.update(120.0, 100.0)

    before_x = float(tracker.state[0, 0])

    predicted_x, predicted_y = tracker.predict()

    assert predicted_x > before_x
    assert abs(predicted_y - 100.0) < 5.0
