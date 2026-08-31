from aegisland.time_sync import (
    SyncMethod,
    TimeSyncBuffer,
)


def test_out_of_order_samples_are_stored_in_timestamp_order() -> None:
    buffer = TimeSyncBuffer()

    buffer.add(2.0, (20.0,))
    buffer.add(0.0, (0.0,))
    buffer.add(1.0, (10.0,))

    timestamps = [
        sample.timestamp_s
        for sample in buffer.samples
    ]

    assert timestamps == [0.0, 1.0, 2.0]


def test_exact_timestamp_returns_exact_sample() -> None:
    buffer = TimeSyncBuffer()

    buffer.add(1.0, (10.0, 20.0))

    result = buffer.sample_at(1.0)

    assert result.valid
    assert result.method == SyncMethod.EXACT
    assert result.values == (10.0, 20.0)


def test_interpolates_between_sensor_samples() -> None:
    buffer = TimeSyncBuffer()

    buffer.add(0.0, (0.0, 0.0))
    buffer.add(1.0, (10.0, 20.0))

    result = buffer.sample_at(0.5)

    assert result.valid
    assert result.method == SyncMethod.INTERPOLATED

    assert result.values is not None
    assert abs(result.values[0] - 5.0) < 1e-6
    assert abs(result.values[1] - 10.0) < 1e-6


def test_short_horizon_extrapolation_uses_recent_velocity() -> None:
    buffer = TimeSyncBuffer(
        max_extrapolation_s=0.2,
    )

    buffer.add(0.0, (0.0,))
    buffer.add(1.0, (10.0,))

    result = buffer.sample_at(1.1)

    assert result.valid
    assert result.method == SyncMethod.EXTRAPOLATED
    assert result.values is not None

    assert abs(result.values[0] - 11.0) < 1e-6


def test_excessive_extrapolation_is_rejected() -> None:
    buffer = TimeSyncBuffer(
        max_extrapolation_s=0.05,
    )

    buffer.add(0.0, (0.0,))
    buffer.add(1.0, (10.0,))

    result = buffer.sample_at(1.2)

    assert not result.valid
    assert result.method == SyncMethod.UNAVAILABLE


def test_buffer_is_bounded() -> None:
    buffer = TimeSyncBuffer(
        max_samples=3,
    )

    buffer.add(0.0, (0.0,))
    buffer.add(1.0, (1.0,))
    buffer.add(2.0, (2.0,))
    buffer.add(3.0, (3.0,))

    assert len(buffer.samples) == 3

    assert [
        sample.timestamp_s
        for sample in buffer.samples
    ] == [1.0, 2.0, 3.0]


def test_out_of_order_insertions_are_counted() -> None:
    buffer = TimeSyncBuffer()

    buffer.add(2.0, (20.0,))
    buffer.add(1.0, (10.0,))
    buffer.add(3.0, (30.0,))

    assert buffer.out_of_order_insertions == 1
