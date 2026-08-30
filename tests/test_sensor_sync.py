from aegisland.sensor_sync import SensorSynchronizer
from aegisland.time_sync import SyncMethod


def test_visual_and_imu_are_aligned_to_same_timestamp() -> None:
    sync = SensorSynchronizer()

    sync.add_visual(
        timestamp_s=1.0,
        x=10.0,
        y=20.0,
        confidence=0.8,
    )

    sync.add_visual(
        timestamp_s=2.0,
        x=20.0,
        y=30.0,
        confidence=0.9,
    )

    sync.add_imu(
        timestamp_s=1.0,
        yaw_rad=0.1,
        yaw_rate_rad_s=0.2,
        ax=1.0,
        ay=2.0,
    )

    sync.add_imu(
        timestamp_s=2.0,
        yaw_rad=0.3,
        yaw_rate_rad_s=0.4,
        ax=3.0,
        ay=4.0,
    )

    snapshot = sync.snapshot(1.5)

    assert snapshot.fully_synchronized

    assert snapshot.visual_sync_method == SyncMethod.INTERPOLATED
    assert snapshot.imu_sync_method == SyncMethod.INTERPOLATED

    assert abs(snapshot.visual_x - 15.0) < 1e-6
    assert abs(snapshot.visual_y - 25.0) < 1e-6

    assert abs(snapshot.imu_yaw_rad - 0.2) < 1e-6
    assert abs(snapshot.imu_ax - 2.0) < 1e-6


def test_out_of_order_imu_arrival_still_synchronizes_correctly() -> None:
    sync = SensorSynchronizer()

    sync.add_imu(
        timestamp_s=2.0,
        yaw_rad=0.2,
        yaw_rate_rad_s=0.0,
        ax=2.0,
        ay=0.0,
    )

    sync.add_imu(
        timestamp_s=1.0,
        yaw_rad=0.1,
        yaw_rate_rad_s=0.0,
        ax=1.0,
        ay=0.0,
    )

    result = sync.imu.sample_at(1.5)

    assert result.valid
    assert result.method == SyncMethod.INTERPOLATED
    assert result.values is not None

    assert abs(result.values[0] - 0.15) < 1e-6
    assert abs(result.values[2] - 1.5) < 1e-6


def test_snapshot_can_degrade_when_visual_data_is_missing() -> None:
    sync = SensorSynchronizer()

    sync.add_imu(
        timestamp_s=1.0,
        yaw_rad=0.1,
        yaw_rate_rad_s=0.2,
        ax=1.0,
        ay=2.0,
    )

    snapshot = sync.snapshot(1.0)

    assert not snapshot.visual_valid
    assert snapshot.imu_valid
    assert not snapshot.fully_synchronized

    assert snapshot.visual_confidence == 0.0
    assert snapshot.imu_yaw_rad == 0.1
