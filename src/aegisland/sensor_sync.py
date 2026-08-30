from __future__ import annotations

from dataclasses import dataclass

from .time_sync import (
    SyncMethod,
    TimeSyncBuffer,
)


@dataclass(frozen=True, slots=True)
class SynchronizedSensorSnapshot:
    timestamp_s: float

    visual_x: float | None
    visual_y: float | None
    visual_confidence: float

    imu_yaw_rad: float | None
    imu_yaw_rate_rad_s: float | None
    imu_ax: float | None
    imu_ay: float | None

    visual_sync_method: SyncMethod
    imu_sync_method: SyncMethod

    visual_valid: bool
    imu_valid: bool

    @property
    def fully_synchronized(self) -> bool:
        return self.visual_valid and self.imu_valid


class SensorSynchronizer:
    """
    Time-align visual localization and IMU state.

    Visual vector:
        x, y, confidence

    IMU vector:
        yaw, yaw_rate, ax, ay
    """

    def __init__(
        self,
        *,
        max_samples: int = 512,
        max_extrapolation_s: float = 0.05,
    ) -> None:
        self.visual = TimeSyncBuffer(
            max_samples=max_samples,
            max_extrapolation_s=max_extrapolation_s,
        )

        self.imu = TimeSyncBuffer(
            max_samples=max_samples,
            max_extrapolation_s=max_extrapolation_s,
        )

    def add_visual(
        self,
        *,
        timestamp_s: float,
        x: float,
        y: float,
        confidence: float,
    ) -> None:
        self.visual.add(
            timestamp_s,
            (
                x,
                y,
                confidence,
            ),
        )

    def add_imu(
        self,
        *,
        timestamp_s: float,
        yaw_rad: float,
        yaw_rate_rad_s: float,
        ax: float,
        ay: float,
    ) -> None:
        self.imu.add(
            timestamp_s,
            (
                yaw_rad,
                yaw_rate_rad_s,
                ax,
                ay,
            ),
        )

    def snapshot(
        self,
        timestamp_s: float,
    ) -> SynchronizedSensorSnapshot:
        visual = self.visual.sample_at(
            timestamp_s
        )

        imu = self.imu.sample_at(
            timestamp_s
        )

        visual_values = visual.values
        imu_values = imu.values

        return SynchronizedSensorSnapshot(
            timestamp_s=timestamp_s,

            visual_x=(
                visual_values[0]
                if visual_values is not None
                else None
            ),
            visual_y=(
                visual_values[1]
                if visual_values is not None
                else None
            ),
            visual_confidence=(
                visual_values[2]
                if visual_values is not None
                else 0.0
            ),

            imu_yaw_rad=(
                imu_values[0]
                if imu_values is not None
                else None
            ),
            imu_yaw_rate_rad_s=(
                imu_values[1]
                if imu_values is not None
                else None
            ),
            imu_ax=(
                imu_values[2]
                if imu_values is not None
                else None
            ),
            imu_ay=(
                imu_values[3]
                if imu_values is not None
                else None
            ),

            visual_sync_method=visual.method,
            imu_sync_method=imu.method,

            visual_valid=visual.valid,
            imu_valid=imu.valid,
        )
