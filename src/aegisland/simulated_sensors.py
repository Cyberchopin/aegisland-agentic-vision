from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import AegisLandAgent


def ingest_out_of_order_imu(
    agent: AegisLandAgent,
    *,
    timestamp_s: float,
    frame_index: int,
) -> None:
    """
    Inject two IMU measurements around the camera timestamp
    in deliberately reversed arrival order.

    Arrival order:
        t + 5 ms
        t - 5 ms

    TimeSyncBuffer should reorder these by measurement timestamp
    and interpolate the IMU state at camera time t.
    """

    future = timestamp_s + 0.005
    past = timestamp_s - 0.005

    agent.ingest_imu(
        timestamp_s=future,
        yaw_rad=0.002 * frame_index + 0.00015,
        yaw_rate_rad_s=0.03,
        ax=0.02,
        ay=0.0,
    )

    agent.ingest_imu(
        timestamp_s=past,
        yaw_rad=0.002 * frame_index - 0.00015,
        yaw_rate_rad_s=0.03,
        ax=0.02,
        ay=0.0,
    )
