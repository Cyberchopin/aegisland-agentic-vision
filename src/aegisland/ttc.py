from __future__ import annotations

from dataclasses import dataclass
from math import hypot, inf


@dataclass(frozen=True, slots=True)
class TTCResult:
    distance_px: float
    closing_speed_px_per_frame: float
    ttc_frames: float
    approaching: bool
    risk: float


def estimate_ttc(
    *,
    object_x: float,
    object_y: float,
    velocity_x: float,
    velocity_y: float,
    target_x: float,
    target_y: float,
) -> TTCResult:
    """
    Estimate time-to-collision with a target point.

    Units:
        position: pixels
        velocity: pixels / frame
        TTC: frames

    This is image-space TTC for simulation/prototyping.
    It is NOT real-world metric TTC in seconds.
    """

    dx = target_x - object_x
    dy = target_y - object_y

    distance = hypot(dx, dy)

    if distance < 1e-6:
        return TTCResult(
            distance_px=0.0,
            closing_speed_px_per_frame=0.0,
            ttc_frames=0.0,
            approaching=True,
            risk=1.0,
        )

    # Unit vector from object toward target.
    direction_x = dx / distance
    direction_y = dy / distance

    # Project velocity onto target direction.
    closing_speed = (
        velocity_x * direction_x
        + velocity_y * direction_y
    )

    if closing_speed <= 1e-6:
        return TTCResult(
            distance_px=distance,
            closing_speed_px_per_frame=closing_speed,
            ttc_frames=inf,
            approaching=False,
            risk=0.0,
        )

    ttc_frames = distance / closing_speed

    # Simple prototype temporal risk curve.
    if ttc_frames <= 3.0:
        risk = 1.0
    elif ttc_frames <= 8.0:
        risk = 0.8
    elif ttc_frames <= 15.0:
        risk = 0.5
    elif ttc_frames <= 30.0:
        risk = 0.25
    else:
        risk = 0.0

    return TTCResult(
        distance_px=distance,
        closing_speed_px_per_frame=closing_speed,
        ttc_frames=ttc_frames,
        approaching=True,
        risk=risk,
    )
