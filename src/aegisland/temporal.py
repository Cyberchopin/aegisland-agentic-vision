from __future__ import annotations

from dataclasses import dataclass

from .tracking import KalmanTracker2D, TrackEstimate
from .ttc import TTCResult, estimate_ttc


@dataclass(frozen=True, slots=True)
class TemporalRisk:
    track: TrackEstimate | None
    ttc: TTCResult | None
    risk: float
    observed: bool


class TemporalRiskAssessor:
    """
    Single-object temporal risk estimator.

    Prototype pipeline:
        observation center
        -> Kalman tracking
        -> velocity estimate
        -> image-space TTC
        -> temporal risk

    This intentionally remains single-target for now.
    """

    def __init__(
        self,
        *,
        dt: float = 1.0,
        process_noise: float = 1e-2,
        measurement_noise: float = 4.0,
    ) -> None:
        self.tracker = KalmanTracker2D(
            dt=dt,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
        )

        self.last_track: TrackEstimate | None = None

    def observe(
        self,
        *,
        object_center: tuple[float, float] | None,
        target_center: tuple[float, float],
    ) -> TemporalRisk:
        if object_center is None:
            if not self.tracker.initialized:
                return TemporalRisk(
                    track=None,
                    ttc=None,
                    risk=0.0,
                    observed=False,
                )

            predicted_x, predicted_y = self.tracker.predict()

            if self.last_track is None:
                return TemporalRisk(
                    track=None,
                    ttc=None,
                    risk=0.0,
                    observed=False,
                )

            predicted_track = TrackEstimate(
                x=predicted_x,
                y=predicted_y,
                vx=float(self.tracker.state[2, 0]),
                vy=float(self.tracker.state[3, 0]),
                predicted_x=(
                    predicted_x
                    + float(self.tracker.state[2, 0])
                ),
                predicted_y=(
                    predicted_y
                    + float(self.tracker.state[3, 0])
                ),
            )

            self.last_track = predicted_track

            ttc = self._estimate(
                predicted_track,
                target_center,
            )

            return TemporalRisk(
                track=predicted_track,
                ttc=ttc,
                risk=ttc.risk,
                observed=False,
            )

        measured_x, measured_y = object_center

        track = self.tracker.update(
            measured_x,
            measured_y,
        )

        self.last_track = track

        ttc = self._estimate(
            track,
            target_center,
        )

        return TemporalRisk(
            track=track,
            ttc=ttc,
            risk=ttc.risk,
            observed=True,
        )

    def _estimate(
        self,
        track: TrackEstimate,
        target_center: tuple[float, float],
    ) -> TTCResult:
        target_x, target_y = target_center

        return estimate_ttc(
            object_x=track.x,
            object_y=track.y,
            velocity_x=track.vx,
            velocity_y=track.vy,
            target_x=target_x,
            target_y=target_y,
        )
