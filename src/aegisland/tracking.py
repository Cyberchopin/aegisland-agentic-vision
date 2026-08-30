from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TrackEstimate:
    x: float
    y: float
    vx: float
    vy: float
    predicted_x: float
    predicted_y: float


class KalmanTracker2D:
    """
    Constant-velocity Kalman filter.

    State:
        [x, y, vx, vy]

    Measurement:
        [x, y]
    """

    def __init__(
        self,
        *,
        dt: float = 1.0,
        process_noise: float = 1e-2,
        measurement_noise: float = 4.0,
    ) -> None:
        self.dt = dt

        self.state = np.zeros((4, 1), dtype=np.float64)

        self.covariance = np.eye(4, dtype=np.float64) * 100.0

        self.transition = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        self.measurement_matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        self.process_covariance = (
            np.eye(4, dtype=np.float64) * process_noise
        )

        self.measurement_covariance = (
            np.eye(2, dtype=np.float64) * measurement_noise
        )

        self.identity = np.eye(4, dtype=np.float64)

        self.initialized = False

    def initialize(
        self,
        x: float,
        y: float,
    ) -> None:
        self.state = np.array(
            [
                [x],
                [y],
                [0.0],
                [0.0],
            ],
            dtype=np.float64,
        )

        self.initialized = True

    def predict(self) -> tuple[float, float]:
        if not self.initialized:
            raise RuntimeError(
                "KalmanTracker2D must be initialized before predict()."
            )

        self.state = self.transition @ self.state

        self.covariance = (
            self.transition
            @ self.covariance
            @ self.transition.T
            + self.process_covariance
        )

        return (
            float(self.state[0, 0]),
            float(self.state[1, 0]),
        )

    def update(
        self,
        measured_x: float,
        measured_y: float,
    ) -> TrackEstimate:
        if not self.initialized:
            self.initialize(
                measured_x,
                measured_y,
            )

        predicted_x, predicted_y = self.predict()

        measurement = np.array(
            [
                [measured_x],
                [measured_y],
            ],
            dtype=np.float64,
        )

        innovation = (
            measurement
            - self.measurement_matrix @ self.state
        )

        innovation_covariance = (
            self.measurement_matrix
            @ self.covariance
            @ self.measurement_matrix.T
            + self.measurement_covariance
        )

        kalman_gain = (
            self.covariance
            @ self.measurement_matrix.T
            @ np.linalg.inv(innovation_covariance)
        )

        self.state = (
            self.state
            + kalman_gain @ innovation
        )

        self.covariance = (
            self.identity
            - kalman_gain @ self.measurement_matrix
        ) @ self.covariance

        x = float(self.state[0, 0])
        y = float(self.state[1, 0])
        vx = float(self.state[2, 0])
        vy = float(self.state[3, 0])

        next_state = (
            self.transition @ self.state
        )

        return TrackEstimate(
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            predicted_x=float(next_state[0, 0]),
            predicted_y=float(next_state[1, 0]),
        )
