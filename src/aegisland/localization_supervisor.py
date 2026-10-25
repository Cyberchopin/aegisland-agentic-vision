"""
Read-only shadow supervisor for `aegisland.localization.v1` messages.

It classifies every message (and every quiet period) as ok / degraded / fault
with explicit reasons. It never produces a command: every assessment carries
CONTROL OUTPUT = DISABLED and COMMAND EXECUTED = FALSE.

Checks:
- contract violations (see localization_message.validate)
- stale data: now - capture_time > max_age
- capture time not strictly increasing (clock/ordering error)
- dropped frames (sequence gap) and frame-interval gaps
- no messages for longer than outage_timeout (camera / pipeline outage)
- tracking lost / quality -1
- pose jump: implied speed or angular rate above physical limits, unless the
  reset_counter changed (then it is reported as a relocalization)
- IMU vs camera clock offset: |gyro| cross-correlated against the angular speed
  implied by the poses over a sliding window
- (v3, degraded only) rotation_inconsistent: angular velocity implied by
  consecutive poses vs mean gyro over the same interval, after subtracting a
  causal running-median bias; 0.5 s mean above threshold
- (v3, degraded only) low_texture: feature matches reported for the frame by an
  independent front end (VisualDeadReckoner) below threshold, 0.5 s mean

The v3 thresholds were fitted on EuRoC MH_01/03/04 and checked on held-out
MH_02/05 (docs/dronehacks/degradation-study.md); they are dataset values, not
values for the competition camera. The gyro must be expressed in the message's
child (body) frame.

Default thresholds are starting points, not tuned values. The 200 ms stale and
400 ms outage defaults mirror PX4 EKF2's EV_MAX_INTERVAL and its 2x timeout.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .degradation_metrics import rotation_log
from .localization_diagnostics import angular_speed_from_poses, estimate_time_offset
from .localization_message import LocalizationMessage, TrackingState, validate
from .trajectory_eval import quaternion_to_matrix

OK, DEGRADED, FAULT = "ok", "degraded", "fault"
FAULT_REASONS = frozenset(
    {
        "stale",
        "capture_time_not_increasing",
        "no_messages",
        "tracking_lost",
        "pose_jump",
        "imu_time_offset",
    }
)
DEGRADATION_REASONS = frozenset({"rotation_inconsistent", "low_texture"})


@dataclass(frozen=True)
class SupervisorConfig:
    expected_period_ns: int
    max_age_ns: int = 200_000_000
    outage_timeout_ns: int = 400_000_000
    gap_factor: float = 1.5
    max_speed_mps: float = 5.0
    max_angular_rate_rad_s: float = 10.0
    imu_check_window_s: float = 5.0
    imu_check_every: int = 20
    imu_search_s: float = 0.15
    imu_step_s: float = 0.0025
    imu_max_offset_s: float = 0.02
    # EuRoC MH_02: wrong window estimates all had correlation < 0.85, good ones >= 0.9.
    imu_min_correlation: float = 0.9
    imu_min_motion_rad_s: float = 0.05
    imu_persistence: int = 2
    # v3 degradation checks; None disables a check.
    # EuRoC train fit (5% of clean training frames): 0.0145 stereo / 0.0139 inertial.
    rotation_residual_rad_s: float | None = 0.014
    # EuRoC train fit: 595 stereo / 599 inertial ORB matches (VDR, 752x480 images).
    min_feature_matches: float | None = 597.0
    degradation_smooth_frames: int = 10
    gyro_bias_frames: int = 200
    gyro_bias_min_frames: int = 20


@dataclass(frozen=True)
class Assessment:
    status: str
    reasons: tuple[str, ...]
    sequence: int | None
    capture_time_ns: int | None
    age_ms: float | None
    imu_offset_s: float | None = None
    rotation_residual_rad_s: float | None = None
    control_output: str = "DISABLED"
    command_executed: bool = False


class LocalizationShadowSupervisor:
    def __init__(self, config: SupervisorConfig) -> None:
        self.config = config
        self._last: LocalizationMessage | None = None
        self._last_valid: LocalizationMessage | None = None
        self._last_publish_ns: int | None = None
        self._imu: deque[tuple[float, float]] = deque()
        self._poses: deque[tuple[float, LocalizationMessage]] = deque()
        self._since_imu_check = 0
        self._imu_offset_s: float | None = None
        self._imu_offset_flagged = False
        self._imu_offset_streak = 0
        self._gyro: deque[tuple[float, np.ndarray]] = deque()
        self._rotation_differences: deque[np.ndarray] = deque(maxlen=config.gyro_bias_frames)
        self._rotation_residuals: deque[float] = deque(maxlen=config.degradation_smooth_frames)
        self._feature_matches: deque[float] = deque(maxlen=config.degradation_smooth_frames)
        self._rotation_residual: float | None = None

    def add_imu(self, timestamp_ns: int, gyro_rad_s: tuple[float, float, float]) -> None:
        self._imu.append((timestamp_ns / 1e9, float(np.linalg.norm(gyro_rad_s))))
        self._trim(self._imu, timestamp_ns / 1e9)
        self._gyro.append((timestamp_ns / 1e9, np.asarray(gyro_rad_s, dtype=float)))
        self._trim(self._gyro, timestamp_ns / 1e9)

    def tick(self, *, now_ns: int) -> Assessment:
        """Call periodically when no message arrived; detects outages."""
        if (
            self._last_publish_ns is not None
            and now_ns - self._last_publish_ns > self.config.outage_timeout_ns
        ):
            return Assessment(FAULT, ("no_messages",), None, None, None)
        return Assessment(OK, (), None, None, None)

    def assess(
        self,
        message: LocalizationMessage,
        *,
        now_ns: int,
        feature_matches: float | None = None,
    ) -> Assessment:
        """Assess one message; `feature_matches` = independent front-end matches (optional)."""
        reasons: list[str] = [f"contract:{error}" for error in validate(message)]
        age_ns = now_ns - message.capture_time_ns
        if age_ns > self.config.max_age_ns:
            reasons.append("stale")

        previous = self._last
        ordered = previous is None or message.capture_time_ns > previous.capture_time_ns
        if not ordered:
            reasons.append("capture_time_not_increasing")
        if previous is not None and ordered:
            if message.sequence - previous.sequence > 1:
                reasons.append("dropped_frames")
            elif (
                message.capture_time_ns - previous.capture_time_ns
                > self.config.gap_factor * self.config.expected_period_ns
            ):
                reasons.append("frame_interval_gap")

        lost = (
            not message.valid
            or message.quality == -1
            or message.tracking_state is TrackingState.LOST
        )
        if lost:
            reasons.append("tracking_lost")

        reset = previous is not None and message.reset_counter != previous.reset_counter
        if reset or message.tracking_state is TrackingState.RELOCALIZED:
            reasons.append("relocalized")
        elif not lost and ordered and self._last_valid is not None and self._jumped(message):
            reasons.append("pose_jump")

        if (
            not lost
            and ordered
            and "pose_jump" not in reasons
            and "relocalized" not in reasons
            and self._rotation_inconsistent(message)
        ):
            reasons.append("rotation_inconsistent")
        if self._low_texture(feature_matches if ordered else None):
            reasons.append("low_texture")

        if not lost and ordered:
            self._last_valid = message
            self._poses.append((message.capture_time_ns / 1e9, message))
            self._trim(self._poses, message.capture_time_ns / 1e9)
            self._since_imu_check += 1
            if self._since_imu_check >= self.config.imu_check_every:
                self._since_imu_check = 0
                self._check_imu_offset()
        if self._imu_offset_flagged:
            reasons.append("imu_time_offset")

        if ordered:
            self._last = message
        self._last_publish_ns = now_ns
        return Assessment(
            status=self._status(reasons),
            reasons=tuple(reasons),
            sequence=message.sequence,
            capture_time_ns=message.capture_time_ns,
            age_ms=age_ns / 1e6,
            imu_offset_s=self._imu_offset_s,
            rotation_residual_rad_s=self._rotation_residual,
        )

    @staticmethod
    def _status(reasons: list[str]) -> str:
        if any(reason in FAULT_REASONS or reason.startswith("contract:") for reason in reasons):
            return FAULT
        return DEGRADED if reasons else OK

    def _jumped(self, message: LocalizationMessage) -> bool:
        previous = self._last_valid
        dt = (message.capture_time_ns - previous.capture_time_ns) / 1e9
        if dt <= 0:
            return False
        step = math.dist(message.position_m, previous.position_m)
        relative = quaternion_to_matrix(np.array(previous.orientation_xyzw)).T @ (
            quaternion_to_matrix(np.array(message.orientation_xyzw))
        )
        angle = math.acos(max(-1.0, min(1.0, (np.trace(relative) - 1.0) / 2.0)))
        return (
            step / dt > self.config.max_speed_mps
            or angle / dt > self.config.max_angular_rate_rad_s
        )

    def _rotation_inconsistent(self, message: LocalizationMessage) -> bool:
        threshold = self.config.rotation_residual_rad_s
        previous = self._last_valid
        if threshold is None or previous is None:
            return False
        start_s, end_s = previous.capture_time_ns / 1e9, message.capture_time_ns / 1e9
        gyro = [g for t, g in self._gyro if start_s < t <= end_s]
        if end_s <= start_s or not gyro:
            return self._smoothed_residual_above(threshold)
        relative = quaternion_to_matrix(np.array(previous.orientation_xyzw)).T @ (
            quaternion_to_matrix(np.array(message.orientation_xyzw))
        )
        difference = rotation_log(relative) / (end_s - start_s) - np.mean(gyro, axis=0)
        history = list(self._rotation_differences)
        self._rotation_differences.append(difference)
        if len(history) < self.config.gyro_bias_min_frames:
            # Gyro bias not estimated yet (EuRoC: ~0.08 rad/s): a residual would only
            # measure the bias, so report nothing rather than a false warning.
            return False
        bias = np.median(history, axis=0)
        self._rotation_residuals.append(float(np.linalg.norm(difference - bias)))
        return self._smoothed_residual_above(threshold)

    def _smoothed_residual_above(self, threshold: float) -> bool:
        if not self._rotation_residuals:
            self._rotation_residual = None
            return False
        self._rotation_residual = float(np.mean(self._rotation_residuals))
        return self._rotation_residual > threshold

    def _low_texture(self, feature_matches: float | None) -> bool:
        threshold = self.config.min_feature_matches
        if threshold is None:
            return False
        if feature_matches is not None:
            self._feature_matches.append(float(feature_matches))
        return bool(self._feature_matches) and np.mean(self._feature_matches) < threshold

    def _trim(self, buffer: deque, newest_s: float) -> None:
        while buffer and newest_s - buffer[0][0] > self.config.imu_check_window_s:
            buffer.popleft()

    def _check_imu_offset(self) -> None:
        offset = self._estimate_imu_offset()
        if offset is None:
            # Not observable right now (hover, low correlation): unknown, not "bad".
            self._imu_offset_s = None
            self._imu_offset_streak = 0
            self._imu_offset_flagged = False
            return
        self._imu_offset_s = offset
        if abs(offset) > self.config.imu_max_offset_s:
            self._imu_offset_streak += 1
        else:
            self._imu_offset_streak = 0
        self._imu_offset_flagged = self._imu_offset_streak >= self.config.imu_persistence

    def _estimate_imu_offset(self) -> float | None:
        if len(self._imu) < 20 or len(self._poses) < 10:
            return None
        times = np.array([t for t, _ in self._poses])
        poses = np.stack([self._pose_matrix(m) for _, m in self._poses])
        mid_times, speeds = angular_speed_from_poses(times, poses)
        imu_times = np.array([t for t, _ in self._imu])
        imu_speed = np.array([s for _, s in self._imu])
        if np.std(speeds) < self.config.imu_min_motion_rad_s:
            return None
        try:
            # Reference = IMU averaged over one frame period (pose finite differences
            # are interval averages); positive offset = poses lag the IMU.
            result = estimate_time_offset(
                imu_times,
                imu_speed,
                mid_times,
                speeds,
                search_s=self.config.imu_search_s,
                step_s=self.config.imu_step_s,
                average_window_s=self.config.expected_period_ns / 1e9,
            )
        except ValueError:
            return None
        if result["correlation"] < self.config.imu_min_correlation:
            return None
        # Report how much later the IMU is stamped than the camera.
        return -float(result["offset_s"])

    @staticmethod
    def _pose_matrix(message: LocalizationMessage) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = quaternion_to_matrix(np.array(message.orientation_xyzw))
        pose[:3, 3] = message.position_m
        return pose
