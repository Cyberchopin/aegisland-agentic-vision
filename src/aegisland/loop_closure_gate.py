"""
Accept legitimate loop-closure corrections, reject everything else.

Observed on EuRoC (cuVSLAM 17.0.1, MH_01/MH_05, sync and async SLAM):
  - every non-zero SLAM-vs-odometry correction happened on the frame where a
    new loop-closure pose appeared (get_loop_closure_poses);
  - between loop closures the SLAM increments equal the odometry increments;
  - corrections were at most ~0.14 m and ~1.1 degrees;
  - `lc_status` stays true for hundreds of frames, so it says nothing about a
    particular frame.

AnchoredSlamOutput therefore publishes  anchor ∘ (odom_anchor⁻¹ ∘ odom_k).
The output is continuous by construction. The anchor moves to the SLAM pose only
when ALL of these hold:
  1. the tracker reported a new loop closure on this frame;
  2. odometry itself did not jump (a plausible step);
  3. the correction is within budget: translation <= max(min_budget,
     drift_fraction * path length since the last anchor) and <= max_translation,
     and rotation <= max_rotation.
An accepted correction increments reset_counter, which is the field PX4 EKF2
uses to take a deliberate jump instead of fighting it.
Any other correction (no loop-closure event, oversized, or odometry jump) is
not applied and is reported, so the supervisor can raise it. A rejected offset
is remembered as a "foreign" world-frame offset of the SLAM stream and removed
before later corrections are judged, so it cannot leak in with the next
legitimate loop closure (found in SITL: it did, before this was added).

Fallback switch: GateConfig(pose_source="odometry") publishes plain odometry and
never touches reset_counter, i.e. the system behaves as if SLAM were off. This is
the one-line way back if loop closure misbehaves or costs too much on the Orin.

No pose is ever sent to hardware from here; this only shapes the shadow message.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

POSE_SOURCES = ("gated_slam", "odometry")


@dataclass(frozen=True)
class GateConfig:
    pose_source: str = "gated_slam"  # "odometry" = fallback, SLAM output ignored
    # Tuned on MH_04 only (MH_05 held out): loop closures come in bursts 0.5-1 s apart,
    # the second refining by 0.07-0.10 m right after an anchor update; the largest
    # legitimate correction seen on MH_04 was 0.21 m.
    min_budget_m: float = 0.15
    drift_fraction: float = 0.05  # 5 % of the path since the last anchor
    max_translation_m: float = 0.5
    max_rotation_deg: float = 3.0
    max_odometry_step_m: float = 0.25  # per frame; 5 m/s at 20 Hz
    continuity_m: float = 1e-3  # below this the SLAM/odometry difference is numerical noise
    continuity_deg: float = 0.05


@dataclass(frozen=True)
class GateDecision:
    output: np.ndarray  # world_from_body to publish
    reset_counter: int
    verdict: str  # "continuous" | "accepted_loop_closure" | rejected_*
    correction_m: float
    correction_deg: float
    budget_m: float


def _angle_deg(rotation: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(rotation) - 1.0) / 2.0))))


class AnchoredSlamOutput:
    def __init__(self, config: GateConfig | None = None):
        self.config = config or GateConfig()
        if self.config.pose_source not in POSE_SOURCES:
            raise ValueError(f"pose_source must be one of {POSE_SOURCES}")
        self.anchor: np.ndarray | None = None
        self.odom_anchor: np.ndarray | None = None
        self.previous_odom: np.ndarray | None = None
        self.path_since_anchor_m = 0.0
        self.reset_counter = 0
        self.foreign = np.eye(4)  # slam = foreign @ (what SLAM should have said)
        self.counts: dict[str, int] = {}

    def _decide(self, verdict, output, correction_m, correction_deg, budget) -> GateDecision:
        self.counts[verdict] = self.counts.get(verdict, 0) + 1
        return GateDecision(output, self.reset_counter, verdict, correction_m, correction_deg,
                            budget)

    def update(self, odom: np.ndarray, slam: np.ndarray, loop_closure_event: bool) -> GateDecision:
        """odom, slam: 4x4 world_from_body of the same frame; returns what to publish."""
        if self.config.pose_source == "odometry":
            return self._decide("odometry_only", odom.copy(), 0.0, 0.0, 0.0)
        if self.anchor is None:
            self.anchor, self.odom_anchor, self.previous_odom = slam.copy(), odom.copy(), odom
            return self._decide("continuous", slam.copy(), 0.0, 0.0, self.config.min_budget_m)
        step = float(np.linalg.norm(odom[:3, 3] - self.previous_odom[:3, 3]))
        self.previous_odom = odom
        self.path_since_anchor_m += step
        anchored = self.anchor @ np.linalg.inv(self.odom_anchor) @ odom
        cleaned = np.linalg.inv(self.foreign) @ slam
        correction = np.linalg.inv(anchored) @ cleaned
        correction_m = float(np.linalg.norm(correction[:3, 3]))
        correction_deg = _angle_deg(correction[:3, :3])
        config = self.config
        budget = min(config.max_translation_m,
                     max(config.min_budget_m, config.drift_fraction * self.path_since_anchor_m))
        if correction_m < config.continuity_m and correction_deg < config.continuity_deg:
            return self._decide("continuous", anchored, correction_m, correction_deg, budget)
        if not loop_closure_event:
            verdict = "rejected_no_loop_closure_event"
        elif step > config.max_odometry_step_m:
            verdict = "rejected_odometry_jump"
        elif correction_m > budget or correction_deg > config.max_rotation_deg:
            verdict = "rejected_oversized_correction"
        else:
            self.anchor, self.odom_anchor = cleaned.copy(), odom.copy()
            self.path_since_anchor_m = 0.0
            self.reset_counter += 1
            return self._decide("accepted_loop_closure", cleaned.copy(), correction_m,
                                correction_deg, budget)
        # Remember the rejected discrepancy so it is not judged (or applied) again.
        self.foreign = slam @ np.linalg.inv(anchored)
        return self._decide(verdict, anchored, correction_m, correction_deg, budget)
