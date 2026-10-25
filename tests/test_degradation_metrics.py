import math

import numpy as np
import pytest

from aegisland.degradation_metrics import (
    causal_mean,
    clean_mask,
    find_events,
    forward_drift_rate,
    interpolate_poses,
    roc_auc,
    rotation_log,
    score_alarms,
)


def yaw_pose(yaw: float, x: float = 0.0) -> np.ndarray:
    pose = np.eye(4)
    c, s = math.cos(yaw), math.sin(yaw)
    pose[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    pose[0, 3] = x
    return pose


def test_rotation_log_recovers_axis_angle_including_near_pi() -> None:
    assert rotation_log(yaw_pose(0.3)[:3, :3]) == pytest.approx([0, 0, 0.3])
    assert np.linalg.norm(rotation_log(yaw_pose(math.pi - 1e-7)[:3, :3])) == pytest.approx(
        math.pi, abs=1e-5
    )


def test_interpolate_poses_midpoint_and_out_of_range() -> None:
    times = np.array([0.0, 1.0])
    poses = np.stack([yaw_pose(0.0, 0.0), yaw_pose(0.4, 2.0)])
    out, valid = interpolate_poses(times, poses, np.array([0.5, 2.0]))
    assert valid.tolist() == [True, False]
    assert out[0, 0, 3] == pytest.approx(1.0)
    assert rotation_log(out[0, :3, :3])[2] == pytest.approx(0.2, abs=1e-3)


def test_forward_drift_rate_is_zero_for_identical_and_measures_extra_motion() -> None:
    times = np.arange(0, 3.0, 0.05)
    reference = np.stack([yaw_pose(0.0, t) for t in times])
    drift, rotation = forward_drift_rate(
        times, reference, reference, np.ones(len(times), bool), horizon_s=1.0
    )
    assert np.nanmax(drift) == pytest.approx(0.0, abs=1e-12)
    assert np.nanmax(rotation) == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(drift[-1])  # incomplete window
    # Estimate moves 10% farther than the reference: 0.1 m of error per second.
    estimate = np.stack([yaw_pose(0.0, 1.1 * t) for t in times])
    drift, _ = forward_drift_rate(times, estimate, reference, np.ones(len(times), bool),
                                  horizon_s=1.0)
    assert np.nanmedian(drift) == pytest.approx(0.1, rel=1e-6)


def test_causal_mean_only_uses_past_values() -> None:
    values = np.array([1.0, np.nan, 3.0, 100.0])
    out = causal_mean(values, 2)
    assert out[:3].tolist() == [1.0, 1.0, 3.0]
    assert out[3] == pytest.approx(51.5)


def test_find_events_merges_short_gaps_and_drops_short_runs() -> None:
    times = np.arange(0, 10.0, 0.1)
    drift = np.zeros_like(times)
    drift[(times >= 2.0) & (times < 2.5)] = 0.2
    drift[(times >= 3.0) & (times < 3.5)] = 0.2  # 0.5 s gap -> merged
    drift[(times >= 8.0) & (times < 8.1)] = 0.2  # single frame -> dropped
    events = find_events(times, drift, threshold=0.05, merge_gap_s=1.0, min_length_s=0.25)
    assert len(events) == 1
    assert events[0].onset_s == pytest.approx(2.0)
    assert events[0].end_s == pytest.approx(3.4)


def test_score_alarms_lead_miss_and_false_alarm_rate() -> None:
    times = np.arange(0, 30.0, 0.1)
    drift = np.zeros_like(times)
    drift[(times >= 10) & (times < 11)] = 0.2
    drift[(times >= 20) & (times < 21)] = 0.2
    events = find_events(times, drift, threshold=0.05, merge_gap_s=1.0, min_length_s=0.25)
    clean = clean_mask(times, np.ones(len(times), bool), events, lead_window_s=3.0,
                       horizon_s=1.0)
    alarm = (times >= 8.0) & (times < 8.5)  # 2 s before the first event
    alarm |= (times >= 0.0) & (times < 0.3)  # a false alarm in clean time
    result = score_alarms(times, alarm, events, clean, lead_window_s=3.0, horizon_s=1.0)
    assert result["events"] == 2
    assert result["detected"] == 1
    assert result["miss_rate"] == pytest.approx(0.5)
    assert result["lead_s"] == pytest.approx([2.0])
    assert result["false_alarm_frame_rate"] == pytest.approx(3 / clean.sum())


def test_roc_auc_perfect_inverse_and_ties() -> None:
    labels = np.array([False, False, True, True])
    assert roc_auc(np.array([0.0, 1.0, 2.0, 3.0]), labels) == 1.0
    assert roc_auc(np.array([3.0, 2.0, 1.0, 0.0]), labels) == 0.0
    assert roc_auc(np.ones(4), labels) == 0.5
    assert roc_auc(np.ones(4), np.zeros(4, bool)) is None
