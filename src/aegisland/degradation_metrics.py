"""
Scoring helpers for "gradual degradation" studies: does an online signal warn
before the localization error starts growing?

Labels come from ground truth (offline only). Signals must be causal: the value
at frame i may use data up to frame i only. Nothing here talks to a vehicle.

Definitions used throughout:
- drift rate at frame i: translational relative-pose error between frame i and
  the first frame >= horizon later, divided by the horizon (m of error per s)
- event: a run of frames whose drift rate exceeds the label threshold; runs
  separated by less than `merge_gap_s` are merged, shorter than `min_length_s`
  are dropped
- an event is detected when an alarm frame falls in
  [onset - lead_window_s, end + horizon_s]; lead = onset - first such alarm
  (positive = warned early, negative = late)
- clean frames: valid-label frames outside every [onset - lead_window_s,
  end + horizon_s + 1 s]; the false-alarm rate is the alarmed fraction of them
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Event:
    onset_s: float
    end_s: float
    peak_drift_mps: float


def rotation_log(rotation: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle) of a 3x3 rotation matrix."""
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    skew = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    if angle < 1e-8:
        return 0.5 * skew
    if np.pi - angle < 1e-6:
        # Near 180 degrees: axis from the symmetric part.
        axis = np.sqrt(np.maximum((np.diag(rotation) + 1.0) / 2.0, 0.0))
        return axis * angle
    return angle / (2.0 * np.sin(angle)) * skew


def interpolate_poses(
    times_s: np.ndarray, poses: np.ndarray, query_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Linear position + normalized-lerp rotation interpolation at query times.

    Returns (poses (M, 4, 4), valid mask); queries outside the time range are invalid.
    Intended for dense references (200 Hz ground truth), where nlerp error is negligible.
    """
    index = np.searchsorted(times_s, query_s)
    valid = (index > 0) & (index < len(times_s))
    out = np.repeat(np.eye(4)[None], len(query_s), axis=0)
    for k in np.flatnonzero(valid):
        i0, i1 = index[k] - 1, index[k]
        weight = (query_s[k] - times_s[i0]) / (times_s[i1] - times_s[i0])
        out[k, :3, 3] = (1 - weight) * poses[i0, :3, 3] + weight * poses[i1, :3, 3]
        q0, q1 = _matrix_to_quaternion(poses[i0, :3, :3]), _matrix_to_quaternion(poses[i1, :3, :3])
        if np.dot(q0, q1) < 0:
            q1 = -q1
        q = (1 - weight) * q0 + weight * q1
        out[k, :3, :3] = _quaternion_to_matrix(q / np.linalg.norm(q))
    return out, valid


def _matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """wxyz quaternion from a rotation matrix (Shepperd)."""
    trace = np.trace(rotation)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        return np.array(
            [
                0.25 * s,
                (rotation[2, 1] - rotation[1, 2]) / s,
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[1, 0] - rotation[0, 1]) / s,
            ]
        )
    i = int(np.argmax(np.diag(rotation)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + rotation[i, i] - rotation[j, j] - rotation[k, k])
    q = np.empty(4)
    q[0] = (rotation[k, j] - rotation[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (rotation[j, i] + rotation[i, j]) / s
    q[1 + k] = (rotation[k, i] + rotation[i, k]) / s
    return q


def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def forward_drift_rate(
    times_s: np.ndarray,
    estimate: np.ndarray,
    reference: np.ndarray,
    reference_valid: np.ndarray,
    *,
    horizon_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame translational and rotational RPE over [t_i, t_i + horizon], per second.

    Returns (translation m/s, rotation deg/s); NaN where the window is incomplete
    or the reference is invalid at either end.
    """
    count = len(times_s)
    translation = np.full(count, np.nan)
    rotation = np.full(count, np.nan)
    ends = np.searchsorted(times_s, times_s + horizon_s)
    for i in range(count):
        j = ends[i]
        if j >= count or not (reference_valid[i] and reference_valid[j]):
            continue
        span = times_s[j] - times_s[i]
        error = np.linalg.inv(np.linalg.inv(reference[i]) @ reference[j]) @ (
            np.linalg.inv(estimate[i]) @ estimate[j]
        )
        translation[i] = np.linalg.norm(error[:3, 3]) / span
        rotation[i] = np.degrees(np.linalg.norm(rotation_log(error[:3, :3]))) / span
    return translation, rotation


def causal_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Mean of the last `window` finite values up to and including each index."""
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1) : i + 1]
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size:
            out[i] = chunk.mean()
    return out


def find_events(
    times_s: np.ndarray,
    drift: np.ndarray,
    *,
    threshold: float,
    merge_gap_s: float,
    min_length_s: float,
) -> list[Event]:
    above = np.isfinite(drift) & (drift > threshold)
    runs: list[list[int]] = []
    for i in np.flatnonzero(above):
        if runs and times_s[i] - times_s[runs[-1][1]] <= merge_gap_s:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return [
        Event(float(times_s[a]), float(times_s[b]), float(np.nanmax(drift[a : b + 1])))
        for a, b in runs
        if times_s[b] - times_s[a] >= min_length_s
    ]


def clean_mask(
    times_s: np.ndarray,
    label_valid: np.ndarray,
    events: list[Event],
    *,
    lead_window_s: float,
    horizon_s: float,
) -> np.ndarray:
    mask = label_valid.copy()
    for event in events:
        mask &= ~(
            (times_s >= event.onset_s - lead_window_s)
            & (times_s <= event.end_s + horizon_s + 1.0)
        )
    return mask


def score_alarms(
    times_s: np.ndarray,
    alarm: np.ndarray,
    events: list[Event],
    clean: np.ndarray,
    *,
    lead_window_s: float,
    horizon_s: float,
) -> dict[str, object]:
    """Detection, lead time, miss rate and clean-frame false-alarm rate."""
    leads = []
    for event in events:
        window = (times_s >= event.onset_s - lead_window_s) & (
            times_s <= event.end_s + horizon_s
        )
        hits = np.flatnonzero(window & alarm)
        if hits.size:
            leads.append(event.onset_s - float(times_s[hits[0]]))
    clean_count = int(clean.sum())
    false_frames = int((alarm & clean).sum())
    onsets = np.flatnonzero(alarm & clean & ~np.concatenate([[False], (alarm & clean)[:-1]]))
    clean_minutes = clean_count * float(np.median(np.diff(times_s))) / 60.0 if clean_count else 0
    return {
        "events": len(events),
        "detected": len(leads),
        "miss_rate": None if not events else 1.0 - len(leads) / len(events),
        "lead_s": leads,
        "lead_s_median": float(np.median(leads)) if leads else None,
        "early_warnings": int(sum(lead > 0 for lead in leads)),
        "clean_frames": clean_count,
        "false_alarm_frame_rate": None if not clean_count else false_frames / clean_count,
        "false_alarm_episodes_per_min": None
        if not clean_minutes
        else len(onsets) / clean_minutes,
    }


def roc_auc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    """Threshold-free ranking quality (Mann-Whitney); 0.5 = useless, 1 = perfect."""
    finite = np.isfinite(scores)
    scores, positive = scores[finite], positive[finite]
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    # Average ranks for ties.
    start = 0
    for end in range(1, len(scores) + 1):
        if end == len(scores) or sorted_scores[end] != sorted_scores[start]:
            ranks[order[start:end]] = 0.5 * (start + end - 1) + 1
            start = end
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
