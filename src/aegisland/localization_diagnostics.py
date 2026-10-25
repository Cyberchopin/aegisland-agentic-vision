"""
Scale-free diagnostics for comparing VisualDeadReckoner with a metric reference.

VisualDeadReckoner reports normalized image motion, not meters, so it cannot be
scored with ATE. These helpers compare it in the only space both share: the
image. A reference relative pose (ground truth or cuVSLAM) is turned into the
predicted motion of the scene point seen at the image center, and compared with
the dead reckoner's homography-derived center motion by direction.

Translation-induced flow depends on scene depth, which the dead reckoner never
knows. Callers must therefore report results for explicit depth assumptions
(`depth_m=None` = rotation only).

Evaluation-only; nothing here feeds the safety decision path.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence

import numpy as np

Projector = Callable[[np.ndarray], np.ndarray]


def predicted_center_flow(
    current_from_previous: np.ndarray,
    project: Projector,
    center_ray: np.ndarray,
    *,
    depth_m: float | None,
) -> np.ndarray:
    """Pixel motion (previous -> current) of the scene point on `center_ray`.

    `center_ray` is the camera-frame ray of the image-center pixel in the previous
    frame; `depth_m` is the assumed distance along it, or None for infinity.
    `project` maps (N, 3) camera-frame points to (N, 2) pixels.
    """
    ray = np.asarray(center_ray, dtype=np.float64)
    ray = ray / np.linalg.norm(ray)
    rotation = current_from_previous[:3, :3]
    if depth_m is None:
        previous = ray
        current = rotation @ ray
    else:
        previous = ray * depth_m
        current = rotation @ previous + current_from_previous[:3, 3]
    if current[2] <= 0:
        return np.array([np.nan, np.nan])
    pixels = project(np.stack([previous, current]))
    return pixels[1] - pixels[0]


def vdr_image_flow(
    vx: np.ndarray, vy: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Invert VisualDeadReckoner's `v = -center_shift / image_size` back to pixels."""
    return np.stack([-np.asarray(vx) * width, -np.asarray(vy) * height], axis=-1)


def direction_agreement(
    estimate_px: np.ndarray,
    reference_px: np.ndarray,
    *,
    minimum_reference_px: float,
) -> dict[str, float | int | None]:
    """Cosine agreement of 2D motion vectors; frames with tiny reference motion are skipped."""
    if estimate_px.shape != reference_px.shape or estimate_px.ndim != 2:
        raise ValueError("estimate and reference must share shape (N, 2)")
    reference_norm = np.linalg.norm(reference_px, axis=1)
    finite_estimate = np.all(np.isfinite(estimate_px), axis=1)
    finite_reference = np.all(np.isfinite(reference_px), axis=1)
    large_enough = finite_reference & (reference_norm >= minimum_reference_px)
    usable = large_enough & finite_estimate
    estimate = estimate_px[usable]
    reference = reference_px[usable]
    estimate_norm = np.linalg.norm(estimate, axis=1)
    cosine = np.where(
        estimate_norm > 0,
        np.sum(estimate * reference, axis=1)
        / np.maximum(estimate_norm * reference_norm[usable], 1e-12),
        0.0,
    )
    ratio = estimate_norm / reference_norm[usable]
    summary: dict[str, float | int | None] = {
        "used": int(usable.sum()),
        "skipped_small_reference": int((~large_enough).sum()),
        "skipped_missing_estimate": int((large_enough & ~finite_estimate).sum()),
        "minimum_reference_px": minimum_reference_px,
        "cosine_median": None,
        "cosine_mean": None,
        "fraction_same_half_plane": None,
        "fraction_within_45_deg": None,
        "magnitude_ratio_median": None,
    }
    if len(cosine):
        summary.update(
            cosine_median=float(np.median(cosine)),
            cosine_mean=float(np.mean(cosine)),
            fraction_same_half_plane=float(np.mean(cosine > 0)),
            fraction_within_45_deg=float(np.mean(cosine > math.cos(math.radians(45)))),
            magnitude_ratio_median=float(np.median(ratio)),
        )
    return summary


def angular_speed_from_poses(
    timestamps_s: np.ndarray, poses: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mid-interval times and |angular velocity| [rad/s] from consecutive 4x4 poses."""
    mid_times, speeds = [], []
    for index in range(len(timestamps_s) - 1):
        dt = float(timestamps_s[index + 1] - timestamps_s[index])
        if dt <= 0:
            raise ValueError("timestamps must be strictly increasing")
        relative = poses[index][:3, :3].T @ poses[index + 1][:3, :3]
        cosine = max(-1.0, min(1.0, (np.trace(relative) - 1.0) / 2.0))
        mid_times.append(0.5 * (timestamps_s[index] + timestamps_s[index + 1]))
        speeds.append(math.acos(cosine) / dt)
    return np.array(mid_times), np.array(speeds)


def estimate_time_offset(
    reference_times_s: np.ndarray,
    reference_signal: np.ndarray,
    other_times_s: np.ndarray,
    other_signal: np.ndarray,
    *,
    search_s: float,
    step_s: float,
    average_window_s: float | None = None,
) -> dict[str, float | int | None]:
    """Offset d maximizing corr(other(t), reference(t - d)); positive d = other lags.

    Use with signals that do not depend on frame conventions, e.g. |gyro| vs the
    angular speed differentiated from a visual trajectory.

    A pose finite difference is an *interval average*. Comparing it with point
    samples of a faster, noisy reference aliases at the reference sample period
    and biases the peak (seen on EuRoC: a spurious -12.5 ms). Pass the slower
    stream's interval as `average_window_s` to compare with the reference's
    moving average over that window instead.
    """
    if average_window_s is not None:
        steps = np.diff(reference_times_s)
        mean_values = 0.5 * (reference_signal[1:] + reference_signal[:-1])
        cumulative = np.concatenate([[0.0], np.cumsum(mean_values * steps)])
        half = average_window_s / 2.0

        def sample(times: np.ndarray) -> np.ndarray:
            upper = np.interp(times + half, reference_times_s, cumulative)
            lower = np.interp(times - half, reference_times_s, cumulative)
            return (upper - lower) / average_window_s

        margin = half
    else:

        def sample(times: np.ndarray) -> np.ndarray:
            return np.interp(times, reference_times_s, reference_signal)

        margin = 0.0
    best = {"offset_s": 0.0, "correlation": -math.inf, "samples": 0}
    for offset in np.arange(-search_s, search_s + step_s / 2, step_s):
        shifted = other_times_s - offset
        inside = (shifted >= reference_times_s[0] + margin) & (
            shifted <= reference_times_s[-1] - margin
        )
        if inside.sum() < 10:
            continue
        reference_values = sample(shifted[inside])
        other_values = other_signal[inside]
        if np.std(reference_values) < 1e-12 or np.std(other_values) < 1e-12:
            continue
        correlation = float(np.corrcoef(reference_values, other_values)[0, 1])
        if correlation > best["correlation"]:
            best = {
                "offset_s": float(offset),
                "correlation": correlation,
                "samples": int(inside.sum()),
            }
    if best["samples"] == 0:
        raise ValueError("no overlapping samples within the search window")
    best["search_s"] = search_s
    best["step_s"] = step_s
    best["average_window_s"] = average_window_s
    return best


def failure_rate_by_bin(
    values: np.ndarray, valid: np.ndarray, *, edges: Sequence[float]
) -> list[dict[str, float | int | None]]:
    """Fraction of invalid frames per [edge_i, edge_i+1) bin of a motion statistic."""
    rows = []
    for low, high in itertools.pairwise(edges):
        selected = (values >= low) & (values < high)
        count = int(selected.sum())
        rows.append(
            {
                "low": float(low),
                "high": float(high),
                "frames": count,
                "invalid_fraction": float(np.mean(~valid[selected])) if count else None,
            }
        )
    return rows
