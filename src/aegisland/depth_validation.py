"""
Depth validation against a flat target at tape-measured distances (hardware day).

Pure analysis, no device code: given depth frames of a flat wall or board seen
roughly head-on, report per distance step
  - ROI median range error vs the measured distance (bias) and relative error
  - temporal noise (std of the ROI median over frames)
  - plane-fit residual RMS (spatial noise / "waviness")
  - invalid-pixel fraction in the ROI and in the whole frame
  - latency (capture -> host receive), if host receive times are given

The tape distance must be measured along the optical axis from the camera's
reference plane (Luxonis: the front glass is close enough at > 0.5 m; state it).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .depth_sectors import Intrinsics


@dataclass(frozen=True)
class StepResult:
    true_distance_m: float
    frames: int
    median_depth_m: float | None
    bias_m: float | None
    relative_error: float | None
    temporal_std_m: float | None
    plane_rms_m: float | None
    roi_invalid_fraction: float
    frame_invalid_fraction: float
    latency_ms_p50: float | None
    latency_ms_p95: float | None


def center_roi(shape: tuple[int, int], fraction: float = 0.2) -> tuple[slice, slice]:
    height, width = shape
    h, w = max(1, int(height * fraction)), max(1, int(width * fraction))
    top, left = (height - h) // 2, (width - w) // 2
    return slice(top, top + h), slice(left, left + w)


def plane_rms(depth_m: np.ndarray, valid: np.ndarray, intrinsics: Intrinsics) -> float | None:
    """RMS distance of back-projected valid points to their least-squares plane."""
    rows, cols = np.nonzero(valid)
    if rows.size < 50:
        return None
    z = depth_m[rows, cols].astype(np.float64)
    x = (cols - intrinsics.cx) / intrinsics.fx * z
    y = (rows - intrinsics.cy) / intrinsics.fy * z
    points = np.column_stack([x, y, z])
    centered = points - points.mean(axis=0)
    normal = np.linalg.svd(centered, full_matrices=False)[2][-1]
    return float(np.sqrt(np.mean((centered @ normal) ** 2)))


def evaluate_step(
    true_distance_m: float,
    frames: list[tuple[np.ndarray, np.ndarray]],
    intrinsics: Intrinsics,
    *,
    latencies_ms: list[float] | None = None,
    roi_fraction: float = 0.2,
) -> StepResult:
    if not frames:
        raise ValueError("no frames for this distance step")
    medians, plane, roi_invalid, frame_invalid = [], [], [], []
    for depth_m, valid in frames:
        rows, cols = center_roi(depth_m.shape, roi_fraction)
        roi_valid = valid[rows, cols]
        roi_invalid.append(1.0 - float(roi_valid.mean()))
        frame_invalid.append(1.0 - float(valid.mean()))
        if roi_valid.sum() >= 10:
            medians.append(float(np.median(depth_m[rows, cols][roi_valid])))
            sub = np.zeros_like(valid)
            sub[rows, cols] = roi_valid
            rms = plane_rms(depth_m, sub, intrinsics)
            if rms is not None:
                plane.append(rms)
    median = float(np.median(medians)) if medians else None
    lat = np.array(latencies_ms or [], dtype=float)
    return StepResult(
        true_distance_m=true_distance_m, frames=len(frames), median_depth_m=median,
        bias_m=None if median is None else median - true_distance_m,
        relative_error=None if median is None else (median - true_distance_m) / true_distance_m,
        temporal_std_m=float(np.std(medians)) if len(medians) > 1 else None,
        plane_rms_m=float(np.median(plane)) if plane else None,
        roi_invalid_fraction=float(np.mean(roi_invalid)),
        frame_invalid_fraction=float(np.mean(frame_invalid)),
        latency_ms_p50=float(np.median(lat)) if lat.size else None,
        latency_ms_p95=float(np.percentile(lat, 95)) if lat.size else None,
    )
