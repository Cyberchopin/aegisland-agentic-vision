import numpy as np
import pytest

from aegisland.depth_sectors import Intrinsics
from aegisland.depth_validation import center_roi, evaluate_step, plane_rms

K = Intrinsics(400.0, 400.0, 319.5, 199.5)


def wall(distance: float, noise: float = 0.0, seed: int = 0, tilt: float = 0.0):
    rng = np.random.default_rng(seed)
    cols = np.arange(640)[None, :].repeat(400, axis=0)
    depth = distance + tilt * (cols - 319.5) / 400.0 + rng.normal(0, noise, (400, 640))
    return depth.astype(np.float32), np.ones((400, 640), bool)


def test_unbiased_wall_has_zero_bias_and_noise_shows_as_temporal_and_plane_rms() -> None:
    frames = [wall(2.0, noise=0.01, seed=s) for s in range(10)]
    result = evaluate_step(2.0, frames, K, latencies_ms=[30, 32, 31, 40])
    assert abs(result.bias_m) < 0.002
    assert result.plane_rms_m == pytest.approx(0.01, rel=0.15)
    assert result.temporal_std_m < 0.002  # median over the ROI averages the noise
    assert result.roi_invalid_fraction == 0.0
    assert result.latency_ms_p50 == pytest.approx(31.5)


def test_bias_invalid_pixels_and_tilted_plane() -> None:
    depth, valid = wall(3.0, tilt=0.3)
    depth = depth * 1.02  # +2 % scale error
    valid[:200, :] = False  # top half invalid (independent of the column tilt)
    result = evaluate_step(3.0, [(depth, valid)], K)
    assert result.relative_error == pytest.approx(0.02, abs=0.003)
    assert result.frame_invalid_fraction == pytest.approx(0.5)
    # A truly flat plane tilted by 20 deg (n . P = D) has (near) zero plane residual.
    theta = np.radians(20.0)
    u = (np.arange(640)[None, :].repeat(400, axis=0) - K.cx) / K.fx
    plane = (3.0 / (np.sin(theta) * u + np.cos(theta))).astype(np.float64)
    assert plane_rms(plane, np.ones_like(valid), K) < 1e-6


def test_empty_roi_and_no_frames() -> None:
    depth, valid = wall(1.0)
    rows, cols = center_roi(depth.shape)
    valid[rows, cols] = False
    result = evaluate_step(1.0, [(depth, valid)], K)
    assert result.median_depth_m is None and result.roi_invalid_fraction == 1.0
    with pytest.raises(ValueError, match="no frames"):
        evaluate_step(1.0, [], K)
