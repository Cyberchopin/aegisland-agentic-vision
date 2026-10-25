"""
Fast-FoundationStereo vs OpenCV StereoSGBM on rectified EuRoC stereo pairs (stage 2 start).

Runs in an isolated PyTorch environment, importing the model code from a pinned
Fast-FoundationStereo checkout (`--ffs-src`); nothing is copied into AegisLand.
The FFS code license is non-commercial / research only; the C-FFS checkpoint is
under the NVIDIA Open Model Agreement. Use here = offline research benchmark.

EuRoC has no depth ground truth, so only PROXY metrics are reported:
  invalid pixel fraction, left-right consistency (flip trick), FFS-vs-SGBM agreement.
None of these is an accuracy validation.

Timing (dev GPU, not Orin; synchronous; first call reported separately):
  preprocess = rectify both images + tensor upload + pad
  model      = forward pass (cuda synchronized before/after)
  postprocess= unpad + device->host + disparity->depth + invalid mask
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

LR_TOLERANCE_PX = 1.0


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_camera(path: Path):
    config = yaml.safe_load(path.read_text())
    fu, fv, cu, cv = config["intrinsics"]
    camera_matrix = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1.0]])
    distortion = np.array(config["distortion_coefficients"], dtype=np.float64).reshape(4, 1)
    transform = np.array(config["T_BS"]["data"], dtype=np.float64).reshape(4, 4)
    return camera_matrix, distortion, transform, tuple(config["resolution"])


def rectification(sequence: Path):
    k0, d0, cam0_from_cam0, size = load_camera(sequence / "cam0" / "sensor_recalibrated.yaml")
    k1, d1, cam0_from_cam1, _ = load_camera(sequence / "cam1" / "sensor_recalibrated.yaml")
    cam1_from_cam0 = np.linalg.inv(cam0_from_cam1) @ cam0_from_cam0
    rotation = np.ascontiguousarray(cam1_from_cam0[:3, :3])
    translation = np.ascontiguousarray(cam1_from_cam0[:3, 3].reshape(3, 1))
    r0, r1, p0, p1, _ = cv2.fisheye.stereoRectify(
        k0, d0, k1, d1, size, rotation, translation,
        flags=cv2.CALIB_ZERO_DISPARITY, balance=0.0, fov_scale=1.0,
    )
    maps = [
        cv2.fisheye.initUndistortRectifyMap(k, d, r, p, size, cv2.CV_32FC1)
        for k, d, r, p in ((k0, d0, r0, p0), (k1, d1, r1, p1))
    ]
    return maps, float(p0[0, 0]), float(np.linalg.norm(translation)), size


def frame_pairs(sequence: Path, count: int, stride: int) -> list[tuple[int, Path, Path]]:
    rows = []
    with (sequence / "cam0" / "data.csv").open() as stream:
        for row in csv.reader(stream):
            if row and not row[0].startswith("#"):
                rows.append((int(row[0]), row[1].strip()))
    picked = rows[::stride][:count]
    return [
        (t, sequence / "cam0" / "data" / name, sequence / "cam1" / "data" / name)
        for t, name in picked
    ]


def lr_consistency(left_disp: np.ndarray, right_disp: np.ndarray) -> float:
    """Fraction of valid left pixels whose right-image disparity agrees within 1 px."""
    height, width = left_disp.shape
    xs = np.arange(width)[None, :].repeat(height, axis=0)
    target = np.round(xs - left_disp).astype(int)
    valid = (left_disp > 0) & (target >= 0) & (target < width)
    rows = np.arange(height)[:, None].repeat(width, axis=1)
    agreement = np.zeros_like(valid)
    agreement[valid] = (
        np.abs(left_disp[valid] - right_disp[rows[valid], target[valid]]) <= LR_TOLERANCE_PX
    )
    return float(agreement.sum() / max(valid.sum(), 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sequence", type=Path, required=True, help="EuRoC mav0/")
    parser.add_argument("--ffs-src", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--valid-iters", type=int, default=8)
    parser.add_argument("--max-disp", type=int, default=192)
    parser.add_argument("--lr-every", type=int, default=4, help="LR check on every Nth frame")
    parser.add_argument("--output", type=Path, required=True, help="new directory")
    args = parser.parse_args()

    sys.path.insert(0, str(args.ffs_src.resolve()))
    import torch
    import Utils as ffs_utils
    from core.utils.utils import InputPadder

    args.output.mkdir(parents=True, exist_ok=False)
    torch.autograd.set_grad_enabled(False)
    # Official checkpoint is a pickled module (weights_only=False runs its pickle code).
    model = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.args.valid_iters = args.valid_iters
    model.args.max_disp = args.max_disp
    # The C-FFS checkpoint's pickled args lack `normalize`, which the current
    # forward() reads. Use the same default as upstream scripts/make_plugin_onnx.py.
    normalize_defaulted = "normalize" not in model.args
    if normalize_defaulted:
        model.args.normalize = True
    model.cuda().eval()

    (map0, map1), focal_px, baseline_m, size = rectification(args.sequence)
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=128, blockSize=5,
        P1=8 * 5 * 5, P2=32 * 5 * 5, uniquenessRatio=10, speckleWindowSize=100,
        speckleRange=2, disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    def ffs(left: np.ndarray, right: np.ndarray):
        sync = torch.cuda.synchronize
        sync()
        t0 = time.perf_counter()
        a = torch.as_tensor(np.tile(left[..., None], (1, 1, 3))).cuda().float()[None]
        b = torch.as_tensor(np.tile(right[..., None], (1, 1, 3))).cuda().float()[None]
        a, b = a.permute(0, 3, 1, 2), b.permute(0, 3, 1, 2)
        padder = InputPadder(a.shape, divis_by=32, force_square=False)
        a, b = padder.pad(a, b)
        sync()
        t1 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=True, dtype=ffs_utils.AMP_DTYPE):
            disp = model.forward(
                a, b, iters=args.valid_iters, test_mode=True, optimize_build_volume="pytorch1"
            )
        sync()
        t2 = time.perf_counter()
        disp = padder.unpad(disp.float()).cpu().numpy().reshape(left.shape).clip(0, None)
        xs = np.arange(disp.shape[1])[None, :]
        invalid = (xs - disp < 0) | (disp <= 0)
        depth = np.where(invalid, np.inf, focal_px * baseline_m / np.maximum(disp, 1e-6))
        t3 = time.perf_counter()
        return disp, depth, invalid, (t1 - t0, t2 - t1, t3 - t2)

    records = []
    torch.cuda.reset_peak_memory_stats()
    with (args.output / "frames.jsonl").open("x") as log:
        for index, (timestamp, left_path, right_path) in enumerate(
            frame_pairs(args.sequence, args.frames, args.stride)
        ):
            raw_left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
            raw_right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
            t0 = time.perf_counter()
            left = cv2.remap(raw_left, *map0, cv2.INTER_LINEAR)
            right = cv2.remap(raw_right, *map1, cv2.INTER_LINEAR)
            rectify_s = time.perf_counter() - t0

            disp, depth, invalid, (upload_s, model_s, post_s) = ffs(left, right)
            record = {
                "index": index,
                "timestamp_ns": timestamp,
                "ffs_rectify_ms": rectify_s * 1e3,
                "ffs_upload_pad_ms": upload_s * 1e3,
                "ffs_model_ms": model_s * 1e3,
                "ffs_post_ms": post_s * 1e3,
                "ffs_invalid_fraction": float(invalid.mean()),
                "ffs_median_depth_m": (
                    float(np.median(depth[~invalid])) if (~invalid).any() else None
                ),
            }
            t0 = time.perf_counter()
            sgbm_disp = sgbm.compute(left, right).astype(np.float32) / 16.0
            record["sgbm_ms"] = (time.perf_counter() - t0) * 1e3
            sgbm_valid = sgbm_disp > 0
            record["sgbm_invalid_fraction"] = float(1.0 - sgbm_valid.mean())
            both = sgbm_valid & ~invalid
            record["ffs_vs_sgbm_median_abs_disp_px"] = (
                float(np.median(np.abs(disp[both] - sgbm_disp[both]))) if both.any() else None
            )
            if index % args.lr_every == 0:
                flipped_disp, _, _, _ = ffs(right[:, ::-1].copy(), left[:, ::-1].copy())
                record["ffs_lr_consistency"] = lr_consistency(disp, flipped_disp[:, ::-1])
                sgbm_right = sgbm.compute(
                    right[:, ::-1].copy(), left[:, ::-1].copy()
                ).astype(np.float32)[:, ::-1] / 16.0
                record["sgbm_lr_consistency"] = lr_consistency(
                    np.where(sgbm_valid, sgbm_disp, 0), sgbm_right
                )
            if index in (0, args.frames // 2):
                def colored(values: np.ndarray) -> np.ndarray:
                    scaled = cv2.convertScaleAbs(np.clip(values, 0, None), alpha=255 / 64)
                    return cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)

                vis = np.concatenate(
                    [cv2.cvtColor(left, cv2.COLOR_GRAY2BGR), colored(disp), colored(sgbm_disp)],
                    axis=1,
                )
                cv2.imwrite(str(args.output / f"vis-{index:03d}-left-ffs-sgbm.jpg"), vis,
                            [cv2.IMWRITE_JPEG_QUALITY, 80])
            log.write(json.dumps(record) + "\n")
            records.append(record)

    def stats(key: str, skip_first: bool = True):
        rows = records[1:] if skip_first else records
        values = [r[key] for r in rows if r.get(key) is not None]
        if not values:
            return None
        array = np.array(values)
        return {"count": int(array.size), "p50": float(np.percentile(array, 50)),
                "p95": float(np.percentile(array, 95)), "mean": float(array.mean())}

    summary = {
        "tool": "benchmarks/gps_denied/run_ffs_euroc.py",
        "claims": "Proxy metrics only (no depth ground truth). Dev GPU, not Orin.",
        "ffs_revision": subprocess.run(
            ["git", "-C", str(args.ffs_src), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "python": sys.version,
        "platform": platform.platform(),
        "sequence": str(args.sequence.resolve()),
        "frames": len(records),
        "stride": args.stride,
        "valid_iters": args.valid_iters,
        "max_disp": args.max_disp,
        "normalize_defaulted_to_true": normalize_defaulted,
        "rectified_size": list(size),
        "rectified_focal_px": focal_px,
        "baseline_m": baseline_m,
        "first_call_ms": {k: records[0][k] for k in ("ffs_upload_pad_ms", "ffs_model_ms")},
        "warm": {key: stats(key) for key in (
            "ffs_rectify_ms", "ffs_upload_pad_ms", "ffs_model_ms", "ffs_post_ms", "sgbm_ms")},
        "proxy": {key: stats(key, skip_first=False) for key in (
            "ffs_invalid_fraction", "sgbm_invalid_fraction", "ffs_lr_consistency",
            "sgbm_lr_consistency", "ffs_vs_sgbm_median_abs_disp_px", "ffs_median_depth_m")},
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
        "timing_note": "LR-check frames run FFS twice; only the first (left) call is timed.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "SHA256SUMS").open("x") as stream:
        for item in sorted(args.output.iterdir()):
            if item.is_file() and item.name != "SHA256SUMS":
                stream.write(f"{sha256_file(item)}  {item.name}\n")
    print(json.dumps({"warm": summary["warm"], "proxy": summary["proxy"],
                      "peak_cuda_memory_mb": summary["peak_cuda_memory_mb"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
