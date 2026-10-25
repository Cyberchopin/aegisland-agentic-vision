"""
Obstacle-sector input on EuRoC with a host-side SGBM stand-in for OAK depth.

EuRoC has no OAK depth, so cam0/cam1 are fisheye-rectified and matched with
OpenCV StereoSGBM; the resulting depth (rectified-left pinhole) goes through
aegisland.depth_sectors exactly as an OAK `depth` frame would (uint16 mm ->
depth_from_oak_mm). Only the input stage changes when OAK depth is available.

There is no depth ground truth in EuRoC Machine Hall: the outputs are
consistency/proxy numbers (valid fraction, nearest-range distribution, timing),
not accuracy. Perception verdict only; no control output.

Run with the AegisLand environment; writes a NEW output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from compare_motion_direction import transform, yaml_list

from aegisland.depth_sectors import (
    Intrinsics,
    SectorConfig,
    depth_from_disparity,
    depth_from_oak_mm,
    min_depth_m,
    sector_report,
)

NUM_DISPARITIES = 96  # like OAK standard mode (0..95)


def rectifier(sequence: Path):
    def camera(name):
        path = sequence / name / "sensor_recalibrated.yaml"
        fx, fy, cx, cy = yaml_list(path, "intrinsics")
        return (np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]),
                np.array(yaml_list(path, "distortion_coefficients")),
                tuple(int(v) for v in yaml_list(path, "resolution")))

    k0, d0, size = camera("cam0")
    k1, d1, _ = camera("cam1")
    cam1_from_cam0 = np.linalg.inv(transform(sequence / "cam1" / "sensor_recalibrated.yaml"))
    r, t = cam1_from_cam0[:3, :3].copy(), cam1_from_cam0[:3, 3].reshape(3, 1).copy()
    r0, r1, p0, p1, _ = cv2.fisheye.stereoRectify(
        k0, d0, k1, d1, size, r, t, flags=cv2.CALIB_ZERO_DISPARITY, balance=0.0, fov_scale=1.0)
    maps = [cv2.fisheye.initUndistortRectifyMap(k, d, rr, p, size, cv2.CV_16SC2)
            for k, d, rr, p in ((k0, d0, r0, p0), (k1, d1, r1, p1))]
    baseline = float(np.linalg.norm(t))
    intrinsics = Intrinsics(p0[0, 0], p0[1, 1], p0[0, 2], p0[1, 2])
    return maps, intrinsics, baseline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sequence", type=Path, required=True, help="mav0/ with *_recalibrated")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--sectors", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True, help="new directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    maps, intrinsics, baseline = rectifier(args.sequence)
    sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=NUM_DISPARITIES, blockSize=7,
                                 P1=8 * 49, P2=32 * 49, uniquenessRatio=10, speckleWindowSize=100,
                                 speckleRange=2, disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM)
    config = SectorConfig(sectors=args.sectors, horizontal_fov_deg=70.0, vertical_band_deg=15.0,
                          # Matches at the top of the search range are "this close or closer"
                          # (saturated), so they count as blind-zone returns, not as obstacles.
                          min_range_m=min_depth_m(intrinsics.fx, baseline, NUM_DISPARITIES - 2))
    rows = [line.split(",") for line in (args.sequence / "cam0" / "data.csv").read_text()
            .splitlines() if line and not line.startswith("#")]
    rows = rows[:: args.stride][: args.frames]
    timings = {"rectify_ms": [], "sgbm_ms": [], "sectors_ms": []}
    with (args.output / "sectors.jsonl").open("x") as stream:
        for stamp, name in rows:
            name = name.strip()
            t0 = time.perf_counter()
            left = cv2.remap(cv2.imread(str(args.sequence / "cam0" / "data" / name), 0),
                             *maps[0], cv2.INTER_LINEAR)
            right = cv2.remap(cv2.imread(str(args.sequence / "cam1" / "data" / name), 0),
                              *maps[1], cv2.INTER_LINEAR)
            t1 = time.perf_counter()
            disparity = sgbm.compute(left, right).astype(np.float32) / 16.0
            z, valid = depth_from_disparity(disparity, intrinsics.fx, baseline)
            # Round-trip through the OAK depth format so the OAK path is exercised.
            depth_mm = np.where(valid, np.clip(z * 1000, 0, 65535), 0).astype(np.uint16)
            t2 = time.perf_counter()
            depth, mask = depth_from_oak_mm(depth_mm)
            report = sector_report(depth, mask, int(stamp), intrinsics,
                                   now_ns=int(stamp) + int((t2 - t0) * 1e9), config=config)
            t3 = time.perf_counter()
            for key, value in zip(timings, (t1 - t0, t2 - t1, t3 - t2)):
                timings[key].append(value * 1e3)
            stream.write(json.dumps({"timestamp_ns": int(stamp), **{
                k: v for k, v in asdict(report).items() if k != "sectors"},
                "nearest_m": report.nearest_m,
                "sectors": [asdict(s) for s in report.sectors]}) + "\n")
    if rows:
        overlay = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
        overlay[mask] = (0.5 * overlay[mask] + [0, 90, 0]).astype(np.uint8)
        cv2.imwrite(str(args.output / "last_frame_valid_depth.jpg"), overlay)
    records = [json.loads(line) for line in (args.output / "sectors.jsonl").read_text()
               .splitlines()]
    per_sector = []
    for k in range(args.sectors):
        values = [r["sectors"][k] for r in records]
        near = [v["nearest_m"] for v in values if v["nearest_m"] is not None]
        per_sector.append({
            "azimuth_deg": values[0]["azimuth_deg"],
            "valid_rate": float(np.mean([v["valid"] for v in values])),
            "nearest_m_p10_p50": [float(np.percentile(near, 10)), float(np.percentile(near, 50))]
            if near else None,
            "reasons": {reason: sum(v["reason"] == reason for v in values)
                        for reason in sorted({v["reason"] for v in values})},
        })
    summary = {
        "tool": "benchmarks/gps_denied/depth_sectors_euroc.py",
        "claims": "SGBM stand-in for OAK depth on EuRoC; no depth ground truth -> proxy "
                  "numbers only; development machine CPU (not Orin); no control output",
        "sequence": str(args.sequence), "frames": len(records), "stride": args.stride,
        "rectified_intrinsics": asdict(intrinsics), "baseline_m": baseline,
        "config": asdict(config),
        "frame_valid_fraction": {"p10": float(np.percentile([r["frame_valid_fraction"]
                                                             for r in records], 10)),
                                 "p50": float(np.median([r["frame_valid_fraction"]
                                                         for r in records]))},
        "timing_ms": {k: {"p50": float(np.median(v)), "p95": float(np.percentile(v, 95))}
                      for k, v in timings.items()},
        "per_sector": per_sector,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))
    with (args.output / "SHA256SUMS").open("x") as stream:
        for item in sorted(args.output.iterdir()):
            if item.name != "SHA256SUMS":
                stream.write(f"{hashlib.sha256(item.read_bytes()).hexdigest()}  {item.name}\n")
    print(json.dumps({k: summary[k] for k in ("frames", "frame_valid_fraction", "timing_ms")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
