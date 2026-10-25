"""
Hardware-day depth validation: OAK StereoDepth presets vs a flat target at
tape-measured distances, in one command.

  oak_depth_validation.py --distances 0.3 0.5 1 2 3 5 --presets nav near accuracy \
      --output benchmarks/results/gps_denied/oak-depth-validation-<model>-<tag>

For every preset and distance it asks the operator to place a flat, textured
board (or wall) head-on at that distance (the tape is measured from the camera
front glass along the optical axis). It then records `--frames` depth frames and
reports, using aegisland.depth_validation:
  - bias and relative error;
  - temporal and plane noise;
  - invalid fraction;
  - capture->host latency;
  - effective FPS.
Results go into a NEW benchmark directory (summary.json + SHA256SUMS).

Presets are starting points, built on DepthAI v3 (3.11) API names and the Luxonis
StereoDepth documentation (see docs/dronehacks/oak-depth-presets.md). They are NOT
validated on hardware yet; this script is how they get validated.

Needs a device: without one it exits with a clear error and writes nothing.
Run with ~/.venvs/depthai and PYTHONPATH=src. Perception only; no control output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from aegisland.depth_sectors import Intrinsics, depth_from_oak_mm
from aegisland.depth_validation import evaluate_step

# name: (base preset, resolution, fps, overrides)
PRESETS = {
    "nav": ("ROBOTICS", (640, 400), 30.0, {"extended": False, "subpixel_bits": 3,
                                            "lr_check": True, "max_range_m": 10.0}),
    "near": ("ROBOTICS", (640, 400), 30.0, {"extended": True, "subpixel_bits": 3,
                                             "lr_check": True, "max_range_m": 4.0}),
    "accuracy": ("HIGH_DETAIL", (1280, 800), 15.0, {"extended": False, "subpixel_bits": 5,
                                                     "lr_check": True, "max_range_m": 15.0}),
}


def build(dai, preset: str):
    base, size, fps, overrides = PRESETS[preset]
    pipeline = dai.Pipeline()
    device = pipeline.getDefaultDevice()
    if device.getIrDrivers():
        device.setIrLaserDotProjectorIntensity(0.0)  # passive stereo, like VIO conditions
    cams = [pipeline.create(dai.node.Camera).build(s)
            for s in (dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C)]
    outs = [c.requestOutput(size, type=dai.ImgFrame.Type.GRAY8, fps=fps) for c in cams]
    stereo = pipeline.create(dai.node.StereoDepth).build(
        outs[0], outs[1], presetMode=getattr(dai.node.StereoDepth.PresetMode, base))
    stereo.setLeftRightCheck(overrides["lr_check"])
    stereo.setExtendedDisparity(overrides["extended"])
    stereo.setSubpixel(True)
    stereo.setSubpixelFractionalBits(overrides["subpixel_bits"])
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_B)
    stereo.initialConfig.postProcessing.thresholdFilter.maxRange = int(
        overrides["max_range_m"] * 1000)
    queue = stereo.depth.createOutputQueue(maxSize=8, blocking=False)
    calibration = device.readCalibration()
    k = np.array(calibration.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, *size))
    intrinsics = Intrinsics(k[0, 0], k[1, 1], k[0, 2], k[1, 2])
    return pipeline, device, queue, intrinsics, {"base": base, "size": size, "fps": fps,
                                                  **overrides}


def capture(dai, queue, frames: int):
    out, latencies, stamps = [], [], []
    while len(out) < frames:
        message = queue.get()
        latencies.append((dai.Clock.now() - message.getTimestamp()).total_seconds() * 1e3)
        stamps.append(message.getTimestampDevice().total_seconds())
        out.append(depth_from_oak_mm(message.getFrame().astype(np.uint16)))
    fps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else None
    return out, latencies, fps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--distances", type=float, nargs="+", required=True, help="metres")
    parser.add_argument("--presets", nargs="+", choices=sorted(PRESETS), default=sorted(PRESETS))
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--no-prompt", action="store_true", help="do not wait for Enter")
    parser.add_argument("--output", type=Path, required=True, help="new directory")
    args = parser.parse_args()
    try:
        import depthai as dai
    except ImportError:
        print("depthai is not installed in this environment", file=sys.stderr)
        return 2
    if not dai.Device.getAllAvailableDevices():
        print("no OAK device found; nothing written", file=sys.stderr)
        return 3
    args.output.mkdir(parents=True, exist_ok=False)
    results = {}
    for preset in args.presets:
        pipeline, device, queue, intrinsics, settings = build(dai, preset)
        pipeline.start()
        steps = []
        try:
            for distance in args.distances:
                if not args.no_prompt:
                    input(f"[{preset}] place the flat target at {distance} m, then press Enter")
                time.sleep(0.5)
                frames, latencies, fps = capture(dai, queue, args.frames)
                result = asdict(evaluate_step(distance, frames, intrinsics,
                                              latencies_ms=latencies))
                result["fps"] = fps
                steps.append(result)
                print(json.dumps({"preset": preset, **{k: result[k] for k in (
                    "true_distance_m", "bias_m", "roi_invalid_fraction", "latency_ms_p50")}}))
        finally:
            pipeline.stop()
        results[preset] = {"settings": settings, "device": device.getDeviceName(),
                           "intrinsics": asdict(intrinsics), "steps": steps}
    summary = {"tool": "benchmarks/gps_denied/oak_depth_validation.py",
               "claims": "flat-target depth check on the connected OAK; perception only",
               "depthai": dai.__version__, "results": results}
    path = args.output / "summary.json"
    path.write_text(json.dumps(summary, indent=1))
    (args.output / "SHA256SUMS").write_text(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  summary.json\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
