"""
Probe cuVSLAM localize_in_map frame by frame (diagnostics for relocalization).

For each (frame, guess json) pair: a fresh Tracker with SLAM (sync mode), track
`--warmup` consecutive frames ending at that frame, then localize_in_map with the
guess, following the official KITTI SLAM example. Records success, the message,
the duration, and the found pose relative to the guess.

Runs in the cuVSLAM environment; offline only. Writes a NEW json file.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--cuvslam-examples", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--guesses", type=Path, nargs="+", required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--radius-m", type=float, default=2.0)
    parser.add_argument("--step-m", type=float, default=0.25)
    parser.add_argument("--angular-step-rad", type=float, default=0.03)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.cuvslam_examples.resolve()))
    import cuvslam
    from dataset_utils import get_rig, load_frame

    rig = get_rig(str(args.sequence))
    rows = [line.split(",") for line in (args.sequence / "cam0" / "data.csv").read_text()
            .splitlines() if line and not line.startswith("#")]
    results = []
    for guess_path in args.guesses:
        guess = json.loads(guess_path.read_text())
        frame = guess["frame"]
        tracker = cuvslam.Tracker(
            rig, cuvslam.Tracker.OdometryConfig(async_sba=False, rectified_stereo_camera=False),
            cuvslam.Tracker.SlamConfig(sync_mode=True))
        images = None
        for k in range(max(0, frame - args.warmup + 1), frame + 1):
            stamp, name = int(rows[k][0]), rows[k][1].strip()
            images = [load_frame(str(args.sequence / cam / "data" / name)) for cam in ("cam0", "cam1")]
            tracker.track(stamp, images)
        settings = cuvslam.Tracker.SlamLocalizationSettings(
            horizontal_search_radius=args.radius_m, vertical_search_radius=args.radius_m / 2,
            horizontal_step=args.step_m, vertical_step=args.step_m / 2,
            angular_step_rads=args.angular_step_rad)
        done, out = threading.Event(), {}

        def finish(pose, message, out=out, done=done):
            out["success"] = pose is not None
            out["message"] = message
            if pose is not None:
                out["translation"] = [float(v) for v in pose.translation]
            done.set()

        started = time.perf_counter()
        tracker.localize_in_map(str(args.map), int(rows[frame][0]),
                                cuvslam.Pose(translation=guess["translation"],
                                             rotation=guess["rotation_xyzw"]),
                                images, settings, lambda: None, finish)
        done.wait(timeout=120)
        out["duration_s"] = time.perf_counter() - started
        if out.get("success"):
            out["distance_from_guess_m"] = float(np.linalg.norm(
                np.array(out["translation"]) - np.array(guess["translation"])))
        results.append({"guess": str(guess_path), "frame": frame,
                        "perturbation": guess.get("perturbation"), **out})
        print(json.dumps(results[-1]))
    args.output.write_text(json.dumps({"tool": "benchmarks/gps_denied/reloc_probe.py",
                                       "map": str(args.map), "sequence": str(args.sequence),
                                       "settings": {"radius_m": args.radius_m,
                                                    "step_m": args.step_m,
                                                    "angular_step_rad": args.angular_step_rad,
                                                    "warmup": args.warmup},
                                       "results": results}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
