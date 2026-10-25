"""
Interleaved A/B timing of CameraMotionCompensator with and without the ORB feature cache.

Both variants process the same consecutive frame pairs; per pair the order of A and B
alternates, so background load affects both equally. Every result (success, counts,
homography, aligned image) must be byte-identical or the run fails. No profiler is
attached; timings are wall-clock per compensate() call, dev machine, 1 OpenCV thread.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from aegisland.motion_compensation import CameraMotionCompensator


class UncachedCompensator(CameraMotionCompensator):
    """Original behavior: detect ORB on both images every call."""

    def _features(self, gray):
        return self.orb.detectAndCompute(gray, None)


def frames(sequence: Path, count: int) -> list[np.ndarray]:
    rows = []
    with (sequence / "cam0" / "data.csv").open() as stream:
        for row in csv.reader(stream):
            if row and not row[0].startswith("#"):
                rows.append(sequence / "cam0" / "data" / row[1].strip())
    return [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in rows[:count]]


def same(a, b) -> bool:
    return (
        a.success == b.success
        and a.match_count == b.match_count
        and a.inlier_count == b.inlier_count
        and (a.homography is None) == (b.homography is None)
        and (a.homography is None or np.array_equal(a.homography, b.homography))
        and np.array_equal(a.aligned_previous, b.aligned_previous)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True, help="new JSON file")
    args = parser.parse_args()

    cv2.setNumThreads(1)
    images = frames(args.sequence, args.frames)
    cached, uncached = CameraMotionCompensator(), UncachedCompensator()
    timings = {"cached": [], "uncached": []}
    for index in range(1, len(images)):
        previous, current = images[index - 1], images[index]
        order = [("cached", cached), ("uncached", uncached)]
        if index % 2:
            order.reverse()
        results = {}
        for name, compensator in order:
            started = time.perf_counter_ns()
            results[name] = compensator.compensate(previous, current)
            timings[name].append((time.perf_counter_ns() - started) / 1e6)
        if not same(results["cached"], results["uncached"]):
            raise SystemExit(f"outputs differ at frame {index}")

    summary = {
        "tool": "benchmarks/gps_denied/bench_orb_cache.py",
        "sequence": str(args.sequence.resolve()),
        "pairs": len(images) - 1,
        "image_shape": list(images[0].shape),
        "opencv": cv2.__version__,
        "threads": 1,
        "outputs_identical": True,
        "note": "dev machine (RTX 4060 Ti host CPU), interleaved A/B, no profiler; not Orin",
    }
    for name, values in timings.items():
        array = np.array(values[1:])  # drop the first call of each variant (cold)
        summary[name] = {
            "p50_ms": float(np.percentile(array, 50)),
            "p95_ms": float(np.percentile(array, 95)),
            "mean_ms": float(array.mean()),
        }
    summary["p50_speedup"] = summary["uncached"]["p50_ms"] / summary["cached"]["p50_ms"]
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
