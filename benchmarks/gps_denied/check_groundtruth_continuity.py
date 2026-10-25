"""
Check EuRoC ground truth for position discontinuities and timestamp problems.

For each sample pair of state_groundtruth_estimate0 it compares the
finite-difference speed with the estimator's own velocity column. A sample is
flagged when the position step exceeds what the reported velocity explains by
more than `--step-tolerance-m`. For each flag it reports the timestamp spacing
(to rule out a timestamp error on our side), whether the step persists (offset)
or returns (spike), and the raw Leica prism track around it (10-20 Hz laser
tracker measurements, independent of the fused estimate).

Uses raw integer nanoseconds from the CSV and, separately, the AegisLand loader,
so a parsing or float-precision problem would show up as a mismatch.

Run with the AegisLand environment; writes a NEW output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from aegisland.trajectory_eval import load_euroc_groundtruth

LEICA_WINDOW_S = 0.5


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [line.split(",") for line in path.read_text().splitlines() if not line.startswith("#")]
    return np.array([int(row[0]) for row in rows], dtype=np.int64), np.array(
        [[float(value) for value in row[1:]] for row in rows]
    )


def leica_around(leica_ns, leica_xyz, center_ns) -> dict:
    window = np.abs(leica_ns - center_ns) <= LEICA_WINDOW_S * 1e9
    times, points = leica_ns[window], leica_xyz[window]
    if len(times) < 3:
        return {"samples": len(times)}
    speeds = np.linalg.norm(np.diff(points, axis=0), axis=1) / (np.diff(times) / 1e9)
    return {
        "samples": len(times),
        "speed_mps_min": float(speeds.min()),
        "speed_mps_max": float(speeds.max()),
        "note": "prism position, not body; only the speed profile is compared",
    }


def check(sequence: Path, tolerance_m: float) -> dict:
    gt_ns, values = load_csv(sequence / "state_groundtruth_estimate0" / "data.csv")
    positions, velocities = values[:, 0:3], values[:, 7:10]
    loaded = load_euroc_groundtruth(sequence / "state_groundtruth_estimate0" / "data.csv")
    loader_dt_error_s = float(
        np.max(np.abs(np.diff(loaded.timestamps_s) - np.diff(gt_ns) / 1e9))
    )
    dt_s = np.diff(gt_ns) / 1e9
    steps = np.diff(positions, axis=0)
    # Velocity-explained step: trapezoid of the reported velocities.
    expected = 0.5 * (velocities[1:] + velocities[:-1]) * dt_s[:, None]
    residual = np.linalg.norm(steps - expected, axis=1)
    flagged = np.flatnonzero(residual > tolerance_m)

    leica = sequence / "leica0" / "data.csv"
    leica_ns, leica_xyz = load_csv(leica) if leica.is_file() else (None, None)
    events = []
    for index in flagged:
        step = steps[index] - expected[index]
        # Persistent offset: positions 0.1 s later still differ from velocity integration.
        after = min(index + 20, len(positions) - 1)
        drift = positions[after] - positions[index] - np.sum(
            expected[index + 1 : after], axis=0
        )
        events.append(
            {
                "sample_index": int(index),
                "t_rel_s": float((gt_ns[index] - gt_ns[0]) / 1e9),
                "timestamp_ns": [int(gt_ns[index]), int(gt_ns[index + 1])],
                "dt_ms": float(dt_s[index] * 1e3),
                "unexplained_step_m": step.tolist(),
                "unexplained_step_norm_m": float(np.linalg.norm(step)),
                "finite_difference_speed_mps": float(np.linalg.norm(steps[index]) / dt_s[index]),
                "reported_speed_mps": float(np.linalg.norm(velocities[index + 1])),
                "offset_persists_after_0p1s_m": float(np.linalg.norm(drift)),
                "leica_raw": None
                if leica_ns is None
                else leica_around(leica_ns, leica_xyz, gt_ns[index]),
            }
        )
    return {
        "samples": len(gt_ns),
        "dt_ms": {
            "min": float(dt_s.min() * 1e3),
            "median": float(np.median(dt_s) * 1e3),
            "max": float(dt_s.max() * 1e3),
            "non_increasing": int(np.sum(dt_s <= 0)),
        },
        "loader_vs_raw_dt_max_error_s": loader_dt_error_s,
        "residual_m": {
            "median": float(np.median(residual)),
            "p99": float(np.percentile(residual, 99)),
        },
        "discontinuities": events,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sequences", type=Path, nargs="+", required=True, help="mav0/ dirs")
    parser.add_argument("--step-tolerance-m", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True, help="new directory")
    args = parser.parse_args()
    report = {
        "tool": "benchmarks/gps_denied/check_groundtruth_continuity.py",
        "step_tolerance_m": args.step_tolerance_m,
        "sequences": {
            sequence.parent.name: check(sequence, args.step_tolerance_m)
            for sequence in args.sequences
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "continuity.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "SHA256SUMS").write_text(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  continuity.json\n", encoding="utf-8"
    )
    for name, result in report["sequences"].items():
        print(name, result["dt_ms"], len(result["discontinuities"]),
              [(round(e["t_rel_s"], 2), round(e["unexplained_step_norm_m"], 3),
                round(e["offset_persists_after_0p1s_m"], 3),
                e["leica_raw"] and round(e["leica_raw"].get("speed_mps_max", 0), 2))
               for e in result["discontinuities"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
