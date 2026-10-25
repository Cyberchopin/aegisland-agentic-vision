"""
Loop closure on/off on EuRoC: ATE, error recovery after the dark segment, and cost.

Reads the run directories of one batch tag (run_cuvslam_euroc.py with/without
--slam, --realtime) and their *-eval-* directories. Writes a NEW directory with
summary.json and error-over-time charts.

Error over time: per-frame position error after one SE(3) alignment per
trajectory (whole run); for MH_05 the windows before / during / after the dark
segment (brightness < 30, cam0 mean) are reported. Development GPU, not Orin.

Run with ~/.venvs/px4 (matplotlib) and PYTHONPATH=src.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aegisland.trajectory_eval import (
    associate,
    load_euroc_groundtruth,
    load_tum,
    umeyama_alignment,
)

BLUE, ORANGE, AQUA, MUTED, GRID, SURFACE, INK = (
    "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#e4e3df", "#fcfcfb", "#0b0b0b")
DARK_BRIGHTNESS = 30.0


def error_series(tum: Path, gt) -> tuple[np.ndarray, np.ndarray]:
    estimate = load_tum(tum)
    pairs = associate(estimate.timestamps_s, gt.timestamps_s, max_time_difference_s=0.005)
    est = np.array([estimate.positions_m[i] for i, _ in pairs])
    ref = np.array([gt.positions_m[j] for _, j in pairs])
    rotation, translation, _ = umeyama_alignment(est, ref, with_scale=False)
    error = np.linalg.norm(est @ rotation.T + translation - ref, axis=1)
    return np.array([estimate.timestamps_s[i] for i, _ in pairs]), error


def dark_windows(sequence: Path, image_stats: Path) -> tuple[float, float] | None:
    if not image_stats.is_file():
        return None
    stats = json.loads(image_stats.read_text())
    times = np.array(stats["timestamps_ns"]) / 1e9
    dark = np.array(stats["brightness"]) < DARK_BRIGHTNESS
    if not dark.any():
        return None
    return float(times[dark].min()), float(times[dark].max())


def window_stats(t, error, start, end) -> float | None:
    mask = (t >= start) & (t <= end)
    return float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--data", type=Path, default=Path.home() / "aegisland-data" / "euroc")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    pattern = re.compile(rf"cuvslam-(stereo|inertial)-(MH_0\d_\w+?)-{args.tag}-"
                         r"(noslam|slam-async|slam-sync)-run(\d)$")
    rows = []
    for run in sorted(args.results.iterdir()):
        match = pattern.match(run.name)
        if not match or not (run / "run.json").is_file():
            continue
        mode, seq, config, repeat = match.groups()
        record = json.loads((run / "run.json").read_text())
        timing = record["timing"]["warm_track_ms"]
        entry = {"run": run.name, "mode": mode, "sequence": seq, "config": config,
                 "repeat": int(repeat), "track_ms_p50": timing["p50"],
                 "track_ms_p95": timing["p95"], "track_ms_max": timing["max"],
                 "sm_clock_mhz": record.get("gpu_clocks", {}).get("sm_clock_mhz"),
                 "valid_for_performance": record["timing"].get("valid_for_performance_claims")}
        slam = record.get("slam")
        if slam:
            entry.update(loop_closures=slam["loop_closure_events"],
                         peak_rss_mib=slam["memory"]["peak_rss_mib"])
        for name in ("trajectory_body", "trajectory_slam_body", "trajectory_slam_smoothed_body"):
            metrics = args.results / f"{run.name}-eval-{name}" / "metrics.json"
            if metrics.is_file():
                entry[f"ate_{name}"] = json.loads(metrics.read_text())["ate_se3"]["rmse"]
        gt = load_euroc_groundtruth(args.data / seq / "mav0" / "state_groundtruth_estimate0"
                                    / "data.csv")
        dark = dark_windows(args.data / seq, args.data / seq / "aegisland_image_stats.json")
        if dark:
            entry["dark_segment_rel_s"] = [dark[0] - gt.timestamps_s[0],
                                           dark[1] - gt.timestamps_s[0]]
            for name in ("trajectory_body", "trajectory_slam_body"):
                path = run / f"{name}.tum"
                if path.is_file():
                    t, e = error_series(path, gt)
                    entry[f"window_rms_{name}"] = {
                        "before_dark_20s": window_stats(t, e, dark[0] - 20, dark[0]),
                        "dark": window_stats(t, e, *dark),
                        "after_dark_20s": window_stats(t, e, dark[1], dark[1] + 20),
                        "last_10s": window_stats(t, e, t[-1] - 10, t[-1]),
                    }
        rows.append(entry)
    groups: dict[str, dict] = {}
    for entry in rows:
        key = f"{entry['sequence']}/{entry['mode']}/{entry['config']}"
        group = groups.setdefault(key, {"runs": 0})
        group["runs"] += 1
        for field in ("ate_trajectory_body", "ate_trajectory_slam_body",
                      "ate_trajectory_slam_smoothed_body", "track_ms_p50", "track_ms_p95",
                      "loop_closures", "peak_rss_mib"):
            if field in entry:
                group.setdefault(field, []).append(round(entry[field], 4))
    summary = {"tool": "benchmarks/gps_denied/slam_compare.py", "tag": args.tag,
               "claims": "EuRoC replay at 20 fps on the development GPU (not Orin)",
               "groups": groups, "runs": rows}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    for seq in sorted({r["sequence"] for r in rows}):
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, facecolor=SURFACE)
        for ax, mode in zip(axes, ("stereo", "inertial")):
            gt = load_euroc_groundtruth(args.data / seq / "mav0" / "state_groundtruth_estimate0"
                                        / "data.csv")
            t0 = gt.timestamps_s[0]
            for config, name, color, label in (
                    ("noslam", "trajectory_body", BLUE, "odometry only"),
                    ("slam-sync", "trajectory_slam_body", ORANGE, "SLAM pose from track()"),
                    ("slam-sync", "trajectory_slam_smoothed_body", AQUA,
                     "SLAM re-optimized at the end (not available live)")):
                run = args.results / f"cuvslam-{mode}-{seq}-{args.tag}-{config}-run1"
                if not (run / f"{name}.tum").is_file():
                    continue
                t, e = error_series(run / f"{name}.tum", gt)
                ax.plot(t - t0, e, color=color, lw=1.2, label=label)
                if name == "trajectory_slam_body":
                    events = [json.loads(line) for line in (run / "frames.jsonl").read_text()
                              .splitlines()]
                    for event in events:
                        if event.get("loop_closure_event"):
                            ax.axvline(event["timestamp_ns"] / 1e9 - t0, color=ORANGE, lw=0.5,
                                       alpha=0.35)
            dark = dark_windows(args.data / seq, args.data / seq / "aegisland_image_stats.json")
            if dark:
                ax.axvspan(dark[0] - t0, dark[1] - t0, color=MUTED, alpha=0.12, lw=0,
                           label="dark segment (brightness < 30)")
            ax.set_facecolor(SURFACE)
            ax.grid(True, color=GRID, lw=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_title(f"{seq} {mode}: position error vs ground truth (SE(3)-aligned); thin "
                         "lines = loop-closure events", loc="left", fontsize=10, color=INK)
            ax.set_ylabel("error (m)", color=MUTED)
            ax.legend(frameon=False, fontsize=8, loc="upper left")
        axes[-1].set_xlabel("time since ground-truth start (s)", color=MUTED)
        fig.tight_layout()
        fig.savefig(args.output / f"slam_{seq}.png", dpi=120)
    print(json.dumps(groups, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
