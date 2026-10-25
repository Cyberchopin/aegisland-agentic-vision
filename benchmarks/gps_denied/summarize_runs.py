"""
Summarize cuVSLAM EuRoC runs into one table (markdown to stdout, JSON to --output).

For each `cuvslam-<mode>-<SEQ>-<tag>-run<N>` directory with a matching `-eval`:
tracking counts, untracked gaps, ATE (SE(3) meters; Sim(3) scale diagnostic), RPE,
track() timing, and run1-vs-run2 trajectory equality per (sequence, mode, tag).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from aegisland.trajectory_eval import load_tum

PATTERN = re.compile(r"cuvslam-(stereo|inertial)-(MH_\d\d_\w+)-(\d{8}-\d{6})-run(\d)$")


def gaps(frames: list[dict]) -> list[int]:
    result, current = [], 0
    for frame in frames:
        if frame["tracked"]:
            if current:
                result.append(current)
            current = 0
        else:
            current += 1
    return result + ([current] if current else [])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="new JSON file")
    args = parser.parse_args()

    rows = []
    for run in sorted(args.results.iterdir()):
        match = PATTERN.match(run.name)
        evaluation = run.with_name(run.name + "-eval") / "metrics.json"
        if not match or not evaluation.is_file():
            continue
        mode, sequence, tag, index = match.groups()
        record = json.loads((run / "run.json").read_text())
        metrics = json.loads(evaluation.read_text())
        frames = [json.loads(line) for line in (run / "frames.jsonl").read_text().splitlines()]
        sibling = run.with_name(run.name[:-1] + ("2" if index == "1" else "1"))
        identical = None
        if sibling.is_dir():
            a, b = load_tum(run / "trajectory_body.tum"), load_tum(sibling / "trajectory_body.tum")
            identical = bool(
                a.positions_m.shape == b.positions_m.shape
                and np.array_equal(a.positions_m, b.positions_m)
            )
        timing = record["timing"]["warm_track_ms"]
        rows.append(
            {
                "sequence": sequence,
                "mode": mode,
                "tag": tag,
                "run": int(index),
                "stereo_frames": record["counts"]["stereo_frames"],
                "untracked_frames": record["counts"]["untracked_frames"],
                "untracked_gaps": gaps(frames),
                "ate_se3_rmse_m": metrics["ate_se3"]["rmse"],
                "ate_se3_max_m": metrics["ate_se3"]["max"],
                "ate_pairs": metrics["ate_se3"]["pairs"],
                "sim3_scale": metrics["ate_sim3_scale_diagnostic"]["scale"],
                "rpe_translation_rmse_m": metrics["rpe"]["translation_rmse_m"],
                "rpe_rotation_rmse_deg": metrics["rpe"]["rotation_rmse_deg"],
                "rpe_delta_frames": metrics["rpe"]["delta_frames"],
                "track_p50_ms": timing.get("p50"),
                "track_p95_ms": timing.get("p95"),
                "track_max_ms": timing.get("max"),
                "first_call_ms": record["timing"]["first_call_ms"],
                "identical_to_other_run": identical,
                "run_dir": run.name,
            }
        )

    print(
        "| seq | mode | run | frames | untracked (gaps) | ATE SE3 RMSE m | max m | Sim3 scale "
        "| RPE20 m / deg | track p50/p95/max ms | run1==run2 |"
    )
    print("|" + " --- |" * 11)
    for row in rows:
        print(
            f"| {row['sequence']} | {row['mode']} | {row['run']} | {row['stereo_frames']} "
            f"| {row['untracked_frames']} ({','.join(map(str, row['untracked_gaps'])) or '-'}) "
            f"| {row['ate_se3_rmse_m']:.3f} | {row['ate_se3_max_m']:.3f} | {row['sim3_scale']:.4f} "
            f"| {row['rpe_translation_rmse_m']:.4f} / {row['rpe_rotation_rmse_deg']:.3f} "
            f"| {row['track_p50_ms']:.2f}/{row['track_p95_ms']:.2f}/{row['track_max_ms']:.1f} "
            f"| {row['identical_to_other_run']} |"
        )
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(rows, stream, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
