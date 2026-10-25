"""
Replay a cuVSLAM run through the localization shadow supervisor with injected faults.

For each scenario (one fault at a time, plus an unmodified baseline) it records
every supervisor assessment and scores: was the fault detected inside its window,
with what latency, and how many non-ok assessments occurred outside the window.

Messages use the recording's own frames (odom_dataset / body_dataset) and the
dataset clock; publish time models one sequential tracker: max(capture, previous
publish) + measured Tracker.track() time, so slow calls queue later frames.
Shadow only: assessments carry CONTROL OUTPUT = DISABLED, nothing is sent.

Supervisor v3 adds degradation-class reasons (rotation_inconsistent, low_texture).
Fault scoring ignores them, so fault false alarms stay comparable with v2; they
are counted separately, and with --degradation-series (degradation_study.py
series.json for the same tracker mode) the warnings of the fault-free scenario
are scored against the ground-truth drift events.

Run with the AegisLand environment; writes a NEW output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

from aegisland.degradation_metrics import clean_mask, find_events, score_alarms
from aegisland.localization_faults import FaultSpec, build_events, score_detection
from aegisland.localization_message import SCHEMA, LocalizationMessage, TrackingState, to_dict
from aegisland.localization_supervisor import (
    DEGRADATION_REASONS,
    LocalizationShadowSupervisor,
    SupervisorConfig,
)
from aegisland.trajectory_eval import load_tum

OUTAGE_SECONDS = 1.0
# Same event definition as degradation_study.py defaults.
LABEL_THRESHOLD_MPS, MERGE_GAP_S, MIN_LENGTH_S, LEAD_WINDOW_S, HORIZON_S = 0.05, 1.0, 0.25, 3.0, 1.0


def load_messages(run: Path) -> tuple[list[LocalizationMessage], int]:
    frames = [json.loads(line) for line in (run / "frames.jsonl").read_text().splitlines()]
    trajectory = load_tum(run / "trajectory_body.tum")
    poses = iter(zip(trajectory.positions_m, trajectory.quaternions_xyzw))
    messages, last_pose = [], ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    previous_publish = 0
    for index, frame in enumerate(frames):
        capture = int(frame["timestamp_ns"])
        # One sequential tracker: a frame starts when it is captured AND the previous
        # frame is finished, so a slow track() call delays (queues) later frames.
        publish = max(capture, previous_publish) + math.ceil(frame["track_ms"] * 1e6)
        previous_publish = publish
        if frame["tracked"]:
            position, quaternion = next(poses)
            last_pose = (tuple(map(float, position)), tuple(map(float, quaternion)))
        messages.append(
            LocalizationMessage(
                schema=SCHEMA,
                source="cuvslam",
                sequence=index,
                frame_id="odom_dataset",
                child_frame_id="body_dataset",
                clock_domain="dataset",
                capture_time_ns=capture,
                publish_time_ns=publish,
                position_m=last_pose[0],
                orientation_xyzw=last_pose[1],
                velocity_mps=None,
                position_variance_m2=None,
                quality=90 if frame["tracked"] else -1,
                valid=bool(frame["tracked"]),
                tracking_state=TrackingState.TRACKING if frame["tracked"] else TrackingState.LOST,
                reset_counter=0,
            )
        )
    periods = sorted(
        b.capture_time_ns - a.capture_time_ns for a, b in itertools.pairwise(messages)
    )
    return messages, periods[len(periods) // 2]


def load_imu(sequence: Path | None) -> list[tuple[int, tuple[float, float, float]]]:
    if sequence is None or not (sequence / "imu0" / "data.csv").is_file():
        return []
    samples = []
    with (sequence / "imu0" / "data.csv").open() as stream:
        for row in csv.reader(stream):
            if row and not row[0].startswith("#"):
                samples.append((int(row[0]), tuple(float(v) for v in row[1:4])))
    return samples


def scenarios(count: int, period_ns: int, has_imu: bool) -> list[tuple[FaultSpec, set[str]]]:
    outage = max(2, round(OUTAGE_SECONDS * 1e9 / period_ns))
    plan = [
        (FaultSpec("none"), set()),
        (FaultSpec("dropped_frames", start=count // 3, length=5), {"dropped_frames"}),
        (FaultSpec("camera_outage", start=count // 2, length=outage), {"no_messages"}),
        (FaultSpec("pose_jump", start=2 * count // 3, magnitude=1.0), {"pose_jump"}),
        (
            FaultSpec("stale_delivery", start=count // 4, length=5, magnitude=0.3),
            {"stale"},
        ),
        (FaultSpec("out_of_order", start=count // 5), {"capture_time_not_increasing"}),
    ]
    if has_imu:
        plan.append((FaultSpec("imu_time_offset", magnitude=0.05), {"imu_time_offset"}))
    return plan


def load_feature_matches(vdr_run: Path | None) -> dict[int, float]:
    if vdr_run is None:
        return {}
    rows = [json.loads(line) for line in (vdr_run / "frames.jsonl").read_text().splitlines()]
    return {int(r["timestamp_ns"]): float(r["match_count"]) for r in rows if r["success"]}


def fault_class(rows):
    """Drop degradation-class reasons; status recomputed as in v2."""
    out = []
    for time_ns, status, reasons, _ in rows:
        kept = tuple(r for r in reasons if r not in DEGRADATION_REASONS)
        out.append((time_ns, status if kept else "ok", kept))
    return out


def score_degradation(rows, series: dict) -> dict:
    import numpy as np

    times = np.array(series["times"])
    drift = np.array([np.nan if v is None else v for v in series["drift"]], dtype=float)
    # Series times are capture times in seconds (trajectory timestamps).
    warned = {round(capture / 1e9, 4) for _, _, reasons, capture in rows
              if capture is not None and DEGRADATION_REASONS & set(reasons)}
    alarm = np.array([round(t, 4) in warned for t in times])
    events = find_events(times, drift, threshold=LABEL_THRESHOLD_MPS, merge_gap_s=MERGE_GAP_S,
                         min_length_s=MIN_LENGTH_S)
    clean = clean_mask(times, np.isfinite(drift), events, lead_window_s=LEAD_WINDOW_S,
                       horizon_s=HORIZON_S)
    return score_alarms(times, alarm, events, clean, lead_window_s=LEAD_WINDOW_S,
                        horizon_s=HORIZON_S)


def replay(events, config: SupervisorConfig, output: Path,
           matches: dict[int, float] | None = None) -> list[tuple]:
    supervisor = LocalizationShadowSupervisor(config)
    rows = []
    with output.open("x", encoding="utf-8") as log:
        for event in events:
            if event.kind == "imu":
                supervisor.add_imu(event.time_ns, event.gyro_rad_s)
                continue
            if event.kind == "tick":
                assessment = supervisor.tick(now_ns=event.time_ns)
            else:
                assessment = supervisor.assess(
                    event.message, now_ns=event.time_ns,
                    feature_matches=(matches or {}).get(event.message.capture_time_ns),
                )
            capture = None if event.message is None else event.message.capture_time_ns
            rows.append((event.time_ns, assessment.status, assessment.reasons, capture))
            if event.kind == "message" or assessment.status != "ok":
                record = {"time_ns": event.time_ns, "event": event.kind, **assessment.__dict__}
                if event.message is not None:
                    record["message"] = to_dict(event.message)
                log.write(json.dumps(record) + "\n")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cuvslam-run", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, help="mav0/ for IMU samples (optional)")
    parser.add_argument("--vdr-run", type=Path, help="VDR run for feature matches (v3)")
    parser.add_argument("--degradation-series", type=Path,
                        help="degradation_study series.json (same mode) to score warnings")
    parser.add_argument("--sequence-name", help="key in the series file, e.g. MH_05_difficult")
    parser.add_argument("--disable-degradation-checks", action="store_true",
                        help="run the v3 supervisor with its v3 checks off (= v2)")
    parser.add_argument("--output", type=Path, required=True, help="new directory")
    args = parser.parse_args()
    matches = load_feature_matches(args.vdr_run)
    series = None
    if args.degradation_series:
        series = json.loads(args.degradation_series.read_text())[args.sequence_name]

    messages, period_ns = load_messages(args.cuvslam_run)
    imu = load_imu(args.sequence)
    config = SupervisorConfig(expected_period_ns=period_ns)
    if args.disable_degradation_checks:  # supervisor v2 behaviour, for A/B comparison
        config = SupervisorConfig(expected_period_ns=period_ns, rotation_residual_rad_s=None,
                                  min_feature_matches=None)
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {
        "tool": "benchmarks/gps_denied/fault_replay.py",
        "control_output": "DISABLED",
        "command_executed": False,
        "cuvslam_run": str(args.cuvslam_run.resolve()),
        "trajectory_sha256": hashlib.sha256(
            (args.cuvslam_run / "trajectory_body.tum").read_bytes()
        ).hexdigest(),
        "messages": len(messages),
        "tracked_messages": sum(m.valid for m in messages),
        "imu_samples": len(imu),
        "feature_match_frames": len(matches),
        "period_ns": period_ns,
        "config": config.__dict__,
        "scenarios": {},
    }
    for spec, expected in scenarios(len(messages), period_ns, bool(imu)):
        events, window = build_events(messages, imu, spec, period_ns)
        all_rows = replay(events, config, args.output / f"{spec.kind}.jsonl", matches)
        rows = fault_class(all_rows)
        reasons: dict[str, int] = {}
        for _, _, row_reasons, _ in all_rows:
            for reason in row_reasons:
                if reason in DEGRADATION_REASONS:
                    reasons[reason] = reasons.get(reason, 0) + 1
        for _, _, row_reasons in rows:
            for reason in row_reasons:
                reasons[reason] = reasons.get(reason, 0) + 1
        score = (
            {"false_alarms": sum(status != "ok" for _, status, _ in rows)}
            if spec.kind == "none"
            else score_detection(
                rows, window=window, expected=expected, grace_ns=config.outage_timeout_ns
            )
        )
        summary["scenarios"][spec.kind] = {
            "spec": spec.__dict__,
            "expected_reasons": sorted(expected),
            "window_ns": list(window),
            **score,
            "reason_counts": reasons,
            "degradation_warnings": sum(
                bool(DEGRADATION_REASONS & set(r)) for _, _, r, _ in all_rows),
        }
        if spec.kind == "none" and series is not None:
            summary["scenarios"]["none"]["degradation_vs_groundtruth"] = score_degradation(
                all_rows, series)
    path = args.output / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.output / "SHA256SUMS").open("x", encoding="utf-8") as stream:
        for item in sorted(args.output.iterdir()):
            if item.is_file() and item.name != "SHA256SUMS":
                stream.write(f"{hashlib.sha256(item.read_bytes()).hexdigest()}  {item.name}\n")
    print(json.dumps({k: {kk: v.get(kk) for kk in ("detected", "latency_ns",
                      "false_alarms_outside_window", "false_alarms")} for k, v in
                      summary["scenarios"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
