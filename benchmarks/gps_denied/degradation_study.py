"""
Which online signals warn before cuVSLAM's error starts growing? (EuRoC MH_01-05)

Label (offline, ground truth): forward drift rate = translational RPE over the
next `--horizon-s` divided by the horizon (m of error per s). Windows that span
a ground-truth discontinuity (check_groundtruth_continuity.py) are excluded.

Signals (causal: value at frame i uses data up to frame i only):
  brightness, contrast, sharpness   cam0 image mean / std / Laplacian variance
  observations, landmarks           cuVSLAM exported counts
  new_landmarks                     landmark ids never seen before (keyframe churn)
  cov_position, cov_rotation        cuVSLAM covariance_xyz_rpy diagonal sums
  track_ms                          Tracker.track() wall time
  vdr_inliers, vdr_matches          VisualDeadReckoner RANSAC inlier ratio / matches
  vdr_disagreement                  1 - cos(VDR image flow, cuVSLAM predicted flow @ 5 m)
  gyro_residual                     |visual angular velocity - gyro (bias-tracked)|
  gyro_rate                         |gyro| (motion aggressiveness, a risk prior)
  partner_disagreement              RPE between this tracker and the other mode over
                                    the last horizon (needs both trackers running)
  partner_scale                     |log| ratio of path lengths, this vs other mode

Each signal's "bad" direction is declared up front (not fitted). Thresholds are
fitted on the TRAIN sequences only: the value that flags `--train-fpr` of the
clean training frames. Test sequences are only scored.

Run with the AegisLand environment; writes a NEW output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from compare_motion_direction import (
    load_imu,
    load_jsonl,
    poses_from_trajectory,
    project_factory,
    transform,
    yaml_list,
)

from aegisland.degradation_metrics import (
    causal_mean,
    clean_mask,
    find_events,
    forward_drift_rate,
    interpolate_poses,
    roc_auc,
    rotation_log,
    score_alarms,
)
from aegisland.localization_diagnostics import predicted_center_flow, vdr_image_flow
from aegisland.trajectory_eval import load_euroc_groundtruth, load_tum

SEQUENCES = ("MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult")
# Declared before any analysis (see docs/dronehacks/degradation-study.md).
TRAIN = ("MH_01_easy", "MH_03_medium", "MH_04_difficult")
TEST = ("MH_02_easy", "MH_05_difficult")
# +1: high values are bad, -1: low values are bad. Declared, not fitted.
DIRECTION = {
    "brightness": -1, "contrast": -1, "sharpness": -1,
    "observations": -1, "landmarks": -1, "new_landmarks": +1,
    "cov_position": +1, "cov_rotation": +1, "track_ms": +1,
    "vdr_inliers": -1, "vdr_matches": -1, "vdr_disagreement": +1,
    "gyro_residual": +1, "gyro_rate": +1,
    "partner_disagreement": +1, "partner_scale": +1,
}
SMOOTH_FRAMES = 10  # 0.5 s at 20 Hz, causal
BIAS_WINDOW_FRAMES = 200  # 10 s causal gyro-bias tracking
VDR_DEPTH_M = 5.0
NULL_SHIFTS = 200
NULL_MIN_SHIFT_S = 15.0
NULL_SEED = 20261024


def image_stats(sequence: Path, timestamps_ns: list[int], cache: Path) -> dict[str, np.ndarray]:
    if cache.is_file():
        stored = json.loads(cache.read_text())
        if stored["timestamps_ns"] == timestamps_ns:
            return {k: np.array(v) for k, v in stored.items() if k != "timestamps_ns"}
    names = {}
    for line in (sequence / "cam0" / "data.csv").read_text().splitlines():
        if line and not line.startswith("#"):
            stamp, name = line.split(",")
            names[int(stamp)] = name.strip()
    stats = {"brightness": [], "contrast": [], "sharpness": []}
    for stamp in timestamps_ns:
        image = cv2.imread(str(sequence / "cam0" / "data" / names[stamp]), cv2.IMREAD_GRAYSCALE)
        stats["brightness"].append(float(image.mean()))
        stats["contrast"].append(float(image.std()))
        stats["sharpness"].append(float(cv2.Laplacian(image, cv2.CV_64F).var()))
    cache.write_text(json.dumps({"timestamps_ns": timestamps_ns, **stats}))
    return {k: np.array(v) for k, v in stats.items()}


def gt_excluded(times_s: np.ndarray, horizon_s: float, discontinuities_s: list[float]):
    excluded = np.zeros(len(times_s), dtype=bool)
    for t in discontinuities_s:
        excluded |= (times_s <= t) & (times_s + horizon_s >= t)
    return excluded


def window_rpe(times_s, poses_a, poses_b, horizon_s) -> tuple[np.ndarray, np.ndarray]:
    """Backward-window RPE between two estimates and |log| path-length ratio (causal)."""
    count = len(times_s)
    rpe, scale = np.full(count, np.nan), np.full(count, np.nan)
    starts = np.searchsorted(times_s, times_s - horizon_s)
    for i in range(count):
        j = starts[i]
        if times_s[i] - times_s[j] < 0.9 * horizon_s:
            continue
        rel_a = np.linalg.inv(poses_a[j]) @ poses_a[i]
        rel_b = np.linalg.inv(poses_b[j]) @ poses_b[i]
        rpe[i] = np.linalg.norm((np.linalg.inv(rel_b) @ rel_a)[:3, 3]) / horizon_s
        length_a = np.sum(np.linalg.norm(np.diff(poses_a[j : i + 1, :3, 3], axis=0), axis=1))
        length_b = np.sum(np.linalg.norm(np.diff(poses_b[j : i + 1, :3, 3], axis=0), axis=1))
        if min(length_a, length_b) > 0.1:
            scale[i] = abs(np.log(length_a / length_b))
    return rpe, scale


def sequence_table(args, name: str, mode: str, partner: str) -> dict:
    sequence = args.data / name / "mav0"
    run = next(args.results.glob(f"cuvslam-{mode}-{name}-{args.signals_tag}-signals"))
    other = next(args.results.glob(f"cuvslam-{partner}-{name}-{args.signals_tag}-signals"))
    frames = load_jsonl(run / "frames.jsonl")
    trajectory = load_tum(run / "trajectory_body.tum")
    times = trajectory.timestamps_s
    stamps = [row["timestamp_ns"] for row in frames if row["tracked"]]
    rows = [row for row in frames if row["tracked"]]
    if len(rows) != len(times):
        raise ValueError(f"{run}: tracked frames and trajectory rows differ")
    estimate = poses_from_trajectory(trajectory)

    gt = load_euroc_groundtruth(sequence / "state_groundtruth_estimate0" / "data.csv")
    reference, valid = interpolate_poses(gt.timestamps_s, poses_from_trajectory(gt), times)
    drift, drift_rot = forward_drift_rate(times, estimate, reference, valid,
                                          horizon_s=args.horizon_s)
    gt_t0 = gt.timestamps_s[0]
    breaks = [gt_t0 + event["t_rel_s"] for event in args.continuity[name]["discontinuities"]]
    excluded = gt_excluded(times, args.horizon_s, breaks)
    drift[excluded] = np.nan
    drift_rot[excluded] = np.nan

    raw: dict[str, np.ndarray] = {}
    raw.update(image_stats(sequence, stamps, args.data / name / "aegisland_image_stats.json"))
    raw["observations"] = np.array([row["observations"][0] for row in rows], dtype=float)
    raw["landmarks"] = np.array([row["landmarks"] for row in rows], dtype=float)
    raw["new_landmarks"] = np.array([row["new_landmarks"] for row in rows], dtype=float)
    covariance = np.array([row["covariance_xyz_rpy_diag"] or [np.nan] * 6 for row in rows])
    raw["cov_position"] = covariance[:, :3].sum(axis=1)
    raw["cov_rotation"] = covariance[:, 3:].sum(axis=1)
    raw["track_ms"] = np.array([row["track_ms"] for row in rows])

    vdr_run = max(args.results.glob(f"vdr-{name}-2026*"))
    vdr = {row["timestamp_ns"]: row for row in load_jsonl(vdr_run / "frames.jsonl")}
    vdr_rows = [vdr.get(stamp) for stamp in stamps]
    raw["vdr_inliers"] = np.array([r["inlier_ratio"] if r and r["success"] else np.nan
                                   for r in vdr_rows])
    raw["vdr_matches"] = np.array([r["match_count"] if r and r["success"] else np.nan
                                   for r in vdr_rows], dtype=float)

    cam0_yaml = sequence / "cam0" / "sensor_recalibrated.yaml"
    width, height = (int(v) for v in yaml_list(cam0_yaml, "resolution"))
    project, center_ray = project_factory(cam0_yaml)
    body_from_cam0 = np.linalg.inv(transform(sequence / "imu0" / "sensor_recalibrated.yaml"))
    cam_poses = estimate @ body_from_cam0
    disagreement = np.full(len(times), np.nan)
    for i in range(1, len(times)):
        row = vdr_rows[i]
        if not row or not row["success"]:
            continue
        flow = predicted_center_flow(np.linalg.inv(cam_poses[i]) @ cam_poses[i - 1], project,
                                     center_ray, depth_m=VDR_DEPTH_M)
        observed = np.array(vdr_image_flow(row["vx_normalized"], row["vy_normalized"],
                                           width, height))
        if np.linalg.norm(flow) >= 1.0 and np.linalg.norm(observed) > 0:
            cosine = observed @ flow / (np.linalg.norm(observed) * np.linalg.norm(flow))
            disagreement[i] = 1.0 - cosine
    raw["vdr_disagreement"] = disagreement

    imu_times, gyro = load_imu(sequence / "imu0" / "data.csv")
    visual_rate = np.full((len(times), 3), np.nan)
    gyro_mean = np.full((len(times), 3), np.nan)
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        visual_rate[i] = rotation_log(estimate[i - 1, :3, :3].T @ estimate[i, :3, :3]) / dt
        inside = (imu_times > times[i - 1]) & (imu_times <= times[i])
        if inside.any():
            gyro_mean[i] = gyro[inside].mean(axis=0)
    difference = visual_rate - gyro_mean
    residual = np.full(len(times), np.nan)
    for i in range(1, len(times)):
        history = difference[max(1, i - BIAS_WINDOW_FRAMES) : i]
        history = history[np.all(np.isfinite(history), axis=1)]
        bias = np.median(history, axis=0) if len(history) >= 20 else np.zeros(3)
        if np.all(np.isfinite(difference[i])):
            residual[i] = np.linalg.norm(difference[i] - bias)
    raw["gyro_residual"] = residual
    raw["gyro_rate"] = np.linalg.norm(gyro_mean, axis=1)

    partner_trajectory = load_tum(other / "trajectory_body.tum")
    partner_index = {round(t, 6): k for k, t in enumerate(partner_trajectory.timestamps_s)}
    partner_poses = poses_from_trajectory(partner_trajectory)
    matched = np.array([partner_index.get(round(t, 6), -1) for t in times])
    if np.all(matched >= 0):
        raw["partner_disagreement"], raw["partner_scale"] = window_rpe(
            times, estimate, partner_poses[matched], args.horizon_s)
    else:
        raw["partner_disagreement"] = raw["partner_scale"] = np.full(len(times), np.nan)

    smoothed = {key: causal_mean(values, SMOOTH_FRAMES) for key, values in raw.items()}
    return {"times": times, "drift": drift, "drift_rot": drift_rot, "signals": smoothed,
            "excluded_gt_frames": int(excluded.sum()), "run": str(run)}


def evaluate(tables: dict, args) -> dict:
    report: dict[str, object] = {}
    events, clean = {}, {}
    for name, table in tables.items():
        events[name] = find_events(table["times"], table["drift"], threshold=args.label_threshold,
                                   merge_gap_s=args.merge_gap_s, min_length_s=args.min_length_s)
        clean[name] = clean_mask(table["times"], np.isfinite(table["drift"]), events[name],
                                 lead_window_s=args.lead_window_s, horizon_s=args.horizon_s)
    report["events"] = {
        name: [event.__dict__ | {"onset_rel_s": event.onset_s - tables[name]["times"][0]}
               for event in values]
        for name, values in events.items()
    }
    report["signals"] = {}
    report["supervisor_v3_rule"] = combined_v3(tables, events, clean, args)
    for signal, direction in DIRECTION.items():
        oriented = {n: direction * t["signals"][signal] for n, t in tables.items()}
        train_values = np.concatenate([oriented[n][clean[n]] for n in TRAIN if n in tables])
        train_values = train_values[np.isfinite(train_values)]
        if train_values.size == 0:
            report["signals"][signal] = {"skipped": "no finite training values"}
            continue
        threshold = float(np.quantile(train_values, 1.0 - args.train_fpr))
        per_sequence = {}
        for name, table in tables.items():
            alarm = np.isfinite(oriented[name]) & (oriented[name] > threshold)
            positive = np.isfinite(table["drift"]) & (table["drift"] > args.label_threshold)
            scored = score_alarms(table["times"], alarm, events[name], clean[name],
                                  lead_window_s=args.lead_window_s, horizon_s=args.horizon_s)
            label_valid = np.isfinite(table["drift"])
            scored["frame_auc"] = roc_auc(oriented[name][label_valid], positive[label_valid])
            scored["spearman_vs_drift"] = spearman(oriented[name], table["drift"])
            per_sequence[name] = scored
        observed = {}
        for split, names in (("train", TRAIN), ("test", TEST)):
            valid = [np.isfinite(tables[n]["drift"]) for n in names]
            observed[split] = {
                "detected": sum(per_sequence[n]["detected"] for n in names),
                "pooled_auc": roc_auc(
                    np.concatenate([oriented[n][v] for n, v in zip(names, valid)]),
                    np.concatenate([tables[n]["drift"][v] > args.label_threshold
                                    for n, v in zip(names, valid)])),
            }
        null = null_baseline(tables, oriented, threshold, events, clean, args, observed)
        report["signals"][signal] = {
            "pooled_auc": {split: observed[split]["pooled_auc"] for split in observed},
            "direction": "high is bad" if direction > 0 else "low is bad",
            "null_shifted": null,
            "threshold_raw": threshold * direction,
            "per_sequence": per_sequence,
            "train": pool([per_sequence[n] for n in TRAIN if n in per_sequence]),
            "test": pool([per_sequence[n] for n in TEST if n in per_sequence]),
        }
    return report


def null_baseline(tables, oriented, threshold, events, clean, args, observed) -> dict:
    """Same signal, same threshold, circularly shifted in time by >= NULL_MIN_SHIFT_S.

    Keeps the signal's own distribution and autocorrelation but breaks its timing
    relation to the label. Per split, each draw shifts every sequence independently;
    p-values = share of draws at least as good as the real signal (with +1 smoothing).
    """
    rng = np.random.default_rng(NULL_SEED)
    out = {}
    for split, names in (("train", TRAIN), ("test", TEST)):
        detections, aucs = [], []
        for _ in range(NULL_SHIFTS):
            detected, scores, labels = 0, [], []
            for name in names:
                table = tables[name]
                count = len(table["times"])
                low = int(NULL_MIN_SHIFT_S / float(np.median(np.diff(table["times"]))))
                shifted = np.roll(oriented[name], int(rng.integers(low, count - low)))
                alarm = np.isfinite(shifted) & (shifted > threshold)
                detected += score_alarms(table["times"], alarm, events[name], clean[name],
                                         lead_window_s=args.lead_window_s,
                                         horizon_s=args.horizon_s)["detected"]
                valid = np.isfinite(table["drift"])
                scores.append(shifted[valid])
                labels.append(table["drift"][valid] > args.label_threshold)
            detections.append(detected)
            aucs.append(roc_auc(np.concatenate(scores), np.concatenate(labels)))
        detections = np.array(detections)
        aucs = np.array([a for a in aucs if a is not None])
        real = observed[split]
        out[split] = {
            "detected_mean": float(detections.mean()),
            "detected_p95": float(np.percentile(detections, 95)),
            "p_detection": float((1 + np.sum(detections >= real["detected"]))
                                 / (1 + len(detections))),
            "pooled_auc_mean": float(aucs.mean()) if aucs.size else None,
            "pooled_auc_p95": float(np.percentile(aucs, 95)) if aucs.size else None,
            "p_auc": None if real["pooled_auc"] is None or not aucs.size
            else float((1 + np.sum(aucs >= real["pooled_auc"])) / (1 + aucs.size)),
        }
    return out


def combined_v3(tables, events, clean, args) -> dict:
    """Supervisor v3 rule on the same causal series: gyro_residual > R OR vdr_matches < M.

    R and M are the supervisor's fixed defaults (fitted on TRAIN earlier); both
    series are shifted by the same amount in the null so their joint timing is kept.
    """
    from aegisland.localization_supervisor import SupervisorConfig

    defaults = SupervisorConfig(expected_period_ns=1)
    residual_max, matches_min = defaults.rotation_residual_rad_s, defaults.min_feature_matches

    def alarm_of(table, shift=0):
        residual = np.roll(table["signals"]["gyro_residual"], shift)
        matches = np.roll(table["signals"]["vdr_matches"], shift)
        return (np.isfinite(residual) & (residual > residual_max)) | (
            np.isfinite(matches) & (matches < matches_min))

    rng = np.random.default_rng(NULL_SEED)
    out = {"rule": f"gyro_residual > {residual_max} OR vdr_matches < {matches_min}"}
    for split, names in (("train", TRAIN), ("test", TEST)):
        per = {n: score_alarms(tables[n]["times"], alarm_of(tables[n]), events[n], clean[n],
                               lead_window_s=args.lead_window_s, horizon_s=args.horizon_s)
               for n in names}
        real = sum(r["detected"] for r in per.values())
        null = []
        for _ in range(NULL_SHIFTS):
            total = 0
            for n in names:
                count = len(tables[n]["times"])
                low = int(NULL_MIN_SHIFT_S / float(np.median(np.diff(tables[n]["times"]))))
                shifted = alarm_of(tables[n], int(rng.integers(low, count - low)))
                total += score_alarms(tables[n]["times"], shifted, events[n], clean[n],
                                      lead_window_s=args.lead_window_s,
                                      horizon_s=args.horizon_s)["detected"]
            null.append(total)
        out[split] = {
            "per_sequence": per,
            **pool(list(per.values())),
            "null_detected_mean": float(np.mean(null)),
            "p_detection": float((1 + np.sum(np.array(null) >= real)) / (1 + len(null))),
        }
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 10:
        return None
    rank_a = np.argsort(np.argsort(a[finite]))
    rank_b = np.argsort(np.argsort(b[finite]))
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def pool(results: list[dict]) -> dict:
    events = sum(r["events"] for r in results)
    detected = sum(r["detected"] for r in results)
    leads = [lead for r in results for lead in r["lead_s"]]
    clean = sum(r["clean_frames"] for r in results)
    false = sum((r["false_alarm_frame_rate"] or 0) * r["clean_frames"] for r in results)
    aucs = [r["frame_auc"] for r in results if r.get("frame_auc") is not None]
    return {
        "events": events,
        "detected": detected,
        "miss_rate": None if not events else 1 - detected / events,
        "early_warnings": int(sum(lead > 0 for lead in leads)),
        "lead_s_median": float(np.median(leads)) if leads else None,
        "false_alarm_frame_rate": None if not clean else false / clean,
        "frame_auc_mean": float(np.mean(aucs)) if aucs else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path.home() / "aegisland-data" / "euroc")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--signals-tag", required=True, help="tag of the *-signals runs")
    parser.add_argument("--continuity", type=Path, required=True, help="continuity.json")
    parser.add_argument("--mode", choices=("stereo", "inertial"), required=True)
    parser.add_argument("--horizon-s", type=float, default=1.0)
    parser.add_argument("--label-threshold", type=float, required=True, help="m/s of drift")
    parser.add_argument("--merge-gap-s", type=float, default=1.0)
    parser.add_argument("--min-length-s", type=float, default=0.25)
    parser.add_argument("--lead-window-s", type=float, default=3.0)
    parser.add_argument("--train-fpr", type=float, default=0.05)
    parser.add_argument("--label-stats-only", action="store_true",
                        help="print TRAIN label distribution and exit (choose threshold)")
    parser.add_argument("--output", type=Path, help="new directory")
    args = parser.parse_args()
    args.continuity = json.loads(args.continuity.read_text())["sequences"]
    partner = "inertial" if args.mode == "stereo" else "stereo"
    names = TRAIN if args.label_stats_only else SEQUENCES
    tables = {name: sequence_table(args, name, args.mode, partner) for name in names}
    if args.label_stats_only:
        for name, table in tables.items():
            drift = table["drift"][np.isfinite(table["drift"])]
            print(name, "drift m/s percentiles 50/90/95/99:",
                  np.round(np.percentile(drift, [50, 90, 95, 99]), 4),
                  "excluded GT frames", table["excluded_gt_frames"])
        return 0
    report = {
        "tool": "benchmarks/gps_denied/degradation_study.py",
        "claims": "Offline EuRoC study; labels use ground truth, signals are causal. "
                  "Development GPU; not field data.",
        "mode": args.mode,
        "partner_mode": partner,
        "split": {"train": TRAIN, "test": TEST},
        "settings": {k: v for k, v in vars(args).items()
                     if k not in ("continuity", "data", "results", "output")},
        "runs": {name: table["run"] for name, table in tables.items()},
        "excluded_gt_frames": {name: t["excluded_gt_frames"] for name, t in tables.items()},
        **evaluate(tables, args),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "study.json"
    path.write_text(json.dumps(report, indent=1, default=float) + "\n")
    series = {name: {"times": t["times"].tolist(), "drift": t["drift"].tolist(),
                     **{k: v.tolist() for k, v in t["signals"].items()}}
              for name, t in tables.items()}
    (args.output / "series.json").write_text(json.dumps(series))
    with (args.output / "SHA256SUMS").open("x") as stream:
        for item in sorted(args.output.iterdir()):
            if item.name != "SHA256SUMS":
                stream.write(f"{hashlib.sha256(item.read_bytes()).hexdigest()}  {item.name}\n")
    rule = report["supervisor_v3_rule"]
    for split in ("train", "test"):
        r = rule[split]
        print(f"v3 rule {split}: detected {r['detected']}/{r['events']} (null mean "
              f"{r['null_detected_mean']:.1f}, p={r['p_detection']:.3f}), early {r['early_warnings']},"
              f" lead median {r['lead_s_median']}, clean-frame FA {r['false_alarm_frame_rate']:.3f}")
    print(f"{'signal':21s} | {'train det/ev':>12s} {'null':>5s} {'p':>5s} {'AUC':>5s} {'p':>5s} | "
          f"{'test det/ev':>11s} {'null':>5s} {'p':>5s} {'early':>5s} {'lead':>5s} {'FA':>5s} "
          f"{'AUC':>5s} {'p':>5s}")
    for signal, result in report["signals"].items():
        if "train" not in result:
            continue
        tr, te, nu, auc = result["train"], result["test"], result["null_shifted"], result["pooled_auc"]

        def f(value, spec=".2f"):
            return "-" if value is None else format(value, spec)

        print(f"{signal:21s} | {tr['detected']:>5d}/{tr['events']:<6d} "
              f"{f(nu['train']['detected_mean'], '.1f'):>5s} {f(nu['train']['p_detection'], '.3f'):>5s} "
              f"{f(auc['train']):>5s} {f(nu['train']['p_auc'], '.3f'):>5s} | "
              f"{te['detected']:>4d}/{te['events']:<6d} {f(nu['test']['detected_mean'], '.1f'):>5s} "
              f"{f(nu['test']['p_detection'], '.3f'):>5s} {te['early_warnings']:>5d} "
              f"{f(te['lead_s_median'], '.1f'):>5s} {f(te['false_alarm_frame_rate'], '.3f'):>5s} "
              f"{f(auc['test']):>5s} {f(nu['test']['p_auc'], '.3f'):>5s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
