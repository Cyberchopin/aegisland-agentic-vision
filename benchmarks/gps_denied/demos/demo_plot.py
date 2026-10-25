"""
Charts + metrics for the three SITL demos (reads sitl_hil_replay.py outputs).

  a  localization: GT vs cuVSLAM (sent) vs EKF2 top view, and position error
  b  degradation: error over time with supervisor v3 warnings, brightness, gyro residual
  c  fault: raw vs supervised EKF2 error, the quality that was sent, EV fusion state

Errors: SE(3)-aligned to ground truth (no scale), in meters; for (c) the
alignment uses only the pre-fault part so the fault's effect is not averaged
away. Run with ~/.venvs/px4 and PYTHONPATH=src. Writes NEW files only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyulog import ULog

from aegisland.trajectory_eval import load_euroc_groundtruth, umeyama_alignment

# Reference palette (dataviz skill), validated light-mode categorical slots 1-4.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
WARNING, CRITICAL = "#fab219", "#d03b3b"
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def load_run(run: Path, sequence: Path) -> dict:
    summary = json.loads((run / "summary.json").read_text())
    sent = [json.loads(line) for line in (run / "sent.jsonl").read_text().splitlines()]
    log = ULog(summary["ulog"])
    data = {d.name: d.data for d in log.data_list if d.multi_id == 0}
    lp = data["vehicle_local_position"]
    t0 = summary["t0_dataset_s"]
    ekf_t = lp["timestamp"] / 1e6 - 1.0  # sim time started at 1 s
    ekf_xyz = np.column_stack([lp["x"], lp["y"], lp["z"]])
    aid = data["estimator_aid_src_ev_pos"]
    gt = load_euroc_groundtruth(sequence / "state_groundtruth_estimate0" / "data.csv")
    return {
        "summary": summary, "sent": sent, "t0": t0,
        "ekf_t": ekf_t, "ekf_xyz": ekf_xyz,
        "fused_t": aid["timestamp"] / 1e6 - 1.0, "fused": aid["fused"].astype(bool),
        "gt_t": gt.timestamps_s - t0, "gt_xyz": gt.positions_m,
    }


def aligned_error(t, xyz, run, fit_mask=None):
    """Align estimate positions to GT (SE3) and return per-sample error (m) + aligned xyz."""
    gt = np.column_stack([np.interp(t, run["gt_t"], run["gt_xyz"][:, k]) for k in range(3)])
    inside = (t >= run["gt_t"][0]) & (t <= run["gt_t"][-1])
    fit = inside if fit_mask is None else inside & fit_mask
    rotation, translation, _ = umeyama_alignment(xyz[fit], gt[fit], with_scale=False)
    aligned = xyz @ rotation.T + translation
    error = np.linalg.norm(aligned - gt, axis=1)
    error[~inside] = np.nan
    return error, aligned, gt


def sent_arrays(run):
    # Positions refer to the capture time, not the delivery time.
    t = np.array([row["capture_rel_s"] for row in run["sent"]])
    xyz = np.array([[row["x"], row["y"], row["z"]] for row in run["sent"]])
    return t, xyz


def rms(values):
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values ** 2)))


def demo_a(run, out: Path) -> dict:
    t, xyz = sent_arrays(run)
    cu_err, cu_aligned, _ = aligned_error(t, xyz, run)
    ekf_err, ekf_aligned, _ = aligned_error(run["ekf_t"], run["ekf_xyz"], run)
    fig, (top, err) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [1, 1.4]})
    gt_t = run["gt_t"]
    keep = (gt_t >= 0) & (gt_t <= run["ekf_t"][-1])
    top.plot(run["gt_xyz"][keep, 0], run["gt_xyz"][keep, 1], color=MUTED, lw=2, label="ground truth")
    top.plot(cu_aligned[:, 0], cu_aligned[:, 1], color=BLUE, lw=1.2, label="cuVSLAM (sent)")
    top.plot(ekf_aligned[:, 0], ekf_aligned[:, 1], color=ORANGE, lw=1.2, label="PX4 EKF2")
    top.set_aspect("equal")
    style(top, "Top view (SE(3)-aligned to ground truth)", "y (m)")
    top.set_xlabel("x (m)", color=MUTED)
    top.legend(frameon=False, fontsize=8)
    err.plot(t, cu_err, color=BLUE, lw=1.2, label=f"cuVSLAM sent  RMS {rms(cu_err):.3f} m")
    err.plot(run["ekf_t"], ekf_err, color=ORANGE, lw=1.2,
             label=f"PX4 EKF2 (EV + EuRoC IMU)  RMS {rms(ekf_err):.3f} m")
    style(err, "Position error vs ground truth", "error (m)")
    err.set_xlabel("time since replay start (s)", color=MUTED)
    err.legend(frameon=False, fontsize=8)
    fig.suptitle("Demo A - EuRoC MH_01 replayed into PX4 SITL EKF2 via MAVLink ODOMETRY "
                 "(offline dataset, SITL only)", fontsize=10, color=MUTED, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    return {"cuvslam_sent_rms_m": rms(cu_err), "ekf2_rms_m": rms(ekf_err),
            "ekf2_max_m": float(np.nanmax(ekf_err)), "frames_sent": len(t)}


def shade(ax, times, mask, color, label):
    """Shade contiguous runs where mask is true (one legend entry)."""
    first = True
    start = None
    for k in range(len(times)):
        if mask[k] and start is None:
            start = times[k]
        if (not mask[k] or k == len(times) - 1) and start is not None:
            ax.axvspan(start, times[k], color=color, alpha=0.25, lw=0,
                       label=label if first else None)
            first, start = False, None


def demo_b(raw, supervised, series: dict, out: Path, threshold: float) -> dict:
    t_raw, xyz_raw = sent_arrays(raw)
    cu_err, _, _ = aligned_error(t_raw, xyz_raw, raw)
    ekf_raw, _, _ = aligned_error(raw["ekf_t"], raw["ekf_xyz"], raw)
    ekf_sup, _, _ = aligned_error(supervised["ekf_t"], supervised["ekf_xyz"], supervised)
    t_sup = np.array([r["t_rel_s"] for r in supervised["sent"]])
    rotation = np.array(["rotation_inconsistent" in r["reasons"] for r in supervised["sent"]])
    texture = np.array(["low_texture" in r["reasons"] for r in supervised["sent"]])
    residual = np.array([np.nan if r["rotation_residual"] is None else r["rotation_residual"]
                         for r in supervised["sent"]])
    series_t = np.array(series["times"]) - raw["t0"]
    brightness = np.array(series["brightness"], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.2), sharex=True, facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [1.6, 1, 1]})
    ax = axes[0]
    shade(ax, t_sup, rotation, CRITICAL, "supervisor: rotation_inconsistent (degraded)")
    shade(ax, t_sup, texture & ~rotation, WARNING, "supervisor: low_texture only (degraded)")
    ax.plot(t_raw, cu_err, color=BLUE, lw=1.2, label="cuVSLAM output (always 'tracking')")
    ax.plot(raw["ekf_t"], ekf_raw, color=ORANGE, lw=1.2, label="EKF2, raw ODOMETRY")
    ax.plot(supervised["ekf_t"], ekf_sup, color=AQUA, lw=1.2,
            label="EKF2, degraded -> variance >= 1 m^2")
    style(ax, "Position error vs ground truth (SE(3)-aligned) - cuVSLAM never reports a "
              "tracking failure", "error (m)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axes[1]
    keep = (series_t >= 0) & (series_t <= t_raw[-1])
    ax.plot(series_t[keep], brightness[keep], color=MUTED, lw=1.2)
    style(ax, "cam0 mean brightness (0-255)", "brightness")
    ax = axes[2]
    ax.plot(t_sup, residual, color=BLUE, lw=1.2, label="|visual - gyro| angular rate, 0.5 s mean")
    ax.axhline(threshold, color=CRITICAL, lw=1, ls="--", label=f"v3 threshold {threshold} rad/s")
    style(ax, "Gyro consistency check (supervisor v3)", "rad/s")
    ax.set_ylim(0, min(0.2, np.nanpercentile(residual, 99.5) * 1.3))
    ax.set_xlabel("time since replay start (s)", color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Demo B - gradual degradation on EuRoC MH_05 (dark segment), PX4 SITL EKF2 "
                 "(offline dataset, SITL only)", fontsize=10, color=MUTED, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    dark = keep & (brightness < 30)
    return {
        "cuvslam_sent_rms_m": rms(cu_err), "ekf2_raw_rms_m": rms(ekf_raw),
        "ekf2_supervised_rms_m": rms(ekf_sup),
        "ekf2_raw_max_m": float(np.nanmax(ekf_raw)), "ekf2_supervised_max_m": float(np.nanmax(ekf_sup)),
        "dark_segment_s": [float(series_t[dark].min()), float(series_t[dark].max())]
        if dark.any() else None,
        "rotation_inconsistent_fraction": float(rotation.mean()),
        "low_texture_fraction": float(texture.mean()),
        "tracking_lost_frames": int(sum(r["quality"] == -1 for r in raw["sent"])),
    }


def demo_c(raw, supervised, out: Path, fault_at: float, length: float) -> dict:
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [1.6, 1, 0.8]})
    window = (fault_at - 8, fault_at + 12)
    metrics = {}
    for run, color, name in ((raw, ORANGE, "raw"), (supervised, AQUA, "supervised")):
        pre = run["ekf_t"] < fault_at
        err, _, _ = aligned_error(run["ekf_t"], run["ekf_xyz"], run, fit_mask=pre)
        after = (run["ekf_t"] >= fault_at) & (run["ekf_t"] <= fault_at + 10)
        metrics[f"ekf2_{name}_max_error_10s_after_fault_m"] = float(np.nanmax(err[after]))
        metrics[f"ekf2_{name}_error_before_fault_rms_m"] = rms(err[pre])
        axes[0].plot(run["ekf_t"], err, color=color, lw=1.4,
                     label=f"EKF2, {name} bridge (max {np.nanmax(err[after]):.2f} m in 10 s)")
        t = np.array([r["t_rel_s"] for r in run["sent"]])
        quality = np.array([r["quality"] for r in run["sent"]])
        axes[1].step(t, quality, where="post", color=color, lw=1.4, label=f"{name} bridge")
        axes[2].step(run["fused_t"], run["fused"].astype(int) + (0.04 if name == "raw" else 0),
                     where="post", color=color, lw=1.4, label=f"{name}")
        statuses = [r for r in run["sent"] if fault_at - 1 <= r["t_rel_s"] <= fault_at + length + 1]
        metrics[f"{name}_reasons_during_fault"] = sorted({x for r in statuses for x in r["reasons"]})
    for ax in axes:
        ax.axvspan(fault_at, fault_at + length, color=CRITICAL, alpha=0.15, lw=0)
        ax.set_xlim(*window)
    axes[0].text(fault_at + 0.1, axes[0].get_ylim()[1] * 0.92, "injected fault:\nVIO output frozen",
                 color=CRITICAL, fontsize=8, va="top")
    style(axes[0], "EKF2 position error vs ground truth (aligned on pre-fault data)", "error (m)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    style(axes[1], "ODOMETRY quality sent (EKF2_EV_QMIN = 1)", "quality")
    axes[1].set_yticks([0, 80])
    axes[1].set_ylim(-8, 88)
    axes[1].legend(frameon=False, fontsize=8, loc="center left")
    style(axes[2], "EKF2 fused the EV position sample (1 = yes)", "fused")
    axes[2].set_yticks([0, 1])
    axes[2].set_xlabel("time since replay start (s)", color=MUTED)
    axes[2].legend(frameon=False, fontsize=8, loc="center left")
    fig.suptitle("Demo C - injected fault (frozen VIO output, 2 s) on EuRoC MH_01, PX4 SITL EKF2 "
                 "(offline dataset, SITL only)", fontsize=10, color=MUTED, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    return metrics


MAGENTA = "#e87ba4"


def demo_marking(runs: dict, out: Path, fault_at: float, length: float) -> dict:
    """Slow drift with an oracle verdict: how much each marking lets EKF2 follow."""
    colors = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
    fig, (err_ax, eph_ax) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, facecolor=SURFACE)
    metrics = {}
    end = fault_at + length
    for color, (name, run) in zip(colors, runs.items()):
        log = ULog(run["summary"]["ulog"])
        lp = next(d.data for d in log.data_list if d.name == "vehicle_local_position")
        t = run["ekf_t"]
        err, _, _ = aligned_error(t, run["ekf_xyz"], run, fit_mask=t < fault_at)
        at_end = (t >= end - 0.5) & (t <= end)
        invalid = t[(t > fault_at) & (lp["xy_valid"] == 0)]
        metrics[name] = {
            "error_at_end_of_drift_m": float(np.nanmean(err[at_end])),
            "eph_max_m": float(np.max(lp["eph"][(t > fault_at) & (t <= end)])),
            "xy_invalid_after_s": float(invalid[0] - fault_at) if invalid.size else None,
        }
        err_ax.plot(t, err, color=color, lw=1.4, label=name)
        err_ax.annotate(name, (end, float(np.nanmean(err[at_end]))), xytext=(4, 0),
                        textcoords="offset points", fontsize=8, color=INK, va="center")
        eph_ax.plot(t, lp["eph"], color=color, lw=1.4, label=name)
    for ax in (err_ax, eph_ax):
        ax.axvspan(fault_at, end, color=CRITICAL, alpha=0.10, lw=0)
        ax.set_xlim(fault_at - 5, end + 8)
    style(err_ax, "EKF2 position error vs ground truth while the vision position drifts "
                  "5 cm/s (shaded)", "error (m)")
    err_ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(eph_ax, "EKF2 reported horizontal position uncertainty (eph)", "eph (m)")
    eph_ax.set_yscale("log")
    eph_ax.set_xlabel("time since replay start (s)", color=MUTED)
    eph_ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Marking a slow drift: variance inflation vs quality 0 (EuRoC MH_01 + injected "
                 "drift, PX4 SITL EKF2, oracle verdict)", fontsize=10, color=MUTED, x=0.01,
                 ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    return metrics



def demo_gate(runs: dict, out: Path, inject_at: dict) -> dict:
    """Loop-closure handling: odometry vs naive SLAM vs gated SLAM (+ injected faults)."""
    colors = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, CRITICAL]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [1.6, 1]})
    metrics = {}
    for color, (name, run) in zip(colors, runs.items()):
        log = ULog(run["summary"]["ulog"])
        data = {d.name: d.data for d in log.data_list if d.multi_id == 0}
        lp, aid = data["vehicle_local_position"], data["estimator_aid_src_ev_pos"]
        t = run["ekf_t"]
        # Align on the part before any injected fault, so a fault is not averaged away.
        fit = t < (min(inject_at.values()) - 2.0) if inject_at else None
        err, _, _ = aligned_error(t, run["ekf_xyz"], run, fit_mask=fit)
        resets = np.diff(lp["xy_reset_counter"].astype(int))
        metrics[name] = {
            "ekf2_rms_m": rms(err), "ekf2_max_m": float(np.nanmax(err)),
            "ekf2_xy_resets": int(np.sum(resets > 0)),
            "ev_rejected_fraction": float(np.mean(aid["innovation_rejected"])),
            "gate_verdicts": run["summary"].get("gate_verdicts"),
            "final_reset_counter_sent": run["summary"].get("final_reset_counter"),
        }
        axes[0].plot(t, err, color=color, lw=1.2, label=f"{name} (RMS {rms(err):.3f} m)")
        reset_times = t[1:][resets > 0]
        axes[1].eventplot([reset_times], lineoffsets=len(metrics), colors=[color],
                          linelengths=0.7)
    for name, at in inject_at.items():
        for ax in axes:
            ax.axvline(at, color=CRITICAL, lw=1, ls="--")
        axes[0].text(at + 0.3, axes[0].get_ylim()[1] * 0.92, name, color=CRITICAL, fontsize=8)
    style(axes[0], "EKF2 position error vs ground truth (SE(3)-aligned on the first 38 s, before "
                   "any injection), MH_05 stereo + cuVSLAM SLAM", "error (m)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    style(axes[1], "EKF2 horizontal position resets (xy_reset_counter increments)", "run")
    axes[1].set_yticks(range(1, len(metrics) + 1))
    axes[1].set_yticklabels(list(metrics), fontsize=8)
    axes[1].set_xlabel("time since replay start (s)", color=MUTED)
    fig.suptitle("Loop-closure corrections into PX4 SITL EKF2: naive jumps vs gated reset_counter "
                 "(offline dataset, SITL only)", fontsize=10, color=MUTED, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    return metrics

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("demo", choices=("a", "b", "c", "m", "g"))
    parser.add_argument("--inject-labels", nargs="*", default=[],
                        help="demo g: label=seconds markers")
    parser.add_argument("--marking-runs", nargs="*", type=Path, help="demo m: run dirs")
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True, help="raw run (b, c) or the run (a)")
    parser.add_argument("--supervised-run", type=Path)
    parser.add_argument("--series", type=Path, help="degradation series.json (b)")
    parser.add_argument("--sequence-name")
    parser.add_argument("--fault-at-s", type=float, default=60.0)
    parser.add_argument("--fault-length-s", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True, help="new directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    run = load_run(args.run, args.sequence)
    png = args.output / f"demo_{args.demo}.png"
    if args.demo == "g":
        runs = {path.name: load_run(path, args.sequence) for path in args.marking_runs}
        marks = {item.split("=")[0]: float(item.split("=")[1]) for item in args.inject_labels}
        metrics = demo_gate(runs, png, marks)
    elif args.demo == "m":
        runs = {path.name: load_run(path, args.sequence) for path in args.marking_runs}
        metrics = demo_marking(runs, png, args.fault_at_s, args.fault_length_s)
    elif args.demo == "a":
        metrics = demo_a(run, png)
    elif args.demo == "b":
        from aegisland.localization_supervisor import SupervisorConfig

        series = json.loads(args.series.read_text())[args.sequence_name]
        metrics = demo_b(run, load_run(args.supervised_run, args.sequence), series, png,
                         SupervisorConfig(expected_period_ns=1).rotation_residual_rad_s)
    else:
        metrics = demo_c(run, load_run(args.supervised_run, args.sequence), png,
                         args.fault_at_s, args.fault_length_s)
    metrics["inputs"] = {"run": str(args.run), "supervised_run": str(args.supervised_run),
                         "ulog_sha256": run["summary"]["ulog_sha256"]}
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
