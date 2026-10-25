"""
Pre-race self-check: the four EKF2 external-vision traps found in SITL, each PASS/FAIL.

  1 frames      we send FRD/NED only; EKF2 heading equals the heading we sent;
                the bridge refuses ENU/FLU and dataset frames
  2 timesync    with TIMESYNC answered, PX4 keeps our capture latency
                (without it PX4 silently uses the arrival time)
  3 slow_drift  EKF2 does NOT reject slow drift by itself, so the protection must be
                ours: EKF2_EV_QMIN >= 1 and an "untrusted" frame stops fusion
  4 failsafe    quality 0 has a cost: the bridge never sends it for "degraded"
                (team decision 2026-10-24), and the time from "untrusted" to
                local-position-invalid is measured and must lie in the planned band

Limit of check 3: SIH's IMU is ideal and stationary, so "EKF2 stops following" here
also means "EKF2 stays near the truth". With a real IMU (HIL replay of EuRoC MH_01,
demo-drift-marking-20261024-121340) 20 s of quality 0 meant 10 m of inertial drift
and an invalid position after 4.6 s: the check proves the marking works, not that
quality 0 is a good answer to a long drift.

Each check runs its own fresh PX4 SITL (SIH, localhost only, never armed; only
HEARTBEAT, TIMESYNC replies and ODOMETRY are sent). Exit code 0 = all PASS.
Run with ~/.venvs/px4 and PYTHONPATH=src (from benchmarks/gps_denied/).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

from sitl_ev_replay import Scenario, run_scenario

from aegisland.localization_message import SCHEMA, LocalizationMessage, TrackingState
from aegisland.mavlink_odometry import UNTRUSTED_QUALITY, to_odometry_fields

HEADING_TOLERANCE_DEG = 2.0
SENT_HEADING_DEG = 35.0  # static_message() default in sitl_ev_replay
LATENCY_S = 0.15
MIN_KEPT_LATENCY_FRACTION = 0.5
DRIFT_TOTAL_M = 1.0  # 0.05 m/s for 20 s in the untrusted_drift plan
MAX_FOLLOWED_DRIFT_M = 0.2
FAILSAFE_BAND_S = (2.0, 15.0)


def message(**overrides) -> LocalizationMessage:
    fields = {
        "schema": SCHEMA, "source": "preflight", "sequence": 0, "frame_id": "odom_frd",
        "child_frame_id": "body_frd", "clock_domain": "host_monotonic", "capture_time_ns": 0,
        "publish_time_ns": 0, "position_m": (0.0, 0.0, 0.0),
        "orientation_xyzw": (0.0, 0.0, 0.0, 1.0), "velocity_mps": None,
        "position_variance_m2": None, "quality": 80, "valid": True,
        "tracking_state": TrackingState.TRACKING, "reset_counter": 0,
    }
    fields.update(overrides)
    return LocalizationMessage(**fields)


def bridge_refuses(frame_id: str, child: str) -> bool:
    try:
        to_odometry_fields(message(frame_id=frame_id, child_frame_id=child), time_usec=0)
    except ValueError:
        return True
    return False


def check_frames(build: Path, out: Path) -> dict:
    result = run_scenario(Scenario("preflight_static", 20), build, out, None)
    samples = result["ekf2"].get("local_position_samples", [])
    valid = [s for s in samples if s["xy_valid"] == 1.0]
    error = (min(abs(((s["heading_deg"] - SENT_HEADING_DEG + 180) % 360) - 180) for s in valid[-3:])
             if valid else math.inf)
    refuses = {"odom_dataset": bridge_refuses("odom_dataset", "body_dataset"),
               "odom_enu": bridge_refuses("odom_enu", "body_flu")}
    passed = error <= HEADING_TOLERANCE_DEG and all(refuses.values())
    return {"pass": passed, "heading_error_deg": error, "bridge_refuses": refuses,
            "detail": f"sent {SENT_HEADING_DEG} deg FRD; EKF2 heading error {error:.2f} deg"}


def check_timesync(build: Path, out: Path) -> dict:
    scenario = Scenario("time_timesync", 20, latency_s=LATENCY_S, warmup_s=70)
    result = run_scenario(scenario, build, out, None)
    age = result["ekf2"].get("visual_odometry_timestamp_minus_sample_ms", {})
    kept = (age.get("median") or 0.0) / 1000.0
    passed = result["timesync_replies"] > 0 and kept >= MIN_KEPT_LATENCY_FRACTION * LATENCY_S
    return {"pass": passed, "timesync_replies": result["timesync_replies"],
            "latency_sent_ms": LATENCY_S * 1e3, "latency_kept_ms": kept * 1e3,
            "detail": "PX4 must keep >= 50% of the capture latency once TIMESYNC converged"}


def _qmin(result) -> int | None:
    lines = result["params_readback"].get("EKF2_EV_QMIN", [])
    try:
        return int(lines[-1].split(":")[-1])
    except (IndexError, ValueError):
        return None


def check_drift_and_failsafe(build: Path, out: Path, qmin: int = 1) -> tuple[dict, dict]:
    result = run_scenario(Scenario("untrusted_drift", 32, params={"EKF2_EV_QMIN": qmin}),
                          build, out, None)
    ekf2 = result["ekf2"]
    samples = ekf2.get("local_position_samples", [])
    sent = result["sent_samples"]
    followed = (samples[-1]["x"] - sent[0]["x"]) if samples and sent else math.inf
    qmin = _qmin(result)
    drift = {"pass": qmin is not None and qmin >= 1 and abs(followed) <= MAX_FOLLOWED_DRIFT_M,
             "EKF2_EV_QMIN": qmin, "drift_sent_m": DRIFT_TOTAL_M,
             "drift_followed_by_ekf2_m": followed,
             "detail": "slow drift marked untrusted must not move the EKF2 position"}
    fused = [r for r in ekf2.get("ev_pos_timeline", []) if "fused" in r]
    stop = next((r["t_s"] for a, r in itertools.pairwise(fused) if a["fused"] and not r["fused"]),
                None)
    invalid = next((r["t_s"] for r in ekf2.get("xy_valid_timeline", [])
                    if stop is not None and r["t_s"] > stop and not r["xy_valid"]), None)
    to_invalid = None if stop is None or invalid is None else invalid - stop
    degraded = to_odometry_fields(message(), time_usec=0, verdict="degraded")["quality"]
    untrusted = to_odometry_fields(message(), time_usec=0, verdict="untrusted")["quality"]
    failsafe = {
        "pass": (degraded != UNTRUSTED_QUALITY and untrusted == UNTRUSTED_QUALITY
                 and to_invalid is not None
                 and FAILSAFE_BAND_S[0] <= to_invalid <= FAILSAFE_BAND_S[1]),
        "quality_for_degraded": degraded, "quality_for_untrusted": untrusted,
        "untrusted_to_position_invalid_s": to_invalid, "expected_band_s": FAILSAFE_BAND_S,
        "detail": "quality 0 makes EKF2 dead-reckon; position goes invalid -> failsafe",
    }
    return drift, failsafe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--px4-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory")
    parser.add_argument("--only", nargs="*", choices=("frames", "timesync", "drift"),
                        help="run a subset (drift = checks 3 and 4)")
    parser.add_argument("--ekf2-ev-qmin", type=int, default=1,
                        help="vehicle setting under test; 0 reproduces the unsafe default")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    build = args.px4_build.resolve()
    out = args.output.resolve()
    wanted = set(args.only or ("frames", "timesync", "drift"))
    checks = {}
    if "frames" in wanted:
        checks["1_frames"] = check_frames(build, out)
    if "timesync" in wanted:
        checks["2_timesync"] = check_timesync(build, out)
    if "drift" in wanted:
        checks["3_slow_drift"], checks["4_quality0_failsafe"] = check_drift_and_failsafe(
            build, out, args.ekf2_ev_qmin)
    report = {"tool": "benchmarks/gps_denied/preflight_sitl_check.py",
              "control_output": "DISABLED", "command_executed": False,
              "scope": "PX4 v1.17.0 SITL (SIH) on localhost; not a check of the real vehicle",
              "ekf2_ev_qmin_under_test": args.ekf2_ev_qmin,
              "all_pass": all(c["pass"] for c in checks.values()), "checks": checks}
    (out / "preflight.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    for name, check in checks.items():
        print(f"{'PASS' if check['pass'] else 'FAIL'}  {name:22s} {check['detail']}")
        print("      " + json.dumps({k: v for k, v in check.items()
                                     if k not in ("pass", "detail")}, default=str))
    print("ALL PASS" if report["all_pass"] else "SOME CHECKS FAILED")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
