"""
Replay a EuRoC sequence into PX4 SITL EKF2 with the sequence's real IMU (HIL).

AegisLand acts as the external simulator of PX4's `none_iris` airframe
(simulator_mavlink, TCP 4560 on localhost, lockstep):
  - HIL_SENSOR at the IMU rate: EuRoC imu0 accel/gyro rotated into body FRD
    (static gyro bias removed, like PX4's on-ground gyro calibration)
  - ODOMETRY: cuVSLAM poses -> aegisland.localization.v1 -> shadow supervisor v3
    -> TRUST_POLICY marking -> FRD/NED ODOMETRY, delivered `--latency-ms` after
    capture (EKF2_EV_DELAY is set to the same value)

So EKF2 fuses moving vision data with the matching IMU, unlike the stationary SIH
checks in sitl_ev_replay.py. Replay only: PX4 is never armed and no command,
setpoint or mode change is sent; the only outgoing messages are HEARTBEAT,
HIL_SENSOR and ODOMETRY to 127.0.0.1. CONTROL OUTPUT = DISABLED.

Note: the simulator link stamps ODOMETRY on arrival (no TIMESYNC on this path);
the TIMESYNC behaviour of the real MAVLink receiver is covered by sitl_ev_replay.py.

Policies:
  raw         every frame sent as the tracker reports it (quality 80 / -1 if lost)
  supervised  supervisor verdict: fault -> "untrusted" (quality 0),
              degraded -> "degraded" (variance floor), else "ok"
Faults (optional, one): freeze = the pipeline republishes the last pose with its
old capture time for `--fault-length-s` starting `--fault-at-s` into the replay.

Run with ~/.venvs/px4 and PYTHONPATH=src. Writes a NEW output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from pymavlink import mavutil
from sitl_ev_replay import Px4, _yaml_matrix

from aegisland.localization_message import SCHEMA, LocalizationMessage, TrackingState
from aegisland.localization_supervisor import LocalizationShadowSupervisor, SupervisorConfig
from aegisland.loop_closure_gate import AnchoredSlamOutput
from aegisland.mavlink_odometry import (
    _matrix_to_quaternion_wxyz,
    dataset_to_frd,
    to_odometry_fields,
)
from aegisland.trajectory_eval import load_euroc_groundtruth, load_tum, quaternion_to_matrix

HIL_ACCEL, HIL_GYRO = 0x7, 0x38
TOPICS = [
    "vehicle_local_position 0", "vehicle_attitude 0", "estimator_aid_src_ev_pos 0",
    "estimator_aid_src_ev_yaw 0", "estimator_status_flags 0", "vehicle_visual_odometry 0",
    "sensor_combined 0", "estimator_status 0",
]
QUALITY_TRACKING = 80
LOOP_CLOSURE_EVENT_WINDOW_NS = 100_000_000


def hil_params(latency_ms: float) -> dict:
    return {
        "SYS_HAS_MAG": 0, "SYS_HAS_GPS": 0, "SYS_HAS_BARO": 0,
        "EKF2_MAG_TYPE": 5, "EKF2_GPS_CTRL": 0, "EKF2_BARO_CTRL": 0, "EKF2_HGT_REF": 3,
        "EKF2_EV_CTRL": 11, "EKF2_EV_DELAY": latency_ms, "EKF2_EV_QMIN": 1,
        "EKF2_EV_NOISE_MD": 0, "EKF2_MULTI_IMU": 0, "SENS_IMU_MODE": 1,
        "IMU_INTEG_RATE": 200, "COM_ARM_WO_GPS": 1,
    }


class FrdConversion:
    """Dataset world/body -> FRD, from a stationary window (gravity) and cam0 forward."""

    def __init__(self, sequence: Path, trajectory, imu: np.ndarray, rest_start_s: float,
                 rest_end_s: float):
        rest = (imu[:, 0] >= rest_start_s) & (imu[:, 0] <= rest_end_s)
        if rest.sum() < 100:
            raise ValueError("stationary window has too few IMU samples")
        up_body = imu[rest, 4:7].mean(axis=0)
        up_body /= np.linalg.norm(up_body)
        self.gyro_bias_imu = imu[rest, 1:4].mean(axis=0)
        index = int(np.searchsorted(trajectory.timestamps_s, rest_start_s))
        world_from_body = quaternion_to_matrix(trajectory.quaternions_xyzw[index])
        forward_body = np.linalg.inv(_yaml_matrix(sequence / "imu0" / "sensor_recalibrated.yaml")
                                     )[:3, :3] @ np.array([0.0, 0.0, 1.0])
        down_body = -up_body
        x_body = forward_body - forward_body.dot(down_body) * down_body
        x_body /= np.linalg.norm(x_body)
        self.body_from_frd = np.column_stack([x_body, np.cross(down_body, x_body), down_body])
        down_world = world_from_body @ down_body
        x_world = world_from_body @ x_body
        x_world = x_world - x_world.dot(down_world) * down_world
        x_world /= np.linalg.norm(x_world)
        self.frd_from_world = np.column_stack(
            [x_world, np.cross(down_world, x_world), down_world]).T
        self.origin = self.frd_from_world @ trajectory.positions_m[index]

    def message(self, raw: LocalizationMessage) -> LocalizationMessage:
        converted = dataset_to_frd(raw, frd_from_dataset_world=self.frd_from_world,
                                   dataset_body_from_frd_body=self.body_from_frd)
        shifted = tuple(float(v) for v in np.array(converted.position_m) - self.origin)
        return replace(converted, position_m=shifted)


def dataset_messages(run: Path) -> list[LocalizationMessage]:
    frames = [json.loads(line) for line in (run / "frames.jsonl").read_text().splitlines()]
    trajectory = load_tum(run / "trajectory_body.tum")
    poses = iter(zip(trajectory.positions_m, trajectory.quaternions_xyzw))
    out, last = [], None
    for index, frame in enumerate(frames):
        if frame["tracked"]:
            position, quaternion = next(poses)
            last = (tuple(map(float, position)), tuple(map(float, quaternion)))
        if last is None:
            continue
        tracked = frame["tracked"]
        out.append(LocalizationMessage(
            schema=SCHEMA, source="cuvslam-17.0.1", sequence=index, frame_id="odom_dataset",
            child_frame_id="body_dataset", clock_domain="dataset",
            capture_time_ns=int(frame["timestamp_ns"]), publish_time_ns=int(frame["timestamp_ns"]),
            position_m=last[0], orientation_xyzw=last[1], velocity_mps=None,
            position_variance_m2=None, quality=QUALITY_TRACKING if tracked else -1,
            valid=tracked, tracking_state=TrackingState.TRACKING if tracked else TrackingState.LOST,
            reset_counter=0))
    return out


def _matrix(position, quaternion_xyzw) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = quaternion_to_matrix(np.asarray(quaternion_xyzw))
    out[:3, 3] = position
    return out


def slam_messages(run: Path, mode: str, *, jump_at_ns=None, jump_m=0.0, false_lc_at_ns=None):
    """Messages from a --slam run.

    naive: SLAM poses as returned by track() (jumps at loop closures, reset_counter 0)
    gated: AnchoredSlamOutput (continuous; reset_counter++ on accepted loop closures)
    Optional injections into the SLAM stream (not into odometry):
      jump_at_ns/jump_m : a +x jump WITHOUT a loop-closure event (a fault)
      false_lc_at_ns    : a loop-closure event carrying a 1 m correction (false closure)
    Returns (messages, per-frame gate log).
    """
    odom = load_tum(run / "trajectory_body.tum")
    slam = load_tum(run / "trajectory_slam_body.tum")
    frames = [json.loads(line) for line in (run / "frames.jsonl").read_text().splitlines()]
    if len(odom.timestamps_s) != len(slam.timestamps_s):
        raise ValueError("odometry and SLAM trajectories must have one row per tracked frame")
    events = {f["timestamp_ns"] for f in frames if f.get("loop_closure_event")}
    gate = AnchoredSlamOutput()
    out, log = [], []
    offset = np.zeros(3)
    for k, t in enumerate(odom.timestamps_s):
        capture = round(t * 1e9)
        # Async SLAM may apply a correction a frame after it reports the loop closure:
        # the event stays valid for LOOP_CLOSURE_EVENT_WINDOW_NS after it was reported.
        event = any(0 <= capture - e < LOOP_CLOSURE_EVENT_WINDOW_NS for e in events)
        slam_pose = _matrix(slam.positions_m[k], slam.quaternions_xyzw[k])
        if jump_at_ns is not None and capture >= jump_at_ns:
            offset = np.array([jump_m, 0.0, 0.0])
        if false_lc_at_ns is not None and abs(capture - false_lc_at_ns) < 25_000_000:
            event, slam_pose = True, slam_pose @ _matrix([1.0, 0, 0], [0, 0, 0, 1])
        slam_pose = slam_pose.copy()
        slam_pose[:3, 3] += offset
        odom_pose = _matrix(odom.positions_m[k], odom.quaternions_xyzw[k])
        if mode == "gated":
            decision = gate.update(odom_pose, slam_pose, event)
            pose, reset, verdict = decision.output, decision.reset_counter, decision.verdict
        else:
            pose, reset, verdict = slam_pose, 0, "naive"
        log.append({"capture_ns": capture, "event": event, "verdict": verdict,
                    "reset_counter": reset})
        w, x, y, z = _matrix_to_quaternion_wxyz(pose[:3, :3])
        out.append(LocalizationMessage(
            schema=SCHEMA, source="cuvslam-17.0.1-slam", sequence=k, frame_id="odom_dataset",
            child_frame_id="body_dataset", clock_domain="dataset", capture_time_ns=capture,
            publish_time_ns=capture, position_m=tuple(map(float, pose[:3, 3])),
            orientation_xyzw=(float(x), float(y), float(z), float(w)), velocity_mps=None,
            position_variance_m2=None, quality=QUALITY_TRACKING, valid=True,
            tracking_state=TrackingState.TRACKING, reset_counter=reset))
    return out, log


def inject_freeze(messages, start_ns: int, length_ns: int):
    """Republish the last pose before `start_ns` with its old capture time."""
    out, frozen = [], None
    for message in messages:
        if start_ns <= message.capture_time_ns < start_ns + length_ns:
            frozen = frozen or out[-1]
            out.append(replace(frozen, sequence=message.sequence,
                               publish_time_ns=message.capture_time_ns))
        else:
            out.append(message)
    return out


def verdict_of(assessment, policy: str) -> str:
    if policy == "raw":
        return "ok"
    if assessment.status == "fault":
        return "untrusted"
    return "degraded" if assessment.status == "degraded" else "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--px4-build", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True, help="mav0/")
    parser.add_argument("--cuvslam-run", type=Path, required=True)
    parser.add_argument("--vdr-run", type=Path, help="feature matches for supervisor v3")
    parser.add_argument("--start-s", type=float, required=True,
                        help="replay start, s after GT start; must begin with >=3 s at rest")
    parser.add_argument("--rest-s", type=float, default=3.0)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--policy", choices=("raw", "supervised"), required=True)
    parser.add_argument("--fault", choices=("none", "freeze", "drift"), default="none")
    parser.add_argument("--drift-mps", type=float, default=0.05, help="drift fault speed (FRD x)")
    parser.add_argument("--oracle-verdict", choices=("ok", "degraded", "untrusted"),
                        help="drift fault: verdict used inside the fault window (assumes the "
                             "supervisor detected it; isolates the effect of the marking)")
    parser.add_argument("--degraded-variance-m2", type=float, default=None,
                        help="override the 'degraded' variance floor (default TRUST_POLICY)")
    parser.add_argument("--fault-at-s", type=float, default=30.0)
    parser.add_argument("--fault-length-s", type=float, default=2.0)
    parser.add_argument("--latency-ms", type=float, default=30.0)
    parser.add_argument("--slam-output", choices=("odometry", "naive", "gated"),
                        default="odometry", help="needs a --slam run for naive/gated")
    parser.add_argument("--inject-slam-jump-at-s", type=float)
    parser.add_argument("--inject-slam-jump-m", type=float, default=0.3)
    parser.add_argument("--inject-false-lc-at-s", type=float)
    parser.add_argument("--speed", type=float, default=1.0, help="wall-clock pacing factor")
    parser.add_argument("--output", type=Path, required=True, help="new directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    gt = load_euroc_groundtruth(args.sequence / "state_groundtruth_estimate0" / "data.csv")
    t0 = gt.timestamps_s[0] + args.start_s
    t_end = t0 + args.duration_s
    imu = np.loadtxt(args.sequence / "imu0" / "data.csv", delimiter=",", comments="#")
    imu[:, 0] = imu[:, 0] / 1e9
    trajectory = load_tum(args.cuvslam_run / "trajectory_body.tum")
    frd = FrdConversion(args.sequence, trajectory, imu, t0, t0 + args.rest_s)

    gate_log = []
    if args.slam_output == "odometry":
        all_messages = dataset_messages(args.cuvslam_run)
    else:
        all_messages, gate_log = slam_messages(
            args.cuvslam_run, args.slam_output,
            jump_at_ns=None if args.inject_slam_jump_at_s is None
            else int((t0 + args.inject_slam_jump_at_s) * 1e9),
            jump_m=args.inject_slam_jump_m,
            false_lc_at_ns=None if args.inject_false_lc_at_s is None
            else int((t0 + args.inject_false_lc_at_s) * 1e9))
    messages = [m for m in all_messages if t0 <= m.capture_time_ns / 1e9 <= t_end]
    if args.fault == "freeze":
        messages = inject_freeze(messages, int((t0 + args.fault_at_s) * 1e9),
                                 int(args.fault_length_s * 1e9))
    matches = {}
    if args.vdr_run:
        for line in (args.vdr_run / "frames.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["success"]:
                matches[int(row["timestamp_ns"])] = float(row["match_count"])
    period = int(np.median(np.diff([m.capture_time_ns for m in messages])))
    supervisor = LocalizationShadowSupervisor(SupervisorConfig(expected_period_ns=period))

    imu = imu[(imu[:, 0] >= t0) & (imu[:, 0] <= t_end)]
    sim_us = lambda t: round((t - t0) * 1e6) + 1_000_000
    latency_s = args.latency_ms / 1000.0

    link = mavutil.mavlink_connection("tcpin:127.0.0.1:4560", source_system=1,
                                      source_component=197)
    px4 = Px4(args.px4_build.resolve(), args.output.resolve(), autostart="10016",
              simulator="none", topics=TOPICS, airframe_params=hil_params(args.latency_ms),
              wait_startup=False)
    px4.start()
    sent_log = (args.output / "sent.jsonl").open("x", encoding="utf-8")
    queue = list(messages)  # sorted by capture time
    next_message = 0
    wall0 = time.monotonic()
    counts = {"hil_sensor": 0, "odometry": 0, "actuator_replies": 0}
    try:
        deadline = time.monotonic() + 60
        while link.port is None:  # tcpin accepts PX4's connection inside recv
            link.recv_msg()
            if time.monotonic() > deadline:
                raise RuntimeError("PX4 did not connect to the simulator port")
            time.sleep(0.05)
        wall0 = time.monotonic()
        for row in imu:
            t = row[0]
            gyro_frd = frd.body_from_frd.T @ (row[1:4] - frd.gyro_bias_imu)
            accel_frd = frd.body_from_frd.T @ row[4:7]
            supervisor.add_imu(round(t * 1e9), tuple(row[1:4]))
            link.mav.hil_sensor_send(sim_us(t), *accel_frd, *gyro_frd, 0, 0, 0, 0, 0, 0, 20.0,
                                     HIL_ACCEL | HIL_GYRO, 0)
            counts["hil_sensor"] += 1
            # Deliver every message whose (capture + latency) has been reached.
            while next_message < len(queue) and (
                    queue[next_message].publish_time_ns / 1e9 + latency_s <= t):
                message = queue[next_message]
                next_message += 1
                now_ns = round(t * 1e9)
                assessment = supervisor.assess(
                    message, now_ns=now_ns, feature_matches=matches.get(message.capture_time_ns))
                verdict = verdict_of(assessment, args.policy)
                converted = frd.message(message)
                in_fault = (args.fault == "drift" and t0 + args.fault_at_s
                            <= message.capture_time_ns / 1e9 <= t0 + args.fault_at_s
                            + args.fault_length_s)
                if args.fault == "drift" and message.capture_time_ns / 1e9 >= t0 + args.fault_at_s:
                    elapsed = min(message.capture_time_ns / 1e9 - t0 - args.fault_at_s,
                                  args.fault_length_s)
                    x, y, z = converted.position_m
                    converted = replace(converted, position_m=(x + args.drift_mps * elapsed, y, z))
                if in_fault and args.oracle_verdict:
                    verdict = args.oracle_verdict
                fields = to_odometry_fields(converted, time_usec=sim_us(t), verdict=verdict)
                if verdict == "degraded" and args.degraded_variance_m2 is not None:
                    for k in (0, 6, 11):
                        fields["pose_covariance"][k] = args.degraded_variance_m2
                link.mav.odometry_send(**fields)
                counts["odometry"] += 1
                sent_log.write(json.dumps({
                    "t_rel_s": t - t0, "capture_rel_s": message.capture_time_ns / 1e9 - t0,
                    "status": assessment.status, "reasons": assessment.reasons,
                    "verdict": verdict, "quality": fields["quality"],
                    "var_x": fields["pose_covariance"][0],
                    "x": fields["x"], "y": fields["y"], "z": fields["z"],
                    "rotation_residual": assessment.rotation_residual_rad_s,
                }) + "\n")
            # Lockstep: wait briefly for PX4's HIL_ACTUATOR_CONTROLS for this step.
            reply = link.recv_match(type="HIL_ACTUATOR_CONTROLS", blocking=True, timeout=0.05)
            counts["actuator_replies"] += reply is not None
            due = wall0 + (t - t0) / args.speed
            if (delay := due - time.monotonic()) > 0:
                time.sleep(delay)
        time.sleep(1.0)
    finally:
        sent_log.close()
        px4.stop()
        link.close()
    logs = sorted((args.output / "rootfs" / "log").rglob("*.ulg"))
    summary = {
        "tool": "benchmarks/gps_denied/sitl_hil_replay.py",
        "control_output": "DISABLED", "command_executed": False,
        "px4": "v1.17.0 none_iris (simulator_mavlink, lockstep), localhost only",
        "args": {k: str(v) for k, v in vars(args).items()},
        "params": hil_params(args.latency_ms),
        "t0_dataset_s": t0, "counts": counts,
        "gate_verdicts": {v: sum(1 for g in gate_log if g["verdict"] == v
                                 and t0 <= g["capture_ns"] / 1e9 <= t_end)
                          for v in sorted({g["verdict"] for g in gate_log})},
        "final_reset_counter": max((m.reset_counter for m in messages), default=0),
        "gyro_bias_removed_rad_s": frd.gyro_bias_imu.tolist(),
        "frd_from_world": frd.frd_from_world.tolist(),
        "body_from_frd": frd.body_from_frd.tolist(),
        "origin_frd_m": frd.origin.tolist(),
        "ulog": str(logs[-1]) if logs else None,
        "ulog_sha256": hashlib.sha256(logs[-1].read_bytes()).hexdigest() if logs else None,
        "wall_s": time.monotonic() - wall0,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("counts", "ulog", "wall_s")}))
    return 0 if logs else 1


if __name__ == "__main__":
    raise SystemExit(main())
