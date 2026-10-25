"""
Replay AegisLand localization messages into PX4 SITL (SIH) as MAVLink ODOMETRY
and record how EKF2 treats them. Replay only: the vehicle is never armed, no
command, setpoint or mode change is sent. Only HEARTBEAT (companion), TIMESYNC
replies and ODOMETRY are transmitted, to 127.0.0.1.

Each scenario starts a fresh PX4 instance (own rootfs, own parameters, own ulog),
streams one scripted sequence and parses the ulog afterwards:

  frames_euroc_static   cuVSLAM MH_01 poses (stationary part) through dataset_to_frd
  enu_pitfall           same poses sent as LOCAL_ENU with an ENU/FLU quaternion
  time_no_timesync      150 ms capture latency, no TIMESYNC replies
  time_timesync         150 ms capture latency, TIMESYNC answered
  quality_qmin0         QMIN=0, quality -1 ("failed") the whole time
  quality_qmin1         QMIN=1, quality 80 -> 0 -> 80 -> -1 -> 80
  variance              NOISE_MD=0, variance NaN -> 1e-4 -> 1 -> 25 m^2
  jump                  1 m step at 8 s, back at 16 s
  slow_drift            0.05 m/s drift from 8 s for 20 s
  reset_jump            1 m step at 8 s with reset_counter + 1
  degraded_drift        slow drift with the "degraded" variance marking
  untrusted_drift       slow drift marked "untrusted" (quality) under QMIN=1

The SIH vehicle sits still on the ground, so only stationary-consistent poses
can be fused; moving replays would test IMU/vision inconsistency, not fusion.

Run with ~/.venvs/px4 (pymavlink, pyulog) and PYTHONPATH=src. Writes a NEW
output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from pymavlink import mavutil
from pyulog import ULog

from aegisland.localization_message import SCHEMA, LocalizationMessage, TrackingState
from aegisland.mavlink_odometry import dataset_to_frd, to_odometry_fields
from aegisland.trajectory_eval import load_euroc_groundtruth, load_tum, quaternion_to_matrix

RATE_HZ = 20.0
COMPANION_CLOCK_OFFSET_US = 7_000_000_000  # companion clock != PX4 clock, on purpose
BASE_PARAMS = {
    "EKF2_GPS_CTRL": 0,
    "EKF2_MAG_TYPE": 5,
    "EKF2_EV_CTRL": 15,
    "EKF2_HGT_REF": 3,
    "EKF2_EV_DELAY": 0.0,
    "EKF2_EV_QMIN": 0,
    "EKF2_EV_NOISE_MD": 0,
}
LOGGED_TOPICS = [
    "estimator_aid_src_ev_pos 0",
    "estimator_aid_src_ev_hgt 0",
    "estimator_aid_src_ev_yaw 0",
    "estimator_status_flags 0",
    "estimator_status 0",
    "vehicle_local_position 0",
    "vehicle_attitude 0",
    "vehicle_visual_odometry 0",
    "vehicle_status 0",
    "timesync_status 0",
]


@dataclass
class Scenario:
    name: str
    duration_s: float
    params: dict = field(default_factory=dict)
    timesync: bool = True
    latency_s: float = 0.02
    warmup_s: float = 0.0  # answer TIMESYNC but send no ODOMETRY yet


def yaw_deg(q_wxyz) -> float:
    w, x, y, z = q_wxyz
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def static_message(position=(1.0, 2.0, -0.6), yaw=35.0) -> LocalizationMessage:
    half = math.radians(yaw) / 2
    return LocalizationMessage(
        schema=SCHEMA, source="sitl-script", sequence=0, frame_id="odom_frd",
        child_frame_id="body_frd", clock_domain="host_monotonic", capture_time_ns=0,
        publish_time_ns=0, position_m=tuple(position),
        orientation_xyzw=(0.0, 0.0, math.sin(half), math.cos(half)), velocity_mps=None,
        position_variance_m2=None, quality=80, valid=True,
        tracking_state=TrackingState.TRACKING, reset_counter=0,
    )


class EurocStatic:
    """cuVSLAM MH_01 poses during the ground-truth-stationary segment, FRD-converted."""

    def __init__(self, run: Path, sequence: Path, start_s: float, end_s: float, yaw0_deg: float):
        trajectory = load_tum(run / "trajectory_body.tum")
        gt = load_euroc_groundtruth(sequence / "state_groundtruth_estimate0" / "data.csv")
        t0 = gt.timestamps_s[0]
        keep = (trajectory.timestamps_s >= t0 + start_s) & (trajectory.timestamps_s <= t0 + end_s)
        self.positions = trajectory.positions_m[keep]
        self.quaternions = trajectory.quaternions_xyzw[keep]
        # World: cuVSLAM world after world_from_body; its z axis is unknown a priori, so
        # derive "down" from the mean IMU specific force while stationary.
        imu = np.loadtxt(sequence / "imu0" / "data.csv", delimiter=",", comments="#")
        rest = (imu[:, 0] / 1e9 >= t0 + start_s) & (imu[:, 0] / 1e9 <= t0 + end_s)
        up_body = imu[rest, 4:7].mean(axis=0)
        up_body /= np.linalg.norm(up_body)
        world_from_body = quaternion_to_matrix(self.quaternions[0])
        up_world = world_from_body @ up_body
        # Body FRD: z = down, x = cam0 optical axis projected on the horizontal plane.
        cam0_from_imu = _yaml_matrix(sequence / "imu0" / "sensor_recalibrated.yaml")
        forward_body = np.linalg.inv(cam0_from_imu)[:3, :3] @ np.array([0.0, 0.0, 1.0])
        down_body = -up_body
        x_body = forward_body - forward_body.dot(down_body) * down_body
        x_body /= np.linalg.norm(x_body)
        self.dataset_body_from_frd_body = np.column_stack(
            [x_body, np.cross(down_body, x_body), down_body]
        )
        # World FRD: z = down; x = initial body forward rotated by yaw0 (known heading).
        down_world = -up_world
        forward_world = world_from_body @ x_body
        x_world = forward_world - forward_world.dot(down_world) * down_world
        x_world /= np.linalg.norm(x_world)
        dataset_from_frd = np.column_stack([x_world, np.cross(down_world, x_world), down_world])
        yaw0 = math.radians(yaw0_deg)
        rotate = np.array([[math.cos(yaw0), -math.sin(yaw0), 0], [math.sin(yaw0), math.cos(yaw0), 0],
                           [0, 0, 1.0]])
        self.frd_from_dataset_world = rotate @ dataset_from_frd.T
        self.origin = self.frd_from_dataset_world @ self.positions[0]

    def message(self, index: int) -> LocalizationMessage:
        i = index % len(self.positions)
        raw = replace(
            static_message(), frame_id="odom_dataset", child_frame_id="body_dataset",
            position_m=tuple(float(v) for v in self.positions[i]),
            orientation_xyzw=tuple(float(v) for v in self.quaternions[i]),
        )
        converted = dataset_to_frd(
            raw, frd_from_dataset_world=self.frd_from_dataset_world,
            dataset_body_from_frd_body=self.dataset_body_from_frd_body,
        )
        # Start near the EKF origin; offsets are irrelevant to the frame check.
        shifted = tuple(float(v) for v in np.array(converted.position_m) - self.origin)
        return replace(converted, position_m=shifted)


def _yaml_matrix(path: Path) -> np.ndarray:
    import re

    text = path.read_text()
    match = re.search(r"data:\s*\[([^\]]*)\]", text, re.DOTALL)
    return np.array([float(v) for v in match.group(1).replace("\n", " ").split(",")]).reshape(4, 4)


def plan(name: str, t: float, euroc: EurocStatic | None, step: int):
    """Return (message, verdict, raw_fields_override) for scenario time t."""
    base = static_message()
    if name in ("frames_euroc_static", "enu_pitfall"):
        return euroc.message(step), "ok"
    if name == "quality_qmin0":
        return replace(base, quality=-1, valid=False, tracking_state=TrackingState.LOST), "ok"
    if name == "quality_qmin1":
        quality = 80 if t < 8 else 0 if t < 16 else 80 if t < 24 else -1 if t < 32 else 80
        if quality == -1:
            return replace(base, quality=-1, valid=False, tracking_state=TrackingState.LOST), "ok"
        return replace(base, quality=quality), "ok"
    if name == "variance":
        variance = None if t < 6 else (1e-4,) * 3 if t < 12 else (1.0,) * 3 if t < 18 else (25.0,) * 3
        return replace(base, position_variance_m2=variance), "ok"
    if name == "jump":
        dx = 1.0 if 8 <= t < 16 else 0.0
        return replace(base, position_m=(1.0 + dx, 2.0, -0.6)), "ok"
    if name == "reset_jump":
        dx, reset = (1.0, 1) if t >= 8 else (0.0, 0)
        return replace(base, position_m=(1.0 + dx, 2.0, -0.6), reset_counter=reset), "ok"
    if name in ("slow_drift", "degraded_drift", "untrusted_drift"):
        dx = 0.05 * min(max(t - 8, 0.0), 20.0)
        verdict = "ok"
        if t >= 8 and name == "degraded_drift":
            verdict = "degraded"
        if t >= 8 and name == "untrusted_drift":
            verdict = "untrusted"
        return replace(base, position_m=(1.0 + dx, 2.0, -0.6)), verdict
    return base, "ok"


SCENARIOS = [
    Scenario("frames_euroc_static", 25),
    Scenario("enu_pitfall", 25),
    Scenario("time_no_timesync", 20, timesync=False, latency_s=0.15),
    # PX4 Timesync converges after 500 accepted samples (CONVERGENCE_WINDOW); until
    # then sync_stamp() returns the arrival time.
    Scenario("time_timesync", 20, latency_s=0.15, warmup_s=70),
    Scenario("quality_qmin0", 20),
    Scenario("quality_qmin1", 40, params={"EKF2_EV_QMIN": 1}),
    Scenario("variance", 24),
    Scenario("jump", 24),
    Scenario("slow_drift", 32),
    Scenario("reset_jump", 16),
    Scenario("degraded_drift", 32),
    Scenario("untrusted_drift", 32, params={"EKF2_EV_QMIN": 1}),
]


class Px4:
    def __init__(self, build: Path, workdir: Path, *, autostart: str = "10040",
                 simulator: str = "sihsim", topics: list[str] | None = None,
                 airframe_params: dict | None = None, wait_startup: bool = True):
        self.build, self.workdir = build, workdir
        self.autostart, self.simulator = autostart, simulator
        self.topics = LOGGED_TOPICS if topics is None else topics
        self.airframe_params = airframe_params or {}
        self.wait_startup = wait_startup
        self.process: subprocess.Popen | None = None
        self.log = None

    def start(self) -> None:
        etc = self.workdir / "etc_src"
        if not etc.exists():
            shutil.copytree(self.build / "etc", etc, symlinks=True)
            logging_dir = etc / "logging"
            logging_dir.mkdir(exist_ok=True)
            (logging_dir / "logger_topics.txt").write_text("\n".join(self.topics) + "\n")
            if self.airframe_params:
                # Parameters for a single boot: appended to this run's copy of the airframe.
                airframe = next((etc / "init.d-posix" / "airframes").glob(f"{self.autostart}_*"))
                airframe.unlink()  # copied as a symlink? replace with a real file
                source = next((self.build / "etc" / "init.d-posix" / "airframes").glob(
                    f"{self.autostart}_*"))
                lines = [f"param set {k} {v}" for k, v in self.airframe_params.items()]
                airframe.write_text(source.read_text() + "\n" + "\n".join(lines) + "\n")
        rootfs = self.workdir / "rootfs"
        rootfs.mkdir(exist_ok=True)
        env = dict(os.environ, PX4_SYS_AUTOSTART=self.autostart, PX4_SIMULATOR=self.simulator)
        self.log = (self.workdir / "px4_stdout.log").open("a")
        self.process = subprocess.Popen(
            [str(self.build / "bin" / "px4"), "-d", str(etc)], cwd=rootfs, env=env,
            stdout=self.log, stderr=subprocess.STDOUT,
        )
        if not self.wait_startup:
            return
        deadline = time.time() + 30
        while time.time() < deadline:
            if "Startup script returned" in (self.workdir / "px4_stdout.log").read_text():
                return
            time.sleep(0.5)
        raise RuntimeError("PX4 did not start")

    def wait_ready(self) -> None:
        deadline = time.time() + 30
        while time.time() < deadline:
            if "PX4 server not running" not in self.command("param", "show", "SYS_AUTOSTART"):
                return
            time.sleep(0.5)
        raise RuntimeError("PX4 command server not reachable")

    def command(self, *args: str) -> str:
        return subprocess.run(
            [str(self.build / "bin" / f"px4-{args[0]}"), *args[1:]], capture_output=True,
            text=True, check=False, timeout=10,
        ).stdout

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.command("shutdown")
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.send_signal(signal.SIGKILL)
                self.process.wait()
        if self.log:
            self.log.close()


def stream(scenario: Scenario, euroc: EurocStatic | None) -> dict:
    link = mavutil.mavlink_connection(
        "udpin:127.0.0.1:14540", source_system=1, source_component=197
    )
    started = time.monotonic()
    companion_us = lambda: time.monotonic_ns() // 1000 + COMPANION_CLOCK_OFFSET_US
    sent = timesync_replies = 0
    next_send = next_heartbeat = started
    sent_log = []
    while (now := time.monotonic()) - started < scenario.warmup_s + scenario.duration_s:
        message = link.recv_match(blocking=False)
        while message is not None:
            if (scenario.timesync and message.get_type() == "TIMESYNC" and message.tc1 == 0):
                link.mav.timesync_send(companion_us() * 1000, message.ts1)
                timesync_replies += 1
            message = link.recv_match(blocking=False)
        if now >= next_heartbeat:
            link.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                                    mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            next_heartbeat += 1.0
        if now - started < scenario.warmup_s:
            next_send = now
        elif now >= next_send and not link.clients:
            next_send += 1.0 / RATE_HZ  # PX4 not heard yet; pymavlink udpin has no peer
        elif now >= next_send:
            t = now - started - scenario.warmup_s
            payload, verdict = plan(scenario.name, t, euroc, sent)
            capture_us = companion_us() - int(scenario.latency_s * 1e6)
            fields = to_odometry_fields(payload, time_usec=capture_us, verdict=verdict) \
                if scenario.name != "enu_pitfall" else enu_fields(payload, capture_us)
            link.mav.odometry_send(**fields)
            sent += 1
            sent_log.append({"t_s": t, "x": fields["x"], "y": fields["y"], "z": fields["z"],
                             "yaw_deg": yaw_deg(fields["q"]), "quality": fields["quality"],
                             "var_x": fields["pose_covariance"][0], "verdict": verdict,
                             "frame_id": fields["frame_id"]})
            next_send += 1.0 / RATE_HZ
        time.sleep(0.002)
    link.close()
    return {"odometry_sent": sent, "timesync_replies": timesync_replies, "sent": sent_log}


def enu_fields(message: LocalizationMessage, capture_us: int) -> dict:
    """Negative test: what a ROS-style ENU/FLU sender would emit (not our policy)."""
    frd = to_odometry_fields(message, time_usec=capture_us)
    # NED->ENU position, FRD->FLU body + NED->ENU world for the quaternion.
    ned_to_enu = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1.0]])
    frd_to_flu = np.diag([1.0, -1.0, -1.0])
    w, x, y, z = frd["q"]
    rotation = ned_to_enu @ quaternion_to_matrix(np.array([x, y, z, w])) @ frd_to_flu
    from aegisland.mavlink_odometry import _matrix_to_quaternion_wxyz

    position = ned_to_enu @ np.array([frd["x"], frd["y"], frd["z"]])
    frd.update(frame_id=mavutil.mavlink.MAV_FRAME_LOCAL_ENU, x=float(position[0]),
               y=float(position[1]), z=float(position[2]),
               q=[float(v) for v in _matrix_to_quaternion_wxyz(rotation)])
    return frd


def analyse(ulog_path: Path, sent: list[dict]) -> dict:
    log = ULog(str(ulog_path))
    data = {d.name: d.data for d in log.data_list if d.multi_id == 0}
    start = log.start_timestamp

    def series(topic: str, key: str):
        if topic not in data or key not in data[topic]:
            return None, None
        return (data[topic]["timestamp"] - start) / 1e6, data[topic][key]

    out: dict[str, object] = {}
    t_flags, ev_pos = series("estimator_status_flags", "cs_ev_pos")
    if t_flags is not None:
        out["cs_ev_pos_fraction"] = float(np.mean(ev_pos))
        out["flags_timeline"] = _timeline(t_flags, {
            key: data["estimator_status_flags"][key]
            for key in ("cs_ev_pos", "cs_ev_yaw", "cs_ev_hgt", "cs_inertial_dead_reckoning")
            if key in data["estimator_status_flags"]
        })
    if "estimator_aid_src_ev_pos" in data:
        aid = data["estimator_aid_src_ev_pos"]
        t = (aid["timestamp"] - start) / 1e6
        out["ev_pos_timeline"] = _timeline(t, {
            "fused": aid["fused"], "innovation_rejected": aid["innovation_rejected"],
            "obs_var_x": np.round(aid["observation_variance[0]"], 5),
            "test_ratio_x": np.round(aid["test_ratio[0]"], 2),
        }, numeric_keys=("obs_var_x", "test_ratio_x"))
    if "vehicle_visual_odometry" in data:
        vo = data["vehicle_visual_odometry"]
        age_ms = (vo["timestamp"].astype(np.int64) - vo["timestamp_sample"].astype(np.int64)) / 1e3
        out["visual_odometry_timestamp_minus_sample_ms"] = {
            "median": float(np.median(age_ms)), "p5": float(np.percentile(age_ms, 5)),
            "p95": float(np.percentile(age_ms, 95)), "count": int(age_ms.size),
        }
        out["visual_odometry_quality_values"] = sorted({int(q) for q in vo["quality"]})
    if "timesync_status" in data:
        ts = data["timesync_status"]
        t = (ts["timestamp"] - start) / 1e6
        out["timesync_status_samples"] = _sample_at(t, {
            "estimated_offset_us": ts["estimated_offset"], "round_trip_time_us": ts["round_trip_time"],
        }, every_s=10.0)
        out["timesync_status_count"] = int(t.size)
    if "vehicle_local_position" in data:
        lp = data["vehicle_local_position"]
        t = (lp["timestamp"] - start) / 1e6
        out["xy_valid_timeline"] = _timeline(t, {"xy_valid": lp["xy_valid"]})
        out["local_position_samples"] = _sample_at(t, {
            "x": lp["x"], "y": lp["y"], "z": lp["z"], "heading_deg": np.degrees(lp["heading"]),
            "xy_valid": lp["xy_valid"],
        }, every_s=2.0)
    return out


def _timeline(t, columns: dict, numeric_keys=()) -> list[dict]:
    """Compress a timeline to the rows where any boolean column changes (+ numbers)."""
    rows, previous = [], None
    for i in range(len(t)):
        if t[i] < 0 or t[i] > 1e6:  # unset (zero) timestamps
            continue
        state = tuple(bool(columns[k][i]) for k in columns if k not in numeric_keys)
        if state != previous:
            row = {"t_s": round(float(t[i]), 2)}
            row.update({k: (float(columns[k][i]) if k in numeric_keys else bool(columns[k][i]))
                        for k in columns})
            rows.append(row)
            previous = state
    if numeric_keys:
        for second in np.arange(0, t[-1], 2.0):
            i = int(np.searchsorted(t, second))
            if i < len(t) and abs(t[i] - second) < 0.5:
                rows.append({"t_s": round(float(t[i]), 2), **{k: float(columns[k][i])
                             for k in numeric_keys}, "periodic": True})
    return sorted(rows, key=lambda r: r["t_s"])


def _sample_at(t, columns: dict, every_s: float) -> list[dict]:
    rows = []
    for second in np.arange(0, t[-1], every_s):
        i = int(np.searchsorted(t, second))
        if i < len(t):
            rows.append({"t_s": round(float(t[i]), 2),
                         **{k: round(float(v[i]), 4) for k, v in columns.items()}})
    return rows


def run_scenario(scenario: Scenario, build: Path, out: Path, euroc) -> dict:
    workdir = out / scenario.name
    workdir.mkdir()
    px4 = Px4(build, workdir)
    params = {**BASE_PARAMS, **scenario.params}
    # Boot 1: set parameters (some need a reboot); boot 2: run the scenario.
    px4.start()
    px4.wait_ready()
    for name, value in params.items():
        px4.command("param", "set", name, str(value))
    time.sleep(2)
    px4.stop()
    px4 = Px4(build, workdir)
    px4.start()
    px4.wait_ready()
    readback = {name: [line.strip() for line in px4.command("param", "show", name).splitlines()
                       if name in line] for name in params}
    time.sleep(8)  # EKF tilt alignment on the ground
    streamed = stream(scenario, euroc)
    time.sleep(1)
    px4.stop()
    logs = sorted((workdir / "rootfs" / "log").rglob("*.ulg"))
    result = {
        "scenario": scenario.__dict__,
        "params_readback": readback,
        "odometry_sent": streamed["odometry_sent"],
        "timesync_replies": streamed["timesync_replies"],
        "sent_first": streamed["sent"][:1],
        "sent_samples": streamed["sent"][:: int(RATE_HZ * 2)],
        "ulog": str(logs[-1]) if logs else None,
        "ulog_sha256": hashlib.sha256(logs[-1].read_bytes()).hexdigest() if logs else None,
    }
    if logs:
        result["ekf2"] = analyse(logs[-1], streamed["sent"])
        # Stream started ~8 s after boot 2; ulog time 0 = logger start (boot).
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--px4-build", type=Path, required=True, help="build/px4_sitl_default")
    parser.add_argument("--cuvslam-run", type=Path, required=True, help="MH_01 run directory")
    parser.add_argument("--sequence", type=Path, required=True, help="MH_01 mav0/")
    parser.add_argument("--only", nargs="*", help="scenario names")
    parser.add_argument("--output", type=Path, required=True, help="new directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    euroc = EurocStatic(args.cuvslam_run, args.sequence, start_s=20.5, end_s=27.5, yaw0_deg=40.0)
    summary = {
        "tool": "benchmarks/gps_denied/sitl_ev_replay.py",
        "control_output": "DISABLED",
        "command_executed": False,
        "note": "PX4 SITL (SIH) on localhost only; vehicle never armed; replay only.",
        "euroc_conversion": {
            "frd_from_dataset_world": euroc.frd_from_dataset_world.tolist(),
            "dataset_body_from_frd_body": euroc.dataset_body_from_frd_body.tolist(),
        },
        "scenarios": {},
    }
    for scenario in SCENARIOS:
        if args.only and scenario.name not in args.only:
            continue
        print("[scenario]", scenario.name, flush=True)
        summary["scenarios"][scenario.name] = run_scenario(
            scenario, args.px4_build.resolve(), args.output.resolve(), euroc
        )
        (args.output / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
