"""
Headless cuVSLAM run on an EuRoC-format (ASL) sequence.

Runs in the isolated cuVSLAM environment, not the AegisLand one. It follows the
official v17.0.1 `examples/euroc/track_euroc.py` configuration, with the Rerun
viewer removed, and writes reproducible artifacts:

  trajectory_body.tum   world_from_body (body = imu0 frame), seconds, meters
  frames.jsonl          per stereo frame: timestamp, tracked flag, track() wall time
                        (+ with --export-signals: covariance diagonal, observation and
                        landmark counts, new landmark ids, gravity in inertial mode)
  run.json              versions, config, input hashes, timing summary
  SHA256SUMS            hashes of every artifact above

The official `dataset_utils.py` is imported from a pinned cuVSLAM checkout
(`--cuvslam-examples`) instead of being copied into AegisLand.

Output is offline evidence only. Nothing here talks to a flight controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from cuvslam_slam import SlamSession
from cuvslam_slam import add_arguments as add_slam_arguments

MODES = {"stereo": "Multicamera", "inertial": "Inertial"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_dirty(path: Path) -> bool | None:
    try:
        return bool(
            subprocess.run(
                ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def gpu_description() -> str | None:
    try:
        return subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def pose_matrix(pose) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(np.asarray(pose.rotation)).as_matrix()
    matrix[:3, 3] = np.asarray(pose.translation)
    return matrix


def paired_metadata(sequence: Path, mode, read_csv_data, cuvslam) -> tuple[list[dict], dict]:
    """Like the official metadata builder, but keep only timestamps present in both cameras.

    The official builder rejects a sequence whose cameras have different frame
    counts (EuRoC MH_04_difficult: cam0 has one frame cam1 lacks). The unpaired
    timestamps are returned so they are recorded, not silently dropped.
    """
    left = {row["timestamp"]: row["filename"] for row in read_csv_data(
        str(sequence / "cam0" / "data.csv"), "camera")}
    right = {row["timestamp"]: row["filename"] for row in read_csv_data(
        str(sequence / "cam1" / "data.csv"), "camera")}
    common = sorted(set(left) & set(right))
    metadata = [
        {
            "type": "stereo",
            "timestamp": timestamp,
            "images_paths": [
                str(sequence / "cam0" / "data" / left[timestamp]),
                str(sequence / "cam1" / "data" / right[timestamp]),
            ],
        }
        for timestamp in common
    ]
    if mode == cuvslam.Tracker.OdometryMode.Inertial:
        metadata.extend(
            {"type": "imu", "timestamp": m["timestamp"], "gyro": m["gyro"], "accel": m["accel"]}
            for m in read_csv_data(str(sequence / "imu0" / "data.csv"), "imu")
        )
    metadata.sort(key=lambda item: item["timestamp"])
    unpaired = {
        "cam0_only": sorted(set(left) - set(right)),
        "cam1_only": sorted(set(right) - set(left)),
    }
    return metadata, unpaired


def tracker_signals(tracker, estimate, seen_landmarks: set[int], cameras: int) -> dict:
    """Per-frame values cuVSLAM exposes online (for degradation studies)."""
    landmark_ids = [landmark.id for landmark in tracker.get_last_landmarks()]
    new = sum(1 for landmark_id in landmark_ids if landmark_id not in seen_landmarks)
    seen_landmarks.update(landmark_ids)
    signals = {
        "observations": [len(tracker.get_last_observations(index)) for index in range(cameras)],
        "landmarks": len(landmark_ids),
        "new_landmarks": new,
        "covariance_xyz_rpy_diag": None,
    }
    if estimate.world_from_rig is not None:
        covariance = np.asarray(estimate.world_from_rig.covariance_xyz_rpy).reshape(-1)
        if covariance.size == 36:
            signals["covariance_xyz_rpy_diag"] = covariance.reshape(6, 6).diagonal().tolist()
    try:
        gravity = tracker.get_last_gravity()
    except ValueError:  # "IMU fusion is disabled" in stereo mode
        gravity = None
    signals["gravity_rig"] = None if gravity is None else np.asarray(gravity).tolist()
    return signals


class GpuClockLog:
    """Sample GPU P-state / clocks with nvidia-smi into gpu_clocks.csv (benchmark rule).

    Development-GPU only: on Jetson nvidia-smi is absent and the log records
    "unavailable" (use tegrastats there).
    """

    QUERY = "timestamp,pstate,clocks.sm,clocks.mem,utilization.gpu,power.draw,memory.used"

    def __init__(self, path: Path, interval_ms: int = 500):
        self.path, self.process = path, None
        try:
            self.stream = path.open("x", encoding="utf-8")
            self.process = subprocess.Popen(
                ["nvidia-smi", f"--query-gpu={self.QUERY}", "--format=csv,noheader",
                 f"-lms={interval_ms}"], stdout=self.stream, stderr=subprocess.DEVNULL)
        except OSError:
            self.stream.write("unavailable\n")

    def stop(self) -> dict:
        if self.process is not None:
            self.process.terminate()
            self.process.wait()
        self.stream.close()
        rows = [line.split(", ") for line in self.path.read_text().splitlines()
                if line.count(",") >= 5]
        clocks = [float(r[2].split()[0]) for r in rows if r[2].split()[0].replace(".", "").isdigit()]
        if not clocks:
            return {"available": False}
        return {"available": True, "samples": len(rows),
                "sm_clock_mhz": {"min": min(clocks), "median": float(np.median(clocks)),
                                 "max": max(clocks)},
                "pstates": sorted({r[1] for r in rows}),
                "note": "clocks not locked (no admin rights in WSL); recorded instead"}


def percentile_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    array = np.asarray(values)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def run(args: argparse.Namespace) -> int:
    output: Path = args.output
    # Never overwrite historical evidence.
    output.mkdir(parents=True, exist_ok=False)

    sys.path.insert(0, str(args.cuvslam_examples.resolve()))
    import cuvslam
    from dataset_utils import get_rig, load_frame, prepare_frame_metadata_euroc, read_csv_data
    from scipy.spatial.transform import Rotation

    sequence = args.sequence.resolve()
    mode = getattr(cuvslam.Tracker.OdometryMode, MODES[args.mode])
    config = cuvslam.Tracker.OdometryConfig(
        async_sba=args.async_sba,
        enable_observations_export=args.export_signals,
        enable_landmarks_export=args.export_signals,
        enable_final_landmarks_export=False,
        rectified_stereo_camera=False,
        odometry_mode=mode,
    )
    rig = get_rig(str(sequence))
    rig_from_imu = pose_matrix(rig.imus[0].rig_from_imu)
    slam = SlamSession(cuvslam, args, pose_matrix, rig_from_imu, Rotation) if args.slam else None
    if args.slam_save_map and args.slam_save_map.exists():
        raise FileExistsError(f"{args.slam_save_map} exists; save_map would overwrite it")
    tracker = cuvslam.Tracker(rig, config, slam.config if slam else None)

    unpaired: dict[str, list[int]] = {}
    if args.pair_by_timestamp:
        metadata, unpaired = paired_metadata(sequence, mode, read_csv_data, cuvslam)
    else:
        metadata = prepare_frame_metadata_euroc(str(sequence), mode)
    if args.start_frame:
        # Drop the first N stereo frames (and the IMU samples before the Nth frame).
        stereo_positions = [i for i, item in enumerate(metadata) if item["type"] != "imu"]
        metadata = metadata[stereo_positions[args.start_frame]:]
    if args.max_frames is not None:
        stereo_seen = 0
        trimmed = []
        for item in metadata:
            if item["type"] != "imu":
                if stereo_seen >= args.max_frames:
                    break
                stereo_seen += 1
            trimmed.append(item)
        metadata = trimmed

    trajectory_rows: list[str] = []
    track_ms: list[float] = []
    stereo_frames = tracked_frames = imu_samples = 0
    first_track_ms = None
    seen_landmarks: set[int] = set()
    clock_log = GpuClockLog(output / "gpu_clocks.csv")
    started = time.perf_counter()
    first_stamp_ns = None
    with (output / "frames.jsonl").open("x", encoding="utf-8") as frames_log:
        for item in metadata:
            if item["type"] == "imu":
                measurement = cuvslam.ImuMeasurement()
                measurement.timestamp_ns = int(item["timestamp"])
                measurement.linear_accelerations = np.asarray(item["accel"])
                measurement.angular_velocities = np.asarray(item["gyro"])
                tracker.register_imu_measurement(0, measurement)
                imu_samples += 1
                continue

            images = [load_frame(path) for path in item["images_paths"]]
            if args.throttle_ms:
                time.sleep(args.throttle_ms / 1000.0)  # diagnostics: stretch wall time only
            if args.realtime:
                # Pace frames to the dataset clock (20 Hz for EuRoC), like a live camera.
                stamp = int(item["timestamp"])
                first_stamp_ns = stamp if first_stamp_ns is None else first_stamp_ns
                due = started + (stamp - first_stamp_ns) / 1e9 / args.realtime
                delay = due - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            timestamp_ns = int(item["timestamp"])
            call_started = time.perf_counter()
            estimate, slam_pose = tracker.track(timestamp_ns, images)
            elapsed_ms = (time.perf_counter() - call_started) * 1000.0
            if slam is not None and args.slam_localize_map and slam.localization is None:
                slam.localize(tracker, timestamp_ns, images)
            if first_track_ms is None:
                first_track_ms = elapsed_ms
            else:
                track_ms.append(elapsed_ms)
            stereo_frames += 1

            tracked = estimate.world_from_rig is not None
            row = {"timestamp_ns": timestamp_ns, "tracked": tracked, "track_ms": round(elapsed_ms, 4)}
            if args.export_signals:
                row.update(tracker_signals(tracker, estimate, seen_landmarks, len(images)))
            if slam is not None:
                row.update(slam.after_track(tracker, timestamp_ns, estimate, slam_pose))
            frames_log.write(json.dumps(row) + "\n")
            if not tracked:
                continue
            tracked_frames += 1
            world_from_body = pose_matrix(estimate.world_from_rig.pose) @ rig_from_imu
            quaternion = Rotation.from_matrix(world_from_body[:3, :3]).as_quat()
            position = world_from_body[:3, 3]
            seconds = timestamp_ns // 1_000_000_000 + (timestamp_ns % 1_000_000_000) * 1e-9
            trajectory_rows.append(
                f"{seconds:.9f} "
                + " ".join(f"{value:.9f}" for value in (*position, *quaternion))
            )
    wall_s = time.perf_counter() - started
    gpu_clocks = clock_log.stop()
    slam_record = slam.finish(tracker, output) if slam is not None else None

    with (output / "trajectory_body.tum").open("x", encoding="utf-8") as stream:
        stream.write("# world_from_body (body=imu0); timestamp_s tx_m ty_m tz_m qx qy qz qw\n")
        stream.write("\n".join(trajectory_rows) + ("\n" if trajectory_rows else ""))

    hashed_inputs = [
        "cam0/data.csv",
        "cam1/data.csv",
        "imu0/data.csv",
        "cam0/sensor_recalibrated.yaml",
        "cam1/sensor_recalibrated.yaml",
        "imu0/sensor_recalibrated.yaml",
        "cam0/sensor.yaml",
        "cam1/sensor.yaml",
        "imu0/sensor.yaml",
        "state_groundtruth_estimate0/data.csv",
    ]
    record = {
        "tool": "benchmarks/gps_denied/run_cuvslam_euroc.py",
        "claims": (
            "Offline dataset replay only. Not field-camera integration, not flight-ready, "
            "no control output."
        ),
        "cuvslam_version": list(cuvslam.get_version()),
        "cuvslam_examples_revision": git_revision(args.cuvslam_examples),
        "aegisland_revision": git_revision(Path(__file__).resolve().parents[2]),
        "aegisland_tracked_files_dirty": git_dirty(Path(__file__).resolve().parents[2]),
        "python": sys.version,
        "platform": platform.platform(),
        "gpu": gpu_description(),
        "sequence": str(sequence),
        "mode": args.mode,
        "export_signals": args.export_signals,
        "slam": slam_record,
        "odometry_config": {
            "async_sba": args.async_sba,
            "rectified_stereo_camera": False,
            "odometry_mode": str(config.odometry_mode),
        },
        "max_frames": args.max_frames,
        "start_frame": args.start_frame,
        "throttle_ms": args.throttle_ms,
        "realtime_rate": args.realtime,
        "gpu_clocks": gpu_clocks,
        "pair_by_timestamp": args.pair_by_timestamp,
        "unpaired_camera_timestamps_ns": unpaired,
        "input_sha256": {
            name: sha256_file(sequence / name)
            for name in hashed_inputs
            if (sequence / name).is_file()
        },
        "counts": {
            "stereo_frames": stereo_frames,
            "tracked_frames": tracked_frames,
            "untracked_frames": stereo_frames - tracked_frames,
            "imu_samples": imu_samples,
        },
        "timing": {
            "note": (
                "Host wall time around Tracker.track(), synchronous; includes upload. "
                "Excludes PNG decode. Development GPU, not Orin."
            ),
            "valid_for_performance_claims": args.realtime is not None,
            "rule": "performance numbers only from --realtime runs with gpu_clocks recorded "
                    "(docs/dronehacks/gps-denied-benchmark-plan.md)",
            "first_call_ms": first_track_ms,
            "warm_track_ms": percentile_summary(track_ms),
            "total_wall_s": wall_s,
        },
    }
    (output / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    with (output / "SHA256SUMS").open("x", encoding="utf-8") as stream:
        for path in sorted(output.iterdir()):
            if path.is_file() and path.name != "SHA256SUMS":
                stream.write(f"{sha256_file(path)}  {path.name}\n")

    print(json.dumps(record["counts"]), json.dumps(record["timing"]["warm_track_ms"]))
    return 0 if tracked_frames > 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sequence", type=Path, required=True, help="path to mav0/")
    parser.add_argument(
        "--cuvslam-examples",
        type=Path,
        required=True,
        help="pinned cuVSLAM checkout's examples/euroc directory",
    )
    parser.add_argument("--mode", choices=sorted(MODES), default="stereo")
    parser.add_argument("--max-frames", type=int, help="smoke runs: limit stereo frames")
    parser.add_argument("--realtime", type=float, nargs="?", const=1.0, default=None,
                        help="pace frames to dataset timestamps (x rate); benchmark rule")
    parser.add_argument("--throttle-ms", type=float, default=0.0,
                        help="sleep before each frame (diagnostics; excluded from track_ms)")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="skip the first N stereo frames (diagnostics)")
    parser.add_argument(
        "--pair-by-timestamp",
        action="store_true",
        help="pair stereo frames by timestamp and record unpaired ones (official: reject)",
    )
    parser.add_argument(
        "--async-sba",
        action="store_true",
        help="run sparse bundle adjustment asynchronously (official example: off)",
    )
    parser.add_argument(
        "--export-signals",
        action="store_true",
        help="enable cuVSLAM observation/landmark export and log per-frame tracker signals",
    )
    add_slam_arguments(parser)
    parser.add_argument("--output", type=Path, required=True, help="new directory; must not exist")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
