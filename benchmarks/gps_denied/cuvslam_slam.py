"""
SLAM additions for run_cuvslam_euroc.py (cuVSLAM v17.0.1 Python API, used as-is).

Records, per stereo frame:
  - the SLAM pose returned by track() (it JUMPS when a loop closure is applied);
  - the part of the SLAM motion that odometry does not explain (`slam_correction_*`),
    i.e. the jump the supervisor would see;
  - SLAM metrics (lc_status, pgo_status, landmark counts);
  - loop-closure events, read from get_loop_closure_poses().

At the end it writes:
  - trajectory_slam_body.tum: per-frame track() SLAM poses;
  - trajectory_slam_smoothed_body.tum: get_all_slam_poses(), re-optimized;
  - optionally a saved map;
  - pose-graph size and memory samples.

Localization in a saved map follows the official KITTI SLAM example (track the
first frame, then localize_in_map with a guess pose and a search grid).

Runs in the cuVSLAM environment. Offline replay only; no control output.
"""

from __future__ import annotations

import json
import resource
import threading
import time
from pathlib import Path

import numpy as np


def rss_mib() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    return float("nan")


def tum_row(timestamp_ns: int, world_from_body: np.ndarray, rotation_cls) -> str:
    quaternion = rotation_cls.from_matrix(world_from_body[:3, :3]).as_quat()
    seconds = timestamp_ns // 1_000_000_000 + (timestamp_ns % 1_000_000_000) * 1e-9
    return f"{seconds:.9f} " + " ".join(f"{v:.9f}" for v in (*world_from_body[:3, 3], *quaternion))


class SlamSession:
    def __init__(self, cuvslam, args, pose_matrix, rig_from_imu, rotation_cls):
        self.cuvslam, self.args = cuvslam, args
        self.pose_matrix, self.rig_from_imu, self.rotation = pose_matrix, rig_from_imu, rotation_cls
        self.config = cuvslam.Tracker.SlamConfig(
            sync_mode=args.slam_sync, max_map_size=args.slam_max_map_size,
            throttling_time_ms=args.slam_throttling_ms)
        self.rows: list[str] = []
        self.previous: tuple[np.ndarray, np.ndarray] | None = None  # (odom, slam) world_from_rig
        self.last_lc_stamp: int | None = None
        self.lc_events: list[dict] = []
        self.memory: list[dict] = []
        self.localization: dict | None = None
        self.frames = 0

    def config_record(self) -> dict:
        return {name: getattr(self.config, name) for name in (
            "sync_mode", "max_map_size", "throttling_time_ms", "map_cell_size",
            "max_landmarks_distance", "retention_time_ms", "use_gpu")}

    def localize(self, tracker, timestamp_ns: int, images) -> None:
        """Localize the first frame in a saved map (official example flow)."""
        guess = json.loads(Path(self.args.slam_guess_json).read_text())
        pose = self.cuvslam.Pose(translation=guess["translation"], rotation=guess["rotation_xyzw"])
        settings = self.cuvslam.Tracker.SlamLocalizationSettings(
            horizontal_search_radius=self.args.loc_search_radius_m,
            vertical_search_radius=self.args.loc_search_radius_m / 2,
            horizontal_step=self.args.loc_step_m, vertical_step=self.args.loc_step_m / 2,
            angular_step_rads=0.03)
        done = threading.Event()
        result: dict = {"guess": guess, "settings": {
            "horizontal_search_radius": settings.horizontal_search_radius,
            "horizontal_step": settings.horizontal_step}}

        def finish(found, message):
            result["success"] = found is not None
            result["message"] = message
            if found is not None:
                result["pose"] = {"translation": list(found.translation),
                                  "rotation_xyzw": list(found.rotation)}
            done.set()

        started = time.perf_counter()
        tracker.localize_in_map(str(self.args.slam_localize_map), timestamp_ns, pose, images,
                                settings, lambda: None, finish)
        done.wait(timeout=60.0)
        result["duration_s"] = time.perf_counter() - started
        result["completed"] = done.is_set()
        self.localization = result

    def after_track(self, tracker, timestamp_ns: int, estimate, slam_pose) -> dict:
        self.frames += 1
        row: dict = {}
        if slam_pose is not None and estimate.world_from_rig is not None:
            odom = self.pose_matrix(estimate.world_from_rig.pose)
            slam = self.pose_matrix(slam_pose)
            self.rows.append(tum_row(timestamp_ns, slam @ self.rig_from_imu, self.rotation))
            if self.previous is not None:
                odom_delta = np.linalg.inv(self.previous[0]) @ odom
                slam_delta = np.linalg.inv(self.previous[1]) @ slam
                correction = np.linalg.inv(odom_delta) @ slam_delta
                cosine = np.clip((np.trace(correction[:3, :3]) - 1) / 2, -1, 1)
                row["slam_correction_m"] = round(float(np.linalg.norm(correction[:3, 3])), 6)
                row["slam_correction_deg"] = round(float(np.degrees(np.arccos(cosine))), 5)
            self.previous = (odom, slam)
        metrics = tracker.get_slam_metrics()
        if metrics is not None:
            row.update(lc_status=bool(metrics.lc_status), pgo_status=bool(metrics.pgo_status),
                       lc_good_landmarks=int(metrics.lc_good_landmarks_count))
        closures = tracker.get_loop_closure_poses()
        if closures:
            stamp = int(closures[-1].timestamp_ns)
            if stamp != self.last_lc_stamp:
                self.last_lc_stamp = stamp
                self.lc_events.append({"frame_timestamp_ns": timestamp_ns, "lc_timestamp_ns": stamp})
                row["loop_closure_event"] = True
        if self.frames % 100 == 0:
            self.memory.append({"frame": self.frames, "rss_mib": rss_mib()})
        return row

    def finish(self, tracker, output: Path) -> dict:
        with (output / "trajectory_slam_body.tum").open("x", encoding="utf-8") as stream:
            stream.write("# SLAM world_from_body from track() (jumps at loop closures)\n")
            stream.write("\n".join(self.rows) + ("\n" if self.rows else ""))
        smoothed = tracker.get_all_slam_poses(0) or []
        with (output / "trajectory_slam_smoothed_body.tum").open("x", encoding="utf-8") as stream:
            stream.write("# get_all_slam_poses(): world_from_body re-optimized at the end\n")
            for item in smoothed:
                matrix = self.pose_matrix(item.pose) @ self.rig_from_imu
                stream.write(tum_row(int(item.timestamp_ns), matrix, self.rotation) + "\n")
        graph = tracker.get_pose_graph()
        saved = None
        if self.args.slam_save_map:
            done, flag = threading.Event(), {}

            def callback(success):
                flag["success"] = bool(success)
                done.set()

            started = time.perf_counter()
            tracker.save_map(str(self.args.slam_save_map), callback)
            done.wait(timeout=120.0)
            size = sum(p.stat().st_size for p in Path(self.args.slam_save_map).rglob("*")
                       if p.is_file())
            saved = {"path": str(self.args.slam_save_map), "success": flag.get("success"),
                     "duration_s": time.perf_counter() - started, "size_bytes": size}
        return {
            "config": self.config_record(),
            "loop_closure_events": len(self.lc_events),
            "loop_closure_event_list": self.lc_events,
            "pose_graph": None if graph is None else {"nodes": len(graph.nodes),
                                                      "edges": len(graph.edges)},
            "smoothed_poses": len(smoothed),
            "localization": self.localization,
            "saved_map": saved,
            "memory": {"samples": self.memory,
                       "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024},
        }


def add_arguments(parser) -> None:
    parser.add_argument("--slam", action="store_true", help="enable cuVSLAM SLAM (loop closure)")
    parser.add_argument("--slam-sync", action="store_true",
                        help="SLAM in the tracking thread (deterministic; official default: async)")
    parser.add_argument("--slam-max-map-size", type=int, default=300)
    parser.add_argument("--slam-throttling-ms", type=int, default=0)
    parser.add_argument("--slam-save-map", type=Path, help="save the map to this NEW folder")
    parser.add_argument("--slam-localize-map", type=Path, help="localize the first frame in map")
    parser.add_argument("--slam-guess-json", type=Path, help="guess pose (map_from_rig)")
    parser.add_argument("--loc-search-radius-m", type=float, default=2.0)
    parser.add_argument("--loc-step-m", type=float, default=0.25)
