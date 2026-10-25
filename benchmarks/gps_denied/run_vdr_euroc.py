"""
Run the core VisualDeadReckoner on an EuRoC-format cam0 sequence, offline.

Uses the unmodified core `CameraMotionCompensator` + `VisualDeadReckoner` with the
same call pattern as `OpenCVLandingPerception._motion` (base pass,
`update_visual_state=True`). EuRoC images are 752 px wide, below the perception
resize cap of 960 px, and already mono, so the input matches what perception sees.

Writes into a NEW directory: frames.jsonl (normalized image units, NOT meters),
run.json, SHA256SUMS. Run with the AegisLand environment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from aegisland.gps_denied import VisualDeadReckoner
from aegisland.motion_compensation import CameraMotionCompensator

REPOSITORY = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def core_revision() -> dict[str, str | bool | None]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(REPOSITORY), *args],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": git("rev-parse", "HEAD"), "tracked_files_dirty": bool(status)}


def camera_rows(sequence: Path) -> list[tuple[int, Path]]:
    rows = []
    with (sequence / "cam0" / "data.csv").open(encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if not row or row[0].startswith("#"):
                continue
            rows.append((int(row[0]), sequence / "cam0" / "data" / row[1].strip()))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sequence", type=Path, required=True, help="path to mav0/")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--output", type=Path, required=True, help="new directory")
    args = parser.parse_args()

    rows = camera_rows(args.sequence)
    if args.max_frames is not None:
        rows = rows[: args.max_frames]
    args.output.mkdir(parents=True, exist_ok=False)

    cv2.setNumThreads(1)
    compensator = CameraMotionCompensator()
    reckoner = VisualDeadReckoner()
    previous = None
    counts = {"frames": 0, "compensation_success": 0, "valid": 0}
    elapsed: list[float] = []
    with (args.output / "frames.jsonl").open("x", encoding="utf-8") as log:
        for timestamp_ns, path in rows:
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise FileNotFoundError(path)
            record: dict[str, object] = {"timestamp_ns": timestamp_ns, "first_frame": False}
            if previous is None:
                pose = reckoner.current()
                record.update(first_frame=True, success=False, inlier_ratio=0.0, match_count=0)
            else:
                started = time.perf_counter()
                result = compensator.compensate(previous, gray)
                pose = reckoner.update(
                    homography=result.homography if result.success else None,
                    image_shape=gray.shape,
                    inlier_ratio=result.inlier_ratio,
                )
                elapsed.append((time.perf_counter() - started) * 1000.0)
                record.update(
                    success=bool(result.success),
                    inlier_ratio=round(float(result.inlier_ratio), 6),
                    match_count=int(result.match_count),
                )
                counts["compensation_success"] += int(result.success)
            record.update(
                vx_normalized=pose.vx,
                vy_normalized=pose.vy,
                x_normalized=pose.x,
                y_normalized=pose.y,
                confidence=pose.confidence,
                valid=pose.valid,
            )
            counts["frames"] += 1
            counts["valid"] += int(pose.valid)
            log.write(json.dumps(record) + "\n")
            previous = gray

    timings = np.asarray(elapsed) if elapsed else np.zeros(0)
    record = {
        "tool": "benchmarks/gps_denied/run_vdr_euroc.py",
        "units": "normalized image units per frame; NOT meters",
        "core": core_revision(),
        "python": sys.version,
        "opencv": cv2.__version__,
        "platform": platform.platform(),
        "sequence": str(args.sequence.resolve()),
        "image_shape": list(gray.shape) if rows else None,
        "max_frames": args.max_frames,
        "input_sha256": {"cam0/data.csv": sha256_file(args.sequence / "cam0" / "data.csv")},
        "counts": counts,
        "timing_ms": {
            "note": "compensate + update per frame, 1 OpenCV thread, excludes PNG decode",
            "count": int(timings.size),
            "p50": float(np.percentile(timings, 50)) if timings.size else None,
            "p95": float(np.percentile(timings, 95)) if timings.size else None,
        },
    }
    (args.output / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    with (args.output / "SHA256SUMS").open("x", encoding="utf-8") as stream:
        for path in sorted(args.output.iterdir()):
            if path.is_file() and path.name != "SHA256SUMS":
                stream.write(f"{sha256_file(path)}  {path.name}\n")
    print(json.dumps(counts), json.dumps(record["timing_ms"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
