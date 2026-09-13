"""Stage-0 replay benchmark. Production perception is deliberately unmodified."""

from __future__ import annotations

import argparse
import cProfile
import csv
import hashlib
import json
import os
import platform
import pstats
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .domain import jsonable
from .perception import OpenCVLandingPerception, PerceptionConfig
from .perception_scenes import build_scenes


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30, check=False)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"unavailable": str(exc)}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def digest(frame):
    return hashlib.sha256(frame.tobytes()).hexdigest()


def inputs(path=None):
    if path is not None:
        with np.load(path, allow_pickle=False) as archive:
            frames = archive["frames"].copy()
        provenance = {
            "kind": "user NPZ ordered replay",
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    else:
        # Existing scene generator, plus a deterministic translated reference pair.
        scenes = build_scenes()
        frames = np.stack([f for s in scenes for f in (s.frame, np.roll(s.frame, 2, axis=1))])
        provenance = {
            "kind": "synthetic smoke corpus",
            "generator": "perception_scenes.build_scenes",
            "seed": 42,
            "translation": "np.roll 2 columns, wrap border",
            "scenes": [s.name for s in scenes],
        }
    if (
        frames.dtype != np.uint8
        or frames.ndim != 4
        or frames.shape[-1] != 3
        or len(frames) == 0
        or min(frames.shape[1:3]) < 12
    ):
        raise ValueError("frames must be nonempty NxHxWx3 uint8 BGR, H/W >= 12")
    return frames, provenance


class SnapshotPerception(OpenCVLandingPerception):
    """Capture only in the separate correctness pass, never in latency runs."""

    def _score_grid(self, gray, edges, texture, motion, boxes):
        self.arrays = {
            k: v.copy()
            for k, v in zip(("gray", "edges", "texture", "motion"), (gray, edges, texture, motion))
        }
        self.boxes = boxes.copy()
        return super()._score_grid(gray, edges, texture, motion, boxes)


def invoke(model, frame, index, retry):
    if retry:
        frame = model.enhance_for_active_perception(frame)
    return model.observe(frame, index, active_perception=retry)


def replay(frames, config, active, seed, factory=OpenCVLandingPerception):
    cv2.setRNGSeed(seed)
    model = factory(config)
    for index, frame in enumerate(frames):
        for retry in [False, True] if active else [False]:
            yield model, frame, index, retry


def snapshot(frames, config, active=False, seed=42):
    arrays, evidence = {}, []
    for model, frame, index, retry in replay(frames, config, active, seed, SnapshotPerception):
        item, annotated = invoke(model, frame, index, retry)
        record = jsonable(item)
        record.pop("processing_ms")  # Only nondeterministic public output.
        record["motion_boxes"] = jsonable(model.boxes)
        evidence.append(record)
        for name, value in {**model.arrays, "annotated": annotated}.items():
            arrays[f"{index}_{int(retry)}_{name}"] = value
    return arrays, evidence


def compare(reference, candidate):
    """Strict equivalence gate; report error magnitudes without forgiving decision changes."""
    ra, re = reference
    ca, ce = candidate
    report = {
        "passed": re == ce and ra.keys() == ca.keys(),
        "evidence_equal": re == ce,
        "arrays": {},
    }
    for key in ra.keys() | ca.keys():
        if key not in ra or key not in ca:
            report["passed"] = False
            report["arrays"][key] = {"error": "missing key"}
            continue
        a, b = ra[key], ca[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            report["passed"] = False
            report["arrays"][key] = {"error": "shape/dtype mismatch"}
            continue
        delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
        metrics = {
            "max_abs_error": float(delta.max()),
            "mae": float(delta.mean()),
            "mismatch_fraction": float(np.mean(a != b)),
        }
        if key.endswith(("_edges", "_motion")):
            union = np.count_nonzero((a != 0) | (b != 0))
            metrics["iou"] = float(np.count_nonzero((a != 0) & (b != 0)) / union) if union else 1.0
        report["arrays"][key] = metrics
        report["passed"] &= bool(np.array_equal(a, b))
    return report


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["mode"], row["frame"], row["retry"], row["stage"])
        groups.setdefault(key, []).append(row["inclusive_ns"])
    return [
        {
            "mode": k[0],
            "frame": k[1],
            "retry": k[2],
            "stage": k[3],
            "count": len(v),
            **{f"p{p}_ns": float(np.percentile(v, p, method="linear")) for p in (50, 95, 99)},
        }
        for k, v in sorted(groups.items())
    ]


def benchmark(
    out,
    frames,
    provenance,
    *,
    warmup=3,
    runs=20,
    threads=1,
    maximum_width=960,
    active=False,
    seed=42,
):
    if warmup < 0 or runs < 1 or threads < 1 or maximum_width < 12:
        raise ValueError("invalid benchmark counts/configuration")
    out.mkdir(parents=True, exist_ok=False)
    old_threads, old_opencl = cv2.getNumThreads(), cv2.ocl.useOpenCL()
    cv2.setNumThreads(threads)
    cv2.ocl.setUseOpenCL(False)
    config = PerceptionConfig(maximum_width=maximum_width)
    rows = []
    try:
        np.savez_compressed(out / "inputs.npz", frames=frames)
        write_json(
            out / "environment.json",
            {
                "schema": 1,
                "argv": sys.argv,
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "python": sys.version,
                "platform": platform.platform(),
                "cpu": platform.processor(),
                "cpu_details": command(
                    [
                        "powershell",
                        "-NoProfile",
                        "-Command",
                        "Get-ItemPropertyValue -LiteralPath 'HKLM:/HARDWARE/DESCRIPTION/System/CentralProcessor/0' -Name ProcessorNameString",
                    ]
                    if sys.platform == "win32"
                    else ["lscpu"]
                ),
                "logical_cpus": os.cpu_count(),
                "opencv": cv2.__version__,
                "numpy": np.__version__,
                "threads": cv2.getNumThreads(),
                "opencl": cv2.ocl.useOpenCL(),
                "config": asdict(config),
                "warmup_sequences_per_mode": warmup,
                "runs": runs,
                "seed_per_sequence": seed,
                "active_retry_every_frame": active,
                "input_shape": list(frames.shape),
                "input_hashes": [digest(f) for f in frames],
                "processed_shape": list(OpenCVLandingPerception(config)._resize(frames[0]).shape),
                "provenance": provenance,
                "git": command(["git", "rev-parse", "HEAD"]),
                "git_status": command(["git", "status", "--porcelain"]),
                "source_sha256": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in Path(__file__).parent.glob("*.py")
                },
                "gpu": command(["nvidia-smi", "-q"]),
                "packages": command([sys.executable, "-m", "pip", "freeze"]),
                "clock": vars(time.get_clock_info("perf_counter")),
            },
        )
        (out / "opencv-build.txt").write_text(cv2.getBuildInformation(), encoding="utf-8")
        for mode in ("baseline", "diagnostic"):
            for run in range(-warmup, runs):
                for model, frame, index, retry in replay(frames, config, active, seed):
                    profiler = cProfile.Profile() if mode == "diagnostic" else None
                    if profiler:
                        profiler.enable()
                    start = time.perf_counter_ns()
                    invoke(model, frame, index, retry)
                    elapsed = time.perf_counter_ns() - start
                    if profiler:
                        profiler.disable()
                    if run < 0:
                        continue
                    base = {"mode": mode, "run": run, "frame": index, "retry": int(retry)}
                    rows.append(
                        {
                            **base,
                            "stage": "end_to_end",
                            "calls": 1,
                            "inclusive_ns": elapsed,
                            "exclusive_ns": elapsed,
                        }
                    )
                    if profiler:
                        profiler.dump_stats(str(out / f"callgraph-{run}-{index}-{int(retry)}.prof"))
                        for (file, line, name), (_, calls, own, total, _) in pstats.Stats(
                            profiler
                        ).stats.items():
                            rows.append(
                                {
                                    **base,
                                    "stage": f"{Path(file).name}:{line}:{name}",
                                    "calls": calls,
                                    "inclusive_ns": round(total * 1e9),
                                    "exclusive_ns": round(own * 1e9),
                                }
                            )
        with (out / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(out / "summary.json", summarize(rows))
        reference = snapshot(frames, config, active, seed)
        np.savez_compressed(out / "reference.npz", **reference[0])
        write_json(out / "reference.json", reference[1])
        check = compare(reference, snapshot(frames, config, active, seed))
        write_json(out / "correctness.json", check)
        if not check["passed"]:
            raise RuntimeError("CPU replay is not exact; see correctness.json")
    finally:
        cv2.setNumThreads(old_threads)
        cv2.ocl.setUseOpenCL(old_opencl)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, help="NPZ with ordered uint8 BGR frames")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--maximum-width", type=int, default=960)
    parser.add_argument("--active", action="store_true")
    args = parser.parse_args()
    frames, provenance = inputs(args.input)
    benchmark(
        args.output,
        frames,
        provenance,
        warmup=args.warmup,
        runs=args.runs,
        threads=args.threads,
        maximum_width=args.maximum_width,
        active=args.active,
    )


if __name__ == "__main__":
    main()
