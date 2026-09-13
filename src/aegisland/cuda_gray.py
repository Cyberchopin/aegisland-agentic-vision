"""Run the first CUDA kernel and validate against the installed CPU oracle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from . import perception
from .profiling import compare, inputs, snapshot, write_json


def errors(reference, actual):
    if reference.shape != actual.shape or reference.dtype != actual.dtype:
        raise ValueError("shape/dtype mismatch")
    difference = actual.astype(np.int16) - reference.astype(np.int16)
    values, counts = np.unique(difference, return_counts=True)
    return {
        "pixels": int(reference.size),
        "passed": bool(np.array_equal(reference, actual)),
        "max_abs_error": int(np.max(np.abs(difference))),
        "mae": float(np.mean(np.abs(difference))),
        "mismatch_fraction": float(np.mean(difference != 0)),
        "signed_error_histogram": {str(int(v)): int(n) for v, n in zip(values, counts)},
    }


def run_case(executable, out, name, frame, *, padding=0, warmup=10, iterations=100):
    height, width = frame.shape[:2]
    source = np.full((height, width * 3 + padding), 0xD3, np.uint8)
    source[:, : width * 3] = frame.reshape(height, width * 3)
    source_path = out / f"{name}.bgr"
    source.tofile(source_path)
    target_path = out / f"{name}.gray"
    args = [
        str(executable.resolve()),
        "--input",
        str(source_path.resolve()),
        "--output",
        str(target_path.resolve()),
        "--samples",
        str((out / f"{name}.csv").resolve()),
        "--width",
        str(width),
        "--height",
        str(height),
        "--stride",
        str(source.shape[1]),
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
    ]
    result = subprocess.run(args, text=True, capture_output=True, timeout=180, check=False)
    write_json(
        out / f"{name}-command.json",
        {
            "argv": args,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
    )
    if result.returncode:
        raise RuntimeError(result.stderr)
    actual = np.fromfile(target_path, np.uint8).reshape(height, width)
    reference = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    reference.tofile(out / f"{name}.cpu-gray")
    report = {
        "name": name,
        "shape": list(frame.shape),
        "source_stride": int(source.shape[1]),
        "source_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
        **errors(reference, actual),
    }
    write_json(out / f"{name}-correctness.json", report)
    return actual, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("invalid counts")
    args.output.mkdir(parents=True, exist_ok=False)
    frames, provenance = inputs(args.input)
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    config = perception.PerceptionConfig()
    resize = perception.OpenCVLandingPerception(config)._resize
    reports, gpu_grays, cpu_rows = [], {}, []
    for index, original in enumerate(frames):
        frame = resize(original)
        name = f"frame-{index:03}"
        actual, report = run_case(
            args.exe, args.output, name, frame, warmup=args.warmup, iterations=args.iterations
        )
        reports.append(report)
        gpu_grays[hashlib.sha256(frame.tobytes()).hexdigest()] = actual
        for iteration in range(-args.warmup, args.iterations):
            start = time.perf_counter_ns()
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elapsed = time.perf_counter_ns() - start
            if iteration >= 0:
                cpu_rows.append({"frame": index, "iteration": iteration, "duration_ns": elapsed})
    # Every possible BGR8 triplet: exhaustive arithmetic and rounding validation.
    colors = np.arange(1 << 24, dtype=np.uint32)
    cube = np.stack((colors & 255, (colors >> 8) & 255, colors >> 16), axis=1)
    cube = cube.astype(np.uint8).reshape(4096, 4096, 3)
    _, report = run_case(args.exe, args.output, "all-bgr8", cube, warmup=1, iterations=1)
    reports.append(report)
    rng = np.random.default_rng(42)
    for width, height, padding in [(1, 1, 7), (33, 9, 17), (319, 241, 5)]:
        frame = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        _, report = run_case(
            args.exe,
            args.output,
            f"stride-{width}-{height}",
            frame,
            padding=padding,
            warmup=1,
            iterations=1,
        )
        reports.append(report)
    # Replay downstream production stages using saved *actual* GPU gray outputs.
    # This is a correctness substitution, NOT an integrated GPU latency benchmark.
    reference = snapshot(frames, config)
    original_cv2 = perception.cv2

    class CachedGpuGray:
        def __getattr__(self, name):
            return getattr(original_cv2, name)

        def cvtColor(self, frame, code):
            if code == cv2.COLOR_BGR2GRAY:
                return gpu_grays[hashlib.sha256(frame.tobytes()).hexdigest()].copy()
            return original_cv2.cvtColor(frame, code)

    try:
        perception.cv2 = CachedGpuGray()
        candidate = snapshot(frames, config)
    finally:
        perception.cv2 = original_cv2
    pipeline_report = compare(reference, candidate)
    write_json(args.output / "pipeline-correctness.json", pipeline_report)
    write_json(args.output / "correctness.json", reports)
    with (args.output / "cpu-gray.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cpu_rows[0]))
        writer.writeheader()
        writer.writerows(cpu_rows)
    write_json(
        args.output / "manifest.json",
        {
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "provenance": provenance,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "exe_sha256": hashlib.sha256(args.exe.read_bytes()).hexdigest(),
            "kernel_source_sha256": hashlib.sha256(
                Path("cpp/cuda_gray/gray_bench.cu").read_bytes()
            ).hexdigest(),
            "contract": "BGR uint8 -> gray uint8; exact equality required, atol=rtol=0",
            "timing": "kernel CUDA events; separate synchronous pageable H2D/kernel/sync/D2H host wall; allocations excluded",
            "synthetic": True,
        },
    )
    if not all(r["passed"] for r in reports) or not pipeline_report["passed"]:
        raise SystemExit(
            "Correctness failure: inspect measured error histograms; tolerance unchanged"
        )
    print(
        json.dumps(
            {
                "cases": len(reports),
                "pixels": sum(r["pixels"] for r in reports),
                "correctness_passed": True,
                "pipeline_gate_passed": True,
            }
        )
    )


if __name__ == "__main__":
    main()
