import json

import cv2
import numpy as np
import pytest

from aegisland.domain import jsonable
from aegisland.profiling import (
    PerceptionConfig,
    benchmark,
    compare,
    inputs,
    invoke,
    replay,
    snapshot,
    summarize,
)


def test_snapshot_matches_production_with_retry_and_state():
    frames, _ = inputs()
    frames = frames[:4]
    config = PerceptionConfig(maximum_width=160)
    cv2.setRNGSeed(42)
    arrays, records = snapshot(frames, config, True)
    expected = []
    for model, frame, index, retry in replay(frames, config, True, 42):
        evidence, annotated = invoke(model, frame, index, retry)
        record = jsonable(evidence)
        record.pop("processing_ms")
        expected.append(record)
        np.testing.assert_array_equal(arrays[f"{index}_{int(retry)}_annotated"], annotated)
    assert [{k: v for k, v in r.items() if k != "motion_boxes"} for r in records] == expected
    assert all(a.shape[:2] == (120, 160) for a in arrays.values())


def test_comparison_rejects_pixel_decision_shape_and_missing_changes():
    a = {"0_0_motion": np.zeros((2, 2), np.uint8)}
    assert compare((a, [{"safe": True}]), (a, [{"safe": True}]))["passed"]
    b = {"0_0_motion": a["0_0_motion"].copy()}
    b["0_0_motion"][0, 0] = 255
    result = compare((a, []), (b, []))
    assert not result["passed"]
    assert result["arrays"]["0_0_motion"]["iou"] == 0
    assert not compare((a, [{"safe": True}]), (a, [{"safe": False}]))["passed"]
    assert not compare((a, []), ({}, []))["passed"]
    assert not compare((a, []), ({"0_0_motion": np.zeros((3, 3))}, []))["passed"]


def test_artifacts_warmup_and_state_reset(tmp_path):
    frames, provenance = inputs()
    out = tmp_path / "run"
    rows = benchmark(out, frames[:2], provenance, warmup=1, runs=2, maximum_width=160)
    baseline = [r for r in rows if r["mode"] == "baseline"]
    assert len(baseline) == 4
    assert {r["run"] for r in rows} == {0, 1}
    assert all(r["inclusive_ns"] >= 0 for r in rows)
    assert json.loads((out / "correctness.json").read_text())["passed"]
    assert len(list(out.glob("*.prof"))) == 4
    with pytest.raises(FileExistsError):
        benchmark(out, frames[:2], provenance)


def test_input_validation_and_percentiles(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, frames=np.zeros((2, 20, 20, 3), np.float32))
    with pytest.raises(ValueError):
        inputs(path)
    rows = [
        {"mode": "baseline", "frame": 0, "retry": 0, "stage": "x", "inclusive_ns": n}
        for n in (1, 2, 3, 4, 5)
    ]
    summary = summarize(rows)[0]
    assert summary["p50_ns"] == 3
    assert summary["p95_ns"] == pytest.approx(4.8)


def test_collector_controlled_clock_nested_residual_and_exception():
    from aegisland.stage_timing import Collector

    ticks = iter([10, 20, 35, 50])
    collector = Collector(clock=lambda: next(ticks))
    with pytest.raises(RuntimeError), collector.span("parent"), collector.span("child"):
        raise RuntimeError("injected")
    assert collector.stack == []
    assert collector.rows[0]["inclusive_ns"] == 40
    assert collector.rows[0]["exclusive_ns"] == 25
    assert collector.rows[1]["inclusive_ns"] == 15
    assert collector.rows[1]["parent"] == "parent"


def test_instrumentation_branches_retry_state_and_restoration(monkeypatch):
    from aegisland.perception import OpenCVLandingPerception
    from aegisland.stage_timing import Collector, instrument

    frames, _ = inputs()
    model = OpenCVLandingPerception(PerceptionConfig(maximum_width=160))
    counter = iter(range(100000))
    collector = Collector(clock=lambda: next(counter))
    updates = []
    update = model.visual_dead_reckoner.update

    def spy_update(**kwargs):
        updates.append(kwargs)
        return update(**kwargs)

    monkeypatch.setattr(model.visual_dead_reckoner, "update", spy_update)
    original_attributes = dict(model.__dict__)
    with instrument(model, collector):
        model.observe(frames[0], 0)
        stages = {r["stage"] for r in collector.rows}
        assert "motion.raw_flow" not in stages
        assert "motion.residual_flow" not in stages
        assert "preprocess.retry_clahe" not in stages
        assert model._active_reference_gray is None
        first_gray = model.previous_gray
        model.observe(frames[1], 1)
        pre_frame = model._active_reference_gray
        assert pre_frame is first_gray
        current = model.previous_gray
        pose = model.visual_dead_reckoner.current()
        count = len(updates)
        collector.rows.clear()
        invoke(model, frames[1], 1, True)
        assert model._active_reference_gray is pre_frame
        assert model.previous_gray is current
        assert model.visual_dead_reckoner.current() == pose
        assert len(updates) == count == 1
        stages = {r["stage"] for r in collector.rows}
        assert {"motion.raw_flow", "motion.residual_flow", "preprocess.retry_clahe"} <= stages
        assert all(r["inclusive_ns"] > 0 for r in collector.rows)
        collector.rows.clear()
        model.observe(np.zeros((48, 48, 3), np.uint8), 2)
        assert "motion.raw_flow" not in {r["stage"] for r in collector.rows}
    for name in ("observe", "_resize", "_motion", "_score_grid", "enhance_for_active_perception"):
        assert (name in model.__dict__) == (name in original_attributes)
    with pytest.raises(RuntimeError), instrument(model, collector):
        raise RuntimeError("injected")
    assert "observe" not in model.__dict__
    assert "compensate" not in model.motion_compensator.__dict__
    assert "analyze" not in model.perception_quality_monitor.__dict__


def test_stage_replay_exact_and_branch_summary_recomputable(tmp_path):
    import csv

    from aegisland.profiling import summarize_branches

    frames, provenance = inputs()
    frames = frames[:2]
    config = PerceptionConfig(maximum_width=160)
    assert compare(snapshot(frames, config, True), snapshot(frames, config, True, stages=True))[
        "passed"
    ]
    out = tmp_path / "stages"
    benchmark(out, frames, provenance, warmup=1, runs=2, active=True, maximum_width=160)
    with (out / "stage-samples.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for field in ("inclusive_ns", "exclusive_ns", "processed_height", "processed_width"):
            row[field] = int(row[field])
    assert summarize_branches(rows) == json.loads((out / "stage-summary.json").read_text())
    assert {int(r["run"]) for r in rows} == {0, 1}
    assert len([r for r in rows if r["stage"] == "end_to_end"]) == 8
    assert not any(
        r["stage"] == "motion.raw_flow" and r["reference_status"] != "valid" for r in rows
    )
    assert json.loads((out / "stage-schema.json").read_text())["version"] == 1
    assert json.loads((out / "stage-correctness.json").read_text())["passed"]
    import hashlib

    for line in (out / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest


def test_benchmark_baseline_unpatched_and_failure_restores_state(tmp_path, monkeypatch):
    from aegisland import profiling
    from aegisland.stage_timing import Collector, instrument

    frames, provenance = inputs()
    seen = []
    original_invoke = profiling.invoke

    def spy(model, *args):
        seen.append("observe" in model.__dict__)
        return original_invoke(model, *args)

    monkeypatch.setattr(profiling, "invoke", spy)
    benchmark(tmp_path / "normal", frames[:1], provenance, warmup=0, runs=1)
    assert seen[:3] == [False, True, False]  # independent baseline/stages/cProfile
    threads, opencl = cv2.getNumThreads(), cv2.ocl.useOpenCL()
    model = profiling.OpenCVLandingPerception()
    original_observe = model.observe

    def fail(*args, **kwargs):
        raise RuntimeError("injected observe failure")

    monkeypatch.setattr(model, "observe", fail)
    # The partially compiled instrumentation must also restore an existing override.
    with pytest.raises(ValueError, match="non-production override"), instrument(model, Collector()):
        pass
    assert model.observe is fail
    monkeypatch.setattr(model, "observe", original_observe)

    def failing_invoke(model, *args):
        raise RuntimeError("injected")

    monkeypatch.setattr(profiling, "invoke", failing_invoke)
    with pytest.raises(RuntimeError, match="injected"):
        benchmark(tmp_path / "failure", frames[:1], provenance, warmup=0, runs=1, threads=2)
    assert cv2.getNumThreads() == threads
    assert cv2.ocl.useOpenCL() == opencl


@pytest.mark.parametrize("failure_call", [1, 2, 3])
def test_round_failure_restores_instrumentation_and_runtime(tmp_path, monkeypatch, failure_call):
    from aegisland import profiling

    frames, provenance = inputs()
    old_threads, old_opencl = cv2.getNumThreads(), cv2.ocl.useOpenCL()
    old_profile = __import__("sys").getprofile()
    original_invoke = profiling.invoke
    models, calls = [], []

    def fail(model, *args):
        models.append(model)
        calls.append(1)
        if len(calls) == failure_call:
            raise RuntimeError("injected round failure")
        return original_invoke(model, *args)

    monkeypatch.setattr(profiling, "invoke", fail)
    with pytest.raises(RuntimeError, match="injected round failure"):
        benchmark(tmp_path / "failure", frames[:1], provenance, warmup=0, runs=1, threads=2)
    assert cv2.getNumThreads() == old_threads
    assert cv2.ocl.useOpenCL() == old_opencl
    assert __import__("sys").getprofile() is old_profile
    for model in models:
        assert "observe" not in model.__dict__
        assert "compensate" not in model.motion_compensator.__dict__
        assert "analyze" not in model.perception_quality_monitor.__dict__


def test_compensation_accept_reject_and_resize_boundaries():
    from aegisland.perception import OpenCVLandingPerception
    from aegisland.stage_timing import Collector, instrument

    rng = np.random.default_rng(42)
    frame = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    model = OpenCVLandingPerception()
    collector = Collector(clock=iter(range(100000)).__next__)
    with instrument(model, collector):
        model.observe(frame, 0)
        collector.rows.clear()
        evidence, _ = model.observe(frame, 1)
        assert evidence.camera_compensation_used
        names = {row["stage"] for row in collector.rows}
        assert {"compensation.warp", "motion.raw_flow", "motion.residual_flow"} <= names
        assert "resize.result" not in names
        collector.rows.clear()
        evidence, _ = model.observe(np.zeros_like(frame), 2)
        assert not evidence.camera_compensation_used
        names = {row["stage"] for row in collector.rows}
        assert "compensation.warp" not in names
        assert {"motion.raw_flow", "motion.residual_flow"} <= names
        collector.rows.clear()
        model.observe(np.zeros((240, 1000, 3), np.uint8), 3)
        assert "resize.result" in {row["stage"] for row in collector.rows}


def test_original_reference_gate(tmp_path):
    frames, provenance = inputs()
    frames = frames[:1]
    old = tmp_path / "before"
    benchmark(old, frames, provenance, warmup=0, runs=1)
    new = tmp_path / "after"
    benchmark(new, frames, provenance, warmup=0, runs=1, reference_dir=old)
    assert json.loads((new / "before-after-equivalence.json").read_text())["passed"]
    records = json.loads((old / "reference.json").read_text())
    records[0]["motion_boxes"] = [[0, 0, 10, 10]]
    (old / "reference.json").write_text(json.dumps(records))
    with pytest.raises(RuntimeError, match="Original baseline differs"):
        benchmark(tmp_path / "mismatch", frames, provenance, warmup=0, runs=1, reference_dir=old)


@pytest.mark.parametrize("active", [False, True])
def test_real_module_cli_regression(tmp_path, active):
    import csv
    import subprocess
    import sys

    frames, _ = inputs()
    input_path = tmp_path / "inputs.npz"
    np.savez_compressed(input_path, frames=frames[:2])
    output = tmp_path / "cli"
    args = [
        sys.executable,
        "-m",
        "aegisland.profiling",
        "--input",
        str(input_path),
        "--output",
        str(output),
        "--warmup",
        "0",
        "--runs",
        "1",
        "--threads",
        "1",
    ]
    if active:
        args.append("--active")
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((output / "correctness.json").read_text())["passed"]
    assert json.loads((output / "stage-correctness.json").read_text())["passed"]
    arrays, evidence = snapshot(frames[:2], PerceptionConfig(), active=active)
    with np.load(output / "reference.npz", allow_pickle=False) as archive:
        actual = (
            {key: archive[key] for key in archive.files},
            json.loads((output / "reference.json").read_text()),
        )
        assert compare((arrays, evidence), actual)["passed"]
    with (output / "stage-samples.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    names = {row["stage"] for row in rows}
    assert {"motion.raw_flow", "motion.residual_flow", "grid.candidate_scoring"} <= names
    assert all(row["frame"] != "0" for row in rows if row["stage"] == "motion.raw_flow")
    assert ("preprocess.retry_clahe" in names) == active
    assert len(list(output.glob("*.prof"))) == (4 if active else 2)


def test_capture_is_explicit_and_partial_setup_restores_overrides():
    from aegisland.profiling import SnapshotPerception
    from aegisland.stage_timing import Collector, instrument

    model = SnapshotPerception()
    callback = model._score_grid.__func__
    with pytest.raises(ValueError, match="non-production override"), instrument(model, Collector()):
        pass
    assert "observe" not in model.__dict__  # rejected after earlier production patches
    with instrument(model, Collector(), capture_grid=True):
        assert model._score_grid.__func__ is callback
        assert "_stage_score_grid" in model.__dict__
    assert "_stage_score_grid" not in model.__dict__
    assert model._score_grid.__func__ is callback
