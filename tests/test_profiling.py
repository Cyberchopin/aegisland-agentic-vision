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
