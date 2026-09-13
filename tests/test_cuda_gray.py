import os
from pathlib import Path

import numpy as np
import pytest

from aegisland.cuda_gray import errors, run_case


def test_error_histogram_detects_one_level_and_unsigned_wrap():
    cpu = np.array([[0, 255, 100]], dtype=np.uint8)
    gpu = np.array([[255, 0, 101]], dtype=np.uint8)
    result = errors(cpu, gpu)
    assert not result["passed"]
    assert result["signed_error_histogram"] == {"-255": 1, "1": 1, "255": 1}
    assert result["max_abs_error"] == 255
    assert result["mismatch_fraction"] == 1


def test_error_gate_rejects_shape_dtype_and_preserves_exact_equality():
    image = np.zeros((3, 4), np.uint8)
    assert errors(image, image.copy())["passed"]
    with pytest.raises(ValueError):
        errors(image, image.astype(np.float32))
    with pytest.raises(ValueError):
        errors(image, image[:, :2])


@pytest.mark.skipif(not os.environ.get("AEGISLAND_GRAY_EXE"), reason="opt-in CUDA integration")
def test_real_gpu_odd_size_padded_rows(tmp_path):
    frame = np.random.default_rng(42).integers(0, 256, (9, 33, 3), np.uint8)
    _, result = run_case(
        Path(os.environ["AEGISLAND_GRAY_EXE"]),
        tmp_path,
        "odd",
        frame,
        padding=17,
        warmup=1,
        iterations=2,
    )
    assert result["passed"]
    assert result["pixels"] == 297
