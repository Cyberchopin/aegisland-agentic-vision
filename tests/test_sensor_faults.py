import cv2
import numpy as np

from aegisland.sensor_faults import (
    CameraFault,
    CameraFaultInjector,
)


def textured_frame() -> np.ndarray:
    frame = np.zeros(
        (240, 320, 3),
        dtype=np.uint8,
    )

    for y in range(20, 220, 40):
        for x in range(20, 300, 40):
            cv2.circle(
                frame,
                (x, y),
                8,
                (
                    (x * 3) % 255,
                    (y * 5) % 255,
                    ((x + y) * 2) % 255,
                ),
                -1,
            )

    return frame


def test_no_fault_preserves_frame() -> None:
    injector = CameraFaultInjector()
    frame = textured_frame()

    result = injector.apply(
        frame,
        CameraFault.NONE,
    )

    assert np.array_equal(
        result.frame,
        frame,
    )

    assert result.affected_fraction == 0.0


def test_overexposure_increases_mean_brightness() -> None:
    injector = CameraFaultInjector()
    frame = textured_frame()

    result = injector.apply(
        frame,
        CameraFault.OVEREXPOSURE,
        severity=0.8,
    )

    assert result.frame.mean() > frame.mean()


def test_underexposure_reduces_mean_brightness() -> None:
    injector = CameraFaultInjector()
    frame = textured_frame()

    result = injector.apply(
        frame,
        CameraFault.UNDEREXPOSURE,
        severity=0.8,
    )

    assert result.frame.mean() < frame.mean()


def test_occlusion_blocks_a_large_region() -> None:
    injector = CameraFaultInjector()
    frame = textured_frame()

    result = injector.apply(
        frame,
        CameraFault.OCCLUSION,
        severity=0.8,
    )

    assert result.affected_fraction > 0.5
    assert np.count_nonzero(result.frame) < np.count_nonzero(frame)


def test_blur_reduces_high_frequency_image_energy() -> None:
    injector = CameraFaultInjector()
    frame = textured_frame()

    result = injector.apply(
        frame,
        CameraFault.BLUR,
        severity=1.0,
    )

    original_gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    blurred_gray = cv2.cvtColor(
        result.frame,
        cv2.COLOR_BGR2GRAY,
    )

    original_laplacian = cv2.Laplacian(
        original_gray,
        cv2.CV_64F,
    ).var()

    blurred_laplacian = cv2.Laplacian(
        blurred_gray,
        cv2.CV_64F,
    ).var()

    assert blurred_laplacian < original_laplacian
