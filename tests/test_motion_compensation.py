import numpy as np

from aegisland.motion_compensation import CameraMotionCompensator
from aegisland.perception import cv2


def textured_frame() -> np.ndarray:
    frame = np.zeros((360, 480), dtype=np.uint8)

    for y in range(30, 330, 45):
        for x in range(30, 450, 55):
            cv2.circle(frame, (x, y), 6, 255, -1)
            cv2.rectangle(
                frame,
                (x + 10, y - 5),
                (x + 22, y + 7),
                180,
                -1,
            )

    cv2.putText(
        frame,
        "AEGIS",
        (120, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        220,
        3,
    )

    return frame


def test_orb_homography_compensates_global_camera_translation() -> None:
    previous = textured_frame()

    transform = np.float32(
        [
            [1, 0, 12],
            [0, 1, 8],
        ]
    )

    current = cv2.warpAffine(
        previous,
        transform,
        (previous.shape[1], previous.shape[0]),
    )

    compensator = CameraMotionCompensator(
        minimum_matches=10,
        minimum_inliers=8,
    )

    result = compensator.compensate(
        previous,
        current,
    )

    assert result.success
    assert result.homography is not None
    assert result.match_count >= 10
    assert result.inlier_count >= 8

    raw_difference = np.mean(
        cv2.absdiff(previous, current)
    )

    compensated_difference = np.mean(
        cv2.absdiff(result.aligned_previous, current)
    )

    assert compensated_difference < raw_difference


def test_rejects_homography_with_excessive_corner_motion() -> None:
    compensator = CameraMotionCompensator(
        maximum_corner_displacement_ratio=0.20,
    )

    huge_translation = np.array(
        [
            [1.0, 0.0, 400.0],
            [0.0, 1.0, 300.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    assert not compensator._homography_is_sane(
        huge_translation,
        (360, 480),
    )


def test_accepts_reasonable_camera_translation() -> None:
    compensator = CameraMotionCompensator(
        maximum_corner_displacement_ratio=0.35,
    )

    reasonable_translation = np.array(
        [
            [1.0, 0.0, 12.0],
            [0.0, 1.0, 8.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    assert compensator._homography_is_sane(
        reasonable_translation,
        (360, 480),
    )


def test_compensation_preserves_independent_foreground_motion() -> None:
    previous = textured_frame()

    # Global camera motion.
    transform = np.float32(
        [
            [1, 0, 12],
            [0, 1, 8],
        ]
    )

    current = cv2.warpAffine(
        previous,
        transform,
        (previous.shape[1], previous.shape[0]),
    )

    # Add an independently moving foreground object.
    # Previous expected camera-shifted object position would be around x=172.
    # We place it farther right to simulate extra object motion.
    cv2.rectangle(
        current,
        (210, 120),
        (250, 170),
        255,
        -1,
    )

    compensator = CameraMotionCompensator(
        minimum_matches=10,
        minimum_inliers=8,
        minimum_inlier_ratio=0.30,
    )

    result = compensator.compensate(
        previous,
        current,
    )

    assert result.success

    residual_flow = cv2.calcOpticalFlowFarneback(
        result.aligned_previous,
        current,
        None,
        0.5,
        3,
        17,
        3,
        5,
        1.2,
        0,
    )

    magnitude, _ = cv2.cartToPolar(
        residual_flow[..., 0],
        residual_flow[..., 1],
    )

    # Foreground region should retain stronger residual motion than background.
    foreground_motion = float(
        magnitude[110:180, 195:265].mean()
    )

    background_motion = float(
        magnitude[20:90, 20:90].mean()
    )

    assert foreground_motion > background_motion
