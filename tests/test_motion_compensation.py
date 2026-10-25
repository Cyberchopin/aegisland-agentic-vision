import itertools

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


def _shifted_sequence(count: int = 4) -> list[np.ndarray]:
    base = textured_frame()
    return [
        cv2.warpAffine(base, np.float32([[1, 0, 3 * i], [0, 1, 2 * i]]), (480, 360))
        for i in range(count)
    ]


class _CountingOrb:
    def __init__(self, orb) -> None:
        self.orb = orb
        self.calls = 0

    def detectAndCompute(self, image, mask):
        self.calls += 1
        return self.orb.detectAndCompute(image, mask)


def test_feature_cache_gives_identical_results_to_fresh_compensators() -> None:
    frames = _shifted_sequence()
    cached = CameraMotionCompensator()

    for previous, current in itertools.pairwise(frames):
        reused = cached.compensate(previous, current)
        fresh = CameraMotionCompensator().compensate(previous, current)

        assert reused.success == fresh.success
        assert reused.match_count == fresh.match_count
        assert reused.inlier_count == fresh.inlier_count
        np.testing.assert_array_equal(reused.homography, fresh.homography)
        np.testing.assert_array_equal(reused.aligned_previous, fresh.aligned_previous)


def test_feature_cache_skips_recomputing_previous_frame() -> None:
    frames = _shifted_sequence()
    compensator = CameraMotionCompensator()
    counter = _CountingOrb(compensator.orb)
    compensator.orb = counter

    for previous, current in itertools.pairwise(frames):
        compensator.compensate(previous, current)

    # 3 frame pairs: 2 detections for the first, 1 new frame for each later pair.
    assert counter.calls == 4


def test_feature_cache_is_not_fooled_by_in_place_mutation() -> None:
    frames = _shifted_sequence(3)
    compensator = CameraMotionCompensator()
    counter = _CountingOrb(compensator.orb)
    compensator.orb = counter
    current = frames[1].copy()

    compensator.compensate(frames[0], current)
    current[:] = frames[2]  # caller reuses its buffer for a different image
    result = compensator.compensate(current, frames[2])

    assert counter.calls == 3  # mutated buffer had to be recomputed
    fresh = CameraMotionCompensator().compensate(frames[2], frames[2])
    assert result.match_count == fresh.match_count
