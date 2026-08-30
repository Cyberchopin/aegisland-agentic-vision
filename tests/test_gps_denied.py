import numpy as np

from aegisland.gps_denied import VisualDeadReckoner


def test_visual_dead_reckoning_tracks_relative_camera_translation() -> None:
    reckoner = VisualDeadReckoner()

    homography = np.array(
        [
            [1.0, 0.0, 12.0],
            [0.0, 1.0, 8.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    pose = reckoner.update(
        homography=homography,
        image_shape=(360, 480),
        inlier_ratio=0.8,
    )

    assert pose.valid
    assert pose.confidence == 0.8

    # Image shifts right/down, so estimated camera motion
    # is left/up in the local visual frame.
    assert pose.x < 0.0
    assert pose.y < 0.0
    assert pose.vx < 0.0
    assert pose.vy < 0.0


def test_visual_dead_reckoning_accumulates_motion() -> None:
    reckoner = VisualDeadReckoner()

    homography = np.array(
        [
            [1.0, 0.0, 10.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    first = reckoner.update(
        homography=homography,
        image_shape=(360, 480),
        inlier_ratio=0.9,
    )

    second = reckoner.update(
        homography=homography,
        image_shape=(360, 480),
        inlier_ratio=0.9,
    )

    assert abs(second.x) > abs(first.x)


def test_visual_dead_reckoning_degrades_when_visual_pose_is_unreliable() -> None:
    reckoner = VisualDeadReckoner()

    good_homography = np.eye(3)

    good = reckoner.update(
        homography=good_homography,
        image_shape=(360, 480),
        inlier_ratio=0.8,
    )

    degraded = reckoner.update(
        homography=None,
        image_shape=(360, 480),
        inlier_ratio=0.0,
    )

    assert good.valid
    assert degraded.confidence < good.confidence
