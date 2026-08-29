from aegisland.domain import VisionEvidence, ZoneCandidate
from aegisland.targeting import LandingTargetManager


def zone(zone_id: str, score: float, safe: bool = True) -> ZoneCandidate:
    return ZoneCandidate(
        zone_id=zone_id,
        bbox_xywh=(0, 0, 10, 10),
        score=score,
        edge_density=0.0,
        motion_occupancy=0.0,
        texture_risk=0.0,
        appearance_occupancy=0.0,
        clearance=1.0 if safe else 0.2,
        safe=safe,
    )


def evidence(*zones: ZoneCandidate) -> VisionEvidence:
    return VisionEvidence(
        evidence_id="target-test",
        frame_index=0,
        confidence=0.9,
        obstacle_risk=0.0,
        motion_risk=0.0,
        candidates=zones,
    )


def test_target_stays_locked_when_another_zone_scores_higher() -> None:
    manager = LandingTargetManager()

    first = manager.select(
        evidence(
            zone("A", 0.90),
            zone("B", 0.80),
        )
    )

    second = manager.select(
        evidence(
            zone("A", 0.70),
            zone("B", 0.95),
        )
    )

    assert first is not None
    assert second is not None
    assert first.zone_id == "A"
    assert second.zone_id == "A"


def test_target_releases_after_three_unsafe_frames() -> None:
    manager = LandingTargetManager(release_after_unsafe_frames=3)

    manager.select(
        evidence(
            zone("A", 0.90),
            zone("B", 0.80),
        )
    )

    first_unsafe = manager.select(
        evidence(
            zone("A", 0.60, safe=False),
            zone("B", 0.95),
        )
    )

    second_unsafe = manager.select(
        evidence(
            zone("A", 0.58, safe=False),
            zone("B", 0.96),
        )
    )

    third_unsafe = manager.select(
        evidence(
            zone("A", 0.55, safe=False),
            zone("B", 0.97),
        )
    )

    assert first_unsafe is not None
    assert second_unsafe is not None
    assert third_unsafe is not None

    assert first_unsafe.zone_id == "A"
    assert second_unsafe.zone_id == "A"
    assert third_unsafe.zone_id == "B"
