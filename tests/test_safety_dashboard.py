import json

import pytest

from aegisland.safety_dashboard import (
    load_evidence,
    write_dashboard,
)


def _write_fake_evidence(
    directory,
) -> None:
    assessment = {
        "status": "PASS",
        "scope": (
            "Deterministic simulation safety "
            "assessment. This is not flight "
            "certification."
        ),
        "gates": [
            {
                "id": "AUTH-01",
                "name": (
                    "No premature GPS-denied "
                    "navigation authority"
                ),
                "passed": True,
                "evidence": (
                    "capability-aware=0, "
                    "confidence-only=2"
                ),
            }
        ],
        "metrics": {
            "capability_authority": {
                "confidence_only": {
                    "premature_authority_frames": 2,
                    "premature_raw_continue_frames": 2,
                    "raw_action_transitions": 5,
                },
                "capability_aware": {
                    "premature_authority_frames": 0,
                    "premature_raw_continue_frames": 0,
                    "raw_action_transitions": 1,
                },
            },
            "sensor_health_hysteresis": {
                "samples_1": {
                    "premature_healthy_grants": 2,
                    "unstable_healthy_frames": 2,
                    "stable_health_latency_frames": 1,
                },
                "samples_3": {
                    "premature_healthy_grants": 0,
                    "unstable_healthy_frames": 0,
                    "stable_health_latency_frames": 3,
                },
            },
            "flicker_recovery": {
                "stable_physical_recovery_frame": 62,
                "stable_trust_frame": 63,
                "stable_health_frame": 65,
                "raw_recovery_frame": 65,
                "final_recovery_frame": 68,
            },
        },
    }

    timeline = [
        {
            "frame": 62,
            "camera_fault_active": False,
            "visual_effective_confidence": 0.0,
            "fused_navigation_confidence": 0.0,
            "perception_failure_type": (
                "geometry_unstable"
            ),
            "visual_health_state": "failed",
            "navigation_mode": "degraded",
            "action": "hold_and_scan",
        },
        {
            "frame": 65,
            "camera_fault_active": False,
            "visual_effective_confidence": 1.0,
            "fused_navigation_confidence": 0.98,
            "perception_failure_type": "healthy",
            "visual_health_state": "healthy",
            "navigation_mode": (
                "visual_inertial_fallback"
            ),
            "action": "hold_and_scan",
        },
        {
            "frame": 68,
            "camera_fault_active": False,
            "visual_effective_confidence": 1.0,
            "fused_navigation_confidence": 0.98,
            "perception_failure_type": "healthy",
            "visual_health_state": "healthy",
            "navigation_mode": (
                "visual_inertial_fallback"
            ),
            "action": "continue_mission",
        },
    ]

    (
        directory
        / "assessment.json"
    ).write_text(
        json.dumps(assessment),
        encoding="utf-8",
    )

    (
        directory
        / "timeline.json"
    ).write_text(
        json.dumps(timeline),
        encoding="utf-8",
    )


def test_dashboard_is_generated(
    tmp_path,
) -> None:
    _write_fake_evidence(
        tmp_path
    )

    output = (
        tmp_path
        / "dashboard.html"
    )

    result = write_dashboard(
        output,
        evidence_dir=tmp_path,
    )

    assert result == output
    assert output.exists()

    document = output.read_text(
        encoding="utf-8"
    )

    normalized = " ".join(
        document.lower().split()
    )

    assert "aegisland" in normalized

    assert (
        "interactive fault timeline"
        in normalized
    )

    assert (
        "capability-aware authority"
        in normalized
    )

    assert (
        "engineering assessment only"
        in normalized
    )

    assert (
        "flight certification"
        in normalized
    )

    assert (
        "airworthiness"
        in normalized
    )


def test_missing_evidence_fails_fast(
    tmp_path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="safety_assessment",
    ):
        load_evidence(
            tmp_path
        )
