import json

from aegisland.safety_assessment import (
    build_assessment,
    write_assessment,
)


def test_deterministic_safety_gates_pass() -> None:
    assessment = build_assessment()

    assert assessment["status"] == "PASS"

    assert all(
        gate["passed"]
        for gate in assessment["gates"]
    )


def test_assessment_preserves_authority_ablation() -> None:
    assessment = build_assessment()

    metrics = assessment[
        "metrics"
    ][
        "capability_authority"
    ]

    assert (
        metrics[
            "confidence_only"
        ][
            "premature_authority_frames"
        ]
        > 0
    )

    assert (
        metrics[
            "capability_aware"
        ][
            "premature_authority_frames"
        ]
        == 0
    )


def test_report_explicitly_limits_certification_claims() -> None:
    assessment = build_assessment()

    assert (
        "not flight certification"
        in assessment["scope"].lower()
    )


def test_assessment_artifacts_are_written(
    tmp_path,
) -> None:
    assessment = write_assessment(
        tmp_path
    )

    json_path = (
        tmp_path
        / "assessment.json"
    )

    markdown_path = (
        tmp_path
        / "assessment.md"
    )

    assert json_path.exists()
    assert markdown_path.exists()

    loaded = json.loads(
        json_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        loaded["status"]
        == assessment["status"]
    )

    markdown = markdown_path.read_text(
        encoding="utf-8"
    )

    assert (
        "NOT FLIGHT CERTIFICATION"
        in markdown
    )
