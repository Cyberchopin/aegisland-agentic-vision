from aegisland.perception_fault_benchmark import run


def test_perception_fault_matrix_passes() -> None:
    results = run()

    assert results

    failures = [
        result
        for result in results
        if not result.passed
    ]

    assert not failures, (
        "Perception fault benchmark failures: "
        f"{failures}"
    )


def test_perception_fault_matrix_covers_expected_cases() -> None:
    results = run()

    cases = {
        result.case
        for result in results
    }

    assert cases == {
        "normal",
        "overexposure",
        "underexposure",
        "occlusion",
        "blur",
        "texture_degenerate",
        "geometry_unstable",
    }
