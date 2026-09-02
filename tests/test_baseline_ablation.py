from aegisland.baseline_ablation import (
    build_comparison,
    summarize,
)
from aegisland.fault_timeline import (
    run_fault_timeline,
)


def _metrics():
    timeline = run_fault_timeline(
        "gps_denied_camera_failure"
    )

    return summarize(
        build_comparison(timeline)
    )


def test_aegisland_reaches_safe_state_faster() -> None:
    metrics = _metrics()

    assert (
        metrics.aegisland_time_to_safe
        <
        metrics.baseline_time_to_safe
    )


def test_aegisland_has_zero_unsafe_continuation() -> None:
    metrics = _metrics()

    assert (
        metrics.aegisland_unsafe_continuation
        == 0
    )

    assert (
        metrics.baseline_unsafe_continuation
        > 0
    )


def test_aegisland_recovers_more_conservatively() -> None:
    metrics = _metrics()

    assert (
        metrics.aegisland_recovery_latency
        >
        metrics.baseline_recovery_latency
    )
