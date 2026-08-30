from aegisland.evaluation import evaluate_scenario
from aegisland.simulator import SCENARIOS


def test_evaluation_returns_metrics_for_known_scenario() -> None:
    name = next(iter(SCENARIOS))

    metrics = evaluate_scenario(name)

    assert metrics.scenario == name
    assert metrics.frames > 0
    assert metrics.action_changes >= 0
    assert metrics.unsafe_landing_frames >= 0
    assert metrics.max_risk_score >= 0.0
    assert metrics.max_motion_risk >= 0.0
    assert metrics.max_temporal_risk >= 0.0
    assert metrics.mean_processing_ms >= 0.0


def test_compound_emergency_scenario_records_recovery() -> None:
    if "critical_battery_collision" not in SCENARIOS:
        return

    metrics = evaluate_scenario(
        "critical_battery_collision"
    )

    assert metrics.emergency_recovery_frames > 0
    assert metrics.critical_frames > 0
    assert metrics.first_critical_frame is not None
