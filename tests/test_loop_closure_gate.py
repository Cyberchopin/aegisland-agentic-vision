import numpy as np
import pytest

from aegisland.loop_closure_gate import AnchoredSlamOutput, GateConfig


def pose(x: float, y: float = 0.0) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = (x, y, 0.0)
    return matrix


def drive(gate, steps, correction_at=None, correction=(0.0, 0.0), event_at=None):
    """Move 0.05 m per frame; SLAM = odometry + correction from `correction_at` on."""
    decisions = []
    for k in range(steps):
        odom = pose(0.05 * k)
        offset = correction if correction_at is not None and k >= correction_at else (0, 0)
        slam = pose(0.05 * k + offset[0], offset[1])
        decisions.append(gate.update(odom, slam, loop_closure_event=(k == event_at)))
    return decisions


def test_continuous_when_slam_equals_odometry() -> None:
    gate = AnchoredSlamOutput()
    decisions = drive(gate, 50)
    assert {d.verdict for d in decisions} == {"continuous"}
    assert decisions[-1].reset_counter == 0
    assert decisions[-1].output[0, 3] == pytest.approx(0.05 * 49)


def test_loop_closure_correction_within_budget_is_accepted_with_reset() -> None:
    gate = AnchoredSlamOutput()
    # 40 frames = 2 m of path -> budget max(0.15 min, 5 % * 2 m) = 0.15 m
    decisions = drive(gate, 60, correction_at=40, correction=(0.08, 0.0), event_at=40)
    assert decisions[40].verdict == "accepted_loop_closure"
    assert decisions[40].reset_counter == 1
    assert decisions[40].budget_m == pytest.approx(0.15, abs=0.01)
    # After the anchor moved, the output follows SLAM continuously again.
    assert {d.verdict for d in decisions[41:]} == {"continuous"}
    assert decisions[-1].output[0, 3] == pytest.approx(0.05 * 59 + 0.08)


def test_jump_without_event_oversized_and_odometry_jump_are_rejected() -> None:
    gate = AnchoredSlamOutput()
    decisions = drive(gate, 30, correction_at=20, correction=(0.08, 0.0), event_at=None)
    assert decisions[20].verdict == "rejected_no_loop_closure_event"
    assert decisions[20].output[0, 3] == pytest.approx(0.05 * 20)  # correction not applied
    assert decisions[-1].reset_counter == 0

    gate = AnchoredSlamOutput()
    decisions = drive(gate, 30, correction_at=20, correction=(1.0, 0.0), event_at=20)
    assert decisions[20].verdict == "rejected_oversized_correction"

    gate = AnchoredSlamOutput(GateConfig(max_odometry_step_m=0.25))
    gate.update(pose(0.0), pose(0.0), False)
    decision = gate.update(pose(1.0), pose(1.05), True)  # odometry itself jumped 1 m
    assert decision.verdict == "rejected_odometry_jump"
    assert gate.counts["rejected_odometry_jump"] == 1


def test_rejected_offset_does_not_leak_into_a_later_loop_closure() -> None:
    gate = AnchoredSlamOutput()
    for k in range(100):
        odom = pose(0.05 * k)
        offset = 0.0
        if k >= 20:
            offset += 0.3  # foreign jump without a loop-closure event
        if k >= 80:
            offset += 0.06  # genuine loop-closure correction at frame 80
        decision = gate.update(odom, pose(0.05 * k + offset), loop_closure_event=(k == 80))
        if k == 20:
            assert decision.verdict == "rejected_no_loop_closure_event"
        if k == 80:
            assert decision.verdict == "accepted_loop_closure"
            assert decision.correction_m == pytest.approx(0.06, abs=1e-9)
    # Output = odometry + the genuine 0.06 m only, never the foreign 0.3 m.
    assert decision.output[0, 3] == pytest.approx(0.05 * 99 + 0.06)
    assert {d for d in gate.counts} >= {"rejected_no_loop_closure_event", "accepted_loop_closure"}


def test_odometry_fallback_switch_ignores_slam_and_never_resets() -> None:
    gate = AnchoredSlamOutput(GateConfig(pose_source="odometry"))
    decisions = drive(gate, 60, correction_at=40, correction=(0.08, 0.0), event_at=40)
    assert {d.verdict for d in decisions} == {"odometry_only"}
    assert decisions[-1].reset_counter == 0
    assert decisions[-1].output[0, 3] == pytest.approx(0.05 * 59)
    with pytest.raises(ValueError):
        AnchoredSlamOutput(GateConfig(pose_source="slam"))
