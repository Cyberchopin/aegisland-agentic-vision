"""
Fault injection for replaying localization messages through the shadow supervisor.

Starts from a recorded message stream (and optional IMU samples) and produces a
time-ordered event list with exactly one injected fault, plus the time window in
which that fault is active. Faults are applied to copies; inputs are not mutated.

Kinds:
  dropped_frames   remove `length` frames starting at frame `start`
  camera_outage    same removal, intended for long gaps (supervisor ticks expose it)
  pose_jump        add `magnitude` meters to x from frame `start` on, no reset_counter change
  stale_delivery   delay publish time by `magnitude` seconds for `length` frames
  out_of_order     deliver frame start+1 before frame start
  imu_time_offset  shift every IMU timestamp by `magnitude` seconds (IMU stamped late)
  none             unmodified stream (false-alarm baseline)
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .localization_message import LocalizationMessage

KINDS = frozenset(
    {
        "none",
        "dropped_frames",
        "camera_outage",
        "pose_jump",
        "stale_delivery",
        "out_of_order",
        "imu_time_offset",
    }
)


@dataclass(frozen=True)
class FaultSpec:
    kind: str
    start: int = 0
    length: int = 1
    magnitude: float = 0.0


@dataclass(frozen=True)
class Event:
    time_ns: int
    kind: str  # "message" | "tick" | "imu"
    message: LocalizationMessage | None = None
    gyro_rad_s: tuple[float, float, float] | None = None


def build_events(
    messages: list[LocalizationMessage],
    imu: list[tuple[int, tuple[float, float, float]]],
    spec: FaultSpec,
    period_ns: int,
) -> tuple[list[Event], tuple[int, int]]:
    if spec.kind not in KINDS:
        raise ValueError(f"unknown fault kind {spec.kind!r}")
    if not messages:
        raise ValueError("no messages to replay")
    stream = list(messages)
    end_index = min(spec.start + spec.length, len(stream) - 1)
    window = (stream[spec.start].capture_time_ns, stream[end_index].capture_time_ns)
    imu_shift_ns = 0

    if spec.kind in ("dropped_frames", "camera_outage"):
        stream = stream[: spec.start] + stream[spec.start + spec.length :]
    elif spec.kind == "pose_jump":
        stream = [
            replace(m, position_m=(m.position_m[0] + spec.magnitude, *m.position_m[1:]))
            if index >= spec.start
            else m
            for index, m in enumerate(stream)
        ]
        window = (stream[spec.start].capture_time_ns, stream[-1].publish_time_ns)
    elif spec.kind == "stale_delivery":
        delay_ns = int(spec.magnitude * 1e9)
        stream = [
            replace(m, publish_time_ns=m.publish_time_ns + delay_ns)
            if spec.start <= index < spec.start + spec.length
            else m
            for index, m in enumerate(stream)
        ]
    elif spec.kind == "out_of_order":
        first, second = stream[spec.start], stream[spec.start + 1]
        stream[spec.start] = replace(second, publish_time_ns=first.publish_time_ns)
        stream[spec.start + 1] = replace(first, publish_time_ns=second.publish_time_ns)
        window = (first.capture_time_ns, second.publish_time_ns)
    elif spec.kind == "imu_time_offset":
        imu_shift_ns = int(spec.magnitude * 1e9)
        window = (messages[0].capture_time_ns, messages[-1].publish_time_ns)
    elif spec.kind == "none":
        window = (0, 0)

    events = [Event(m.publish_time_ns, "message", message=m) for m in stream]
    events += [Event(t + imu_shift_ns, "imu", gyro_rad_s=gyro) for t, gyro in imu]
    first_ns = messages[0].capture_time_ns
    last_ns = max(m.publish_time_ns for m in stream)
    tick = first_ns + period_ns // 2
    while tick <= last_ns:
        events.append(Event(tick, "tick"))
        tick += period_ns
    # Stable sort: at equal times IMU and ticks keep their insertion order after messages.
    events.sort(key=lambda event: event.time_ns)
    return events, window


def score_detection(
    assessments: list[tuple[int, str, tuple[str, ...]]],
    *,
    window: tuple[int, int],
    expected: set[str],
    grace_ns: int = 0,
) -> dict[str, object]:
    """Detection latency for `expected` reasons inside the window, and alarms outside it."""
    start, end = window
    detection_time = None
    false_alarms = 0
    for time_ns, status, reasons in assessments:
        inside = start <= time_ns <= end + grace_ns
        if inside and detection_time is None and expected & set(reasons):
            detection_time = time_ns
        if not inside and status != "ok":
            false_alarms += 1
    return {
        "detected": detection_time is not None,
        "latency_ns": None if detection_time is None else detection_time - start,
        "false_alarms_outside_window": false_alarms,
    }
