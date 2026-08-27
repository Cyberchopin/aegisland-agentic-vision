from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Action(StrEnum):
    CONTINUE_MISSION = "continue_mission"
    RETURN_HOME = "return_home"
    HOLD_AND_SCAN = "hold_and_scan"
    EVADE_AND_HOLD = "evade_and_hold"
    REQUEST_HUMAN_APPROVAL = "request_human_approval"
    LAND = "land"
    EMERGENCY_LAND = "emergency_land"


class SafetyLevel(StrEnum):
    NOMINAL = "nominal"
    CAUTION = "caution"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class Telemetry:
    battery_percent: float
    altitude_m: float
    horizontal_speed_mps: float = 0.0
    gps_available: bool = True
    home_link_available: bool = True
    timestamp_s: float = 0.0

    def __post_init__(self) -> None:
        if not 0 <= self.battery_percent <= 100:
            raise ValueError("battery_percent must be in [0, 100]")
        if self.altitude_m < 0:
            raise ValueError("altitude_m cannot be negative")


@dataclass(frozen=True, slots=True)
class ZoneCandidate:
    zone_id: str
    bbox_xywh: tuple[int, int, int, int]
    score: float
    edge_density: float
    motion_occupancy: float
    texture_risk: float
    appearance_occupancy: float
    clearance: float
    safe: bool


@dataclass(frozen=True, slots=True)
class VisionEvidence:
    evidence_id: str
    frame_index: int
    confidence: float
    obstacle_risk: float
    motion_risk: float
    candidates: tuple[ZoneCandidate, ...] = ()
    active_perception_used: bool = False
    processing_ms: float = 0.0
    notes: tuple[str, ...] = ()

    @property
    def best_zone(self) -> ZoneCandidate | None:
        return max(self.candidates, key=lambda candidate: candidate.score, default=None)


@dataclass(frozen=True, slots=True)
class Decision:
    action: Action
    safety_level: SafetyLevel
    risk_score: float
    requires_human_approval: bool
    reasons: tuple[str, ...]
    evidence_id: str
    target_zone_id: str | None = None
    command_executed: bool = False


@dataclass(frozen=True, slots=True)
class TraceEvent:
    trace_id: str
    sequence: int
    telemetry: Telemetry
    evidence: VisionEvidence
    decision: Decision
    command: dict[str, Any] = field(default_factory=dict)


def jsonable(value: Any) -> Any:
    """Convert nested dataclasses, tuples and enums into JSON-safe values."""
    if hasattr(value, "__dataclass_fields__"):
        return jsonable(asdict(value))
    if isinstance(value, StrEnum):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value
