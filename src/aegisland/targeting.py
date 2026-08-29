from __future__ import annotations

from dataclasses import dataclass

from .domain import VisionEvidence, ZoneCandidate


@dataclass(slots=True)
class TargetState:
    locked_zone_id: str | None = None
    unsafe_frames: int = 0


class LandingTargetManager:
    def __init__(self, *, release_after_unsafe_frames: int = 3) -> None:
        self.state = TargetState()
        self.release_after_unsafe_frames = release_after_unsafe_frames

    def select(self, evidence: VisionEvidence) -> ZoneCandidate | None:
        candidates = {candidate.zone_id: candidate for candidate in evidence.candidates}

        if self.state.locked_zone_id is None:
            best = evidence.best_zone
            if best is not None:
                self.state.locked_zone_id = best.zone_id
            return best

        locked = candidates.get(self.state.locked_zone_id)

        if locked is None:
            self.state.locked_zone_id = None
            self.state.unsafe_frames = 0
            return evidence.best_zone

        if locked.safe:
            self.state.unsafe_frames = 0
            return locked

        self.state.unsafe_frames += 1

        if self.state.unsafe_frames < self.release_after_unsafe_frames:
            return locked

        best = evidence.best_zone
        self.state.locked_zone_id = best.zone_id if best else None
        self.state.unsafe_frames = 0
        return best
