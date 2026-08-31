from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

from .domain import Decision


class ApprovalStatus(StrEnum):
    REQUESTED = "requested"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    approval_id: str
    action: str
    target_zone_id: str | None
    status: ApprovalStatus
    reason: str


class ApprovalManager:
    def __init__(self) -> None:
        self._requests: dict[str, ApprovalRequest] = {}

    def request(self, decision: Decision) -> ApprovalRequest | None:
        if not decision.requires_human_approval:
            return None

        key = (
            f"{decision.evidence_id}:"
            f"{decision.action.value}:"
            f"{decision.target_zone_id}"
        )
        approval_id = hashlib.sha1(key.encode()).hexdigest()[:12]

        existing = self._requests.get(approval_id)
        if existing is not None:
            return existing

        request = ApprovalRequest(
            approval_id=approval_id,
            action=decision.action.value,
            target_zone_id=decision.target_zone_id,
            status=ApprovalStatus.REQUESTED,
            reason="Autonomous action requires explicit human authorization.",
        )

        self._requests[approval_id] = request
        return request

    def get(self, approval_id: str) -> ApprovalRequest | None:
        return self._requests.get(approval_id)


    def is_approved(self, approval_id: str) -> bool:
        request = self.get(approval_id)
        return request is not None and request.status == ApprovalStatus.APPROVED

    def is_terminally_blocked(self, approval_id: str) -> bool:
        request = self.get(approval_id)
        return (
            request is not None
            and request.status in {
                ApprovalStatus.REJECTED,
                ApprovalStatus.EXPIRED,
            }
        )

    def mark_pending(self, request: ApprovalRequest) -> ApprovalRequest:
        return self._update(request, ApprovalStatus.PENDING)

    def approve(self, request: ApprovalRequest) -> ApprovalRequest:
        return self._update(request, ApprovalStatus.APPROVED)

    def reject(self, request: ApprovalRequest) -> ApprovalRequest:
        return self._update(request, ApprovalStatus.REJECTED)

    def expire(self, request: ApprovalRequest) -> ApprovalRequest:
        return self._update(request, ApprovalStatus.EXPIRED)

    def _update(
        self,
        request: ApprovalRequest,
        status: ApprovalStatus,
    ) -> ApprovalRequest:
        updated = ApprovalRequest(
            approval_id=request.approval_id,
            action=request.action,
            target_zone_id=request.target_zone_id,
            status=status,
            reason=request.reason,
        )

        self._requests[request.approval_id] = updated
        return updated
