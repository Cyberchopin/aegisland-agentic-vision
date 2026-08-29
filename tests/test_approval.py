from aegisland.approval import ApprovalManager, ApprovalStatus
from aegisland.domain import Action, Decision, SafetyLevel


def decision(*, approval: bool = True) -> Decision:
    return Decision(
        action=Action.REQUEST_HUMAN_APPROVAL,
        safety_level=SafetyLevel.HIGH,
        risk_score=0.7,
        requires_human_approval=approval,
        reasons=("operator review required",),
        evidence_id="approval-frame",
        target_zone_id="Z1-2",
    )


def test_approval_request_has_stable_identity() -> None:
    manager = ApprovalManager()

    first = manager.request(decision())
    second = manager.request(decision())

    assert first is not None
    assert second is not None
    assert first.approval_id == second.approval_id


def test_approval_lifecycle_can_reach_approved() -> None:
    manager = ApprovalManager()

    request = manager.request(decision())

    assert request is not None
    assert request.status == ApprovalStatus.REQUESTED

    request = manager.mark_pending(request)
    assert request.status == ApprovalStatus.PENDING

    request = manager.approve(request)
    assert request.status == ApprovalStatus.APPROVED


def test_non_approval_decision_creates_no_request() -> None:
    manager = ApprovalManager()

    request = manager.request(
        decision(approval=False)
    )

    assert request is None
