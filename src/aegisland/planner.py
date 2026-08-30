from __future__ import annotations

from dataclasses import dataclass

from .domain import Action, Decision, SafetyLevel, Telemetry, VisionEvidence


@dataclass(frozen=True, slots=True)
class SafetyPolicy:
    critical_battery: float = 3.0
    emergency_battery: float = 7.0
    return_battery: float = 15.0
    minimum_vision_confidence: float = 0.48
    safe_zone_score: float = 0.64
    collision_risk: float = 0.78
    human_review_risk: float = 0.58


class SafetyPlanner:
    """Deterministic, auditable policy layer for perception-driven action.

    A learned model may suggest an action later, but it cannot bypass this policy.
    This makes every hazardous transition testable and replayable.
    """

    def __init__(self, policy: SafetyPolicy | None = None) -> None:
        self.policy = policy or SafetyPolicy()

    def decide(
        self,
        telemetry: Telemetry,
        evidence: VisionEvidence,
        *,
        target_zone=None,
    ) -> Decision:
        policy = self.policy
        best = target_zone if target_zone is not None else evidence.best_zone
        zone_score = best.score if best else 0.0
        zone_safe = bool(best and best.safe and best.score >= policy.safe_zone_score)

        battery_risk = max(0.0, min(1.0, (policy.return_battery - telemetry.battery_percent) / 15))
        navigation_mode = evidence.navigation_mode

        # Compatibility path for direct planner tests or legacy evidence
        # that has not passed through DynamicConfidenceFusion yet.
        if navigation_mode == "unknown":
            if telemetry.gps_available:
                navigation_mode = "gps_primary"
            elif (
                evidence.visual_localization_valid
                and evidence.visual_localization_confidence >= 0.55
            ):
                navigation_mode = "visual_fallback"
            else:
                navigation_mode = "degraded"

        visual_navigation_available = navigation_mode in {
            "visual_fallback",
            "visual_inertial_fallback",
        }

        if navigation_mode == "gps_primary":
            gps_risk = 0.0
        elif navigation_mode == "visual_inertial_fallback":
            gps_risk = 0.05
        elif navigation_mode == "visual_fallback":
            gps_risk = 0.08
        else:
            gps_risk = 0.18

        risk_score = min(
            1.0,
            0.34 * battery_risk
            + 0.28 * evidence.obstacle_risk
            + 0.22 * evidence.motion_risk
            + 0.16 * (1.0 - evidence.confidence)
            + gps_risk,
        )

        def result(
            action: Action,
            level: SafetyLevel,
            reasons: tuple[str, ...],
            *,
            approval: bool = False,
            target: str | None = None,
        ) -> Decision:
            return Decision(
                action=action,
                safety_level=level,
                risk_score=round(risk_score, 4),
                requires_human_approval=approval,
                reasons=reasons,
                evidence_id=evidence.evidence_id,
                target_zone_id=target,
            )

        local_zone_motion = (
            best.motion_occupancy
            if best is not None
            else 0.0
        )

        local_zone_clearance = (
            best.clearance
            if best is not None
            else 1.0
        )

        landing_zone_intrusion = (
            best is not None
            and (
                local_zone_motion >= 0.08
                or local_zone_clearance < 0.35
            )
        )

        temporal_collision_risk = evidence.temporal_risk >= 0.8

        collision_critical = (
            evidence.obstacle_risk >= policy.collision_risk
            or evidence.motion_risk >= 0.85
            or landing_zone_intrusion
            or temporal_collision_risk
        )

        battery_critical = (
            telemetry.battery_percent <= policy.critical_battery
        )

        if collision_critical and battery_critical:
            return result(
                Action.EMERGENCY_RECOVERY,
                SafetyLevel.CRITICAL,
                (
                    "Critical battery and immediate collision risk are active simultaneously.",
                    "Long-duration hold is unsafe because remaining energy is critically limited.",
                    "The vehicle must evade the immediate hazard while transitioning toward emergency landing.",
                ),
                target=best.zone_id if best else None,
            )

        if collision_critical:
            return result(
                Action.EVADE_AND_HOLD,
                SafetyLevel.CRITICAL,
                (
                    "Immediate collision or moving-object risk exceeds the safety envelope.",
                ),
            )

        if telemetry.battery_percent <= policy.critical_battery:
            reasons = ["Battery is below the critical reserve; delay is more dangerous than landing."]
            if zone_safe:
                reasons.append("OpenCV evidence confirms a currently clear landing zone.")
            else:
                reasons.append("No fully safe zone exists; selecting the least-risk observed zone.")
            return result(
                Action.EMERGENCY_LAND,
                SafetyLevel.CRITICAL,
                tuple(reasons),
                target=best.zone_id if best else None,
            )

        if evidence.confidence < policy.minimum_vision_confidence:
            return result(
                Action.HOLD_AND_SCAN,
                SafetyLevel.HIGH,
                ("Vision confidence is too low for an irreversible landing decision.",),
            )

        if telemetry.battery_percent <= policy.emergency_battery:
            if zone_safe:
                return result(
                    Action.LAND,
                    SafetyLevel.HIGH,
                    (
                        "Battery crossed the emergency threshold.",
                        "The highest-scoring OpenCV landing zone passes all safety gates.",
                    ),
                    target=best.zone_id,
                )
            return result(
                Action.REQUEST_HUMAN_APPROVAL,
                SafetyLevel.HIGH,
                (
                    "Battery requires landing soon, but vision has not found a fully safe zone.",
                    f"Best observed zone score is {zone_score:.2f}.",
                ),
                approval=True,
                target=best.zone_id if best else None,
            )

        if telemetry.battery_percent <= policy.return_battery:
            if telemetry.gps_available and telemetry.home_link_available:
                return result(
                    Action.RETURN_HOME,
                    SafetyLevel.CAUTION,
                    ("Battery reserve is low, while GPS and the home link remain available.",),
                )
            if zone_safe:
                return result(
                    Action.REQUEST_HUMAN_APPROVAL,
                    SafetyLevel.HIGH,
                    ("Return-to-home is unavailable; OpenCV found a viable contingency zone.",),
                    approval=True,
                    target=best.zone_id,
                )
            return result(
                Action.HOLD_AND_SCAN,
                SafetyLevel.HIGH,
                ("Return-to-home is unavailable and no safe contingency zone is confirmed.",),
            )

        if not telemetry.gps_available:
            if visual_navigation_available:
                return result(
                    Action.CONTINUE_MISSION,
                    SafetyLevel.CAUTION,
                    (
                        "GPS is unavailable.",
                        "Visual dead reckoning remains above the localization confidence gate.",
                        "Only bounded local autonomy is permitted while GPS-denied.",
                    ),
                    target=best.zone_id if best else None,
                )

            return result(
                Action.HOLD_AND_SCAN,
                SafetyLevel.HIGH,
                (
                    "GPS is unavailable and visual localization is not reliable enough for bounded autonomy.",
                    "The vehicle holds position while perception attempts to recover localization.",
                ),
            )

        if risk_score >= policy.human_review_risk:
            return result(
                Action.REQUEST_HUMAN_APPROVAL,
                SafetyLevel.HIGH,
                ("Combined visual and telemetry risk requires operator review.",),
                approval=True,
                target=best.zone_id if best else None,
            )

        return result(
            Action.CONTINUE_MISSION,
            SafetyLevel.NOMINAL,
            ("Telemetry and current OpenCV evidence remain inside the safety envelope.",),
        )

