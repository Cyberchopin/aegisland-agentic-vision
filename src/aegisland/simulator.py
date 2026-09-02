from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from .domain import Telemetry
from .perception import cv2, np, require_opencv
from .sensor_faults import CameraFault, CameraFaultInjector


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    description: str
    frame_count: int
    start_battery: float
    end_battery: float
    gps_loss_frame: int | None = None
    intrusion_frame: int | None = None
    low_light_frame: int | None = None
    camera_fault: CameraFault = CameraFault.NONE
    camera_fault_start_frame: int | None = None
    camera_fault_end_frame: int | None = None
    camera_fault_severity: float = 1.0
    camera_fault_frames: frozenset[int] = frozenset()


def camera_fault_active(
    scenario: Scenario,
    frame_index: int,
) -> bool:
    if scenario.camera_fault == CameraFault.NONE:
        return False

    if frame_index in scenario.camera_fault_frames:
        return True

    if scenario.camera_fault_start_frame is None:
        return False

    return (
        frame_index >= scenario.camera_fault_start_frame
        and (
            scenario.camera_fault_end_frame is None
            or frame_index < scenario.camera_fault_end_frame
        )
    )


SCENARIOS = {
    "low_battery_intrusion": Scenario(
        name="low_battery_intrusion",
        description="Battery collapses while a moving person enters the best landing zone.",
        frame_count=90,
        start_battery=12,
        end_battery=2,
        intrusion_frame=38,
    ),
    "critical_battery_collision": Scenario(
        name="critical_battery_collision",
        description=(
            "Critical battery coincides with an immediate moving obstacle, "
            "forcing compound emergency recovery."
        ),
        frame_count=45,
        start_battery=3.0,
        end_battery=1.5,
        intrusion_frame=2,
    ),
    "gps_loss_low_light": Scenario(
        name="gps_loss_low_light",
        description="GPS is lost and illumination drops, forcing an active-perception retry.",
        frame_count=80,
        start_battery=18,
        end_battery=6,
        gps_loss_frame=22,
        low_light_frame=30,
    ),
    "gps_denied_camera_failure": Scenario(
        name="gps_denied_camera_failure",
        description=(
            "GPS is lost before severe camera overexposure, "
            "testing visual localization failure and graceful degradation."
        ),
        frame_count=80,
        start_battery=82,
        end_battery=72,
        gps_loss_frame=20,
        camera_fault=CameraFault.OVEREXPOSURE,
        camera_fault_start_frame=40,
        camera_fault_end_frame=56,
        camera_fault_severity=0.96,
    ),
    "gps_denied_camera_flicker": Scenario(
        name="gps_denied_camera_flicker",
        description=(
            "GPS is lost before camera overexposure. "
            "During recovery, camera quality flickers between "
            "healthy and failed states before stabilizing."
        ),
        frame_count=80,
        start_battery=82,
        end_battery=72,
        gps_loss_frame=20,
        camera_fault=CameraFault.OVEREXPOSURE,
        camera_fault_start_frame=40,
        camera_fault_end_frame=56,
        camera_fault_severity=0.96,
        camera_fault_frames=frozenset(
            {
                58,
                61,
            }
        ),
    ),
    "nominal": Scenario(
        name="nominal",
        description="Healthy mission with stable telemetry and clear ground.",
        frame_count=60,
        start_battery=88,
        end_battery=76,
    ),
}


def generate(scenario: Scenario, size: tuple[int, int] = (960, 540)) -> Iterator[tuple[Any, Telemetry]]:
    require_opencv()
    width, height = size
    camera_fault_injector = CameraFaultInjector()

    for index in range(scenario.frame_count):
        progress = index / max(1, scenario.frame_count - 1)
        battery = scenario.start_battery + progress * (scenario.end_battery - scenario.start_battery)
        frame = np.full((height, width, 3), (87, 114, 92), dtype=np.uint8)

        # Deterministic texture gives Canny/Laplacian meaningful but reproducible input.
        for y in range(0, height, 36):
            cv2.line(frame, (0, y), (width, y), (83, 108, 87), 1)
        cv2.rectangle(frame, (58, 225), (285, 495), (113, 151, 116), -1)
        cv2.rectangle(frame, (355, 240), (605, 500), (126, 160, 126), -1)
        cv2.rectangle(frame, (670, 235), (915, 495), (105, 139, 110), -1)
        cv2.putText(frame, "LZ-A", (425, 375), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (174, 195, 174), 2)

        # Static obstacles create structural risk in the left and right candidates.
        cv2.circle(frame, (170, 365), 62, (52, 64, 70), -1)
        cv2.rectangle(frame, (752, 280), (825, 470), (63, 73, 79), -1)

        if scenario.intrusion_frame is not None and index >= scenario.intrusion_frame:
            dx = min(260, (index - scenario.intrusion_frame) * 7)
            person_x = 635 - dx
            cv2.circle(frame, (person_x, 325), 22, (45, 48, 54), -1)
            cv2.rectangle(frame, (person_x - 16, 346), (person_x + 16, 425), (48, 51, 58), -1)

        if scenario.low_light_frame is not None and index >= scenario.low_light_frame:
            factor = 0.34 + 0.08 * ((index // 8) % 2)
            frame = cv2.convertScaleAbs(frame, alpha=factor, beta=0)

        fault_active = camera_fault_active(
            scenario,
            index,
        )

        if fault_active:
            frame = camera_fault_injector.apply(
                frame,
                scenario.camera_fault,
                severity=scenario.camera_fault_severity,
            ).frame

        gps_available = scenario.gps_loss_frame is None or index < scenario.gps_loss_frame
        telemetry = Telemetry(
            battery_percent=round(battery, 2),
            altitude_m=max(1.2, 18.0 - progress * 6.0),
            horizontal_speed_mps=2.0 if battery > 15 else 0.4,
            gps_available=gps_available,
            home_link_available=gps_available,
            timestamp_s=index / 15.0,
        )
        yield frame, telemetry

