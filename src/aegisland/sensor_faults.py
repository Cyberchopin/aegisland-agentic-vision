from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np


class CameraFault(StrEnum):
    NONE = "none"
    OVEREXPOSURE = "overexposure"
    UNDEREXPOSURE = "underexposure"
    OCCLUSION = "occlusion"
    BLUR = "blur"


@dataclass(frozen=True, slots=True)
class CameraFaultResult:
    frame: np.ndarray
    fault: CameraFault
    severity: float
    affected_fraction: float


class CameraFaultInjector:
    """
    Deterministic image-space camera fault injection.

    This does not fake perception outputs. It corrupts the actual frame
    before the OpenCV perception pipeline sees it.
    """

    def apply(
        self,
        frame: np.ndarray,
        fault: CameraFault,
        *,
        severity: float = 1.0,
    ) -> CameraFaultResult:
        if frame is None or frame.size == 0:
            raise ValueError("frame must be a non-empty numpy array")

        severity = self._clamp(severity)

        if fault == CameraFault.NONE or severity == 0.0:
            return CameraFaultResult(
                frame=frame.copy(),
                fault=CameraFault.NONE,
                severity=0.0,
                affected_fraction=0.0,
            )

        if fault == CameraFault.OVEREXPOSURE:
            corrupted = self._overexpose(
                frame,
                severity,
            )

            return CameraFaultResult(
                frame=corrupted,
                fault=fault,
                severity=severity,
                affected_fraction=1.0,
            )

        if fault == CameraFault.UNDEREXPOSURE:
            corrupted = self._underexpose(
                frame,
                severity,
            )

            return CameraFaultResult(
                frame=corrupted,
                fault=fault,
                severity=severity,
                affected_fraction=1.0,
            )

        if fault == CameraFault.OCCLUSION:
            corrupted, fraction = self._occlude(
                frame,
                severity,
            )

            return CameraFaultResult(
                frame=corrupted,
                fault=fault,
                severity=severity,
                affected_fraction=fraction,
            )

        if fault == CameraFault.BLUR:
            corrupted = self._blur(
                frame,
                severity,
            )

            return CameraFaultResult(
                frame=corrupted,
                fault=fault,
                severity=severity,
                affected_fraction=1.0,
            )

        raise ValueError(
            f"Unsupported camera fault: {fault}"
        )

    @staticmethod
    def _overexpose(
        frame: np.ndarray,
        severity: float,
    ) -> np.ndarray:
        target = np.full_like(
            frame,
            255,
        )

        return cv2.addWeighted(
            frame,
            1.0 - severity,
            target,
            severity,
            0.0,
        )

    @staticmethod
    def _underexpose(
        frame: np.ndarray,
        severity: float,
    ) -> np.ndarray:
        return cv2.convertScaleAbs(
            frame,
            alpha=max(
                0.0,
                1.0 - severity,
            ),
            beta=0.0,
        )

    @staticmethod
    def _occlude(
        frame: np.ndarray,
        severity: float,
    ) -> tuple[np.ndarray, float]:
        corrupted = frame.copy()

        height, width = corrupted.shape[:2]

        fraction = 0.15 + 0.75 * severity
        fraction = min(
            0.90,
            max(0.0, fraction),
        )

        occlusion_width = max(
            1,
            int(width * fraction),
        )

        start_x = max(
            0,
            (width - occlusion_width) // 2,
        )

        end_x = min(
            width,
            start_x + occlusion_width,
        )

        corrupted[
            :,
            start_x:end_x,
        ] = 0

        actual_fraction = (
            (end_x - start_x)
            / width
        )

        return (
            corrupted,
            actual_fraction,
        )

    @staticmethod
    def _blur(
        frame: np.ndarray,
        severity: float,
    ) -> np.ndarray:
        kernel = int(
            3 + severity * 28
        )

        if kernel % 2 == 0:
            kernel += 1

        kernel = max(
            3,
            kernel,
        )

        return cv2.GaussianBlur(
            frame,
            (kernel, kernel),
            0,
        )

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:
        return max(
            0.0,
            min(1.0, float(value)),
        )
