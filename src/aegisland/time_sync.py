from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from enum import StrEnum


class SyncMethod(StrEnum):
    EXACT = "exact"
    INTERPOLATED = "interpolated"
    EXTRAPOLATED = "extrapolated"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class TimedVectorSample:
    timestamp_s: float
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SyncResult:
    timestamp_s: float
    values: tuple[float, ...] | None
    method: SyncMethod
    valid: bool
    source_time_error_s: float


class TimeSyncBuffer:
    """
    Bounded timestamp-ordered sensor buffer.

    Supports:
    - out-of-order insertion
    - exact lookup
    - linear interpolation
    - short-horizon constant-velocity extrapolation

    Designed for sensor-state synchronization, not raw image buffering.
    """

    def __init__(
        self,
        *,
        max_samples: int = 512,
        max_extrapolation_s: float = 0.05,
    ) -> None:
        if max_samples < 2:
            raise ValueError("max_samples must be at least 2")

        self.max_samples = max_samples
        self.max_extrapolation_s = max_extrapolation_s
        self._samples: list[TimedVectorSample] = []
        self._out_of_order_insertions = 0

    @property
    def samples(self) -> tuple[TimedVectorSample, ...]:
        return tuple(self._samples)

    @property
    def out_of_order_insertions(self) -> int:
        return self._out_of_order_insertions

    def add(
        self,
        timestamp_s: float,
        values: tuple[float, ...],
    ) -> None:
        sample = TimedVectorSample(
            timestamp_s=float(timestamp_s),
            values=tuple(float(value) for value in values),
        )

        if (
            self._samples
            and sample.timestamp_s
            < self._samples[-1].timestamp_s
        ):
            self._out_of_order_insertions += 1

        timestamps = [
            existing.timestamp_s
            for existing in self._samples
        ]

        index = bisect_left(
            timestamps,
            sample.timestamp_s,
        )

        if (
            index < len(self._samples)
            and self._samples[index].timestamp_s
            == sample.timestamp_s
        ):
            self._samples[index] = sample
        else:
            self._samples.insert(index, sample)

        overflow = len(self._samples) - self.max_samples

        if overflow > 0:
            del self._samples[:overflow]

    def sample_at(
        self,
        timestamp_s: float,
    ) -> SyncResult:
        timestamp_s = float(timestamp_s)

        if not self._samples:
            return self._unavailable(timestamp_s)

        timestamps = [
            sample.timestamp_s
            for sample in self._samples
        ]

        index = bisect_left(
            timestamps,
            timestamp_s,
        )

        # Exact timestamp.
        if (
            index < len(self._samples)
            and self._samples[index].timestamp_s
            == timestamp_s
        ):
            sample = self._samples[index]

            return SyncResult(
                timestamp_s=timestamp_s,
                values=sample.values,
                method=SyncMethod.EXACT,
                valid=True,
                source_time_error_s=0.0,
            )

        # Interpolate between two surrounding samples.
        if 0 < index < len(self._samples):
            before = self._samples[index - 1]
            after = self._samples[index]

            if len(before.values) != len(after.values):
                return self._unavailable(timestamp_s)

            duration = (
                after.timestamp_s
                - before.timestamp_s
            )

            if duration <= 0.0:
                return self._unavailable(timestamp_s)

            alpha = (
                timestamp_s - before.timestamp_s
            ) / duration

            values = tuple(
                left + alpha * (right - left)
                for left, right in zip(
                    before.values,
                    after.values,
                )
            )

            return SyncResult(
                timestamp_s=timestamp_s,
                values=values,
                method=SyncMethod.INTERPOLATED,
                valid=True,
                source_time_error_s=0.0,
            )

        # Short-horizon extrapolation past newest sample.
        if (
            index == len(self._samples)
            and len(self._samples) >= 2
        ):
            older = self._samples[-2]
            newest = self._samples[-1]

            horizon = (
                timestamp_s
                - newest.timestamp_s
            )

            if (
                horizon < 0.0
                or horizon > self.max_extrapolation_s
            ):
                return self._unavailable(timestamp_s)

            dt = (
                newest.timestamp_s
                - older.timestamp_s
            )

            if (
                dt <= 0.0
                or len(older.values) != len(newest.values)
            ):
                return self._unavailable(timestamp_s)

            velocity = tuple(
                (new - old) / dt
                for old, new in zip(
                    older.values,
                    newest.values,
                )
            )

            values = tuple(
                value + rate * horizon
                for value, rate in zip(
                    newest.values,
                    velocity,
                )
            )

            return SyncResult(
                timestamp_s=timestamp_s,
                values=values,
                method=SyncMethod.EXTRAPOLATED,
                valid=True,
                source_time_error_s=horizon,
            )

        return self._unavailable(timestamp_s)

    @staticmethod
    def _unavailable(
        timestamp_s: float,
    ) -> SyncResult:
        return SyncResult(
            timestamp_s=timestamp_s,
            values=None,
            method=SyncMethod.UNAVAILABLE,
            valid=False,
            source_time_error_s=float("inf"),
        )
