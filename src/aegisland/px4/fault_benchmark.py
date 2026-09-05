from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mavsdk import System

from aegisland.px4.shadow_policy import (
    ShadowAuthority,
)
from aegisland.px4.supervisor import (
    Px4ShadowSupervisor,
)
from aegisland.px4.watchdog import (
    LinkHealth,
    evaluate_telemetry_freshness,
)


@dataclass(frozen=True, slots=True)
class Px4FaultBenchmarkMetrics:
    baseline_samples: int
    fault_samples: int
    recovery_samples: int

    observed_fault: bool
    observed_recovery: bool

    authority_revocation_latency_ms: float | None
    authority_restore_latency_ms: float | None

    unsafe_continuation_samples: int
    premature_authority_samples: int
    telemetry_false_positive_samples: int

    command_execution_enabled: bool


async def wait_connected(drone: System) -> None:
    async for state in drone.core.connection_state():
        if state.is_connected:
            return


async def collect_stream(
    stream: Any,
    key: str,
    values: dict[str, Any],
    timestamps: dict[str, float],
) -> None:
    async for item in stream:
        values[key] = item
        timestamps[key] = time.monotonic()


async def run_benchmark(
    server_address: str,
    grpc_port: int,
    output: Path,
) -> Px4FaultBenchmarkMetrics:
    drone = System(
        mavsdk_server_address=server_address,
        port=grpc_port,
    )

    await drone.connect()
    await wait_connected(drone)

    values: dict[str, Any] = {}
    timestamps: dict[str, float] = {}

    tasks = [
        asyncio.create_task(
            collect_stream(
                drone.telemetry.position(),
                "position",
                values,
                timestamps,
            )
        ),
        asyncio.create_task(
            collect_stream(
                drone.telemetry.velocity_ned(),
                "velocity",
                values,
                timestamps,
            )
        ),
        asyncio.create_task(
            collect_stream(
                drone.telemetry.health(),
                "health",
                values,
                timestamps,
            )
        ),
    ]

    supervisor = Px4ShadowSupervisor(
        recovery_samples=3,
    )

    baseline_samples = 0
    fault_samples = 0
    recovery_samples = 0

    unsafe_continuation_samples = 0
    premature_authority_samples = 0
    telemetry_false_positive_samples = 0

    observed_fault = False
    observed_recovery = False

    first_invalid_time: float | None = None
    first_revoked_time: float | None = None

    first_recovery_valid_time: float | None = None
    first_restored_time: float | None = None

    baseline_ready = False
    previous_valid: bool | None = None

    print()
    print("PX4 FAULT BENCHMARK")
    print("=" * 72)
    print("CONTROL OUTPUT: DISABLED")
    print()
    print("Waiting for stable healthy baseline...")

    try:
        while True:
            await asyncio.sleep(0.1)

            if not {
                "position",
                "velocity",
                "health",
            }.issubset(values):
                continue

            now = time.monotonic()

            last_fast_update = max(
                timestamps["position"],
                timestamps["velocity"],
            )

            watchdog = evaluate_telemetry_freshness(
                max(
                    0.0,
                    now - last_fast_update,
                )
            )

            global_position_valid = (
                values[
                    "health"
                ].is_global_position_ok
            )

            decision = supervisor.step(
                telemetry_health=watchdog.health,
                global_position_valid=(
                    global_position_valid
                ),
            )

            if not baseline_ready:
                if (
                    global_position_valid
                    and watchdog.health
                    == LinkHealth.HEALTHY
                    and decision.authority
                    == ShadowAuthority.GRANTED
                ):
                    baseline_samples += 1
                else:
                    baseline_samples = 0

                if baseline_samples >= 5:
                    baseline_ready = True
                    print(
                        "BASELINE_READY "
                        "→ now inject: failure gps off"
                    )

                previous_valid = (
                    global_position_valid
                )
                continue

            if (
                watchdog.health
                != LinkHealth.HEALTHY
                and global_position_valid
            ):
                telemetry_false_positive_samples += 1

            # Fault onset = first observed valid -> invalid transition.
            if (
                previous_valid is True
                and not global_position_valid
                and not observed_fault
            ):
                observed_fault = True
                first_invalid_time = now

                print()
                print("FAULT_OBSERVED")
                print(
                    "global_position_valid=False"
                )

            if observed_fault and not observed_recovery:
                fault_samples += 1

                if (
                    not global_position_valid
                    and decision.authority
                    == ShadowAuthority.GRANTED
                ):
                    unsafe_continuation_samples += 1

                if (
                    first_revoked_time is None
                    and decision.authority
                    == ShadowAuthority.REVOKED
                ):
                    first_revoked_time = now

            # First valid sample after the observed failure.
            if (
                observed_fault
                and not observed_recovery
                and previous_valid is False
                and global_position_valid
            ):
                observed_recovery = True
                first_recovery_valid_time = now

                print()
                print("RECOVERY_OBSERVED")
                print(
                    "global_position_valid=True"
                )

            if observed_recovery:
                recovery_samples += 1

                if (
                    decision.authority
                    == ShadowAuthority.GRANTED
                ):
                    if first_restored_time is None:
                        first_restored_time = now

                    # Benchmark complete once authority
                    # has safely returned.
                    break

                if (
                    global_position_valid
                    and decision.authority
                    == ShadowAuthority.GRANTED
                    and decision.global_position_state.value
                    != "healthy"
                ):
                    premature_authority_samples += 1

            print(
                "SAMPLE "
                f"valid={global_position_valid} "
                f"telemetry={watchdog.health.value} "
                f"estimator="
                f"{decision.global_position_state.value} "
                f"recovery="
                f"{decision.recovery_streak}/3 "
                f"authority={decision.authority.value} "
                f"action={decision.action.value}"
            )

            previous_valid = (
                global_position_valid
            )

    finally:
        for task in tasks:
            task.cancel()

        await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

    revoke_latency_ms = None

    if (
        first_invalid_time is not None
        and first_revoked_time is not None
    ):
        revoke_latency_ms = (
            first_revoked_time
            - first_invalid_time
        ) * 1000.0

    restore_latency_ms = None

    if (
        first_recovery_valid_time is not None
        and first_restored_time is not None
    ):
        restore_latency_ms = (
            first_restored_time
            - first_recovery_valid_time
        ) * 1000.0

    metrics = Px4FaultBenchmarkMetrics(
        baseline_samples=baseline_samples,
        fault_samples=fault_samples,
        recovery_samples=recovery_samples,
        observed_fault=observed_fault,
        observed_recovery=observed_recovery,
        authority_revocation_latency_ms=(
            None
            if revoke_latency_ms is None
            else round(revoke_latency_ms, 3)
        ),
        authority_restore_latency_ms=(
            None
            if restore_latency_ms is None
            else round(restore_latency_ms, 3)
        ),
        unsafe_continuation_samples=(
            unsafe_continuation_samples
        ),
        premature_authority_samples=(
            premature_authority_samples
        ),
        telemetry_false_positive_samples=(
            telemetry_false_positive_samples
        ),
        command_execution_enabled=False,
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            asdict(metrics),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--server-address",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--grpc-port",
        type=int,
        default=50051,
    )

    parser.add_argument(
        "--output",
        default=(
            "runs/px4-fault-benchmark/"
            "gps-failure.json"
        ),
    )

    args = parser.parse_args()

    metrics = asyncio.run(
        run_benchmark(
            args.server_address,
            args.grpc_port,
            Path(args.output),
        )
    )

    print()
    print("=" * 72)
    print("PX4 GPS FAILURE BENCHMARK RESULT")
    print("=" * 72)

    for key, value in asdict(metrics).items():
        print(
            f"{key:40} {value}"
        )

    print()
    print(
        f"Evidence written to: {args.output}"
    )


if __name__ == "__main__":
    main()
