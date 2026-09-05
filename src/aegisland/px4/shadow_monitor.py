from __future__ import annotations

import argparse
import asyncio
import time
from math import hypot
from typing import Any

from mavsdk import System

from aegisland.px4.supervisor import (
    Px4ShadowSupervisor,
)
from aegisland.px4.telemetry import (
    Px4TelemetrySnapshot,
    normalize_px4_telemetry,
)
from aegisland.px4.watchdog import (
    evaluate_telemetry_freshness,
)


async def wait_connected(
    drone: System,
) -> None:
    async for state in drone.core.connection_state():
        if state.is_connected:
            return


async def monitor_connection(
    drone: System,
    state: dict[str, Any],
) -> None:
    async for connection in (
        drone.core.connection_state()
    ):
        state["connected"] = (
            connection.is_connected
        )
        state["timestamp"] = (
            time.monotonic()
        )


async def collect_stream(
    stream: Any,
    key: str,
    values: dict[str, Any],
    timestamps: dict[str, float],
) -> None:
    async for item in stream:
        values[key] = item
        timestamps[key] = time.monotonic()


async def run(
    server_address: str,
    grpc_port: int,
) -> None:
    drone = System(
        mavsdk_server_address=server_address,
        port=grpc_port,
    )

    print()
    print("AEGISLAND PX4 SHADOW MODE")
    print("=" * 72)
    print("CONTROL OUTPUT: DISABLED")
    print("COMMAND EXECUTED: FALSE")
    print()

    await asyncio.wait_for(
        drone.connect(),
        timeout=5.0,
    )

    await asyncio.wait_for(
        wait_connected(drone),
        timeout=5.0,
    )

    print("PX4_CONNECTION_OK")
    print()

    supervisor = Px4ShadowSupervisor(
        recovery_samples=3,
    )

    values: dict[str, Any] = {}
    timestamps: dict[str, float] = {}

    connection_state: dict[str, Any] = {
        "connected": True,
        "timestamp": time.monotonic(),
    }

    tasks = [
        asyncio.create_task(
            monitor_connection(
                drone,
                connection_state,
            )
        ),
        asyncio.create_task(
            collect_stream(
                drone.telemetry.battery(),
                "battery",
                values,
                timestamps,
            )
        ),
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

    try:
        while True:
            now = time.monotonic()

            # Position and velocity are our fast telemetry
            # signals. They represent transport liveness
            # much better than slower battery/health topics.
            fast_times = [
                timestamps[key]
                for key in (
                    "position",
                    "velocity",
                )
                if key in timestamps
            ]

            if fast_times:
                last_fast_update = max(
                    fast_times
                )

                telemetry_age_s = max(
                    0.0,
                    now - last_fast_update,
                )
            else:
                telemetry_age_s = float(
                    "inf"
                )

            watchdog = (
                evaluate_telemetry_freshness(
                    telemetry_age_s
                    if telemetry_age_s
                    != float("inf")
                    else 999.0
                )
            )

            required = {
                "battery",
                "position",
                "velocity",
                "health",
            }

            if not required.issubset(values):
                missing = sorted(
                    required - values.keys()
                )

                print(
                    "SAFETY "
                    f"link={watchdog.health.value} "
                    f"age={watchdog.age_s:.3f}s "
                    "nav=degraded "
                    "authority=revoked "
                    "action=hold_and_scan "
                    "executed=False "
                    f"missing={','.join(missing)}"
                )

                print("-" * 72)

                await asyncio.sleep(1.0)
                continue

            battery = values["battery"]
            position = values["position"]
            velocity = values["velocity"]
            health = values["health"]

            snapshot = Px4TelemetrySnapshot(
                battery_percent=(
                    battery.remaining_percent
                ),
                relative_altitude_m=(
                    position.relative_altitude_m
                ),
                velocity_north_mps=(
                    velocity.north_m_s
                ),
                velocity_east_mps=(
                    velocity.east_m_s
                ),
                global_position_ok=(
                    health.is_global_position_ok
                ),
                connected=(
                    watchdog.health.value
                    != "failed"
                ),
                timestamp_s=(
                    max(fast_times)
                    if fast_times
                    else now
                ),
            )

            telemetry = (
                normalize_px4_telemetry(
                    snapshot
                )
            )

            global_position_valid = (
                telemetry.gps_available
            )

            shadow_decision = supervisor.step(
                telemetry_health=watchdog.health,
                global_position_valid=(
                    global_position_valid
                ),
            )

            freshness_label = (
                "FRESH"
                if not watchdog.stale
                else "CACHED"
            )

            print(
                "PX4 "
                f"[{freshness_label}] "
                f"battery={snapshot.battery_percent:6.1f}% "
                f"alt={snapshot.relative_altitude_m:7.2f}m "
                f"speed="
                f"{hypot(snapshot.velocity_north_mps, snapshot.velocity_east_mps):5.2f}m/s "
                f"global_position_valid={snapshot.global_position_ok}"
            )

            print(
                "AEGISLAND "
                f"battery={telemetry.battery_percent:6.1f}% "
                f"alt={telemetry.altitude_m:7.2f}m "
                f"speed={telemetry.horizontal_speed_mps:5.2f}m/s "
                f"global_position_valid={telemetry.gps_available}"
            )

            transport = (
                "connected"
                if connection_state.get(
                    "connected",
                    False,
                )
                else "disconnected"
            )

            print(
                "SAFETY "
                f"transport={transport} "
                f"raw_telemetry={watchdog.health.value} "
                f"telemetry_state={shadow_decision.telemetry_state.value} "
                f"telemetry_recovery={shadow_decision.telemetry_recovery_streak}/3 "
                f"age={watchdog.age_s:.3f}s "
                f"estimator={shadow_decision.global_position_state.value} "
                f"estimator_recovery={shadow_decision.recovery_streak}/3 "
                f"nav={shadow_decision.navigation_mode.value} "
                f"authority={shadow_decision.authority.value} "
                f"action={shadow_decision.action.value} "
                f"executed={shadow_decision.executed}"
            )

            print("-" * 72)

            await asyncio.sleep(1.0)

    finally:
        for task in tasks:
            task.cancel()

        await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )


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

    args = parser.parse_args()

    try:
        asyncio.run(
            run(
                args.server_address,
                args.grpc_port,
            )
        )
    except KeyboardInterrupt:
        print(
            "\nShadow monitor stopped."
        )


if __name__ == "__main__":
    main()
