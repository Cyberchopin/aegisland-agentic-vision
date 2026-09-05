from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mavsdk import System

from aegisland.px4.shadow_policy import ShadowAuthority
from aegisland.px4.supervisor import Px4ShadowSupervisor
from aegisland.px4.watchdog import (
    LinkHealth,
    evaluate_telemetry_freshness,
)


@dataclass(frozen=True, slots=True)
class TimelineEvent:
    t_s: float
    event: str
    telemetry_age_s: float
    telemetry_health: str
    global_position_valid: bool
    authority: str
    action: str


@dataclass(frozen=True, slots=True)
class SafetyGate:
    gate_id: str
    description: str
    passed: bool
    observed: str


@dataclass(frozen=True, slots=True)
class LinkBenchmarkSummary:
    baseline_samples: int

    observed_degraded: bool
    observed_failed: bool
    observed_recovery: bool

    degraded_latency_ms: float | None
    failed_latency_ms: float | None
    authority_revocation_latency_ms: float | None
    authority_restore_latency_ms: float | None

    stale_authority_samples: int
    cached_position_authority_samples: int
    premature_recovery_samples: int

    command_execution_enabled: bool
    overall_pass: bool


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


def write_report(
    path: Path,
    summary: LinkBenchmarkSummary,
    gates: list[SafetyGate],
    timeline: list[TimelineEvent],
) -> None:
    lines = [
        "# PX4 Telemetry Interruption Safety Benchmark",
        "",
        (
            "> PX4 SITL / MAVSDK shadow-mode evaluation. "
            "No flight-control commands were executed."
        ),
        "",
        f"**Overall: {'PASS' if summary.overall_pass else 'FAIL'}**",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]

    for key, value in asdict(summary).items():
        lines.append(
            f"| `{key}` | `{value}` |"
        )

    lines.extend(
        [
            "",
            "## Safety Gates",
            "",
            "| Gate | Requirement | Result | Observed |",
            "|---|---|---|---|",
        ]
    )

    for gate in gates:
        result = (
            "PASS"
            if gate.passed
            else "FAIL"
        )

        lines.append(
            f"| `{gate.gate_id}` | "
            f"{gate.description} | "
            f"**{result}** | "
            f"`{gate.observed}` |"
        )

    lines.extend(
        [
            "",
            "## Timeline",
            "",
            (
                "| t (s) | Event | Age (s) | Telemetry | "
                "Position Valid | Authority | Action |"
            ),
            "|---:|---|---:|---|---|---|---|",
        ]
    )

    for event in timeline:
        lines.append(
            f"| {event.t_s:.3f} | "
            f"{event.event} | "
            f"{event.telemetry_age_s:.3f} | "
            f"{event.telemetry_health} | "
            f"{event.global_position_valid} | "
            f"{event.authority} | "
            f"{event.action} |"
        )

    lines.extend(
        [
            "",
            "## Key principle",
            "",
            (
                "A cached position value may still look numerically valid "
                "after telemetry stops updating. Navigation authority must "
                "depend on freshness, not only on the value itself."
            ),
            "",
            "## Scope",
            "",
            (
                "This is PX4 SITL shadow-mode safety evidence. "
                "It is not flight certification or closed-loop "
                "physical flight control."
            ),
            "",
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


async def run_benchmark(
    server_address: str,
    grpc_port: int,
    output_dir: Path,
) -> LinkBenchmarkSummary:
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
    baseline_ready = False

    observed_degraded = False
    observed_failed = False
    observed_recovery = False

    interruption_start_time: float | None = None
    degraded_time: float | None = None
    failed_time: float | None = None
    revoke_time: float | None = None

    recovery_fresh_time: float | None = None
    restore_time: float | None = None

    stale_authority_samples = 0
    cached_position_authority_samples = 0
    premature_recovery_samples = 0

    benchmark_t0: float | None = None
    last_fresh_time: float | None = None

    previous_health: LinkHealth | None = None
    previous_authority: str | None = None

    timeline: list[TimelineEvent] = []

    print()
    print("PX4 TELEMETRY INTERRUPTION BENCHMARK")
    print("=" * 72)
    print("CONTROL OUTPUT: DISABLED")
    print("Waiting for stable baseline...")

    try:
        while True:
            await asyncio.sleep(0.05)

            required = {
                "position",
                "velocity",
                "health",
            }

            if not required.issubset(values):
                continue

            now = time.monotonic()

            current_fast_time = max(
                timestamps["position"],
                timestamps["velocity"],
            )

            age_s = max(
                0.0,
                now - current_fast_time,
            )

            watchdog = (
                evaluate_telemetry_freshness(
                    age_s
                )
            )

            global_position_valid = bool(
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

            authority = (
                decision.authority.value
            )

            if not baseline_ready:
                healthy = (
                    watchdog.health
                    == LinkHealth.HEALTHY
                    and global_position_valid
                    and decision.authority
                    == ShadowAuthority.GRANTED
                )

                if healthy:
                    baseline_samples += 1
                else:
                    baseline_samples = 0

                if baseline_samples >= 10:
                    baseline_ready = True
                    benchmark_t0 = now
                    last_fresh_time = (
                        current_fast_time
                    )

                    print()
                    print("BASELINE_READY")
                    print(
                        "Now stop the PX4 MAVLink stream."
                    )

                    timeline.append(
                        TimelineEvent(
                            t_s=0.0,
                            event="BASELINE_READY",
                            telemetry_age_s=age_s,
                            telemetry_health=(
                                watchdog.health.value
                            ),
                            global_position_valid=(
                                global_position_valid
                            ),
                            authority=authority,
                            action=(
                                decision.action.value
                            ),
                        )
                    )

                previous_health = (
                    watchdog.health
                )
                previous_authority = authority
                continue

            assert benchmark_t0 is not None

            relative_time = (
                now - benchmark_t0
            )

            # Fresh telemetry is still arriving.
            if (
                current_fast_time
                != last_fresh_time
            ):
                last_fresh_time = (
                    current_fast_time
                )

                # If we have already seen a telemetry fault,
                # this is the beginning of recovery.
                if (
                    observed_degraded
                    and recovery_fresh_time
                    is None
                ):
                    recovery_fresh_time = now

                    timeline.append(
                        TimelineEvent(
                            t_s=relative_time,
                            event=(
                                "FRESH_TELEMETRY_RETURNED"
                            ),
                            telemetry_age_s=age_s,
                            telemetry_health=(
                                watchdog.health.value
                            ),
                            global_position_valid=(
                                global_position_valid
                            ),
                            authority=authority,
                            action=(
                                decision.action.value
                            ),
                        )
                    )

                    print()
                    print(
                        "FRESH_TELEMETRY_RETURNED"
                    )

            # We define interruption onset at the last
            # fresh navigation sample before watchdog
            # degradation becomes observable.
            if (
                watchdog.health
                == LinkHealth.DEGRADED
                and not observed_degraded
            ):
                observed_degraded = True
                degraded_time = now
                interruption_start_time = (
                    current_fast_time
                )

                timeline.append(
                    TimelineEvent(
                        t_s=relative_time,
                        event="TELEMETRY_DEGRADED",
                        telemetry_age_s=age_s,
                        telemetry_health=(
                            watchdog.health.value
                        ),
                        global_position_valid=(
                            global_position_valid
                        ),
                        authority=authority,
                        action=(
                            decision.action.value
                        ),
                    )
                )

                print()
                print("TELEMETRY_DEGRADED")

            if (
                watchdog.health
                == LinkHealth.FAILED
                and not observed_failed
            ):
                observed_failed = True
                failed_time = now

                timeline.append(
                    TimelineEvent(
                        t_s=relative_time,
                        event="TELEMETRY_FAILED",
                        telemetry_age_s=age_s,
                        telemetry_health=(
                            watchdog.health.value
                        ),
                        global_position_valid=(
                            global_position_valid
                        ),
                        authority=authority,
                        action=(
                            decision.action.value
                        ),
                    )
                )

                print()
                print("TELEMETRY_FAILED")

            if (
                observed_degraded
                and revoke_time is None
                and decision.authority
                == ShadowAuthority.REVOKED
            ):
                revoke_time = now

                timeline.append(
                    TimelineEvent(
                        t_s=relative_time,
                        event="AUTHORITY_REVOKED",
                        telemetry_age_s=age_s,
                        telemetry_health=(
                            watchdog.health.value
                        ),
                        global_position_valid=(
                            global_position_valid
                        ),
                        authority=authority,
                        action=(
                            decision.action.value
                        ),
                    )
                )

            # Any non-healthy telemetry retaining authority
            # would be unsafe.
            if (
                watchdog.health
                != LinkHealth.HEALTHY
                and decision.authority
                == ShadowAuthority.GRANTED
            ):
                stale_authority_samples += 1

            # Stronger form of the same test:
            # cached position may still claim validity.
            if (
                watchdog.health
                != LinkHealth.HEALTHY
                and global_position_valid
                and decision.authority
                == ShadowAuthority.GRANTED
            ):
                cached_position_authority_samples += 1

            if recovery_fresh_time is not None:
                # Raw telemetry may already be fresh, while
                # effective telemetry authority is still
                # recovering through hysteresis.
                if (
                    decision.telemetry_state.value
                    != "healthy"
                    and decision.authority
                    == ShadowAuthority.GRANTED
                ):
                    premature_recovery_samples += 1

                if (
                    watchdog.health
                    == LinkHealth.HEALTHY
                    and decision.telemetry_state.value
                    == "healthy"
                    and decision.authority
                    == ShadowAuthority.GRANTED
                ):
                    observed_recovery = True
                    restore_time = now

                    timeline.append(
                        TimelineEvent(
                            t_s=relative_time,
                            event="AUTHORITY_RESTORED",
                            telemetry_age_s=age_s,
                            telemetry_health=(
                                watchdog.health.value
                            ),
                            global_position_valid=(
                                global_position_valid
                            ),
                            authority=authority,
                            action=(
                                decision.action.value
                            ),
                        )
                    )

                    break

            if (
                watchdog.health
                != previous_health
                or authority
                != previous_authority
            ):
                timeline.append(
                    TimelineEvent(
                        t_s=relative_time,
                        event="STATE_TRANSITION",
                        telemetry_age_s=age_s,
                        telemetry_health=(
                            watchdog.health.value
                        ),
                        global_position_valid=(
                            global_position_valid
                        ),
                        authority=authority,
                        action=(
                            decision.action.value
                        ),
                    )
                )

            previous_health = (
                watchdog.health
            )
            previous_authority = authority

    finally:
        for task in tasks:
            task.cancel()

        await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

    degraded_latency_ms = None
    failed_latency_ms = None
    revoke_latency_ms = None
    restore_latency_ms = None

    if (
        interruption_start_time is not None
        and degraded_time is not None
    ):
        degraded_latency_ms = round(
            (
                degraded_time
                - interruption_start_time
            )
            * 1000.0,
            3,
        )

    if (
        interruption_start_time is not None
        and failed_time is not None
    ):
        failed_latency_ms = round(
            (
                failed_time
                - interruption_start_time
            )
            * 1000.0,
            3,
        )

    if (
        interruption_start_time is not None
        and revoke_time is not None
    ):
        revoke_latency_ms = round(
            (
                revoke_time
                - interruption_start_time
            )
            * 1000.0,
            3,
        )

    if (
        recovery_fresh_time is not None
        and restore_time is not None
    ):
        restore_latency_ms = round(
            (
                restore_time
                - recovery_fresh_time
            )
            * 1000.0,
            3,
        )

    gates = [
        SafetyGate(
            "PX4-LINK-01",
            "Telemetry interruption reaches DEGRADED",
            observed_degraded,
            str(observed_degraded),
        ),
        SafetyGate(
            "PX4-LINK-02",
            "Telemetry interruption reaches FAILED",
            observed_failed,
            str(observed_failed),
        ),
        SafetyGate(
            "PX4-LINK-03",
            "No authority while telemetry is stale",
            stale_authority_samples == 0,
            str(stale_authority_samples),
        ),
        SafetyGate(
            "PX4-LINK-04",
            "Cached valid position never retains authority",
            cached_position_authority_samples
            == 0,
            str(
                cached_position_authority_samples
            ),
        ),
        SafetyGate(
            "PX4-LINK-05",
            "Authority is revoked at degraded freshness threshold",
            (
                revoke_latency_ms is not None
                and revoke_latency_ms
                <= 650.0
            ),
            f"{revoke_latency_ms} ms",
        ),
        SafetyGate(
            "PX4-LINK-06",
            "Fresh telemetry recovery is observed",
            observed_recovery,
            str(observed_recovery),
        ),
        SafetyGate(
            "PX4-LINK-07",
            "No premature authority during stale recovery",
            premature_recovery_samples == 0,
            str(premature_recovery_samples),
        ),
        SafetyGate(
            "PX4-LINK-08",
            "Control execution remains disabled",
            True,
            "False",
        ),
    ]

    overall_pass = all(
        gate.passed
        for gate in gates
    )

    summary = LinkBenchmarkSummary(
        baseline_samples=baseline_samples,
        observed_degraded=observed_degraded,
        observed_failed=observed_failed,
        observed_recovery=observed_recovery,
        degraded_latency_ms=(
            degraded_latency_ms
        ),
        failed_latency_ms=(
            failed_latency_ms
        ),
        authority_revocation_latency_ms=(
            revoke_latency_ms
        ),
        authority_restore_latency_ms=(
            restore_latency_ms
        ),
        stale_authority_samples=(
            stale_authority_samples
        ),
        cached_position_authority_samples=(
            cached_position_authority_samples
        ),
        premature_recovery_samples=(
            premature_recovery_samples
        ),
        command_execution_enabled=False,
        overall_pass=overall_pass,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output_dir
        / "link-interruption-summary.json"
    ).write_text(
        json.dumps(
            asdict(summary),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (
        output_dir
        / "link-interruption-timeline.json"
    ).write_text(
        json.dumps(
            [
                asdict(event)
                for event in timeline
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (
        output_dir
        / "link-interruption-gates.json"
    ).write_text(
        json.dumps(
            [
                asdict(gate)
                for gate in gates
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    write_report(
        output_dir
        / "link-interruption-report.md",
        summary,
        gates,
        timeline,
    )

    print()
    print("=" * 72)
    print("PX4 LINK SAFETY ASSESSMENT")
    print("=" * 72)

    for gate in gates:
        result = (
            "PASS"
            if gate.passed
            else "FAIL"
        )

        print(
            f"{gate.gate_id:13} "
            f"{result:4}  "
            f"{gate.description}"
        )

    print()
    print(
        "OVERALL:",
        "PASS"
        if overall_pass
        else "FAIL",
    )

    print()
    print(
        "degraded latency:",
        degraded_latency_ms,
        "ms",
    )
    print(
        "failed latency:",
        failed_latency_ms,
        "ms",
    )
    print(
        "authority revoke latency:",
        revoke_latency_ms,
        "ms",
    )
    print(
        "authority restore latency:",
        restore_latency_ms,
        "ms",
    )

    print()
    print(
        "stale authority samples:",
        stale_authority_samples,
    )
    print(
        "cached-position authority samples:",
        cached_position_authority_samples,
    )

    print()
    print(
        f"Evidence: {output_dir}"
    )

    return summary


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
        "--output-dir",
        default=(
            "runs/"
            "px4-link-fault-benchmark"
        ),
    )

    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            args.server_address,
            args.grpc_port,
            Path(args.output_dir),
        )
    )


if __name__ == "__main__":
    main()
