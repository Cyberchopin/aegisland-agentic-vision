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
    global_position_valid: bool
    telemetry_health: str
    estimator_state: str
    recovery_streak: int
    authority: str
    action: str


@dataclass(frozen=True, slots=True)
class SafetyGate:
    gate_id: str
    description: str
    passed: bool
    observed: str


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
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


def write_markdown(
    path: Path,
    summary: BenchmarkSummary,
    gates: list[SafetyGate],
    timeline: list[TimelineEvent],
) -> None:
    lines = [
        "# PX4 GPS Failure Safety Benchmark",
        "",
        (
            "> PX4 SITL / MAVSDK shadow-mode evaluation. "
            "No flight-control commands were executed."
        ),
        "",
        "## Result",
        "",
        f"**Overall: {'PASS' if summary.overall_pass else 'FAIL'}**",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]

    for key, value in asdict(summary).items():
        lines.append(f"| `{key}` | `{value}` |")

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
        result = "PASS" if gate.passed else "FAIL"
        lines.append(
            f"| `{gate.gate_id}` | "
            f"{gate.description} | "
            f"**{result}** | "
            f"`{gate.observed}` |"
        )

    lines.extend(
        [
            "",
            "## Event Timeline",
            "",
            (
                "| t (s) | Event | Position | Telemetry | "
                "Estimator | Recovery | Authority | Action |"
            ),
            "|---:|---|---|---|---|---:|---|---|",
        ]
    )

    for event in timeline:
        lines.append(
            f"| {event.t_s:.3f} | "
            f"{event.event} | "
            f"{event.global_position_valid} | "
            f"{event.telemetry_health} | "
            f"{event.estimator_state} | "
            f"{event.recovery_streak}/3 | "
            f"{event.authority} | "
            f"{event.action} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "- Failure authority revocation is expected to be immediate "
                "once invalid global position is observed."
            ),
            (
                "- Recovery is intentionally slower and requires sustained "
                "healthy observations."
            ),
            (
                "- Zero unsafe continuation samples means navigation "
                "authority was never retained while the observed global "
                "position capability was invalid."
            ),
            (
                "- Zero premature authority samples means authority was "
                "not restored before estimator-health hysteresis completed."
            ),
            "",
            "## Scope",
            "",
            (
                "This is deterministic PX4 SITL shadow-mode safety evidence. "
                "It is not flight certification and does not demonstrate "
                "closed-loop physical flight control."
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
) -> BenchmarkSummary:
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

    previous_valid: bool | None = None
    previous_estimator: str | None = None
    previous_authority: str | None = None

    baseline_ready = False
    benchmark_t0: float | None = None

    timeline: list[TimelineEvent] = []

    print()
    print("PX4 FAULT BENCHMARK V2")
    print("=" * 72)
    print("CONTROL OUTPUT: DISABLED")
    print("Waiting for stable baseline...")

    try:
        while True:
            await asyncio.sleep(0.1)

            required = {
                "position",
                "velocity",
                "health",
            }

            if not required.issubset(values):
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
                values["health"].is_global_position_ok
            )

            decision = supervisor.step(
                telemetry_health=watchdog.health,
                global_position_valid=global_position_valid,
            )

            estimator_state = (
                decision.global_position_state.value
            )
            authority = decision.authority.value

            if not baseline_ready:
                healthy_baseline = (
                    global_position_valid
                    and watchdog.health
                    == LinkHealth.HEALTHY
                    and decision.authority
                    == ShadowAuthority.GRANTED
                )

                if healthy_baseline:
                    baseline_samples += 1
                else:
                    baseline_samples = 0

                if baseline_samples >= 5:
                    baseline_ready = True
                    benchmark_t0 = now

                    timeline.append(
                        TimelineEvent(
                            t_s=0.0,
                            event="BASELINE_READY",
                            global_position_valid=True,
                            telemetry_health=(
                                watchdog.health.value
                            ),
                            estimator_state=estimator_state,
                            recovery_streak=(
                                decision.recovery_streak
                            ),
                            authority=authority,
                            action=decision.action.value,
                        )
                    )

                    print()
                    print(
                        "BASELINE_READY"
                    )
                    print(
                        "Now run in PX4: failure gps off"
                    )

                previous_valid = (
                    global_position_valid
                )
                previous_estimator = estimator_state
                previous_authority = authority
                continue

            assert benchmark_t0 is not None
            relative_time = now - benchmark_t0

            # GPS-only fault separation check:
            # while global-position capability is failed,
            # the independent telemetry freshness watchdog
            # should remain healthy.
            if (
                observed_fault
                and not observed_recovery
                and watchdog.health
                != LinkHealth.HEALTHY
            ):
                telemetry_false_positive_samples += 1

            if (
                previous_valid is True
                and not global_position_valid
                and not observed_fault
            ):
                observed_fault = True
                first_invalid_time = now

                timeline.append(
                    TimelineEvent(
                        t_s=relative_time,
                        event="FAULT_OBSERVED",
                        global_position_valid=False,
                        telemetry_health=(
                            watchdog.health.value
                        ),
                        estimator_state=estimator_state,
                        recovery_streak=(
                            decision.recovery_streak
                        ),
                        authority=authority,
                        action=decision.action.value,
                    )
                )

                print()
                print("FAULT_OBSERVED")

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

            if (
                observed_fault
                and not observed_recovery
                and previous_valid is False
                and global_position_valid
            ):
                observed_recovery = True
                first_recovery_valid_time = now

                timeline.append(
                    TimelineEvent(
                        t_s=relative_time,
                        event="RECOVERY_STARTED",
                        global_position_valid=True,
                        telemetry_health=(
                            watchdog.health.value
                        ),
                        estimator_state=estimator_state,
                        recovery_streak=(
                            decision.recovery_streak
                        ),
                        authority=authority,
                        action=decision.action.value,
                    )
                )

                print()
                print("RECOVERY_STARTED")

            if observed_recovery:
                recovery_samples += 1

                if (
                    global_position_valid
                    and estimator_state != "healthy"
                    and decision.authority
                    == ShadowAuthority.GRANTED
                ):
                    premature_authority_samples += 1

                if (
                    estimator_state != previous_estimator
                    or authority != previous_authority
                ):
                    timeline.append(
                        TimelineEvent(
                            t_s=relative_time,
                            event="STATE_TRANSITION",
                            global_position_valid=(
                                global_position_valid
                            ),
                            telemetry_health=(
                                watchdog.health.value
                            ),
                            estimator_state=estimator_state,
                            recovery_streak=(
                                decision.recovery_streak
                            ),
                            authority=authority,
                            action=decision.action.value,
                        )
                    )

                if (
                    decision.authority
                    == ShadowAuthority.GRANTED
                    and estimator_state == "healthy"
                ):
                    first_restored_time = now

                    timeline.append(
                        TimelineEvent(
                            t_s=relative_time,
                            event="AUTHORITY_RESTORED",
                            global_position_valid=True,
                            telemetry_health=(
                                watchdog.health.value
                            ),
                            estimator_state=estimator_state,
                            recovery_streak=(
                                decision.recovery_streak
                            ),
                            authority=authority,
                            action=decision.action.value,
                        )
                    )

                    break

            previous_valid = global_position_valid
            previous_estimator = estimator_state
            previous_authority = authority

    finally:
        for task in tasks:
            task.cancel()

        await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

    revoke_latency_ms = None
    restore_latency_ms = None

    if (
        first_invalid_time is not None
        and first_revoked_time is not None
    ):
        revoke_latency_ms = round(
            (
                first_revoked_time
                - first_invalid_time
            )
            * 1000.0,
            3,
        )

    if (
        first_recovery_valid_time is not None
        and first_restored_time is not None
    ):
        restore_latency_ms = round(
            (
                first_restored_time
                - first_recovery_valid_time
            )
            * 1000.0,
            3,
        )

    gates = [
        SafetyGate(
            "PX4-GPS-01",
            "GPS/global-position failure is observed",
            observed_fault,
            str(observed_fault),
        ),
        SafetyGate(
            "PX4-GPS-02",
            "Authority is revoked on the first observed invalid sample",
            (
                revoke_latency_ms is not None
                and revoke_latency_ms <= 1.0
            ),
            f"{revoke_latency_ms} ms",
        ),
        SafetyGate(
            "PX4-GPS-03",
            "No unsafe continuation while position is invalid",
            unsafe_continuation_samples == 0,
            str(unsafe_continuation_samples),
        ),
        SafetyGate(
            "PX4-GPS-04",
            "No authority grant before recovery hysteresis completes",
            premature_authority_samples == 0,
            str(premature_authority_samples),
        ),
        SafetyGate(
            "PX4-GPS-05",
            "No telemetry false-positive during GPS-only fault",
            telemetry_false_positive_samples == 0,
            str(telemetry_false_positive_samples),
        ),
        SafetyGate(
            "PX4-GPS-06",
            "Estimator recovery is observed and authority returns",
            (
                observed_recovery
                and first_restored_time is not None
            ),
            str(observed_recovery),
        ),
        SafetyGate(
            "PX4-GPS-07",
            "Control execution remains disabled",
            True,
            "False",
        ),
    ]

    overall_pass = all(
        gate.passed for gate in gates
    )

    summary = BenchmarkSummary(
        baseline_samples=baseline_samples,
        fault_samples=fault_samples,
        recovery_samples=recovery_samples,
        observed_fault=observed_fault,
        observed_recovery=observed_recovery,
        authority_revocation_latency_ms=(
            revoke_latency_ms
        ),
        authority_restore_latency_ms=(
            restore_latency_ms
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
        overall_pass=overall_pass,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output_dir / "gps-failure-summary.json"
    ).write_text(
        json.dumps(
            asdict(summary),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (
        output_dir / "gps-failure-timeline.json"
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
        output_dir / "gps-failure-gates.json"
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

    write_markdown(
        output_dir / "gps-failure-report.md",
        summary,
        gates,
        timeline,
    )

    print()
    print("=" * 72)
    print("PX4 SAFETY ASSESSMENT")
    print("=" * 72)

    for gate in gates:
        result = (
            "PASS"
            if gate.passed
            else "FAIL"
        )

        print(
            f"{gate.gate_id:12} "
            f"{result:4}  "
            f"{gate.description}"
        )

    print()
    print(
        "OVERALL:",
        "PASS" if overall_pass else "FAIL",
    )

    print()
    print(
        f"authority revoke latency: "
        f"{revoke_latency_ms} ms"
    )
    print(
        f"authority restore latency: "
        f"{restore_latency_ms} ms"
    )
    print(
        "unsafe continuation samples:",
        unsafe_continuation_samples,
    )
    print(
        "premature authority samples:",
        premature_authority_samples,
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
        default="runs/px4-fault-benchmark-v2",
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
