from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .flicker_benchmark import (
    run_benchmark as run_flicker_benchmark,
)
from .fusion_authority_ablation import (
    run_ablation as run_fusion_authority_ablation,
)
from .health_hysteresis_ablation import (
    run_ablation as run_health_hysteresis_ablation,
)
from .sensor_failure_benchmark import (
    run_static_cases,
)

ASSESSMENT_SCOPE = (
    "Deterministic simulation safety assessment. "
    "This is not flight certification."
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            [
                "git",
                "rev-parse",
                "--short",
                "HEAD",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _gate(
    gate_id: str,
    name: str,
    passed: bool,
    evidence: str,
) -> dict[str, Any]:
    return {
        "id": gate_id,
        "name": name,
        "passed": passed,
        "evidence": evidence,
    }


def build_assessment() -> dict[str, Any]:
    flicker = run_flicker_benchmark()
    authority = run_fusion_authority_ablation()
    health = run_health_hysteresis_ablation()
    sensor_cases = run_static_cases()

    capability = authority.capability_aware
    confidence_only = authority.confidence_only

    health_stateful = health.aegisland_hysteresis
    health_stateless = health.no_hysteresis

    sensor_matrix_passed = all(
        case.passed
        for case in sensor_cases
    )

    gates = [
        _gate(
            "AUTH-01",
            "No premature GPS-denied navigation authority",
            capability.premature_authority_frames == 0,
            (
                "capability-aware="
                f"{capability.premature_authority_frames}, "
                "confidence-only="
                f"{confidence_only.premature_authority_frames}"
            ),
        ),
        _gate(
            "AUTH-02",
            "No premature raw mission continuation",
            capability.premature_raw_continue_frames == 0,
            (
                "capability-aware="
                f"{capability.premature_raw_continue_frames}, "
                "confidence-only="
                f"{confidence_only.premature_raw_continue_frames}"
            ),
        ),
        _gate(
            "HEALTH-01",
            "No premature HEALTHY recovery during flicker",
            flicker.premature_healthy_grants == 0,
            (
                "premature_healthy_grants="
                f"{flicker.premature_healthy_grants}"
            ),
        ),
        _gate(
            "CONTROL-01",
            "No final mission continuation during unstable recovery",
            (
                flicker
                .final_continue_frames_during_unstable_recovery
                == 0
            ),
            (
                "unstable_final_continue_frames="
                f"{flicker.final_continue_frames_during_unstable_recovery}"
            ),
        ),
        _gate(
            "MATRIX-01",
            "Sensor failure matrix matches expected safe modes",
            sensor_matrix_passed,
            (
                f"{sum(case.passed for case in sensor_cases)}"
                f"/{len(sensor_cases)} cases passed"
            ),
        ),
    ]

    overall_pass = all(
        gate["passed"]
        for gate in gates
    )

    return {
        "schema_version": "1.0",
        "project": "AegisLand",
        "status": (
            "PASS"
            if overall_pass
            else "FAIL"
        ),
        "scope": ASSESSMENT_SCOPE,
        "metadata": {
            "generated_at_utc": datetime.now(
                UTC
            ).isoformat(),
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "gates": gates,
        "metrics": {
            "capability_authority": {
                "confidence_only": {
                    "premature_authority_frames": (
                        confidence_only
                        .premature_authority_frames
                    ),
                    "premature_raw_continue_frames": (
                        confidence_only
                        .premature_raw_continue_frames
                    ),
                    "raw_action_transitions": (
                        confidence_only
                        .raw_action_transitions
                    ),
                    "raw_recovery_latency_frames": (
                        confidence_only
                        .raw_recovery_latency_frames
                    ),
                    "final_recovery_latency_frames": (
                        confidence_only
                        .final_recovery_latency_frames
                    ),
                },
                "capability_aware": {
                    "premature_authority_frames": (
                        capability
                        .premature_authority_frames
                    ),
                    "premature_raw_continue_frames": (
                        capability
                        .premature_raw_continue_frames
                    ),
                    "raw_action_transitions": (
                        capability
                        .raw_action_transitions
                    ),
                    "raw_recovery_latency_frames": (
                        capability
                        .raw_recovery_latency_frames
                    ),
                    "final_recovery_latency_frames": (
                        capability
                        .final_recovery_latency_frames
                    ),
                },
            },
            "sensor_health_hysteresis": {
                "samples_1": {
                    "premature_healthy_grants": (
                        health_stateless
                        .premature_healthy_grants
                    ),
                    "unstable_healthy_frames": (
                        health_stateless
                        .unstable_healthy_frames
                    ),
                    "stable_health_latency_frames": (
                        health_stateless
                        .stable_health_latency_frames
                    ),
                },
                "samples_3": {
                    "premature_healthy_grants": (
                        health_stateful
                        .premature_healthy_grants
                    ),
                    "unstable_healthy_frames": (
                        health_stateful
                        .unstable_healthy_frames
                    ),
                    "stable_health_latency_frames": (
                        health_stateful
                        .stable_health_latency_frames
                    ),
                },
            },
            "flicker_recovery": {
                "stable_physical_recovery_frame": (
                    flicker
                    .stable_physical_recovery_frame
                ),
                "stable_trust_frame": (
                    flicker.stable_trust_frame
                ),
                "stable_health_frame": (
                    flicker.stable_health_frame
                ),
                "raw_recovery_frame": (
                    flicker.raw_recovery_frame
                ),
                "final_recovery_frame": (
                    flicker.final_recovery_frame
                ),
            },
            "sensor_failure_matrix": [
                {
                    "case": case.case,
                    "expected_mode": case.expected_mode,
                    "actual_mode": case.actual_mode,
                    "action": case.action,
                    "passed": case.passed,
                }
                for case in sensor_cases
            ],
        },
        "limitations": [
            (
                "All results in this assessment are generated "
                "from deterministic simulation or synthetic inputs."
            ),
            (
                "The assessment does not establish real-aircraft "
                "safety, airworthiness, or flight certification."
            ),
            (
                "Hardware timing, aerodynamic dynamics, vibration, "
                "sensor calibration drift, and environmental effects "
                "require separate SITL/HIL/flight validation."
            ),
        ],
    }


def _markdown(
    assessment: dict[str, Any],
) -> str:
    gates = assessment["gates"]
    metrics = assessment["metrics"]

    lines = [
        "# AegisLand Flight Safety Assessment",
        "",
        f"**Verdict:** {assessment['status']}",
        "",
        f"**Scope:** {assessment['scope']}",
        "",
        "## Safety gates",
        "",
        "| Gate | Requirement | Result | Evidence |",
        "|---|---|---|---|",
    ]

    for gate in gates:
        result = (
            "PASS"
            if gate["passed"]
            else "FAIL"
        )

        lines.append(
            "| "
            f"{gate['id']} | "
            f"{gate['name']} | "
            f"{result} | "
            f"{gate['evidence']} |"
        )

    authority = metrics[
        "capability_authority"
    ]

    health = metrics[
        "sensor_health_hysteresis"
    ]

    flicker = metrics[
        "flicker_recovery"
    ]

    lines.extend(
        [
            "",
            "## Controlled ablations",
            "",
            "### Capability-aware authority",
            "",
            "| Metric | Confidence-only | Capability-aware |",
            "|---|---:|---:|",
            (
                "| Premature authority frames | "
                f"{authority['confidence_only']['premature_authority_frames']} | "
                f"{authority['capability_aware']['premature_authority_frames']} |"
            ),
            (
                "| Premature raw continue frames | "
                f"{authority['confidence_only']['premature_raw_continue_frames']} | "
                f"{authority['capability_aware']['premature_raw_continue_frames']} |"
            ),
            (
                "| Raw action transitions | "
                f"{authority['confidence_only']['raw_action_transitions']} | "
                f"{authority['capability_aware']['raw_action_transitions']} |"
            ),
            (
                "| Raw recovery latency | "
                f"{authority['confidence_only']['raw_recovery_latency_frames']} | "
                f"{authority['capability_aware']['raw_recovery_latency_frames']} |"
            ),
            "",
            "### SensorHealth hysteresis",
            "",
            "| Metric | samples=1 | samples=3 |",
            "|---|---:|---:|",
            (
                "| Premature HEALTHY grants | "
                f"{health['samples_1']['premature_healthy_grants']} | "
                f"{health['samples_3']['premature_healthy_grants']} |"
            ),
            (
                "| Unstable HEALTHY frames | "
                f"{health['samples_1']['unstable_healthy_frames']} | "
                f"{health['samples_3']['unstable_healthy_frames']} |"
            ),
            (
                "| Stable health latency | "
                f"{health['samples_1']['stable_health_latency_frames']} | "
                f"{health['samples_3']['stable_health_latency_frames']} |"
            ),
            "",
            "## Recovery chain",
            "",
            (
                "- Physical recovery: frame "
                f"{flicker['stable_physical_recovery_frame']}"
            ),
            (
                "- Trust recovery: frame "
                f"{flicker['stable_trust_frame']}"
            ),
            (
                "- Health recovery: frame "
                f"{flicker['stable_health_frame']}"
            ),
            (
                "- Raw navigation recovery: frame "
                f"{flicker['raw_recovery_frame']}"
            ),
            (
                "- Final control recovery: frame "
                f"{flicker['final_recovery_frame']}"
            ),
            "",
            "## Interpretation",
            "",
            (
                "AegisLand separates measurement return, "
                "sensor health, capability observability, "
                "authority restoration, and control resumption."
            ),
            "",
            (
                "The measured safety improvements trade faster "
                "availability for more conservative recovery."
            ),
            "",
            "## Limitations",
            "",
        ]
    )

    for limitation in assessment["limitations"]:
        lines.append(
            f"- {limitation}"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            (
                "**NOT FLIGHT CERTIFICATION.** "
                "This report is an engineering assessment "
                "of the current prototype."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def write_assessment(
    output_dir: Path,
) -> dict[str, Any]:
    assessment = build_assessment()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output_dir / "assessment.json"
    ).write_text(
        json.dumps(
            assessment,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_dir / "assessment.md"
    ).write_text(
        _markdown(assessment),
        encoding="utf-8",
    )

    return assessment


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate AegisLand deterministic "
            "flight-safety assessment artifacts."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/safety-assessment"
        ),
    )

    args = parser.parse_args()

    assessment = write_assessment(
        args.output
    )

    print()
    print("AEGISLAND FLIGHT SAFETY ASSESSMENT")
    print("=" * 72)

    for gate in assessment["gates"]:
        marker = (
            "PASS"
            if gate["passed"]
            else "FAIL"
        )

        print(
            f"{marker:4} "
            f"{gate['id']:12} "
            f"{gate['name']}"
        )

    print("-" * 72)
    print(
        "OVERALL:",
        assessment["status"],
    )
    print(
        "SCOPE:",
        assessment["scope"],
    )
    print(
        "OUTPUT:",
        args.output,
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
