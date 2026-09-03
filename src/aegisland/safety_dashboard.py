from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def _json(data: Any) -> str:
    return json.dumps(
        data,
        separators=(",", ":"),
    )


def load_evidence(
    evidence_dir: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    assessment_path = (
        evidence_dir
        / "assessment.json"
    )

    timeline_path = (
        evidence_dir
        / "timeline.json"
    )

    missing = [
        path.name
        for path in (
            assessment_path,
            timeline_path,
        )
        if not path.exists()
    ]

    if missing:
        names = ", ".join(missing)

        raise FileNotFoundError(
            "Missing safety evidence: "
            f"{names}. Run "
            "`python -m aegisland.safety_assessment` "
            "first."
        )

    assessment = json.loads(
        assessment_path.read_text(
            encoding="utf-8"
        )
    )

    timeline = json.loads(
        timeline_path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(assessment, dict):
        raise TypeError(
            "assessment.json must contain "
            "a JSON object."
        )

    if not isinstance(timeline, list):
        raise TypeError(
            "timeline.json must contain "
            "a JSON array."
        )

    return assessment, timeline


def render_dashboard(
    *,
    assessment: dict[str, Any],
    timeline: list[dict[str, Any]],
    output: Path,
) -> Path:

    authority = assessment[
        "metrics"
    ][
        "capability_authority"
    ]

    health = assessment[
        "metrics"
    ][
        "sensor_health_hysteresis"
    ]

    recovery = assessment[
        "metrics"
    ][
        "flicker_recovery"
    ]

    gates = assessment["gates"]

    gate_cards = "".join(
        (
            "<div class='gate'>"
            f"<span class='gate-id'>{html.escape(gate['id'])}</span>"
            f"<strong>{html.escape(gate['name'])}</strong>"
            f"<span class='pass'>{'PASS' if gate['passed'] else 'FAIL'}</span>"
            f"<small>{html.escape(gate['evidence'])}</small>"
            "</div>"
        )
        for gate in gates
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta
    name="viewport"
    content="width=device-width,initial-scale=1"
>
<title>AegisLand Safety Console</title>

<style>
:root {{
    --bg:#061019;
    --panel:#0c1a25;
    --panel2:#102432;
    --line:#243846;
    --text:#eef8ff;
    --muted:#8fa6b6;
    --green:#52e0a4;
    --amber:#ffc35c;
    --red:#ff646d;
    --blue:#66b7ff;
    --purple:#b99cff;
}}

* {{
    box-sizing:border-box;
}}

body {{
    margin:0;
    background:
        radial-gradient(
            circle at 15% 0%,
            #173349,
            var(--bg) 38%
        );
    color:var(--text);
    font-family:
        Inter,
        ui-sans-serif,
        system-ui,
        sans-serif;
}}

main {{
    max-width:1320px;
    margin:auto;
    padding:38px 26px 80px;
}}

.eyebrow {{
    color:var(--green);
    font-weight:800;
    letter-spacing:.18em;
    text-transform:uppercase;
    font-size:12px;
}}

h1 {{
    margin:8px 0 5px;
    font-size:46px;
}}

.subtitle {{
    color:var(--muted);
    max-width:860px;
    line-height:1.6;
}}

.grid {{
    display:grid;
    gap:14px;
}}

.metrics {{
    grid-template-columns:
        repeat(5,1fr);
    margin:28px 0;
}}

.card,
.gate {{
    border:1px solid var(--line);
    background:
        rgba(12,26,37,.92);
    border-radius:18px;
    padding:18px;
}}

.metric {{
    font-size:30px;
    font-weight:850;
    margin-top:8px;
}}

.label {{
    color:var(--muted);
    font-size:11px;
    text-transform:uppercase;
    letter-spacing:.13em;
}}

.pass {{
    color:var(--green);
    font-weight:850;
}}

.gates {{
    grid-template-columns:
        repeat(5,1fr);
}}

.gate {{
    display:flex;
    flex-direction:column;
    gap:9px;
}}

.gate small {{
    color:var(--muted);
    line-height:1.4;
}}

.gate-id {{
    font-size:11px;
    color:var(--blue);
    font-weight:800;
}}

.section {{
    margin-top:34px;
}}

.two {{
    grid-template-columns:
        1fr 1fr;
}}

table {{
    width:100%;
    border-collapse:collapse;
}}

th,
td {{
    padding:10px 12px;
    text-align:left;
    border-bottom:
        1px solid var(--line);
}}

th {{
    color:var(--muted);
    text-transform:uppercase;
    letter-spacing:.1em;
    font-size:11px;
}}

.bar-row {{
    display:grid;
    grid-template-columns:
        220px 1fr 60px;
    gap:12px;
    align-items:center;
    margin:13px 0;
}}

.bar-track {{
    height:12px;
    border-radius:99px;
    background:#07121b;
    overflow:hidden;
}}

.bar {{
    height:100%;
    border-radius:99px;
    background:
        linear-gradient(
            90deg,
            var(--blue),
            var(--purple)
        );
}}

.chain {{
    display:grid;
    grid-template-columns:
        repeat(5,1fr);
    gap:10px;
}}

.stage {{
    position:relative;
    padding:16px;
    border-radius:14px;
    background:var(--panel2);
    border:1px solid var(--line);
}}

.stage strong {{
    display:block;
    font-size:25px;
    margin:4px 0;
}}

.stage small {{
    color:var(--muted);
}}

.canvas-wrap {{
    padding:14px;
}}

canvas {{
    width:100%;
    height:320px;
    background:#07131d;
    border-radius:14px;
}}

.scrubber {{
    width:100%;
    margin-top:16px;
}}

.frame-detail {{
    display:grid;
    grid-template-columns:
        repeat(5,1fr);
    gap:10px;
    margin-top:14px;
}}

.detail {{
    padding:11px;
    background:#081722;
    border-radius:12px;
}}

.detail b {{
    display:block;
    margin-top:4px;
}}

.pill {{
    display:inline-block;
    border-radius:99px;
    padding:4px 9px;
    background:#183143;
}}

.warning {{
    margin-top:30px;
    border-left:
        4px solid var(--amber);
    padding:14px 16px;
    background:
        rgba(255,195,92,.07);
}}

@media(max-width:900px) {{
    .metrics,
    .gates,
    .chain,
    .frame-detail {{
        grid-template-columns:
            repeat(2,1fr);
    }}

    .two {{
        grid-template-columns:1fr;
    }}
}}
</style>
</head>

<body>
<main>

<div class="eyebrow">
Flight Safety Assessment Console
</div>

<h1>AegisLand</h1>

<p class="subtitle">
A replayable safety-evidence dashboard for
GPS-denied navigation degradation and recovery.
The dashboard separates perception trust,
sensor health, capability authority,
planner decisions, and final control actions.
</p>

<section class="grid metrics">

<div class="card">
<div class="label">Assessment</div>
<div class="metric pass">
{assessment['status']}
</div>
</div>

<div class="card">
<div class="label">Premature authority</div>
<div class="metric">
{authority['capability_aware']['premature_authority_frames']}
</div>
</div>

<div class="card">
<div class="label">Planner transitions</div>
<div class="metric">
{authority['capability_aware']['raw_action_transitions']}
</div>
</div>

<div class="card">
<div class="label">Stable health</div>
<div class="metric">
F{recovery['stable_health_frame']}
</div>
</div>

<div class="card">
<div class="label">Final resume</div>
<div class="metric">
F{recovery['final_recovery_frame']}
</div>
</div>

</section>

<section class="section">

<h2>Safety gates</h2>

<div class="grid gates">
{gate_cards}
</div>

</section>

<section class="section grid two">

<div class="card">

<h2>
Capability-aware authority ablation
</h2>

<p class="subtitle">
Same sensor stream and same SensorHealth
recovery point. Only authority semantics change.
</p>

<div class="bar-row">
<span>Premature authority · confidence</span>
<div class="bar-track">
<div
    class="bar"
    style="width:100%"
></div>
</div>
<b>
{authority['confidence_only']['premature_authority_frames']}
</b>
</div>

<div class="bar-row">
<span>Premature authority · capability</span>
<div class="bar-track">
<div
    class="bar"
    style="width:0%"
></div>
</div>
<b>
{authority['capability_aware']['premature_authority_frames']}
</b>
</div>

<div class="bar-row">
<span>Planner transitions · confidence</span>
<div class="bar-track">
<div
    class="bar"
    style="width:100%"
></div>
</div>
<b>
{authority['confidence_only']['raw_action_transitions']}
</b>
</div>

<div class="bar-row">
<span>Planner transitions · capability</span>
<div class="bar-track">
<div
    class="bar"
    style="width:20%"
></div>
</div>
<b>
{authority['capability_aware']['raw_action_transitions']}
</b>
</div>

</div>

<div class="card">

<h2>SensorHealth hysteresis</h2>

<p class="subtitle">
Recovery samples = 3 trades availability
for temporal stability.
</p>

<table>

<thead>
<tr>
<th>Metric</th>
<th>samples=1</th>
<th>samples=3</th>
</tr>
</thead>

<tbody>

<tr>
<td>Premature HEALTHY</td>
<td>
{health['samples_1']['premature_healthy_grants']}
</td>
<td class="pass">
{health['samples_3']['premature_healthy_grants']}
</td>
</tr>

<tr>
<td>Unstable HEALTHY frames</td>
<td>
{health['samples_1']['unstable_healthy_frames']}
</td>
<td class="pass">
{health['samples_3']['unstable_healthy_frames']}
</td>
</tr>

<tr>
<td>Stable health latency</td>
<td>
{health['samples_1']['stable_health_latency_frames']}
</td>
<td>
{health['samples_3']['stable_health_latency_frames']}
</td>
</tr>

</tbody>
</table>

</div>

</section>

<section class="section">

<h2>Recovery chain</h2>

<div class="chain">

<div class="stage">
<small>Physical</small>
<strong>
{recovery['stable_physical_recovery_frame']}
</strong>
<small>camera remains physically usable</small>
</div>

<div class="stage">
<small>Trust</small>
<strong>
{recovery['stable_trust_frame']}
</strong>
<small>semantic localization trust restored</small>
</div>

<div class="stage">
<small>Health</small>
<strong>
{recovery['stable_health_frame']}
</strong>
<small>sustained recovery confirmed</small>
</div>

<div class="stage">
<small>Authority</small>
<strong>
{recovery['raw_recovery_frame']}
</strong>
<small>raw planner may resume</small>
</div>

<div class="stage">
<small>Control</small>
<strong>
{recovery['final_recovery_frame']}
</strong>
<small>stabilized control released</small>
</div>

</div>

</section>

<section class="section card canvas-wrap">

<h2>Interactive fault timeline</h2>

<p class="subtitle">
Red regions mark injected camera faults.
Blue = visual effective confidence.
Green = fused navigation confidence.
</p>

<canvas
    id="timeline"
    width="1200"
    height="320"
></canvas>

<input
    class="scrubber"
    id="scrubber"
    type="range"
    min="0"
    max="{max(0, len(timeline) - 1)}"
    value="0"
/>

<div
    id="frameDetail"
    class="frame-detail"
></div>

</section>

<div class="warning">

<strong>Engineering assessment only.</strong>

This dashboard reports deterministic synthetic
and simulation evidence. It is not flight
certification and does not establish
real-aircraft airworthiness.

</div>

</main>

<script>

const rows = {_json(timeline)};

const canvas =
    document.getElementById("timeline");

const ctx =
    canvas.getContext("2d");

const scrubber =
    document.getElementById("scrubber");

const frameDetail =
    document.getElementById("frameDetail");

const initialIndexRaw = rows.findIndex(
    row => row.camera_fault_active
);

const initialIndex =
    initialIndexRaw >= 0
        ? initialIndexRaw
        : 0;

scrubber.value = String(initialIndex);


function draw() {{
    const W = canvas.width;
    const H = canvas.height;
    const p = 38;

    ctx.clearRect(0,0,W,H);

    ctx.strokeStyle = "#203441";
    ctx.lineWidth = 1;

    for (
        let i = 0;
        i <= 4;
        i++
    ) {{
        const y =
            p +
            i *
            (H - 2*p) /
            4;

        ctx.beginPath();
        ctx.moveTo(p,y);
        ctx.lineTo(W-p,y);
        ctx.stroke();
    }}

    rows.forEach(
        (row,index) => {{
            if (!row.camera_fault_active) {{
                return;
            }}

            const x =
                p +
                index *
                (W - 2*p) /
                Math.max(
                    1,
                    rows.length - 1
                );

            const w =
                (W - 2*p) /
                Math.max(
                    1,
                    rows.length
                );

            ctx.fillStyle =
                "rgba(255,100,109,.18)";

            ctx.fillRect(
                x,
                p,
                Math.max(2,w),
                H - 2*p
            );
        }}
    );

    function line(
        values,
        color
    ) {{
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.beginPath();

        values.forEach(
            (value,index) => {{
                const x =
                    p +
                    index *
                    (W - 2*p) /
                    Math.max(
                        1,
                        values.length - 1
                    );

                const y =
                    H -
                    p -
                    value *
                    (H - 2*p);

                if (index === 0) {{
                    ctx.moveTo(x,y);
                }} else {{
                    ctx.lineTo(x,y);
                }}
            }}
        );

        ctx.stroke();
    }}

    line(
        rows.map(
            row =>
                row.visual_effective_confidence
        ),
        "#66b7ff"
    );

    line(
        rows.map(
            row =>
                row.fused_navigation_confidence
        ),
        "#52e0a4"
    );

    const selectedIndex =
        Number(scrubber.value);

    const selectedX =
        p +
        selectedIndex *
        (W - 2*p) /
        Math.max(
            1,
            rows.length - 1
        );

    ctx.strokeStyle =
        "rgba(255,255,255,.92)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(selectedX, p);
    ctx.lineTo(selectedX, H - p);
    ctx.stroke();
}}


function detail(index) {{
    const row = rows[index];

    if (!row) {{
        return;
    }}

    frameDetail.innerHTML = `
        <div class="detail">
            <span class="label">Frame</span>
            <b>${{row.frame}}</b>
        </div>

        <div class="detail">
            <span class="label">Perception</span>
            <b>${{row.perception_failure_type}}</b>
        </div>

        <div class="detail">
            <span class="label">Visual health</span>
            <b>${{row.visual_health_state}}</b>
        </div>

        <div class="detail">
            <span class="label">Navigation</span>
            <b>${{row.navigation_mode}}</b>
        </div>

        <div class="detail">
            <span class="label">Control</span>
            <b>${{row.action}}</b>
        </div>
    `;
}}


scrubber.addEventListener(
    "input",
    () => {{
        detail(
            Number(scrubber.value)
        );
    }}
);


draw();
detail(initialIndex);

</script>

</body>
</html>
"""

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        document,
        encoding="utf-8",
    )

    return output


def write_dashboard(
    output: Path,
    *,
    evidence_dir: Path | None = None,
) -> Path:
    if evidence_dir is None:
        evidence_dir = output.parent

    assessment, timeline = load_evidence(
        evidence_dir
    )

    return render_dashboard(
        assessment=assessment,
        timeline=timeline,
        output=output,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the AegisLand "
            "interactive safety dashboard."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/safety-assessment/dashboard.html"
        ),
    )

    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=Path(
            "runs/safety-assessment"
        ),
    )

    args = parser.parse_args()

    output = write_dashboard(
        args.output,
        evidence_dir=args.evidence_dir,
    )

    print()
    print("AEGISLAND SAFETY DASHBOARD")
    print("=" * 72)
    print(f"Written: {output}")
    print()
    print(
        "Serve with:"
    )
    print(
        "python -m aegisland.cli serve "
        "--directory runs/safety-assessment"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
