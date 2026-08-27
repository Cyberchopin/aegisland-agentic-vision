from __future__ import annotations

import html
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from statistics import mean

from .domain import TraceEvent, jsonable


def write_report(events: Iterable[TraceEvent], output: str | Path, scenario: str) -> Path:
    events = list(events)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    actions = Counter(event.decision.action.value for event in events)
    avg_latency = mean(event.evidence.processing_ms for event in events) if events else 0.0
    peak_risk = max((event.decision.risk_score for event in events), default=0.0)
    approval_count = sum(event.decision.requires_human_approval for event in events)
    rows = "".join(
        f"<tr><td>{event.sequence}</td><td>{event.telemetry.battery_percent:.1f}%</td>"
        f"<td>{event.evidence.confidence:.2f}</td><td>{event.decision.risk_score:.2f}</td>"
        f"<td><span class='tag {event.decision.safety_level.value}'>{html.escape(event.decision.action.value)}</span></td>"
        f"<td>{html.escape('; '.join(event.decision.reasons))}</td></tr>"
        for event in events
    )
    data = json.dumps([jsonable(event) for event in events])
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AegisLand Mission Evidence</title>
<style>
:root{{--bg:#071019;--panel:#0d1a26;--line:#203246;--text:#eef6ff;--muted:#91a4b8;--green:#48d99a;--amber:#ffbd5a;--red:#ff5f65;--purple:#bc8cff}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 20% 0,#10273a,var(--bg) 42%);color:var(--text);font:15px Inter,ui-sans-serif,system-ui}}
main{{max-width:1200px;margin:auto;padding:36px 24px 70px}}h1{{font-size:42px;margin:5px 0}}.eyebrow{{color:var(--green);letter-spacing:.2em;text-transform:uppercase;font-weight:800}}.sub{{color:var(--muted);max-width:720px}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:28px 0}}.card{{background:rgba(13,26,38,.9);border:1px solid var(--line);border-radius:16px;padding:18px}}.metric{{font-size:29px;font-weight:800;margin-top:8px}}.label{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.12em}}
canvas{{width:100%;height:240px;background:#09141f;border:1px solid var(--line);border-radius:16px}}
.table-wrap{{overflow:auto;margin-top:22px;border:1px solid var(--line);border-radius:16px}}table{{width:100%;border-collapse:collapse;background:var(--panel)}}th,td{{padding:12px 14px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{color:var(--muted);font-size:12px;text-transform:uppercase}}.tag{{display:inline-block;padding:4px 9px;border-radius:99px;background:#183040;white-space:nowrap}}.critical{{color:var(--red)}}.high{{color:var(--purple)}}.caution{{color:var(--amber)}}.nominal{{color:var(--green)}}
@media(max-width:800px){{.grid{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><main>
<div class="eyebrow">Agentic Vision Evidence Console</div><h1>AegisLand</h1>
<p class="sub">Replayable perception → decision → action trace for scenario <b>{html.escape(scenario)}</b>. The default adapter is simulation-only and sends no flight command.</p>
<section class="grid">
<div class="card"><div class="label">Frames evaluated</div><div class="metric">{len(events)}</div></div>
<div class="card"><div class="label">Peak risk</div><div class="metric">{peak_risk:.2f}</div></div>
<div class="card"><div class="label">Mean vision latency</div><div class="metric">{avg_latency:.1f} ms</div></div>
<div class="card"><div class="label">Human approvals</div><div class="metric">{approval_count}</div></div>
</section>
<div class="card"><div class="label">Action distribution</div><p>{html.escape(json.dumps(actions, sort_keys=True))}</p></div>
<h2>Risk and battery timeline</h2><canvas id="chart" width="1120" height="240"></canvas>
<div class="table-wrap"><table><thead><tr><th>#</th><th>Battery</th><th>Vision</th><th>Risk</th><th>Action</th><th>Why</th></tr></thead><tbody>{rows}</tbody></table></div>
</main><script>
const events={data};const c=document.getElementById('chart'),x=c.getContext('2d'),W=c.width,H=c.height,p=24;
x.strokeStyle='#203246';for(let i=0;i<=4;i++){{let y=p+i*(H-2*p)/4;x.beginPath();x.moveTo(p,y);x.lineTo(W-p,y);x.stroke()}}
function line(values,color,max){{x.strokeStyle=color;x.lineWidth=3;x.beginPath();values.forEach((v,i)=>{{let px=p+i*(W-2*p)/Math.max(1,values.length-1),py=H-p-(v/max)*(H-2*p);i?x.lineTo(px,py):x.moveTo(px,py)}});x.stroke()}}
line(events.map(e=>e.decision.risk_score),'#ff5f65',1);line(events.map(e=>e.telemetry.battery_percent),'#48d99a',100);
</script></body></html>"""
    output.write_text(document, encoding="utf-8")
    return output

