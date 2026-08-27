from __future__ import annotations

import argparse
import json
import time
import webbrowser
from collections import Counter
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .agent import AegisLandAgent
from .domain import jsonable
from .perception import OpenCVLandingPerception, cv2, require_opencv
from .planner import SafetyPlanner
from .report import write_report
from .simulator import SCENARIOS, generate
from .trace import CloudTraceStore, JsonlTraceStore, MemoryTraceStore


def run_demo(scenario_name: str, output: Path, display: bool = False) -> dict[str, object]:
    require_opencv()
    scenario = SCENARIOS[scenario_name]
    output.mkdir(parents=True, exist_ok=True)
    memory = MemoryTraceStore()
    local = JsonlTraceStore(output / "trace.jsonl", truncate=True)

    class Fanout:
        def write(self, event):
            memory.write(event)
            CloudTraceStore(local).write(event)

    agent = AegisLandAgent(OpenCVLandingPerception(), SafetyPlanner(), Fanout())
    writer = cv2.VideoWriter(
        str(output / "annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        15,
        (960, 540),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the MP4 writer")

    started = time.perf_counter()
    try:
        for index, (frame, telemetry) in enumerate(generate(scenario)):
            _event, annotated = agent.step(frame, telemetry, index)
            writer.write(annotated)
            if display:
                cv2.imshow("AegisLand", annotated)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        writer.release()
        if display:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    write_report(memory.events, output / "report.html", scenario_name)
    actions = Counter(event.decision.action.value for event in memory.events)
    summary = {
        "scenario": scenario_name,
        "description": scenario.description,
        "frames": len(memory.events),
        "wall_seconds": round(elapsed, 3),
        "frames_per_second": round(len(memory.events) / max(elapsed, 0.001), 2),
        "actions": dict(actions),
        "active_perception_frames": sum(event.evidence.active_perception_used for event in memory.events),
        "human_approval_frames": sum(event.decision.requires_human_approval for event in memory.events),
        "final_decision": jsonable(memory.events[-1].decision) if memory.events else None,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def serve(directory: Path, port: int) -> None:
    directory = directory.resolve()
    handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(*args, directory=str(directory), **kwargs)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/report.html"
    print(f"Serving {directory} at {url}. Press Ctrl+C to stop.")
    webbrowser.open(url)
    server.serve_forever()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aegisland", description="AegisLand agentic vision")
    subcommands = parser.add_subparsers(dest="command", required=True)
    demo = subcommands.add_parser("demo", help="Run a deterministic synthetic safety scenario")
    demo.add_argument("--scenario", choices=sorted(SCENARIOS), default="low_battery_intrusion")
    demo.add_argument("--output", type=Path, default=Path("runs/latest"))
    demo.add_argument("--display", action="store_true")
    web = subcommands.add_parser("serve", help="Open the generated evidence report")
    web.add_argument("--directory", type=Path, default=Path("runs/latest"))
    web.add_argument("--port", type=int, default=8080)
    evaluate = subcommands.add_parser("evaluate", help="Run all deterministic benchmark scenarios")
    evaluate.add_argument("--output", type=Path, default=Path("runs/evaluation"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "demo":
        print(json.dumps(run_demo(args.scenario, args.output, args.display), indent=2))
    elif args.command == "serve":
        serve(args.directory, args.port)
    elif args.command == "evaluate":
        results = [run_demo(name, args.output / name) for name in SCENARIOS]
        (args.output / "benchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
