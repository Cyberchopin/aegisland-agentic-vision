from __future__ import annotations

import json
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "AegisLand_AWS_Compute_Grant_Proposal.pdf"
BENCHMARK = ROOT / "runs" / "evaluation-final" / "benchmark.json"

NAVY = colors.HexColor("#07131F")
PANEL = colors.HexColor("#102235")
TEAL = colors.HexColor("#43D7A3")
CYAN = colors.HexColor("#66C7FF")
AMBER = colors.HexColor("#FFC469")
INK = colors.HexColor("#172331")
MUTED = colors.HexColor("#5E6E7F")
PALE = colors.HexColor("#EEF4F8")
LINE = colors.HexColor("#CAD8E3")


def architecture_drawing() -> Drawing:
    drawing = Drawing(500, 155)
    nodes = [
        (8, 90, 90, 45, "Video +\ntelemetry", CYAN),
        (110, 90, 92, 45, "OpenCV 5\nperception", TEAL),
        (214, 90, 92, 45, "Confidence\ngate", AMBER),
        (318, 90, 82, 45, "Safety\npolicy", TEAL),
        (412, 90, 80, 45, "Human or\naction", CYAN),
        (214, 15, 92, 42, "CLAHE\nre-scan", AMBER),
        (364, 15, 128, 42, "AWS trace evidence\nAPI + Lambda + DDB + S3", CYAN),
    ]
    for x, y, w, h, label, accent in nodes:
        drawing.add(Rect(x, y, w, h, rx=7, ry=7, fillColor=PALE, strokeColor=accent, strokeWidth=1.5))
        lines = label.split("\n")
        for index, text in enumerate(lines):
            drawing.add(
                String(
                    x + w / 2,
                    y + h / 2 + 5 - index * 12,
                    text,
                    textAnchor="middle",
                    fontName="Helvetica-Bold" if index == 0 else "Helvetica",
                    fontSize=8.5,
                    fillColor=INK,
                )
            )

    def arrow(x1: float, y1: float, x2: float, y2: float, color=LINE) -> None:
        drawing.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=1.5))
        drawing.add(Polygon([x2, y2, x2 - 5, y2 + 3, x2 - 5, y2 - 3], fillColor=color, strokeColor=color))

    arrow(98, 112, 110, 112)
    arrow(202, 112, 214, 112)
    arrow(306, 112, 318, 112)
    arrow(400, 112, 412, 112)
    drawing.add(Line(260, 90, 260, 57, strokeColor=AMBER, strokeWidth=1.5))
    drawing.add(Polygon([260, 57, 257, 62, 263, 62], fillColor=AMBER, strokeColor=AMBER))
    drawing.add(Line(214, 36, 170, 36, strokeColor=AMBER, strokeWidth=1.5))
    drawing.add(Line(170, 36, 170, 90, strokeColor=AMBER, strokeWidth=1.5))
    drawing.add(Polygon([170, 90, 167, 85, 173, 85], fillColor=AMBER, strokeColor=AMBER))
    arrow(452, 90, 452, 57, CYAN)
    return drawing


def header_footer(canvas, doc) -> None:
    canvas.saveState()
    width, height = letter
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 0.30 * inch, width, 0.30 * inch, stroke=0, fill=1)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.white)
    canvas.drawString(0.60 * inch, height - 0.20 * inch, "AEGISLAND  /  OPENCV AI COMPETITION 2026")
    canvas.setFillColor(MUTED)
    canvas.drawRightString(width - 0.60 * inch, 0.38 * inch, f"Page {doc.page}")
    canvas.setStrokeColor(LINE)
    canvas.line(0.60 * inch, 0.52 * inch, width - 0.60 * inch, 0.52 * inch)
    canvas.restoreState()


def build() -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=28,
        leading=32,
        textColor=colors.white,
        alignment=TA_LEFT,
        spaceAfter=8,
    )
    subtitle = ParagraphStyle(
        "Subtitle",
        parent=styles["BodyText"],
        fontSize=12,
        leading=17,
        textColor=colors.HexColor("#D6E5F1"),
    )
    h1 = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        textColor=INK,
        spaceBefore=10,
        spaceAfter=7,
    )
    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#087C68"),
        spaceBefore=7,
        spaceAfter=4,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.35,
        leading=13.2,
        textColor=INK,
        spaceAfter=6,
    )
    small = ParagraphStyle("Small", parent=body, fontSize=7.9, leading=10.5, textColor=MUTED)
    callout = ParagraphStyle(
        "Callout",
        parent=body,
        fontName="Helvetica-Bold",
        fontSize=10.2,
        leading=14.5,
        textColor=INK,
        alignment=TA_CENTER,
    )
    bullet = ParagraphStyle("Bullet", parent=body, leftIndent=13, firstLineIndent=-8, bulletIndent=3)

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        rightMargin=0.62 * inch,
        leftMargin=0.62 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.68 * inch,
        title="AegisLand AWS Compute Grant Proposal",
        author="Shiyue Wang",
        subject="OpenCV AI Competition 2026",
    )
    story = []

    cover = Table(
        [[
            [
                Paragraph("AWS COMPUTE GRANT PROPOSAL", ParagraphStyle("eyebrow", parent=small, textColor=TEAL, fontName="Helvetica-Bold", letterSpacing=1.5)),
                Spacer(1, 10),
                Paragraph("AegisLand", title),
                Paragraph("Agentic Vision for Emergency Drone Landing", ParagraphStyle("coverSub", parent=subtitle, fontName="Helvetica-Bold", fontSize=15, leading=19)),
                Spacer(1, 11),
                Paragraph("A safety-first perception-decision-action system where OpenCV evidence changes the next tool call, landing plan, or human-approval request.", subtitle),
                Spacer(1, 15),
                Paragraph("Solo builder: Shiyue Wang  |  UCLA Mathematics of Computation  |  United States", ParagraphStyle("meta", parent=small, textColor=colors.white)),
            ]
        ]],
        colWidths=[7.25 * inch],
    )
    cover.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("BOX", (0, 0), (-1, -1), 1, TEAL),
        ("LEFTPADDING", (0, 0), (-1, -1), 24),
        ("RIGHTPADDING", (0, 0), (-1, -1), 24),
        ("TOPPADDING", (0, 0), (-1, -1), 24),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 24),
    ]))
    story.extend([cover, Spacer(1, 13)])

    cards = Table(
        [[
            Paragraph("PRIMARY PATH<br/><b>Agentic Vision</b>", callout),
            Paragraph("CORE<br/><b>OpenCV 5</b>", callout),
            Paragraph("CLOUD<br/><b>AWS serverless</b>", callout),
        ]],
        colWidths=[2.35 * inch] * 3,
    )
    cards.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE),
        ("BOX", (0, 0), (-1, -1), 0.75, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.75, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.extend([cards, Spacer(1, 8)])

    story.append(Paragraph("Problem and real-world impact", h1))
    story.append(Paragraph(
        "Emergency drone landing is not only route planning. When battery reserve collapses, GPS disappears, illumination changes, or people move into a candidate landing region, a system must convert uncertain visual evidence into a constrained action quickly and preserve an auditable explanation. AegisLand evaluates potential landing regions, reacts to moving hazards, requests human approval for ambiguous high-risk cases, and stores every perception-to-action trace for replay.",
        body,
    ))
    story.append(Paragraph("Target users and beneficiaries", h2))
    story.append(Paragraph(
        "Emergency-response teams, infrastructure inspectors, robotics researchers, and student drone teams operating in degraded navigation conditions. The competition build is a research prototype and is not represented as flight-certified.",
        body,
    ))
    story.append(Paragraph("Why this project is differentiated", h2))
    for text in [
        "Visual evidence changes the action: motion can reject a zone, cancel a landing, or trigger evasive hold.",
        "Low confidence changes the tool plan: the agent invokes a second OpenCV pass with CLAHE exposure recovery.",
        "Ambiguous high-risk cases stop at a human gate; a language model cannot bypass the deterministic policy.",
        "The system exposes failures and latency instead of presenting a polished but unauditable demo.",
    ]:
        story.append(Paragraph(text, bullet, bulletText="•"))

    story.append(PageBreak())
    story.append(Paragraph("Technical solution", h1))
    story.append(Paragraph("Substantive OpenCV 5 workload", h2))
    story.append(Paragraph(
        "The core pipeline uses Gaussian filtering, Canny structure, morphological closing, Laplacian texture risk, dense Farneback optical flow, contour-based moving-region extraction, median-relative appearance occupancy, candidate-zone clearance, and annotated video generation. The outputs are measurable safety cues, not semantic certainty claims.",
        body,
    ))
    story.append(architecture_drawing())
    story.append(Paragraph("Agentic perception-decision-action loop", h2))
    loop_table = Table(
        [
            ["Evidence state", "Next plan / action"],
            ["Low vision confidence", "Enhance frame and invoke OpenCV again"],
            ["Motion occupies the leading zone", "Reject zone and re-rank alternatives"],
            ["Emergency battery + verified zone", "Simulate landing at the evidence-linked target"],
            ["No verified zone + remaining reserve", "Hold or request operator approval"],
            ["Collision risk above envelope", "Evasive hold overrides the mission plan"],
        ],
        colWidths=[2.45 * inch, 4.60 * inch],
    )
    loop_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PANEL),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("LEADING", (0, 0), (-1, -1), 11),
        ("GRID", (0, 0), (-1, -1), 0.6, LINE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([loop_table, Spacer(1, 8)])

    story.append(Paragraph("Meaningful AWS architecture", h2))
    story.append(Paragraph(
        "The local safety loop continues if the network fails. It sends IAM SigV4-signed events to Amazon API Gateway. An Arm64 AWS Lambda function validates and indexes each event in DynamoDB and stores the complete trace in a private encrypted S3 bucket. AWS X-Ray provides function observability; S3 lifecycle and DynamoDB TTL bound retention to 30 days. Client-side cloud failure is recorded locally and never interrupts the safety loop.",
        body,
    ))
    story.append(Paragraph("Planned COOL path", h2))
    story.append(Paragraph(
        "After freezing the OpenCV 5 baseline, the same committed clips and operations will be evaluated on the OpenCV COOL AMI and AWS Graviton. The report will include warm-up, repetitions, hardware, latency, throughput, utilization, and cost. No performance gain will be claimed before verified execution and reproducible comparison.",
        body,
    ))

    story.append(PageBreak())
    story.append(Paragraph("Evaluation, demonstration, and responsible operation", h1))
    story.append(Paragraph("Evaluation method", h2))
    story.append(Paragraph(
        "The deterministic suite injects nominal flight, battery decline, GPS loss, low light, static obstacles, and a moving intrusion into the leading landing region. Primary metrics are unsafe-landing rate, hazard reaction frames, scenario task success, appropriate human-control routing, p50/p95 vision latency, and trace completeness. Final evidence will separate synthetic and licensed real-video results and include failure clips plus ablations for optical flow, active perception, and the policy layer.",
        body,
    ))

    if BENCHMARK.exists():
        results = json.loads(BENCHMARK.read_text(encoding="utf-8"))
        rows = [["Current prototype baseline", "Frames", "FPS", "Active re-scan", "Approval frames"]]
        for result in results:
            rows.append([
                result["scenario"].replace("_", " "),
                str(result["frames"]),
                f"{result['frames_per_second']:.2f}",
                str(result["active_perception_frames"]),
                str(result["human_approval_frames"]),
            ])
        benchmark_table = Table(rows, colWidths=[2.25 * inch, 0.70 * inch, 0.70 * inch, 1.25 * inch, 1.35 * inch])
        benchmark_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PANEL),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.6, LINE),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.extend([benchmark_table, Paragraph("Unoptimized local container baseline measured on deterministic synthetic inputs. These FPS values expose the current limitation; they are not real-time or COOL performance claims.", small)])

    story.append(Paragraph("Five-minute judge demonstration", h2))
    for text in [
        "Show identical low-battery telemetry producing different actions after OpenCV evidence changes.",
        "Darken the scene and show low confidence triggering the CLAHE re-scan tool call.",
        "Move an intrusion into the best zone and show re-ranking plus the human approval stop.",
        "Replay the evidence ID, reasons, target zone, simulated command, and signed AWS trace.",
        "Close with measured limitations and the next real-video / COOL experiments.",
    ]:
        story.append(Paragraph(text, bullet, bulletText="•"))

    story.append(Paragraph("Safety and responsible-use limits", h2))
    story.append(Paragraph(
        "Simulation-only command adapter; no autopilot or motor command is sent. No face recognition or identity tracking. Raw bystander video is not uploaded. A landing score is a heuristic, not semantic certainty. A future PX4 or DroneKit adapter must add geofencing, acknowledgements, independent failsafes, and hardware-in-the-loop validation before any physical test.",
        body,
    ))

    story.append(Paragraph("Team bio and grant use", h2))
    story.append(Paragraph(
        "Shiyue Wang is an incoming UCLA Mathematics of Computation student building toward computer vision, robotics, and reliable AI systems. Her recent work includes an OpenCV-based Stanford DroneHacks preparation stack, AWS serverless projects CreditBridge AI and Continuum, and hackathon applications spanning human-centered AI and data visualization. The $150 grant will support short-lived Arm64/Graviton experiments, COOL comparisons, encrypted trace storage, and reproducible demonstration infrastructure. Resources will be budget-alerted, tagged, and stopped after experiments.",
        body,
    ))

    references = Table(
        [[Paragraph("Official competition", small), Paragraph("https://opencv26.devpost.com/", small)],
         [Paragraph("Technical rules", small), Paragraph("https://opencv26.devpost.com/rules", small)],
         [Paragraph("COOL on AWS", small), Paragraph("https://aws.amazon.com/marketplace/pp/prodview-fdvbfiewzuehs", small)]],
        colWidths=[1.35 * inch, 5.70 * inch],
    )
    references.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), PALE),
        ("GRID", (0, 0), (-1, -1), 0.5, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.extend([Spacer(1, 4), KeepTogether([Paragraph("Official references", h2), references])])

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    return OUTPUT


if __name__ == "__main__":
    print(build())

