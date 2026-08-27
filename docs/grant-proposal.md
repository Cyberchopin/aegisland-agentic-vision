# AWS Compute Grant Proposal — AegisLand

## Team name

AegisLand — Shiyue Wang (solo builder)

## Problem and real-world impact

Emergency drone landing is a perception-and-control problem, not merely a route-planning problem. When battery reserve collapses, GPS disappears, illumination changes, or people move into a candidate landing region, a system must turn uncertain visual evidence into a constrained action quickly and leave an auditable explanation. AegisLand is a safety-first agentic-vision prototype that evaluates potential landing regions, reacts to moving hazards, requests human approval for ambiguous high-risk cases, and preserves evidence for replay. Target beneficiaries include emergency-response, inspection, and research-drone operators working in degraded navigation conditions.

## OpenCV 5 image/video analysis

The core workload uses OpenCV 5 for Gaussian filtering, Canny structure, morphological operations, Laplacian texture risk, dense Farneback optical flow, contour-based moving-region extraction, candidate-zone clearance, and annotated video generation. When exposure causes confidence to fall, the agent changes its plan and invokes a second OpenCV tool pass using CLAHE enhancement. Motion or obstacle evidence can reject a landing region, cancel a planned landing, trigger evasive hold, or route the action to human approval. This is a substantive multi-step perception–decision–action loop rather than a chatbot that describes a fixed result.

## AWS architecture

The local OpenCV loop remains operational without network access. It sends IAM SigV4-signed evidence events to Amazon API Gateway. An Arm64 AWS Lambda function validates and indexes each event in Amazon DynamoDB while storing the complete trace in a private, encrypted Amazon S3 bucket. AWS X-Ray provides function observability. S3 lifecycle rules and DynamoDB TTL bound retention to 30 days. Phase 3 will evaluate the same committed vision operations on the OpenCV COOL AMI and AWS Graviton against the baseline OpenCV 5 build, reporting latency, throughput, and cost without claiming gains before measurement.

## Architecture description

Video/telemetry → OpenCV 5 multi-cue perception → confidence gate → optional CLAHE re-scan → deterministic safety policy → human approval or simulated command → IAM-signed API Gateway → Arm64 Lambda → DynamoDB index + S3 evidence.

## Target users

Emergency-response teams, infrastructure inspectors, robotics researchers, and student drone teams that need a transparent contingency-landing layer. The competition prototype is simulation-only and is not represented as flight-certified.

## Evaluation and judge demonstration

The initial deterministic suite injects nominal flight, battery decline, GPS loss, low light, static obstacles, and a moving intrusion into the current best landing region. We will measure unsafe-landing rate, hazard reaction frames, task success, appropriate human-control routing, p50/p95 vision latency, and trace completeness. The five-minute judge demo will show identical telemetry producing different actions after OpenCV evidence changes, a low-confidence active re-scan, a human approval stop, and an AWS-backed trace replay. Final evaluation will include failure clips and ablations, with synthetic and licensed real-video results reported separately.

## Focus path

Primary: Agentic Vision Award. Planned secondary path: Best Use of COOL, contingent on verified COOL execution and reproducible baseline comparison.

## Team bio

Shiyue Wang is an incoming UCLA Mathematics of Computation student building toward computer vision, robotics, and reliable AI systems. Her recent work includes an OpenCV-based Stanford DroneHacks preparation stack, AWS serverless projects such as CreditBridge AI and Continuum, and multiple hackathon applications spanning human-centered AI and data visualization. She brings hands-on Python, TypeScript, OpenCV, AWS SAM, DynamoDB, S3, and agent-workflow experience, plus a strong motivation to turn drone-safety study into a rigorously evaluated open prototype.

## Requested compute use

The $150 grant will support short-lived Arm64/Graviton experiments, COOL baseline comparisons, encrypted trace storage, and reproducible demonstration infrastructure. Resources will be budget-alerted, tagged, and stopped or deleted after experiments; raw evidence has a 30-day retention limit.

