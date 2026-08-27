# Build-to-win roadmap

Final submission safety deadline: **October 26, 2026 at 11:45 PM PDT**. The overview also displays 11:59 PM; the project uses the earlier Devpost deadline.

## Week 0 - August 26 to 30: lock the foundation

- Join Devpost and submit the AWS Compute Grant proposal PDF.
- Create the public GitHub repository and protect `main` with CI.
- Record the three deterministic baseline scenarios.
- Learn: evidence objects, optical flow, safety-policy precedence, and tests.

Exit gate: a fresh clone produces a trace, MP4, HTML report, and eight passing tests.

## Week 1 - August 31 to September 6: ground-motion compensation

- Detect ORB features and estimate camera homography.
- Subtract global camera motion before motion occupancy.
- Add failure cases for blur, low texture, and homography rejection.
- Learn: keypoints, descriptors, RANSAC, homography, inliers.

Exit gate: camera pan no longer looks like a person/object intrusion in the test clips.

## Week 2 - September 7 to 13: temporal reliability

- Add Kalman-filtered tracks and time-to-collision.
- Add score hysteresis so the target zone does not flicker between frames.
- Add seeded scenario variants and reaction-frame metrics.
- Learn: state estimation, covariance, gating, data association.

Exit gate: target switching and hazard reaction are reproducible and quantified.

## Week 3 - September 14 to 20: appropriate autonomy

- Build an operator approval UI with evidence thumbnails and expiry.
- Add policy configuration with signed versions and replay.
- Add command acknowledgement / timeout semantics to the adapter contract.
- Learn: finite-state machines, idempotency, fail-closed design.

Exit gate: every approval-required decision is blocked until a valid, unexpired approval exists.

## Week 4 - September 21 to 27: AWS observability

- Deploy the confirmation-gated SAM stack after checking AWS Budget alerts.
- Add CloudWatch dashboards, error alarms, trace-query endpoint, and tags.
- Rehearse the 30-minute grant check-in if selected.
- Learn: IAM SigV4, Lambda, DynamoDB keys/TTL, S3 lifecycle, X-Ray.

Exit gate: local operation survives cloud failure and the cloud trace can be replayed by evidence ID.

## Week 5 - September 28 to October 4: OpenCV 5 and COOL benchmark

- Freeze clips, operations, resolution, warm-up, and repetitions.
- Profile the x86 OpenCV 5 baseline.
- Run the same workload on verified COOL + AWS Graviton if access is approved.
- Learn: p50/p95, throughput, utilization, cost normalization, honest baselines.

Exit gate: raw results and scripts reproduce every chart; otherwise skip the COOL prize claim.

## Week 6 - October 5 to 11: external validity

- Add licensed real downward-facing drone clips.
- Label landing-region hazards and review failure clips.
- Run ablations: no flow, no active scan, no policy.
- Learn: dataset licenses, annotation protocol, precision/recall, ablation design.

Exit gate: synthetic and real-video results are separated and limitations are written before filming.

## Week 7 - October 12 to 18: judge experience

- Turn the evidence report into a responsive endpoint.
- Draft the technical report and architecture figures.
- Record a rough five-minute demo and cut every non-evidence second.
- Learn: technical storytelling, counterfactual demo, user-control explanation.

Exit gate: a new viewer can explain exactly how one OpenCV result changed one later action.

## Final week - October 19 to 26: freeze and submit

- October 19: feature freeze.
- October 20-22: clean-machine reproduction and AWS deployment check.
- October 23: final failure/limitation review and responsible-use check.
- October 24: final video and report.
- October 25: Devpost draft complete with judge-access tests.
- October 26 by noon PDT: final upload; reserve the rest of the day for verification only.

## Stanford DroneHacks transfer value

This roadmap trains the exact reusable layers needed for DroneHacks: OpenCV regions and contours, temporal tracking, uncertainty handling, safety state machines, human-in-the-loop control, AWS observability, reproducible testing, and a technically honest robotics demo narrative.
