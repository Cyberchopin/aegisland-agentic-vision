# AegisLand Engineering Study Notes

> Learning log for building a safety-first agentic vision system for simulated emergency drone landing.

---

## 1. System Mental Model

A robotics/autonomy system can be viewed as:

```text
Sensors / Simulator
→ Perception
→ State Estimation
→ Target Selection
→ Planning
→ Safety Policy
→ Command Execution
→ Verification
→ Trace

AegisLand is simulation-only. It does not send commands to real drone hardware.

2. Python Engineering Skills
Dataclasses

Used for structured immutable system state:

Telemetry
VisionEvidence
ZoneCandidate
Decision
TraceEvent
RecoveryPlan

Important syntax:

@dataclass(frozen=True, slots=True)
class Decision:
    ...

frozen=True helps prevent accidental mutation.

Type hints

Examples:

str | None
tuple[float, float]
list[ZoneCandidate]
dataclasses.replace

Useful when deriving a modified version of immutable state:

updated = dataclasses.replace(
    evidence,
    temporal_risk=0.8,
)
3. Testing / TDD

Core workflow:

python -m pytest -q
git diff --check
python -m py_compile path/to/file.py

Engineering principle:

Add a regression test for safety behavior before or alongside the implementation.

Tests currently cover areas including:

compound emergency recovery
target locking
action stabilization
command lifecycle
injected failures
fail-closed behavior
human approval
camera-motion compensation
Kalman tracking
TTC
temporal risk
4. OpenCV Perception

Current pipeline uses:

grayscale conversion
edge / texture cues
morphology
contour extraction
optical flow
candidate landing zones

Important lesson:

A visual cue is evidence, not ground truth.

5. Optical Flow

Optical flow estimates apparent pixel motion between frames.

Problem:

Camera movement can make the entire image appear to move.

Therefore:

raw optical flow
≠ independent object motion

This motivated camera-motion compensation.

6. ORB

ORB provides:

keypoints
+
binary descriptors

A keypoint is a visually distinctive image location.

A descriptor is a compact representation of the local image patch.

Because ORB descriptors are binary, matching uses:

Hamming distance
7. Feature Matching

The system progressed from basic brute-force matching to:

KNN matching
→ Lowe ratio test
→ RANSAC

Lowe ratio test rejects ambiguous matches when the best match is not clearly better than the second-best match.

Engineering lesson:

More matches do not necessarily mean better matches.

8. Homography

A homography is a 3×3 projective transformation:

previous image
→ current camera viewpoint

Used here to estimate global camera motion.

Pipeline:

ORB
→ feature matching
→ RANSAC
→ homography
→ warp previous frame
→ residual optical flow

Quality gates include:

minimum match count
minimum inlier count
minimum inlier ratio
finite matrix values
corner-displacement sanity check

Important principle:

A model being successfully estimated does not mean the model is trustworthy.

9. Target Lock / Hysteresis

Important distinction:

best candidate this frame
≠ committed mission target

A target manager keeps the selected landing zone stable until it remains unsafe for multiple frames.

This prevents target flapping.

Safety principle:

escalation fast
de-escalation slow
10. Kalman Filter

Current state:

[x, y, vx, vy]

Measurements:

[x, y]

The Kalman filter combines:

prediction
+
measurement

to estimate state under noisy observations.

Important matrices:

F: state transition
H: measurement model
P: covariance / uncertainty
K: Kalman gain

Kalman filtering is state estimation, not object detection.

11. Detection vs Tracking vs Prediction
Detection

Where is something in this frame?

Tracking

Is this the same object across frames?

Prediction

Where is it likely to go next?

AegisLand now begins moving from frame-level perception toward temporal reasoning.

12. TTC — Time To Collision

Current TTC is image-space TTC.

Units:

position = pixels
velocity = pixels/frame
TTC = frames

It is NOT yet metric TTC in seconds.

Basic idea:

TTC = distance / closing speed

But closing speed must be velocity projected toward the target:

closing_speed =
velocity · direction_to_target

An object moving quickly sideways should not automatically have high TTC risk.

13. Temporal Risk

Current temporal pipeline:

motion observation
→ Kalman
→ velocity
→ committed landing target
→ TTC
→ temporal risk

This allows the planner to react before an object physically occupies the landing zone.

14. Safety Planner

The planner considers the whole system state instead of using independent event handlers.

Examples:

collision risk
+
normal battery
→ EVADE_AND_HOLD
collision risk
+
critical battery
→ EMERGENCY_RECOVERY

Compound hazards require compound policies.

15. Action Stabilization

Safety actions should not oscillate every frame.

Principle:

danger increases
→ escalate immediately

danger appears to disappear
→ require persistence before de-escalating
16. Emergency Recovery

Prototype recovery sequence:

BRAKE (if needed)
→ EVADE
→ ALIGN_SAFE_ZONE
→ DESCEND
→ LAND

This is a simulation recovery plan, not a physical flight controller.

17. Command Runtime

Command states include:

PLANNED
DISPATCHED
ACKNOWLEDGED
COMPLETED
TIMEOUT
FAILED

Important principle:

Command sent does not mean command executed.

18. Fail-Closed Safety

If command execution becomes unverified:

timeout / failure
→ FAIL_CLOSED

Normal autonomous operation should not silently continue.

Recovery requires explicit health checks and operator acknowledgment.

19. Human Approval

Approval lifecycle:

REQUESTED
→ PENDING
→ APPROVED / REJECTED / EXPIRED

Important principle:

Human approval must actually gate command execution.

Policy requirement and execution authorization are separate concepts.

20. Trace / Observability

A safety system should explain:

what was observed
which target was selected
raw decision
stabilized decision
command ID
execution result
failure state
approval state
recovery plan
temporal risk

A system that cannot explain critical actions is difficult to debug or trust.

Core Rules I Should Remember
Never trust one frame.
Detection is not tracking.
Tracking is not prediction.
Best candidate is not necessarily the committed target.
Camera motion is not object motion.
Model output is not automatically trustworthy.
Safety escalation should be fast.
De-escalation should require persistence.
Command sent is not command executed.
Unverified execution should fail closed.
Human approval must actually block execution.
Every critical action should be traceable.
Tests protect safety behavior.
Simulation results must not be presented as physical-flight validation.
Measure improvements instead of merely naming algorithms.
Current Learning Frontier

Next areas to strengthen:

multi-object association
calibrated TTC
camera calibration / depth
quantitative evaluation
ROS2 concepts
PX4 concepts
VIO / SLAM
real-time system constraints
C++ robotics development
