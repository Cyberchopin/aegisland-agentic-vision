# Day 02 — From Simulation Safety to PX4 Shadow Supervision

## What I built

Today AegisLand crossed an important boundary: it moved from an internal autonomy simulation into an external PX4 SITL supervision loop.

The current integration is:

```text
PX4 SITL
  ↓
MAVLink
  ↓
MAVSDK
  ↓
PX4 telemetry adapter
  ↓
Telemetry freshness watchdog
  ↓
Estimator health
  ↓
Navigation authority
  ↓
Shadow safety decision
AegisLand is still operating in shadow mode:

CONTROL OUTPUT: DISABLED
COMMAND EXECUTED: FALSE

It observes PX4 and makes safety decisions, but does not send flight-control commands.

Key milestone

The system can now distinguish several concepts that initially looked similar:

Connection
≠ Telemetry freshness
≠ Estimator validity
≠ Navigation authority

This was one of the most important lessons of the integration.

A MAVLink connection can still exist while navigation telemetry is stale.

Telemetry can remain fresh while the global-position estimator becomes invalid.

An estimator can start producing valid measurements again without immediately deserving navigation authority.

Bug #1 — Slow telemetry is not a dead connection

The first watchdog implementation waited for several MAVSDK streams together:

battery
position
velocity
health

This caused false failures because these topics do not update at the same frequency.

A slow battery or health topic could make the whole telemetry pipeline look stale.

The fix was to separate the streams and use fast navigation telemetry such as position and velocity for freshness monitoring.

Lesson
One slow topic
≠ dead transport
Bug #2 — Cached data can look healthy

When telemetry stopped updating, the monitor continued displaying the last received PX4 values.

For example:

gps=True

could still be printed even though the sample was old.

The fix was to explicitly distinguish:

PX4 [FRESH]

from:

PX4 [CACHED]
Lesson
Data value
≠ data freshness
Transport vs telemetry

AegisLand now treats these separately:

transport=connected

telemetry=healthy
age=0.01s

A connection tells us that MAVSDK/PX4 communication exists.

Telemetry freshness tells us whether the information used for navigation is still recent enough to trust.

This separation prevents false confidence in stale state estimates.

GPS failure experiment

Using PX4 SITL failure injection:

failure gps off

the PX4 communication link remained healthy while the global-position estimator became invalid.

AegisLand observed:

transport=connected
telemetry=healthy
global_position_valid=False
estimator=failed
nav=degraded
authority=revoked
action=hold_and_scan
executed=False

This demonstrated that AegisLand could distinguish a navigation-capability failure from a transport failure.

After:

failure gps ok

global position became valid again.

Why gps=True was renamed

Originally the PX4 adapter displayed:

gps=True

However, the MAVSDK value being used was:

health.is_global_position_ok

This does not directly mean that the physical GPS sensor itself is healthy.

It means that PX4 currently considers its global-position estimate usable.

Therefore the more accurate semantic name is:

global_position_valid

This distinction matters in robotics because:

sensor availability
≠ estimator validity
≠ navigation capability
Fail fast, recover conservatively

AegisLand already had a stateful SensorHealthMonitor.

Instead of creating a new recovery system for PX4, the existing safety abstraction was reused.

The policy is asymmetric:

Failure:
one invalid observation
→ FAILED immediately
→ authority revoked

but recovery requires sustained evidence:

valid observation 1
→ DEGRADED 1/3

valid observation 2
→ DEGRADED 2/3

valid observation 3
→ HEALTHY
→ authority restored

This prevents one noisy good sample from immediately restoring navigation authority.

PX4 fault benchmark

The first PX4-in-the-loop GPS failure benchmark produced:

baseline_samples                         5
fault_samples                            292
recovery_samples                         3

observed_fault                           True
observed_recovery                        True

authority_revocation_latency_ms          0.0
authority_restore_latency_ms             201.917

unsafe_continuation_samples              0
premature_authority_samples              0
telemetry_false_positive_samples         0

command_execution_enabled                False
Interpretation

authority_revocation_latency_ms = 0.0

means that once AegisLand observed the first invalid global-position sample, authority was revoked during the same supervisor step.

unsafe_continuation_samples = 0

means the system never continued the mission while the observed global-position capability was invalid.

premature_authority_samples = 0

means authority was not restored during the 1/3 or 2/3 recovery stages.

authority_restore_latency_ms ≈ 202 ms

is intentional.

The system pays a small recovery delay in exchange for sustained evidence that the estimator has actually recovered.

Architecture principle

The most important idea I learned today is that autonomy safety is not just about perception accuracy.

A robust system asks several different questions:

What did I observe?

Is the measurement fresh?

Is the sensor healthy?

Is the estimator valid?

Is the capability observable?

Do I still have navigation authority?

Should the mission continue?

AegisLand is gradually becoming a system that reasons about these questions separately.

Current AegisLand safety philosophy
Confidence ≠ Health

Connection ≠ Freshness

Freshness ≠ Estimator validity

Estimator validity ≠ Navigation authority

Sensor failure ≠ System failure

and:

Fail fast.
Recover conservatively.
Current project status

Approximate hackathon-prototype maturity:

~70%

Completed or working:

OpenCV landing perception
landing-zone candidate scoring
camera motion compensation
temporal collision / TTC reasoning
visual dead reckoning
sensor synchronization
SensorHealthMonitor
PerceptionTrustGate
dynamic confidence fusion
capability-aware navigation authority
recovery hysteresis
camera / IMU / GPS fault simulation
deterministic fault matrix
severity calibration
cross-scene robustness
baseline ablation
component ablation
Safety Assessment framework
PX4 SITL
MAVLink
MAVSDK
PX4 telemetry adapter
shadow supervision
telemetry freshness watchdog
transport/freshness separation
estimator health supervision
PX4 GPS fault injection
PX4 GPS fault benchmark
Biggest limitation today

The PX4 integration is still read-only shadow supervision.

AegisLand can currently observe an external autopilot and decide whether navigation authority should exist, but it does not control PX4.

Also, the rich visual / IMU / GPS fallback architecture already tested internally has not yet been fully connected to external PX4 sensor streams.

Therefore this project should currently be described as:

PX4-in-the-loop shadow safety supervision

not:

autonomous closed-loop flight control

and definitely not:

flight-certified software

Next steps

The next development phase is to turn individual successful experiments into repeatable safety evidence.

Planned work:

PX4 fault benchmark timeline
automatic PASS / FAIL safety gates
repeated N-run experiments
latency distributions instead of one-run numbers
additional PX4 fault modes
connect PX4 IMU state
connect visual localization into the PX4-facing supervisor
demonstrate GPS-denied fallback using multiple capabilities
ROS2 / VIO / depth integration
eventually validate on approved physical hardware
One-sentence takeaway

AegisLand has evolved from a simulated perception-safety prototype into a PX4-in-the-loop shadow supervisor that separates transport health, telemetry freshness, estimator validity, and navigation authority while demonstrating immediate fail-safe revocation and conservative recovery.
