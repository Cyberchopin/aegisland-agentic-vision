# Stage 0: reproducible CPU perception baseline (scope ends at Stage 4)

Inspected upstream revision: `54d4fb170e893929ea0a24cc4f90c347845464aa`.
No production perception code is changed. Run the benchmark as a standalone process.

## Actual call graph

`agent.py` calls `perception.observe(frame, frame_index)`. A low-confidence retry
calls `enhance_for_active_perception` (BGR→Lab, split, CLAHE, merge, Lab→BGR), then
`observe(..., active_perception=True)`. The harness can force this retry on every
frame for reproducible branch coverage; it does not benchmark the agent policy.

`OpenCVLandingPerception.observe`:

1. `_resize`: aspect-preserving cap at 960 pixels wide by default; no upscale.
2. BGR→gray; retry-only CLAHE; GaussianBlur 5×5.
3. Canny 55/145; morphology close 3×3.
4. Laplacian CV_32F → convertScaleAbs texture map.
5. `_motion`: first frame/size mismatch returns zero mask without optical flow.
   Otherwise raw Farneback → cartToPolar; `CameraMotionCompensator.compensate`
   (two ORB detectAndCompute calls, BFMatcher knnMatch, ratio filtering,
   RANSAC findHomography, sanity checks, conditional warpPerspective);
   `VisualDeadReckoner.update` (retry uses current pose);
   second Farneback on aligned reference → cartToPolar; magnitude > 1.8;
   elliptical 7×7 open, two dilations, external contours and area-filtered boxes.
6. `PerceptionQualityMonitor.analyze`: brightness/dark masks, Laplacian variance,
   histogram entropy, ORB detect, dark-region contours, geometry/image scores and classification.
7. `_score_grid`: median-relative dark appearance mask, elliptical 5×5 open,
   eight lower-grid ROIs; edges/texture/motion/appearance/clearance scores and safe flags;
   sorted `ZoneCandidate` results.
8. Risk statistics, confidence, evidence hash, `VisionEvidence` construction.
9. Frame copy and `_annotate`; reference-state update; return evidence and BGR image.

There is no HSV conversion, color-range threshold or color-mask fusion in this CPU pipeline.

## Run and timing contract

From the repository, Python >=3.11 with the pinned project dependencies:

```sh
python -m pip install -e '.[dev]'
python -m aegisland.profiling --output runs/cpu-stage0 --warmup 3 --runs 20 --threads 1
python -m aegisland.profiling --input runs/cpu-stage0/inputs.npz --output runs/cpu-retry --warmup 3 --runs 20 --threads 1 --active
python -m pytest --basetemp=work/pytest
python -m ruff check src tests
```

Use a new output directory each time (overwriting is rejected). The default corpus
contains 14 frames: each existing 320×240 scene and a two-column cyclic translation.
Transitions between different scenes are intentional discontinuities, not camera trajectories.
This is synthetic smoke coverage, not a representative flight dataset. For a real
baseline supply an NPZ `frames` array, ordered N×H×W×3 uint8 BGR, with no pickle.
Keep sequence length and order unchanged across backends. Record camera/video source,
capture/extraction procedure, license, timestamps and any dropped frames separately.
The archive and each decoded frame are hashed; actual replay pixels are saved.
For 720p/1080p tests supply real inputs of those dimensions. The default 960 cap still
applies: use `--maximum-width 1920` to preserve native 1080p, and use the same cap
for every backend. Do not label resized processing as native 1080p.

Each warmup/run reinitializes all perception state and OpenCV RNG (seed 42).
Each sequence preserves temporal state between frames, and retries preserve the
pre-frame reference. Warmups are complete sequences, excluded from samples and
performed separately for each timing mode. First-frame timings remain separate
from later frames. Inputs are loaded before timing; model construction, seeding,
disk I/O, correctness copies and serialization are outside timing.

`baseline/end_to_end` uses perf_counter_ns around the actual call and includes
allocation, annotation, and state updates. Retry totals also include the explicit
Lab enhancement. This is synchronous CPU wall latency, not camera/decode latency
or full flight-agent latency. Existing `processing_ms` ends before annotation and
some evidence construction, so it is not the comparable end-to-end boundary.

`diagnostic` separately uses cProfile on the same workload. `samples.csv` contains
run/frame/retry, call count, inclusive and exclusive nanoseconds for every function,
including OpenCV C calls. These are instrumented diagnostics, not speedup evidence.
Two Farneback calls are aggregated in their function row (calls=2); similarly repeated
morphology calls aggregate. `.prof` files preserve caller edges so morphology in
motion, quality and grid scoring can be distinguished using pstats.print_callers().
Python/NumPy work remains visible instead of being silently omitted. Inclusive times
overlap; never sum them. Exclusive times attribute work excluding traced callees,
but include instrumentation distortion. cProfile does not expose OpenCV internal
native worker call graphs. Use baseline wall time for comparisons.

`summary.json` uses NumPy linear percentiles p50/p95/p99 by mode, frame, retry and
function. Missing branch calls are absent, not zero-duration executions. Counts are
reported; 20 observations are only a smoke benchmark, not a reliable tail estimate.
Increase runs and repeat independent experiments on a quiet machine for conclusions.

## Correctness contract

`reference.npz` stores gray, edges, texture, motion mask and annotated frame per
frame/pass. `reference.json` stores all public evidence except processing_ms and adds
motion boxes. `correctness.json` compares two independent CPU replays.
`profiling.compare((arrays, evidence), (candidate_arrays, candidate_evidence))`
is the future backend comparison API; identical keys, shapes, dtypes and sequence
ordering are required. Reuse `inputs.npz` and the saved configuration.

The initial acceptance tolerance is **zero**, not an assumed GPU error allowance:
these intermediate arrays are uint8, including binary 0/255 masks. Report max absolute
error, MAE, pixel mismatch fraction and mask IoU (both empty → 1). Exact pixel equality
is required. All serialized evidence must match, including candidate ordering, boxes,
safe flags, best-zone implications, failure classes, localization and evidence_id.
This is a conservative equivalence gate, not a claim that arbitrary GPU algorithms
will be bit-identical. Upstream has no measured GPU error distribution supporting a
looser tolerance. Rounded values use four decimals for most scores, three for brightness
and sharpness, six for localization coordinates/velocity; rounding precision is **not**
a safety margin. One changed mask pixel can cross a discrete grid/classification gate.
Do not excuse decision changes using a generic epsilon. If a future algorithm differs,
retain failed results and calibrate a separately reviewed field-specific contract
against real data and threshold-boundary cases before accepting it.

Tests cover production-vs-capture equivalence with temporal state and retries, resize,
pixel/decision/schema failures, warmup exclusion, artifact creation and percentiles.
The existing full test suite remains the policy/behavior regression check.

## Minimal first custom kernel measurement plan (no CUDA implementation)

Select **BGR8→GRAY8** as the first one-thread-per-pixel kernel: it actually exists here,
has a simple input/output contract, and can be compared with the saved gray output
on base passes. Use the exact OpenCV integer rounding behavior as the oracle. Do not
start with HSV. Validate black/white, channel extremes, rounding boundaries, odd widths,
non-contiguous source strides and all replay frames before measuring. Resize stays CPU
for this first isolated experiment. The future driver must export candidate outputs.

Planned driver contract (not an existing executable): `gray_bench --input INPUT.npz
--warmup 10 --iterations 20`, loading/resizing the same BGR pixels before measurement,
launching exactly one `bgr_to_gray` kernel per iteration (including warmups), recording
dimensions/strides/grid/block/compiler flags and CUDA-event kernel timings. Separately
record synchronized host wall latency including H2D+kernel+D2H, and whole perception
latency including untouched CPU stages. Never compare kernel-only time with observe().
Use no fusion, tiling or asynchronous overlap in this first measurement.

On the target NVIDIA machine, from a shell with ncu/nvcc/nvidia-smi available:

```sh
nvidia-smi -q > gpu.txt
nvcc --version > cuda-version.txt
ncu --version > ncu-version.txt
ncu --list-sections > ncu-sections.txt
ncu --query-metrics --query-metrics-mode all > ncu-metrics.txt
ncu --kernel-name-base function --kernel-name bgr_to_gray --launch-skip 10 --launch-count 1 --section LaunchStats --section Occupancy --section MemoryWorkloadAnalysis --section SpeedOfLight --export first-gray ./gray_bench --input inputs.npz --warmup 10 --iterations 20
ncu --import first-gray.ncu-rep --page raw --csv > first-gray.csv
ncu --kernel-name-base function --kernel-name bgr_to_gray --launch-skip 10 --launch-count 1 --metrics sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,launch__registers_per_thread,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_blocks,launch__occupancy_limit_warps --export first-gray-metrics ./gray_bench --input inputs.npz --warmup 10 --iterations 20
ncu --import first-gray-metrics.ncu-rep --page raw --csv > first-gray-metrics.csv
```

These commands are a plan for that future driver, not commands already measured here.
Windows uses `./gray_bench.exe`. Metric availability depends on GPU/tool version:
verify every requested name in ncu-metrics.txt, preserve any unsupported-metric error,
and use the available section report instead of substituting zero. Record profiler
replay/cache settings and permissions errors. Nsight replay/collection perturbs timing;
collect ordinary CUDA-event and end-to-end timings without ncu as separate runs.
Capture achieved occupancy (active warps percentage), theoretical occupancy and its
register/shared-memory/block/warp limits, registers/thread, DRAM throughput percentage,
read/write bytes and duration. Bytes/duration yields measured DRAM bytes/sec; it is
not the same as logical 4 bytes/pixel bandwidth because caches affect DRAM traffic.
High occupancy alone does not establish better performance.

References: [NVIDIA CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
and [profiling metrics guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
No Stage 5+ work, GPU speedup targets or measured GPU results are included.

## CPU named-stage milestone (schema version 1)

The default harness now executes three independent rounds, in this order:
`baseline`, `stages`, `diagnostic` (cProfile). Each round has its own complete
warmup sequences and fresh model/RNG state per sequence. Baseline and cProfile
use the unchanged production methods. Baseline never constructs a stage collector.
The stage round parses trusted local production method source and compiles
statement spans into temporary instance methods. No alternate perception algorithm
is maintained; production perception/compensation/quality files are unchanged.
Compilation, installation and restoration of methods happen outside timed calls.
All temporary instance overrides are restored in `finally`, including partial setup
and perception failures. Thread and OpenCL settings are restored on failures too.

`stage-schema.json` is the versioned boundary contract. Each stable semantic name
lists the exact production statement(s) it covers; repeated branches retain all
definitions. Source hashes in `environment.json` identify the implementation.
`motion.raw_flow` and `motion.residual_flow` are separate Farneback calls;
`motion.raw_magnitude` and `motion.residual_magnitude` are separate conversions.
Compensation covers ORB on each image, matching, filtering, point construction,
homography, validation and conditional warp. Quality covers exposure, texture,
features, connected dark regions, scoring and classification. Grid scoring includes
appearance masks, all candidate calculations and sorting. Enhancement, resize,
CLAHE, blur, edges, texture conversion, evidence, frame copy, annotation and
reference updates have named boundaries as well. Some Python operations remain
intentionally grouped in a statement; these are not individual native-call timings.

Only a taken branch body produces a sample. Untaken retry CLAHE, skipped flow,
rejected-homography warp and no-op resize do not produce fake zero durations.
Condition evaluation is included in the nearest enclosing span's residual rather
than separately attributed. A no-reference result has its own executed span.
First-frame/no-reference, shape mismatch and valid-reference observations are
separate branch groups; a valid reference does not imply a continuous camera
trajectory or successful compensation. Warmed sequence-first frames are not
process cold-start measurements. Retry enhancement belongs to the retry total;
retries retain the pre-frame image and reuse rather than integrate the pose.

`stage-samples.csv` contains per-call spans, sample/parent IDs, run/frame/pass,
reference and compensation status, localization operation and processed shape.
IDs are local to one invocation. Inclusive durations overlap. Exclusive durations
are inclusive minus direct child inclusive durations, including condition work,
Python overhead and timing instrumentation. They are residual attribution, not
pure native compute. Never add parent and child durations or subtract an assumed
instrumentation cost. All samples stay in memory until timing ends. The independent
baseline wall samples remain the comparison boundary, including annotation and
state updates; `processing_ms` remains an incomplete internal boundary.

`stage-summary.json` groups by stage, parent, pass, reference/compensation/localization
branches and processed shape; it records counts and NumPy linear p50/p95/p99 for
inclusive and exclusive nanoseconds. Recompute it with
`profiling.summarize_branches(rows)` after parsing integer duration/shape columns
from `stage-samples.csv`. The existing summary can be recomputed with
`profiling.summarize(rows)` from `samples.csv`. Stage mode's `end_to_end` samples
also remain in that file, explicitly labeled `mode=stages`; use `mode=baseline`
for uninstrumented comparisons. No latency threshold is an acceptance test.

Separate correctness passes compare baseline replays and an instrumented replay
using the existing exact array/boxes/public-evidence contract (only `processing_ms`
is excluded). `stage-correctness.json` preserves the instrumented comparison.
`--reference PATH` additionally compares saved inputs/configuration and all outputs
against an existing baseline, saving `before-after-equivalence.json`. Differences
fail the command without changing tolerance. `SHA256SUMS` hashes every artifact
except the manifest itself, including partial output on a failed benchmark.
Installed distributions and available installer/direct-URL metadata are saved
without requiring pip in a uv-created environment.

Tests use controlled clocks for nested timing and exception unwinding, check
baseline/stages/cProfile separation, resize and compensation branches, skipped
flow, retry reference identity and single pose update, exact replay equivalence,
artifact hashes, original-reference rejection and summary recomputation. No GPU
backend or dependency installation is part of this milestone.

Example using the preserved pre-instrumentation inputs (choose fresh output paths):

```sh
.venv/bin/python -m aegisland.profiling --input runs/original-baseline-20261016T233804473881Z/base/inputs.npz --reference runs/original-baseline-20261016T233804473881Z/base --output runs/cpu-stages-base --warmup 3 --runs 20 --threads 1
.venv/bin/python -m aegisland.profiling --input runs/original-baseline-20261016T233804473881Z/base/inputs.npz --reference runs/original-baseline-20261016T233804473881Z/retry --output runs/cpu-stages-retry --warmup 3 --runs 20 --threads 1 --active
```

Earlier CUDA plan text in this document is historical; `first-cuda-kernel.md`
supersedes its plan-only status. This milestone makes no performance claim.

The CLI capture regression is covered by subprocess tests invoking the real
`python -m aegisland.profiling` entry point for both base and retry workloads.
Instrumentation obtains function bodies only from the explicit production classes
(`OpenCVLandingPerception`, `CameraMotionCompensator`, `PerceptionQualityMonitor`).
Snapshot replay explicitly selects `capture_grid=True`, which binds the compiled
production grid method to `_stage_score_grid` while leaving the capture callback
untouched. Module names, including `__main__`, play no role in selection. Other
non-production overrides are rejected with cleanup of any earlier patches.
