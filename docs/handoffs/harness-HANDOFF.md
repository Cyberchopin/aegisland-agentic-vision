# DroneHacks benchmark harness handoff

## Checkout and scope

- Writable checkout: `<worktrees>/harness` only.
- Branch at start: `work/dronehacks-harness`.
- Starting HEAD: `e13b6984b2479691e9178223fbbaa5a3460585e8`.
- Initial tracked/untracked state: clean. No `.codegraph` was present.
- Existing benchmark/evaluation code, scripts, tests, CI and relevant DroneHacks
  documentation were read before implementation. No existing script or result
  was edited. No other checkout was modified.
- No push, merge, deployment, dataset download or flight-controller communication.
  All harness execution flags remain false. No vendor inference was run.

## Changes

| New file | Purpose |
| --- | --- |
| `src/aegisland/benchmark_contract.py` | Versioned adapter capabilities, strict JSON/JSONL validation, explicit platforms, counts, latency tails, FPS, drop/coverage and tracking/recovery metrics |
| `src/aegisland/benchmark_adapters.py` | Existing VDR core replay; legacy cuVSLAM stereo/inertial, VDR and optional FFS artifact import; canonical recorded OAK/DepthAI interface |
| `src/aegisland/benchmark_harness.py` | `run`/`verify` CLI; unique directories, raw evidence, source snapshots, command/config/HEAD/environment, SHA-256 sealing and failure state |
| `tests/test_benchmark_contract.py` | Focused corruption, count, timestamp, unit/platform, adapter, lifecycle and end-to-end tests |
| `docs/dronehacks-benchmark-contract.md` | Runnable commands, producer/result contract and precise metric definitions |
| `docs/benchmark-migration-notes.md` | Additive migration from existing formats and limits of historical evidence |
| `docs/full-suite-timeout-triage.md` | Closeout inventory, bounded shard durations/exit codes, timeout diagnosis and command/path checks |
| `HANDOFF.md` | Scope, validation, remaining gaps and integration order |

The new code reuses the production `CameraMotionCompensator` and
`VisualDeadReckoner`, plus existing ATE/RPE functions. It does not wire an estimator
into the flight or safety decision path. Failed and successful runs get separate
UTC/UUID directories; no historical result is overwritten.

## Validation

The existing environment was borrowed read-only via its interpreter at
`<main-checkout>/.venv/bin/python`. Commands used
`PYTHONPATH=src` and `PYTHONDONTWRITEBYTECODE=1`; nothing was installed into it.

**Harness-related validation: 53 tests passed.** This is 38 new contract/harness
cases plus 15 existing GPS-denied/trajectory cases, not 53 newly added tests.
The closeout pass reruns this same selection independently; exact timing and
exit code are in [the timeout triage](../../docs/full-suite-timeout-triage.md).

Focused tests:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest \
  tests/test_benchmark_contract.py tests/test_trajectory_eval.py tests/test_gps_denied.py \
  -o addopts='' -q
```

Result: **53 passed**. Coverage includes trailing-newline and whole-line truncation,
duplicate/decreasing timestamps, NaN/Infinity/overflow/duplicate keys, contradictory
summary/pose counts, unavailable VDR timings, FFS memory scope, OAK expected
schedule, source SHA mismatch, resealed-but-inconsistent result summaries,
error-state rejection, unique output directories, Orin mislabelling, and known
synthetic ATE/RPE correctness. These controlled test values are not vendor
benchmark measurements.

Repository Ruff check: **passed**, using `ruff check --config pyproject.toml src tests`.
Formatting is applied only to new Python files. `git diff --check` also passed.

Additional integration checks passed:

- Native eight-frame CPU VDR CLI run, followed by CLI `verify`.
- Read-only imports and hash/result verification of existing artifacts for
  `cuvslam-stereo`, `cuvslam-inertial`, `aegisland-vdr`, and `ffs-research`.
- Actual archived fixtures: `cuvslam-stereo-20261024-synthetic-s7-run1`,
  `cuvslam-inertial-MH_04_difficult-20261024-102659-signals`,
  `vdr-20261024-synthetic-s7-run1`, and `ffs-c-ffs-MH_01_easy-20261024-032728`.
  This validates ingestion, not new GPU performance.

Smoke/import evidence is local under ignored `runs/dronehacks-contract-*` roots.
Run `verify` on an explicitly selected UUID directory. No timing or accuracy
numbers from these checks are promoted to performance claims.

**Expanded regression (previous pass): 286 passed, 1 skipped, 53 deselected in
54.91 seconds; exit code 0.** This overlaps the 53 harness-related tests above;
the counts must not be added. The 53 deselected cases are a different set of
legacy matrix/timeline tests, not the harness-related selection.
The skip is the existing opt-in CUDA executable integration test. This selection
excludes the expensive repeated scenario/timeline/ablation matrix modules:

```sh
timeout 180s env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  OPENCV_FOR_THREADS_NUM=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest -o addopts='' -q \
  -k 'not ablation and not timeline and not test_reliability_benchmark and not test_sensor_failure_benchmark and not test_perception_fault_benchmark and not test_perception_scene_benchmark and not test_perception_severity_benchmark and not test_flicker_benchmark and not test_safety_assessment and not test_safety_dashboard'
```

**Complete suite: unresolved because of the 300-second timeout.** Collected
**340 tests**; attempted with a **300-second** limit and
OpenCV/BLAS/OMP set to one thread. It exceeded the bound and exited **124**;
there were no reported failures before timeout, but **the full suite did not
complete and is not claimed passing**. An earlier unbounded/default-thread run
was interrupted because the repeated simulations were slow. No tests were
weakened or changed to force completion. The historical bounded command is
retained below for reference; closeout does not remove or increase its timeout:

```sh
timeout 300s env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  OPENCV_FOR_THREADS_NUM=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest -o addopts='' -q
```

Closeout uses disjoint bounded shards, a full node-ID inventory and partial
per-test timing/stack evidence to diagnose the timeout. It does not convert
partial shard passes into a complete-suite pass. See
[docs/full-suite-timeout-triage.md](../../docs/full-suite-timeout-triage.md) for exact
budgets, durations, exit codes, unresolved cases and preservation checks.
No harness defect has been demonstrated during closeout; implementation and
existing tests remain unchanged.

Closeout collected all 340 cases across 59 modules and ran 18 disjoint bounded
shards: 310 fully passed, 1 skipped, 9 interrupted and 20 not reached before
their shard limits. Nine legacy simulation shards exited 124; no failed test
phase reports were observed. These diagnostic counts do not resolve the
complete-suite timeout. Repeated CPU optical-flow scenario replays are the
observed slow path. Exact per-shard/per-test durations and exit codes are in
the triage report. All 877 pre-existing output files matched their before/after
SHA-256 hashes; documented command/path checks and final whitespace checks passed.

## Known gaps

- No Orin board, OAK hardware capture, cuVSLAM inference, FFS inference or large
  dataset replay was performed. Vendor adapters normalize preserved artifacts;
  the native runnable estimator is the existing CPU VDR baseline.
- OAK's old flat-target aggregate summaries cannot supply per-frame samples.
  Future capture/replay producers must emit canonical metadata, a full timestamp
  schedule and explicit dropped slots. The included OAK fixture is synthetic.
- Archived upstream camera drops, missing historical commands/HEADs, VDR timing
  samples and unrecorded RAM/VRAM peaks cannot be reconstructed. Missing values
  stay null with scope/reason. Imported platform declarations are not attestation.
- Native cold VDR frame is reference initialization (`current()`), not a cold
  ORB update; model/import/decode time is outside that latency scope. RAM is a
  process-lifetime RSS high watermark, not isolated algorithm memory.
- All new results set `valid_for_performance_claims=false`. Actual paced runs,
  GPU/Jetson clocks/power modes and repeat-session evidence still require review
  under the existing benchmark plan. Desktop evidence is never Orin evidence.
- SHA-256 checks detect corruption, not authenticity if an actor rewrites both
  data and checksums. Hard kills may leave running/unsealed directories; readers
  reject them. Malformed CLI syntax/unwritable roots cannot emit a normal run.

## Suggested cherry-pick order

No commits or Git refs were created; changes are new reviewable working-tree
files. An integrator can commit these groups in order, then cherry-pick those
commits without touching old result directories:

1. Contract and adapters: `benchmark_contract.py`, `benchmark_adapters.py`.
2. Run lifecycle/CLI and tests: `benchmark_harness.py`, `test_benchmark_contract.py`.
3. Contract/migration documentation and this handoff.

Keep the first two groups together when running tests. No schema migration,
existing test modification, dependency update or result regeneration is required.
