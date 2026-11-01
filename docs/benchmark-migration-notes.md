# Migrating GPS-denied runs to the unified contract

This change adds three modules and tests; it does not replace existing benchmark
entrypoints or historical results. Start with the
[contract](dronehacks-benchmark-contract.md) and keep the existing
[pacing and performance rules](dronehacks/gps-denied-benchmark-plan.md).

## Existing result adapters

Create a small JSON file describing the **original** measurement platform, using
the schema in the contract. For the archived development runs, the source metadata
records a WSL2 x86_64 host and RTX 4060 Ti; it is a desktop result. Use its actual
hardware description in `runs/origin-platform.json`. Then import into a new root:

```sh
PYTHONPATH=src python -m aegisland.benchmark_harness run \
  --adapter cuvslam-stereo --format legacy \
  --source benchmarks/results/gps_denied/cuvslam-stereo-20261024-synthetic-s7-run1 \
  --expected-frames 200 --origin-platform runs/origin-platform.json \
  --output-root runs/dronehacks-imports
```

This command reads committed fixture artifacts only; it does not need EuRoC images
or a working cuVSLAM installation. It performs no new inference. The `200` argument
is the archived run's declared count, not a newly measured benchmark outcome.
The old source manifest/HEAD/clock/config metadata stays in `source/`; the new
`revision.json` describes the harness revision used to parse it.

| Existing format | Required files | Normalization/limitations |
| --- | --- | --- |
| `run_vdr_euroc.py` | `run.json`, `frames.jsonl` | `valid` becomes ok/lost; normalized x/y retained. Timing is null because individual elapsed values were not logged. Summary valid count must agree. |
| `run_cuvslam_euroc.py --mode stereo` | `run.json`, `frames.jsonl`, `trajectory_body.tum` when tracked | `track_ms` retained; mode and total/tracked/untracked/pose counts must agree. |
| `run_cuvslam_euroc.py --mode inertial` | Same | Use `cuvslam-inertial`; stereo cannot be silently relabelled inertial. |
| `run_ffs_euroc.py` | `summary.json`, `frames.jsonl` | Optional `ffs-research`; sum rectify/upload/model/post for defined latency; preserve invalid-pixel coverage and allocated CUDA peak. No localization accuracy. |
| `oak_depth_validation.py` | Aggregate summary only | Cannot recover per-frame latency, timestamps or drops. Keep historical validation separate; export canonical producer records for future unified runs. |

Existing indexes, when present, are checked for every consumed file. Optional
`gpu_clocks.csv` is copied. No old file is rewritten and no old checksum is updated.
Do not add newlines to a damaged historical JSONL to make it pass: preserve the
failure artifact and regenerate a new source run if authorized.

TUM trajectories contain seconds as floating point while JSONL contains integer
nanoseconds. The adapter requires strictly increasing TUM timestamps, one pose per
tracked frame, and agreement within 1 microsecond (to tolerate epoch-second float
serialization). Frame JSONL timestamps themselves are exact integer comparisons.

Ground truth is optional and must be the same body frame and clock as the estimate:
add `--groundtruth /path/to/mav0/state_groundtruth_estimate0/data.csv` to compute
ATE/RPE. It is copied and hashed; original calibration and transform responsibility
remain with the producer. Do not pass VDR normalized positions as metric poses.

## OAK/DepthAI and future producer changes

Use `--format contract` with the required producer `metadata.json` and per-frame
records. The producer should persist its expected capture schedule before replay,
retain device timestamps, and write an explicit dropped slot on queue overflow or
missing processing. Save runtime/device versions, camera calibration/preset,
resolution, exposure, IMU availability, synchronization and clock/power information
in its environment/config metadata. CPU host timing and on-device stereo timing
must have distinct named scopes.

Keep capture and hardware setup outside this importer. The provided OAK contract
test uses a synthetic producer record solely to test ingestion; it is not evidence
of OAK performance or a substitute for a hardware recording. Future recorders can
adopt the contract without changing the old scripts. Aggregate OAK flat-target
validation remains useful as a separate depth-quality experiment.

## Historical uncertainty stays explicit

- Old replay counts describe logged selected inputs, not upstream camera losses.
  The importer cannot recover input slots missing from both logs and summaries.
- Old VDR per-frame timings and process RAM peaks are unavailable. FFS's CUDA
  allocator peak has narrower scope than total VRAM. These remain null or explicitly
  scoped rather than estimated.
- Imported environment/command information is only as complete as the preserved
  source metadata. The harness does not invent a historical command or HEAD.
- No Orin, camera, cuVSLAM or FFS runtime measurement is performed during import.
  Verify source artifacts and labels before making any comparison.
- All new results remain ineligible for performance claims until the existing
  paced-run, clock and repeat-session requirements are reviewed separately.

## Verification and adoption

Run `python -m aegisland.benchmark_harness verify NEW_RUN_DIR` before consuming a
result. Reusing an output root is safe: each invocation gets a new UUID directory.
Consume only explicitly identified `complete` runs; retain `failed` directories
for diagnostics. There is no "latest result" overwrite or automatic pooled table.

Downstream code can use `verify_run(Path(...))` to receive a validated result dict.
The additive modules are `benchmark_contract` (validation/metrics),
`benchmark_adapters` (core replay and artifact adapters) and `benchmark_harness`
(CLI, provenance, run lifecycle and integrity). Existing `summarize_runs.py` retains
its original behavior; it does not gain the new strict guarantees. Migrate consumers
to the new reader explicitly instead of assuming old summaries were retrovalidated.
