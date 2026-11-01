# DroneHacks Flight Forensics & Replay Workbench — HANDOFF

Only writable checkout: `<worktrees>/forensics`.
Branch: `work/dronehacks-forensics`.
Starting/current HEAD: `e13b6984b2479691e9178223fbbaa5a3460585e8`.
No commit, push, merge or deployment was performed. No other worktree was modified.

Implemented replay-only file ingestion, validation, diagnostic replay and evidence
sealing. All authority/action records remain `executed=False`, `authority=NONE`.
**Local sensing planes do not establish 3D traversability:** every report keeps
`full_3d_traversability=UNKNOWN`, `supports_3d_traversal=false`.
No Orin, OAK, PX4, SITL or flight behavior was validated on actual hardware.

## Refreshed B / D handoffs and provisional audit

The latest B and D HANDOFF files were reread before finalizing contracts. Their
hashes and the later bridge plan are recorded in
[docs/forensics/integration-plan.md](../../docs/forensics/integration-plan.md).

- B: `e944592ba8d0dbcdc00488a47bc7694740fa22d443f364bf9c11abfe205a85f5`.
  Preserve complete expected schedules, explicit dropped slots, original platform,
  unavailable metrics and its unresolved complete-suite timeout. Its reported test
  selections overlap; none are added to this worktree's test counts.
- D: `ae134ec733f224d879c2ef8c7da808c049a6c03f60bcd87107a4c2e2af78b46c`.
  This is the refreshed v2 adversarial / 36-frame handoff, not the historical v1
  example. Preserve all top-level and opening-embedded local-plane objects,
  rejection reasons, RGB/depth ambiguity, sampled ROI, probe plane and unassessed
  volumes. Positive local support never becomes 3D, descent or landing permission.
- Baseline audit handoff was read at
  `<agent-output>/baseline-audit/HANDOFF.md`;
  SHA-256 `78cf9e417f6540f2b485451e98d7417868f354d61f3d5e8f239f92c0df0cd020`.
  Its final exit-0 provenance verification is **still pending**. Dependency
  decisions remain provisional, regardless of earlier handoff wording.

Sibling documentation and relevant definitions were read only; no sibling module,
result, environment or worktree was copied or modified. The integration extension
used no network, download, hardware access or additional dependency installation.
Earlier dependencies installed for the initial ULog task are isolated in this
worktree's `.venv`; no sudo or global installation was used. The local PyULog /
NumPy pins are not an approved shared Jetson dependency decision.

## Files changed and contracts

| Files / directory | Purpose |
| --- | --- |
| `src/aegisland/forensics/model.py`, `ingest.py` | Strict JSONL / PyULog local-file ingestion, hashes, incomplete/unknown evidence |
| `alignment.py`, `analysis.py` | Hash-bound paired-clock interpolation, native clock preservation and anomaly rules |
| `integration.py` | OAK RGB/depth, PX4 attitude/telemetry, VL53L1X and advisory replay contracts; time/frame/calibration/transform checks |
| `integration_fixtures.py`, `fixtures.py` | Byte-deterministic sensor NPZ/JSONL and minimal binary ULog fixtures |
| `platform_manifest.py` | Separate desktop / Jetson Orin Nano / replay / SITL / unknown declarations and static no-hardware preflight |
| `recorder.py` | Exclusive run directories, immutable source snapshots, SHA-256 seals, implementation hashes and read-only verification |
| `scope.py` | Shared local-sensing-plane boundary; 3D traversability UNKNOWN |
| `report.py`, `cli.py`, `__init__.py`, `__main__.py` | Self-contained HTML / JSON / Markdown, diagnostic sliders and offline CLI |
| `tests/test_forensics.py`, `tests/test_forensics_integration.py`, `tests/forensics_replay_smoke.cjs` | Corruption / stale / missing / contradictory / authority / local-plane / DOM regression tests |
| `fixtures.py`, `integration_fixtures.py` | Deterministic in-memory synthetic ULog, damaged-input, clock-pair and four-stream fixture generators; generated binary fixtures are not committed |
| `docs/forensics/README.md`, `jetson-perception.schema.json` | CLI and original Jetson JSONL schema |
| `docs/forensics/integration-replay.md`, `integration-plan.md` | Exact four-stream contracts, recorder lifecycle and later B/D bridge plan |
| `docs/competition-data-boundary.md` | Data, safety, execution, local-plane and future-flight-input boundaries |
| `requirements-forensics.txt`, `pyproject.toml` | Isolated requirements, optional PyULog extra, console entry |
| `.github/workflows/ci.yml`, `README.md`, `HANDOFF.md` | Bounded isolated CI job, documentation entry and this handoff |

The exact untracked file inventory, including each fixture and report, appears in
the final git status below. Existing flight-control/planner modules are unchanged.

Canonical sensor records require explicit nanosecond clock domains, replay
assessment time, scheduled frame IDs, declared calibration/transform IDs and
literal false execution. OAK NPZ uses declared RGB8/BGR8, aligned optical-axis-z
metres and a boolean validity mask; pickle/nonfinite/shape/hash failures are
rejected. PX4 attitude is normalized **wxyz**, `ned_from_frd`; static sensor
transforms use normalized **xyzw**, `parent_from_child` into body FRD.
Downward range disagreement is only an inferred candidate when time, target,
geometry and uncertainty permit comparison. Different clocks/targets/rays stay
unknown. Configuration hashes/calibration IDs are producer declarations until
independently corroborated; they do not prove physical calibration.

Every advisory retains its source-frame binding and `local_sensing_planes` list.
Each nested plane must preserve local_camera_z_probe_plane scope, sampled ROI,
probe z, local width, all five unassessed volume limits, UNKNOWN 3D traversal,
false traversal support and false/NONE execution/authority. High confidence,
matching ranges, low-light recovery or local width cannot increase authority.
No native B/D bridge is claimed implemented; the later plan preserves raw traces,
all nested gaps, different hash algorithms and the ray-range versus camera-z distinction.

## CLI and example reports

Use the existing local environment; no install is needed for these commands:

```bash
PYTHONPATH=src .venv/bin/python -m aegisland.forensics --help
PYTHONPATH=src .venv/bin/python -m aegisland.forensics fixtures \
  --output runs/forensics-fixtures
PYTHONPATH=src .venv/bin/python -m aegisland.forensics integration \
  --bundle runs/forensics-fixtures/integration-consistent \
  --output-root runs/integration --run-id review-001
PYTHONPATH=src .venv/bin/python -m aegisland.forensics verify runs/integration/review-001
PYTHONPATH=src .venv/bin/python -m aegisland.forensics preflight \
  --environment replay --output-root runs/preflight --run-id metadata-001
```

Run IDs must be new; omission allocates a UTC/UUID ID. Output cannot be inside the
read-only input bundle. Sources are copied from the exact hashed snapshots.
The seal is written last; unsealed directories are rejected and never resumed.
`verify` checks identity, exact inventory, hashes, authority and recomputes replay
counts and finding categories/references. Hashes are not authenticity against an
actor rewriting both data and seals. A verified degraded run remains degraded.

The old `report --px4 ... --jetson ... --output ...` command still provides
regenerable, unsealed ULog diagnostic pages; it is distinct from recorded runs.
It marks unproved alignment unknown and only conditionally interpolates within
hash-bound clock anchors. Physical synchronization remains unknown.

Generated local artifacts used during development were reviewed but are excluded
from the checkpoint commit. The committed generators and tests reproduce their
synthetic inputs in temporary or ignored run storage; no benchmark output,
dataset, third-party binary, model, or cache is included.

Observed local exits before exclusion:

- `examples/forensics/unknown-alignment/`: synthetic anomalies, alignment unknown;
  report command exit 0, not a health claim.
- `examples/forensics/bounded-alignment/`: synthetic clock mapping inferred,
  physical sync unknown; report command exit 0.
- `examples/forensics/integration/consistent/`: 12 synthetic usable records;
  integration exit 0; seal/recomputed verification exit 0, 8 files.
- `examples/forensics/integration/adverse/`: stale, missing, out-of-order,
  calibration mismatch, skew and range contradiction candidates;
  integration exit **2 by design**; seal/recomputed verification exit 0, 8 files.
- `examples/forensics/preflight/no-hardware/`: static OS/package metadata only;
  preflight and verification exit 0, 5 files; hardware readiness unknown.

Open HTML from disk. Integration replay shows diagnostic records, never commands.
Old development evidence was moved into ignored `runs/forensics-early/`, not
rewritten as final sealed evidence. Final examples are small, entirely synthetic
except for explicitly scoped local OS/package/memory/storage preflight metadata.

## Bounded test and check evidence

Environment: Python 3.12, PyULog 1.2.2, NumPy 2.5.4, pytest 8.4.2, Ruff 0.14.3.

```bash
timeout 60s .venv/bin/python -m pytest \
  tests/test_forensics.py tests/test_forensics_integration.py \
  tests/test_domain.py tests/test_time_sync.py tests/test_sensor_sync.py \
  tests/test_px4_telemetry.py
# 177 passed in 0.60s, exit 0.
# 163 forensics cases + 14 related existing cases; B/D test counts are not included.

.venv/bin/ruff check src/aegisland/forensics \
  tests/test_forensics.py tests/test_forensics_integration.py
# All checks passed; exit 0.

.venv/bin/ruff format --check src/aegisland/forensics \
  tests/test_forensics.py tests/test_forensics_integration.py
# 16 files already formatted; exit 0.

git diff --check
# No output; exit 0.
```

Coverage includes actual PyULog decoding of invented binary fixtures; damaged,
empty, nonfinite, wrong-type, duplicate-key, out-of-order and missing inputs;
stale/future/mixed-clock evidence; schedule tail loss; payload/calibration/
transform/advisory bindings; contradictory ranges; malicious authority and 3D
claims; positive local-plane support with UNKNOWN 3D traversal; exact source
preservation; no-overwrite sealing; tampering/resealed inconsistent summaries;
no-hardware preflight and package-import restrictions; and both HTML sliders via
Node DOM smoke. DOM smoke is not actual browser rendering or device validation.

The complete OpenCV suite was not run in this minimal environment. An earlier
full-repository Ruff check observed 6 existing errors in untouched files:
2 E731 in `tests/test_localization_supervisor.py`, 4 E402 in
`tests/test_perception_trust.py`. Those unrelated tests were not changed. The
scoped checks above passed; no complete-suite success is claimed.

## Unresolved hardware and integration questions

- Exact Orin Nano SKU, JetPack/CUDA/DepthAI versions, native driver loadability,
  clocks/power settings and memory limits remain unknown. No runtime was probed.
- Actual OAK-D S2 intrinsics, image/depth registration, channel order, frame rate,
  USB behavior, timestamp epoch/session reset and camera mounting are unverified.
- PX4 firmware/topic schema, attitude convention and copied-log completeness must
  be checked against actual post-flight evidence. No SITL or PX4 link was started.
- VL53L1X status/units, FoV, mounting offset, invalid returns, sigma meaning and
  same-surface comparability are unverified. Parser bounds are not device specs.
- Real clock pairs need independent event and uncertainty evidence; calibration
  IDs, matching file hashes and equal domain strings do not prove synchronization.
- Local-plane evidence does not assess ROI-external, beyond-plane, occluded,
  dynamic or swept vehicle volume. Full 3D traversability and landing safety stay UNKNOWN.
- Provenance final exit-0 and shared dependency choices remain pending. Native
  B/D bridges are planned only; no vendor performance numbers are promoted.

## Onsite read-only verification steps

1. With the site's permission, obtain copies of already recorded sensor logs,
   frame assets, full schedules, calibration/config snapshots and existing board
   metadata through the approved read-only evidence process. Record their source
   and hashes. Do not launch a capture pipeline or connect the flight controller
   from this tool; do not change safety settings or permissions.
2. Run `preflight` into a new directory. Review observed host/package/capacity
   metadata and unknown device readiness. Retain expected USB, serial, I2C and GPU
   permission requirements for an administrator; this tool neither tests device
   access nor changes groups, udev rules or device permissions.
3. Review units, frame ID/session identity, clock-domain evidence, quaternion
   order, camera view, actual intrinsics and static transforms offline. If any
   link is unproved, leave it unknown; never repair it using competition route
   knowledge, best-looking offsets or safety-parameter changes.
4. Only a reviewed canonical replay bundle can be passed to `integration` with
   a new run ID. Check the ledger for missing/stale/conflicting evidence, then
   run `verify` on that exact directory. Review raw records and source snapshots.
5. Preserve local-plane and UNKNOWN 3D limits at every nesting level. Do not feed
   any result, map, trajectory, decision or recorded control into a later flight.
   The laptop remains a read-only monitoring/evidence-analysis station. Actual
   hardware/SITL/flight validation requires a separately scoped measured activity.

## Pre-checkpoint inventory

The following historical status block records the agent handoff before checkpoint
coordination. Generated `examples/forensics/` and `tests/fixtures/forensics/`
payloads in that block are intentionally excluded from the final commit.

```text
 M .github/workflows/ci.yml
 M README.md
 M pyproject.toml
?? HANDOFF.md
?? docs/competition-data-boundary.md
?? docs/forensics/README.md
?? docs/forensics/integration-plan.md
?? docs/forensics/integration-replay.md
?? docs/forensics/jetson-perception.schema.json
?? examples/forensics/README.md
?? examples/forensics/bounded-alignment/findings.md
?? examples/forensics/bounded-alignment/report.html
?? examples/forensics/bounded-alignment/summary.json
?? examples/forensics/integration/adverse/findings.md
?? examples/forensics/integration/adverse/implementation.json
?? examples/forensics/integration/adverse/report.html
?? examples/forensics/integration/adverse/seal.json
?? examples/forensics/integration/adverse/source/frames/frame.npz
?? examples/forensics/integration/adverse/source/manifest.json
?? examples/forensics/integration/adverse/source/records.jsonl
?? examples/forensics/integration/adverse/summary.json
?? examples/forensics/integration/consistent/findings.md
?? examples/forensics/integration/consistent/implementation.json
?? examples/forensics/integration/consistent/report.html
?? examples/forensics/integration/consistent/seal.json
?? examples/forensics/integration/consistent/source/frames/frame.npz
?? examples/forensics/integration/consistent/source/manifest.json
?? examples/forensics/integration/consistent/source/records.jsonl
?? examples/forensics/integration/consistent/summary.json
?? examples/forensics/preflight/no-hardware/findings.md
?? examples/forensics/preflight/no-hardware/implementation.json
?? examples/forensics/preflight/no-hardware/report.html
?? examples/forensics/preflight/no-hardware/seal.json
?? examples/forensics/preflight/no-hardware/summary.json
?? examples/forensics/unknown-alignment/findings.md
?? examples/forensics/unknown-alignment/report.html
?? examples/forensics/unknown-alignment/summary.json
?? requirements-forensics.txt
?? src/aegisland/forensics/__init__.py
?? src/aegisland/forensics/__main__.py
?? src/aegisland/forensics/alignment.py
?? src/aegisland/forensics/analysis.py
?? src/aegisland/forensics/cli.py
?? src/aegisland/forensics/fixtures.py
?? src/aegisland/forensics/ingest.py
?? src/aegisland/forensics/integration.py
?? src/aegisland/forensics/integration_fixtures.py
?? src/aegisland/forensics/model.py
?? src/aegisland/forensics/platform_manifest.py
?? src/aegisland/forensics/recorder.py
?? src/aegisland/forensics/report.py
?? src/aegisland/forensics/scope.py
?? tests/fixtures/forensics/README.md
?? tests/fixtures/forensics/aligned-jetson.jsonl
?? tests/fixtures/forensics/aligned.ulg
?? tests/fixtures/forensics/alignment.json
?? tests/fixtures/forensics/corrupt.ulg
?? tests/fixtures/forensics/empty.jsonl
?? tests/fixtures/forensics/empty.ulg
?? tests/fixtures/forensics/integration-adverse/frames/frame.npz
?? tests/fixtures/forensics/integration-adverse/manifest.json
?? tests/fixtures/forensics/integration-adverse/records.jsonl
?? tests/fixtures/forensics/integration-consistent/frames/frame.npz
?? tests/fixtures/forensics/integration-consistent/manifest.json
?? tests/fixtures/forensics/integration-consistent/records.jsonl
?? tests/fixtures/forensics/invalid-jetson.jsonl
?? tests/fixtures/forensics/jetson.jsonl
?? tests/fixtures/forensics/missing-fields.ulg
?? tests/fixtures/forensics/nonfinite.ulg
?? tests/fixtures/forensics/sync-evidence.jsonl
?? tests/fixtures/forensics/synthetic.ulg
?? tests/fixtures/forensics/truncated.ulg
?? tests/forensics_replay_smoke.cjs
?? tests/test_forensics.py
?? tests/test_forensics_integration.py
```
