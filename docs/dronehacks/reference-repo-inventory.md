# Reference repositories: what can serve DroneHacks stages 1-3

Inspected 2026-10-24, read-only. No code from either repository is merged into the core.

| Source | Where | State |
| --- | --- | --- |
| `aegisland-opencv-2026` public | `~/work/aegisland-opencv-2026`, `4ba11c1` | README only; nothing to reuse |
| `aegisland-opencv-2026` local | `~/aegisland-opencv-2026`, branch `codex-a/stage-a1-baseline` | almost everything is **uncommitted** (`cloud/`, `scripts/`, `tests/`, `evidence/`, `approval/`, `submission/`); a Codex process was still running there, so tests were run on a copy |
| `aegisland-nebius-2026` public | `~/work/aegisland-nebius-2026`, `508238f` | main; branch `codex-b/stage-b1-nebius` also on remote |
| `aegisland-nebius-2026` local | `~/aegisland-nebius-2026`, `codex-b/stage-b1-nebius` @ `a668acf` | 7 modified submission docs + untracked media/handoff |

Tests run this session on snapshot copies (not in the original directories), with
credentials unset and HTTP(S) proxy pointed at a dead port so any accidental network call
would fail instead of reaching AWS or Nebius:

- opencv local: **121 passed** (`pytest -q`, 0.75 s). Matches its own handoff claim.
- nebius local: **74 tests OK** (`python -m unittest discover -s tests`, 0.18 s).

Both suites test cloud/packaging/review logic. Neither tests localization, depth,
calibration or GPU code. "Maturity" below means "has tests that passed here", nothing more.

## Inventory

| Module | Path | Maturity (tested here) | Serves stage | Reuse? | What would have to change |
| --- | --- | --- | --- | --- | --- |
| Core pin + clean-tree check | opencv-local `scripts/competition_demo.py` `check_core`, `core-pin.json` | 5 tests in `test_competition_demo.py` pass (wrong commit, dirty core, bad core creates no output, existing output preserved, output cannot change core) | 1, 3, 4 | **Pattern yes** | Add `git rev-parse HEAD` + `status --porcelain` of the core into `run_cuvslam_euroc.py`'s `run.json`; reject dirty tree only in "release" mode so day-to-day dev still works |
| Command recorder | opencv-local `scripts/record_command.py` | No dedicated test; used to produce its evidence JSONs | 1, 4 | **Pattern yes** (≈30 lines) | Exclusive-create (`"x"`) JSON with argv, cwd, UTC start/end, exit code, stdout/stderr. Useful for test-suite and benchmark evidence. Needs a license decision first (see below) or a clean rewrite |
| Decision parity check | opencv-local `evidence/hardening/decision-and-hash-parity.json` (+ script in `scripts/build_hardening_submission.py`) | Recorded result: 80 decisions identical | 3 | **Idea yes** | When a localization message is added to the shadow path, replay the 6 core scenarios and require identical decision dictionaries vs `eda23d3`. Must be rewritten against the core's own `simulator`, not the competition harness |
| Artifact validation (trace count = summary = decoded video frames; `hardware_command_sent is False`) | opencv-local `competition_demo.validate_artifacts` | Covered indirectly by the 5 demo tests | 3 | **Pattern yes** | Same idea for localization replay: frames.jsonl count = stereo frames; every record `control_output == "DISABLED"` |
| Versioned bounded packet + ACK | opencv-local `cloud/competition_cloud/protocol.py` | `test_storage_adversarial.py` etc. pass (missing fields rejected, nested JSON rejected, truncated bodies never indexed) | 3 | **Design reference only** | The *contract style* (protocol string, required fields, size bounds, reject-before-use) fits a `localization-v1` message. The S3/DynamoDB/ACK part does not belong in a flight stack |
| S3/DynamoDB storage, Lambda, SAM, IAM, quota scripts, approval plans | opencv-local `cloud/`, `infra/`, `approval/`, `scripts/aws_*`, `request_lambda_quota_102.py` | tests pass (stubbed) | none | **No** | Competition-specific cloud infrastructure; brief says do not copy it |
| Submission packaging / archive verification | opencv-local `scripts/build_submission_package.py`, `verify_submission_archive.py` | tests pass | 4 (freeze) | **Maybe later** | Only the "manifest + checksum + independent re-verify" idea, at freeze time |
| Strict AegisLand trace reader | nebius `trace_reader.py`, `docs/TRACE_READER_CONTRACT.md` | 74 unittest pass (includes reader tests); stdlib only; MIT | 3 | **Yes, as an offline tool** | Validates the core's JSONL trace (ranges, required fields, separate decision/command/execution counters, never maps unknown to success). Could run as a standalone check on replay traces. Would need new fields once localization messages exist; copy only with MIT notice |
| Deterministic no-model baseline | nebius `model_evaluation.py` `BaselineProvider` | covered by tests | — | **No** | Specific to LLM review comparison |
| LLM evidence review | nebius `nebius_review.py`, `live_smoke*.py` | tests pass with mocks; live calls are paid and guarded by `--allow-paid-api` | none | **No** | Paid API, unverified grounded review; out of scope |
| Demo runbook (hash-before/after, no overwrite) | nebius `scripts/demo_runbook.py` | not separately tested | 4 | **Idea only** | Same "refuse existing output, hash inputs" policy already used by the new runner |

## Nothing found for

Stereo depth, VIO/SLAM, camera/IMU calibration, time synchronization, GPU profiling,
or PX4 integration. Those exist only in the core (`sensor_sync.py`, `time_sync.py`,
`px4/`, `profiling.py`, `cuda_gray.py`) or are new work.

## Licensing notes

- nebius: MIT (`LICENSE`, "AegisLand contributors"); copying requires keeping the notice.
- opencv local: **no LICENSE file** in the working tree. Even though it is your own code,
  decide its license before copying text from it into the Apache-2.0 core; rewriting the
  ~10-line patterns from scratch avoids the question.

## Recommended order (only after stage 1 runs on real EuRoC)

1. Add core commit + dirty flag to `run.json` (small, in-core, no copying).
2. Stage 3: define `localization-v1` message with explicit frame/unit/validity fields,
   validate it the way `protocol.validate` does, and gate it with a decision-parity replay.
3. Optionally run nebius `trace_reader` as an external checker on replay traces.
