# GPS-denied benchmark: Stage 0 record and Stage 1 (cuVSLAM + EuRoC)

Primary brief: [`AegisLand-DroneHacks-Codex-Brief.md`](AegisLand-DroneHacks-Codex-Brief.md).
Branch `work/dronehacks-gps-denied-c`, based on `eda23d3`. Local only; push URL disabled.

Scope of this document: offline dataset replay. Nothing here is field-camera
integration, flight-ready localization, or a control path. PX4 stays read-only
shadow supervision (`CONTROL OUTPUT: DISABLED`, `COMMAND EXECUTED: FALSE`).

## 0. Performance measurement rule (binding from 2026-10-24)

On 2026-10-24 it was found that the `track()` "0.8 → 8 ms step" was GPU DVFS, not cuVSLAM:
- the GPU dropped from P2 / 2535 MHz to P5 / 210 MHz about 7 s into a batch replay;
- a replay paced at 20 fps never boosted in the first place.

Batch-replay timings therefore describe a boosted GPU, not live operation. From now on:

1. **Pace like the camera.** Every timing number that is reported comes from a run with `run_cuvslam_euroc.py --realtime` (frames paced to the dataset timestamps; 20 fps for EuRoC). Use `--realtime 2` only for explicitly labelled stress runs.
2. **Lock or record the GPU clock.**
   - Where possible, lock it: `jetson_clocks` plus a fixed `nvpmodel` on Orin; `nvidia-smi -lgc` with admin rights on a desktop.
   - Where that is not possible (WSL2, no admin), record it. The runner always writes `gpu_clocks.csv` (P-state, SM/mem clock, utilization and power every 500 ms) and summarizes it in `run.json` → `gpu_clocks`.
   - On Jetson, record `tegrastats` instead.
3. **`run.json` → `timing.valid_for_performance_claims` must be true** before a number goes into a report as performance. Batch runs are still fine for accuracy, ATE and determinism; their timings must not be quoted as performance.
4. Report p50/p95 together with the SM clock range of the run, and always say on which machine it was measured ("开发机实测" is not Orin).
5. Determinism: same-session repeats can be identical while runs in different sessions differ (MH_01 Stereo-Inertial gave 0.031, 0.040 and 0.0335 m). Report ≥3 runs from different sessions.

## 1. Stage 0: what exists today

### Development machine (measured 2026-10-24, not the competition Orin)

| Item | Value |
| --- | --- |
| GPU | RTX 4060 Ti 8 GB, SM 8.9, driver 610.60 (CUDA UMD 13.3) |
| OS | WSL2 kernel 6.18, Ubuntu 24.04.1, Python 3.12.3 |
| CPU / RAM / disk | 28 threads / 15 GB / ~945 GB free |
| CUDA toolkit | none system-wide (no `nvcc`); CUDA math libs come from pip in the isolated venv |
| Docker | not available in this WSL distro |

The competition Orin Nano (JetPack, memory, camera model, IMU) is still unconfirmed;
no number below is an Orin number.

### Current GPS-denied call chain

```text
agent.py  AegisLandAgent.step
  -> perception.observe(frame)                       src/aegisland/perception.py
       -> _motion(gray, reference_gray, update_visual_state=not retry)
            -> CameraMotionCompensator.compensate   ORB(1200) + BFMatcher + RANSAC homography
            -> VisualDeadReckoner.update            src/aegisland/gps_denied.py
               (retry pass calls .current(): no double integration; tested)
       -> VisionEvidence.visual_relative_x/y, visual_localization_confidence/valid
  -> PerceptionTrustGate.evaluate                    gates on quality + confidence
  -> SensorSynchronizer.add_visual(t, x, y, conf)    only if localization_trusted
  -> SensorHealthMonitor.assess("visual", ...)
  -> DynamicConfidenceFusion.fuse(...)               uses confidence/valid only, not x/y
  -> planner / safety decision (simulation) ; PX4 shadow monitor is read-only
```

### `VisualDeadReckoner` output: units and limits

- `vx = -dx_center / image_width`, `vy = -dy_center / image_height`, where `dx/dy` is the
  displacement of the image center under the frame-to-frame homography. `x/y` are the
  running sum: **normalized image units per frame, not meters**.
- No camera intrinsics, no depth, no scale, no rotation/translation separation (a yaw or
  pitch shows up as "translation"), no IMU, no z, no loop closure.
- `confidence` is the RANSAC inlier ratio (decays by 0.85 per failed frame). It is not a
  calibrated probability.
- Consequence: it can only be compared with a metric estimator through normalized,
  scale-free relative drift on the same recording, never through ATE in meters.

### Reusable mechanisms already in the repo

- `src/aegisland/profiling.py`: run directory + `environment.json` + input hashes +
  `SHA256SUMS`; reused here as a pattern (new code does not import it).
- `src/aegisland/trace.py`: JSONL trace store.
- Missing before this change: any trajectory loader, timestamp association, or ATE/RPE.

## 2. Stage 1: cuVSLAM

### Candidate check

| Check | Finding |
| --- | --- |
| Version | v17.0.1, released 2026-10-13, tag commit `51631ba` |
| License | NVIDIA Community License: commercial use OK, **only on NVIDIA hardware**; keep LICENSE + attribution if redistributing; do not vendor into AegisLand |
| Platforms | x86_64 Ubuntu 22.04/24.04; Orin with JetPack 6.x (cu12 wheel `cp310 aarch64`) |
| Dependencies | Python wheel only; no ROS, no Docker; CUDA libs (cuBLAS 13, cuSOLVER 12, cuSPARSE 12) |
| Dataset | EuRoC MAV, ETH Research Collection DOI 10.3929/ethz-b-000690084, **"In Copyright - Non-Commercial Use Permitted"**: never commit or redistribute images |
| Calibration | official example ships `sensor_recalibrated.yaml` (fisheye model, transforms relative to cam0) |
| Ground truth | `state_groundtruth_estimate0/data.csv`, body = imu0 frame, meters, quaternion wxyz |

### Isolated environment (does not touch the AegisLand venv)

```sh
mkdir -p ~/.cache/aegisland-dl/cuvslam-17.0.1 && cd ~/.cache/aegisland-dl/cuvslam-17.0.1
gh release download v17.0.1 -R nvidia-isaac/cuVSLAM \
  -p 'cuvslam-17.0.1+cu13-cp312-abi3-manylinux_2_39_x86_64.whl'
python3 -m venv ~/.venvs/cuvslam-17.0.1
~/.venvs/cuvslam-17.0.1/bin/pip install ./cuvslam-17.0.1+cu13-cp312-abi3-manylinux_2_39_x86_64.whl \
  "nvidia-cublas>=13,<14" "nvidia-cusolver>=12,<13" "nvidia-cusparse>=12,<13" \
  numpy==2.2.4 pillow==12.2.0 pyyaml==6.0.2 scipy==1.14.1

cd ~/.cache/aegisland-dl
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch v17.0.1 --filter=blob:none --sparse \
  https://github.com/nvidia-isaac/cuVSLAM.git cuVSLAM-src-v17.0.1
git -C cuVSLAM-src-v17.0.1 sparse-checkout set --no-cone /examples/euroc/ /LICENSE

# every shell that runs cuVSLAM:
export LD_LIBRARY_PATH=$HOME/.venvs/cuvslam-17.0.1/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/lib/wsl/lib
```

Exact versions: [`benchmarks/gps_denied/cuvslam-env.lock.txt`](../../benchmarks/gps_denied/cuvslam-env.lock.txt).

### Benchmark boundary

```text
recorded sequence (EuRoC ASL layout)          read-only input
   -> benchmarks/gps_denied/run_cuvslam_euroc.py   cuVSLAM venv, separate process
        trajectory_body.tum, frames.jsonl, run.json, SHA256SUMS   (new dir, never overwritten)
   -> benchmarks/gps_denied/evaluate_trajectory.py AegisLand venv
        uses src/aegisland/trajectory_eval.py (numpy only, evaluation-only)
        metrics.json, SHA256SUMS                    (new dir)
```

The runner imports the official `dataset_utils.py` from the pinned checkout and uses the
official configuration (`async_sba=False`, `rectified_stereo_camera=False`), minus the
Rerun viewer. No cuVSLAM code is copied into AegisLand and nothing feeds the safety core.

Frame conventions:

- cuVSLAM rig = cam0 (x right, y down, z forward); world = rig at the first frame.
- Output is converted to `world_from_body = world_from_rig @ rig_from_imu` so it is in the
  same body frame (imu0) as EuRoC ground truth.
- Timestamps: EuRoC integer ns -> seconds; association tolerance 5 ms by default.

Metric rules (enforced by `trajectory_eval.py` labels):

- `ate_se3`: rigid alignment, meters. This is the headline number only when the estimate is
  metric (stereo or stereo-inertial) and ground truth is metric.
- `ate_sim3_scale_diagnostic`: labelled "after scale correction; not a metric-scale result".
  Use its `scale` to detect scale error; never report its RMSE as metric accuracy.
- `rpe`: per `delta_frames` estimate frames; translation m, rotation deg.
- `VisualDeadReckoner` output must not be fed to ATE in meters.

### Results so far (measured this session)

Synthetic smoke sequence (textured plane, zero-distortion fisheye, 200 frames at 20 Hz,
baseline 0.11 m, known metric motion; generator `make_synthetic_euroc.py`, seed 7):

| Run | Tracked | ATE SE(3) RMSE | Sim(3) scale | RPE(20) trans / rot | warm `track()` p50 / p95 |
| --- | --- | --- | --- | --- | --- |
| run1 | 200/200 | 0.00998 m | 0.989 | 0.0074 m / 0.084 deg | 0.89 / 1.12 ms |
| run2 | 200/200 | 0.00998 m | 0.989 | 0.0074 m / 0.084 deg | 0.82 / 1.02 ms |

- run1 and run2 trajectories are byte-identical (same input, same config).
- First `track()` call: 18.2 ms (cold) vs <1.5 ms warm.
- Artifacts: `benchmarks/results/gps_denied/cuvslam-stereo-20261024-synthetic-s7-run{1,2}[-eval]/`.
- What this proves: install, GPU path, calibration parsing, frame/units conversion and the
  evaluator agree end to end; stereo scale is metric to ~1% on this easy scene.
- What it does **not** prove: anything about EuRoC, real cameras, motion blur, lighting,
  IMU fusion, or Orin performance. Synthetic extrinsics are identity, so the
  `rig_from_imu` conversion is only exercised by real EuRoC calibration.

### EuRoC download (manual)

ETH Research Collection rate-limits / bot-checks automated downloads from this machine
(HTTP 429/500); do not try to work around it. Only full archives are offered (Machine
Hall ZIP 12096.15 MB, Vicon Room 1 ZIP 5762.35 MB). The record page shows sizes but no
checksum, so integrity is checked by size plus the ZIP's per-entry CRC32
(`unzip -t` on the outer entry and on the inner `MH_01_easy.zip`).
Download in a browser, then:

```sh
mkdir -p ~/aegisland-data/euroc
cd ~/aegisland-data/euroc
unzip -j /mnt/c/Users/wangs/Downloads/machine_hall.zip 'machine_hall/MH_01_easy/MH_01_easy.zip' -d .
unzip -q MH_01_easy.zip -d MH_01_easy        # yields MH_01_easy/mav0
E=~/.cache/aegisland-dl/cuVSLAM-src-v17.0.1/examples/euroc
cp $E/sensor_cam0.yaml MH_01_easy/mav0/cam0/sensor_recalibrated.yaml
cp $E/sensor_cam1.yaml MH_01_easy/mav0/cam1/sensor_recalibrated.yaml
cp $E/sensor_imu0.yaml MH_01_easy/mav0/imu0/sensor_recalibrated.yaml

cd ~/aegisland-agentic-vision
R=benchmarks/results/gps_denied/cuvslam-stereo-MH_01_easy-$(date +%Y%m%d-%H%M%S)
~/.venvs/cuvslam-17.0.1/bin/python benchmarks/gps_denied/run_cuvslam_euroc.py \
  --sequence ~/aegisland-data/euroc/MH_01_easy/mav0 --cuvslam-examples $E \
  --mode stereo --max-frames 200 --output $R-smoke      # short smoke first
~/.venvs/cuvslam-17.0.1/bin/python benchmarks/gps_denied/run_cuvslam_euroc.py \
  --sequence ~/aegisland-data/euroc/MH_01_easy/mav0 --cuvslam-examples $E \
  --mode stereo --output $R
.venv/bin/python benchmarks/gps_denied/evaluate_trajectory.py \
  --estimate $R/trajectory_body.tum \
  --groundtruth ~/aegisland-data/euroc/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv \
  --output $R-eval
# then repeat with --mode inertial, and repeat each run once to check determinism
```

### VisualDeadReckoner vs cuVSLAM: scale-free comparison

VDR reports normalized image motion, so it is never converted to meters or scored with
ATE. `benchmarks/gps_denied/compare_motion_direction.py` compares both in image space:

1. **Direction consistency.** For each consecutive cam0 frame pair, the reference relative
   pose (EuRoC ground truth, and separately cuVSLAM) predicts where the scene point at the
   image center moves, through the cam0 fisheye model. VDR's own center motion is
   `(-vx * W, -vy * H)` pixels (the inverse of its normalization). Score: median cosine,
   fraction in the same half-plane, fraction within 45 degrees; magnitude ratio is
   informative only.
2. **Depth assumption made explicit.** Translation-induced flow scales with 1/depth, which
   VDR never knows. Results are reported for depth = inf (rotation only), 2, 5 and 10 m.
   cuVSLAM is scored by the same procedure as a ceiling for the metric itself.
3. **Failure behavior.** VDR valid fraction, VDR invalid fraction binned by ground-truth
   angular speed, cuVSLAM tracked fraction and lengths of untracked gaps.
4. **Clock alignment.** Cross-correlation of |gyro| (IMU clock) with angular speed
   differentiated from the cuVSLAM and GT trajectories; IMU samples per camera interval.
5. **Extrinsic chain.** `inv(cam0_from_imu)` from the recalibrated file vs the original
   EuRoC `cam0/sensor.yaml` `T_BS` (body_from_cam0): rotation and translation difference.

Limits: direction agreement says nothing about accumulated drift or position error; one
homography on a distorted fisheye image is part of what is measured; small motions
(< 1 px predicted) are skipped; single sequence.

Synthetic check (seed 7, plane at 4 m, measured this session; real EuRoC pending):

| depth assumption | VDR vs GT cos median (used) | cuVSLAM vs GT cos median |
| --- | --- | --- |
| inf (rotation only) | no frames above 1 px | no frames above 1 px |
| 2 m | 0.995 (199) | 0.99995 |
| 5 m | 0.998 (137) | 0.99999 |
| 10 m | 0.985 (103) | 0.9999 |

VDR valid 199/200 frames (first frame has no reference); compensate + update p50 23.0 ms,
p95 30.0 ms on 1 OpenCV thread. Artifacts:
`benchmarks/results/gps_denied/vdr-20261024-synthetic-s7-run1/`,
`.../compare-vdr-cuvslam-20261024-synthetic-s7/`.

## 3. Not in this stage

- Field camera, OAK/Luxonis driver, Orin deployment, any PX4 write path or external-vision
  injection (Stage 3 is read-only/SITL first).
- Wiring cuVSLAM into `perception.py`, fusion, or the planner.
- Other SLAM frameworks (ORB-SLAM3, OpenVINS, ...): one candidate first, per the brief.
- Fast-FoundationStereo (Stage 2), vLLM/SGLang, Kimi, training.
- Copying code or infrastructure from `aegisland-opencv-2026` / `aegisland-nebius-2026`.
- Claiming VisualDeadReckoner is VIO/SLAM or metric.

## 4. Open questions that block later steps

| Unknown | Blocks |
| --- | --- |
| Exact Luxonis model, IMU presence/rate, global vs rolling shutter | Stereo-inertial on real hardware, calibration plan |
| Orin Nano memory + JetPack version | Choosing cu12 aarch64 wheel; Orin timing |
| Scoring rules / arena task | What accuracy and latency actually matter |
| PX4 firmware version on the kit | External-vision message format (Stage 3) |
| Rules on pre-written code | How much of this harness may be used at the event |
