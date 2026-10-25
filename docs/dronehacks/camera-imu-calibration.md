# Camera–IMU calibration procedure (OAK stereo + IMU → cuVSLAM)

**For:** Chelsea (electronics & integration), Darion (mechanical & mounting)
**Status:** draft, 2026-10-24.
- The software side has been tested on EuRoC data and on synthetic Kalibr files.
- **No OAK device and no real Kalibr run have been done yet.**
- Steps marked ⚠️ depend on the exact camera model, which is still unconfirmed.

## 0. What we need at the end

| Quantity | Why | Produced by |
|---|---|---|
| Intrinsics of left and right camera (focal, principal point, 4 distortion coefficients) | cuVSLAM undistortion | Kalibr camera calibration |
| Stereo extrinsics `cam0_from_cam1` | stereo depth / scale | Kalibr camera calibration |
| Camera–IMU extrinsics `cam0_from_imu` | stereo-inertial mode; supervisor gyro check | Kalibr camera–IMU calibration |
| Camera–IMU **time offset** | IMU and images must line up within ~1 ms | Kalibr camera–IMU calibration |
| IMU noise densities and random walks | stereo-inertial weighting | Allan variance (or datasheet) |

The OAK factory calibration (EEPROM) is what `oak_record.py` writes by default. Treat it as a **starting point only**:
- It keeps only 4 of up to 14 distortion coefficients.
- The direction of its IMU↔camera transform still has to be confirmed against Kalibr.
- It has no time offset.

## 1. Tools

- **Kalibr**, the ethz-asl/kalibr repository on GitHub.
  - Use its Docker image (ROS1 Noetic) on a laptop, not on the drone.
  - Commands used: `kalibr_bagcreater`, `kalibr_calibrate_cameras`, `kalibr_calibrate_imu_camera`.
- **Allan variance:** `allan_variance_ros` (ori-drs, GitHub), or any Allan-deviation script on a static IMU log.
- **AegisLand helpers** in `benchmarks/gps_denied/`:
  - `oak_record.py` records into the EuRoC layout;
  - `kalibr_tools.py to-kalibr` makes a folder for `kalibr_bagcreater`;
  - `kalibr_tools.py from-kalibr` turns Kalibr results into cuVSLAM yaml files and applies the time offset;
  - `run_cuvslam_euroc.py` is used for validation.

## 2. Calibration target

- **AprilGrid 6×6**, tag size 0.088 m, spacing ratio 0.3 (the Kalibr default). Print it on A1/A0 at 100 % scale.
- Print on **matte** paper and mount it on something **flat and rigid** (aluminium composite or thick foam board). The board must not bend.
- **Measure the printed tag size with calipers** and put the measured value into `aprilgrid.yaml`:
  ```yaml
  target_type: 'aprilgrid'
  tagCols: 6
  tagRows: 6
  tagSize: 0.088      # measured, metres
  tagSpacing: 0.3
  ```
- Lighting: even and bright. That way the exposure can stay short (see below).

## 3. Camera settings during all calibration recordings

- Use the **same resolution and frame rate as in flight**. Intrinsics change with resolution and binning: 640×400 and 1280×800 on OV9282 are different calibrations ⚠️.
- **Fixed, short exposure** (≤ 5 ms) and fixed gain, to keep motion blur below ~1 px.
- IR dot projector **off** (Pro models ⚠️). `oak_record.py` forces it off.
- IMU at 200 Hz, recorded together with the images on the device clock. This is the `oak_record.py` default.

## 4. Recordings (three of them, each in its own new directory)

All three use `oak_record.py --output <new dir> --duration-s <T>`.

| # | What | Duration | How to move |
|---|---|---|---|
| R1 | Cameras only | 60–90 s | Hold the camera still and **move the board**, or the reverse. Slowly cover the whole image, especially the corners. Vary distance (0.5–2 m) and tilt (±45° in all directions). |
| R2 | Camera + IMU | 60–120 s | The board is **fixed on a wall**; move the camera rig. Start and end with 5 s still. In between, excite **all 6 axes** in turn: roll, pitch, yaw rotations and x, y, z translations. Be smooth but energetic (~1–2 Hz oscillations). Always keep the board fully in view, and avoid blur. |
| R3 | IMU noise | ≥ 2 h (3 h better) | Rig completely still on a vibration-free surface, at stable temperature. Nothing else running on the bench. |

Repeat R2 once more on another day, or after remounting. The two results are compared in §6.

## 5. Running Kalibr

```bash
# 1. EuRoC recording -> Kalibr folder -> rosbag (inside the Kalibr container)
python benchmarks/gps_denied/kalibr_tools.py to-kalibr --mav0 R1/mav0 --output R1-kalibr
kalibr_bagcreater --folder R1-kalibr --output-bag R1.bag

# 2. cameras (R1). Model: pinhole-radtan for normal lenses, pinhole-equi for wide/fisheye (OAK W) ⚠️
kalibr_calibrate_cameras --bag R1.bag --target aprilgrid.yaml \
  --topics /cam0/image_raw /cam1/image_raw --models pinhole-radtan pinhole-radtan

# 3. IMU noise (R3): Allan variance -> imu.yaml (noise densities, random walks, update_rate: 200)
#    Kalibr guidance: inflate the Allan values (x2..x10) before use.

# 4. camera-IMU (R2); estimates extrinsics and the time offset
kalibr_bagcreater --folder R2-kalibr --output-bag R2.bag
kalibr_calibrate_imu_camera --bag R2.bag --target aprilgrid.yaml \
  --cam camchain-R1.yaml --imu imu.yaml
```

## 6. Acceptance checks (all must hold; write the numbers into `notes.md`)

| Check | Pass if | Why |
|---|---|---|
| Camera reprojection error (R1) | RMS < 0.3 px; no structure in the error plot (e.g. larger at the edges) | wrong lens model or a bent board shows up here |
| Stereo baseline | within 1 mm of the datasheet (e.g. 75 mm) ⚠️ | gross error check |
| Camera–IMU reprojection (R2) | RMS < 0.5 px | — |
| Gyro / accel residuals (R2) | look like white noise; no periodic pattern | vibration or IMU clipping |
| `T_cam_imu` vs CAD (Darion's numbers) | translation within ~5 mm, rotation within ~1° | EuRoC's two published calibrations differ by 0.36° / 7.6 mm; much larger means an error |
| Time offset | \|shift\| < 5 ms, and two R2 runs agree within 1 ms | a *changing* offset breaks VIO (§4 of the interface spec) |
| Repeatability | two R2 runs agree within 2 mm / 0.2° | — |

## 7. Converting to cuVSLAM and checking in our pipeline

```bash
# Kalibr results -> NEW recording copy with cuVSLAM yaml files and time-corrected IMU stamps
python benchmarks/gps_denied/kalibr_tools.py from-kalibr --mav0 FLIGHT/mav0 \
  --camchain camchain-imucam-R2.yaml --imu imu.yaml --output FLIGHT-kalibr
```

Conventions the converter applies (unit-tested in `tests/test_kalibr_import.py`):
- `cam0.T_cam_imu` is `cam0_from_imu`.
- `cam1.T_cn_cnm1` is `cam1_from_cam0`; the converter inverts it.
- `timeshift_cam_imu` means `t_imu = t_cam + shift`, so the IMU stamps are rewritten as `t_imu − shift`.
- The output file name depends on the lens model:
  - pinhole-equi → `sensor_recalibrated.yaml` (fisheye, cam0-relative);
  - pinhole-radtan → `sensor.yaml` (Brown, body = IMU-relative).

  **cuVSLAM's EuRoC loader picks the distortion model from the file name**, so this matters.

Then validate on a separate recording, not the calibration one:
1. `run_cuvslam_euroc.py --mode stereo` and `--mode inertial`: 0 untracked frames while walking the rig around.
2. Fault replay / supervisor:
   - The `imu_time_offset` estimate should be within ±3 ms; EuRoC gives ±3 ms.
   - The rotation-consistency residual median should be well below the v3 threshold. EuRoC's threshold is 0.014 rad/s; it is re-fitted for our camera.
3. Walk a closed loop (start = end) and compare the start-to-end gap between stereo and stereo-inertial modes.

## 8. When to recalibrate

- After any remount, any impact, or a lens or focus change. Also when the resolution or binning changes.
- When check 2 above drifts (time offset or rotation residual).
- Keep every raw recording (R1–R3) and every Kalibr report PDF in the team drive, next to `notes.md`.
