# AegisLand GPS-denied localization — sensor, mounting, timing and logging interface

**For:** Chelsea (electronics & integration), Darion (mechanical & mounting)
**From:** AegisLand software (Shiori Wang)
**Status:** draft, 2026-10-24. Based on offline EuRoC replays and PX4 SITL only. Nothing here has been tested on our hardware yet.

## 0. Scope and safety boundary

- The localization stack (cuVSLAM stereo / stereo-inertial odometry + AegisLand shadow supervisor) currently runs **offline and in SITL only**.
- It sends **no commands**. CONTROL OUTPUT = DISABLED. A MAVLink ODOMETRY bridge exists, but it has only been run against PX4 SITL on localhost.
- Whether vision is ever fused on the real vehicle is a separate team decision. It needs bench and tethered tests first.
- This document lists what software needs from the hardware so that any later decision can be made on real data.

## 1. What we need from each of you (summary)

| # | Item | Owner | Format | Why |
|---|------|-------|--------|-----|
| M1 | Camera + IMU mounting drawing with axes | Darion | PDF/PNG + CAD export (STEP) | extrinsic seed values, sanity check of calibration |
| M2 | `T_body_cam0`, `T_body_imu` as numbers | Darion | YAML (template §3) | frame conversion to PX4 FRD |
| M3 | Rigidity / vibration isolation design | Darion | short note + photos | VIO fails on flex and vibration |
| M4 | Field-of-view obstruction check | Darion | photo from each camera, props/legs visible? | masks, mounting changes |
| E1 | Camera + IMU part numbers, firmware versions | Chelsea | text | calibration model, timestamp behaviour |
| E2 | Time-sync design (camera ↔ IMU ↔ Jetson ↔ flight controller) | Chelsea | block diagram | timestamps must share one clock (§4) |
| E3 | Jetson ↔ flight-controller link | Chelsea | port, baud rate, MAVLink instance | ODOMETRY rate and TIMESYNC (§4.3) |
| E4 | Power / USB bandwidth budget for the camera | Chelsea | numbers | frame drops show up as localization faults |
| C1 | Calibration recordings (§5) | together | raw recordings in the layout of §6 | intrinsics, extrinsics, camera–IMU time offset |
| L1 | One bench recording + matching PX4 ulog | together | §6 | first end-to-end replay on our hardware |

## 2. Camera and IMU requirements

These requirements come from what our evaluation showed matters, not from a datasheet wish list.

### Camera (stereo)
- **Global shutter**, stereo pair **hardware-synchronized**.
  - cuVSLAM requires left/right timestamps within 1 ms.
  - EuRoC MH_04 has one unpaired left frame, which the official loader rejects outright.
  - We now pair frames by timestamp and log every unpaired frame, so please do not "fix" mismatches silently in firmware.
- Resolution: VGA (640×480) or higher. Frame rate: 20–30 Hz. EuRoC is 752×480 at 20 Hz, and all our numbers so far come from that.
- **Exposure:** prefer a fixed or bounded auto-exposure, and **log exposure time and gain per frame** if the camera exposes them.
  - Our degradation study found the worst error growth right after a dark segment in EuRoC MH_05.
  - In that segment mean brightness dropped to about 14/255 and error rose to 0.21 m.
- **IR dot projector OFF** while doing visual odometry. The projected pattern moves with the camera and creates false features.
- Lens: the field of view must stay clear of propellers and landing gear (see M4). If something must be visible, tell us; it will need an image mask.

### IMU
- Rate ≥ 200 Hz. EuRoC is 200 Hz; the stereo-inertial mode and the supervisor's rotation-consistency check both use it.
- It must be on the **same rigid body** as the cameras. The supervisor compares camera-derived angular velocity with the gyro; any flex between the two shows up as a false "rotation inconsistent" warning.
- We need the noise parameters: gyro/accel noise density and random walk. They can come from the datasheet, or better, from a 2–3 hour static recording for Allan variance.
- Range and saturation: please tell us the configured gyro and accel full-scale ranges. Clipping during vibration is a known failure mode.

## 3. Mounting information (Darion)

Please give us the following. Units: metres and degrees. Frames: right-handed.

```yaml
# All transforms: T_A_B maps a point in frame B to frame A (A_from_B).
body_frame: "FRD at vehicle centre of gravity (x forward, y right, z down)"
T_body_imu:            # IMU frame as printed on the chip / module axes drawing
  translation_m: [x, y, z]
  rotation_rpy_deg: [roll, pitch, yaw]   # or a 3x3 matrix; say which convention
T_body_cam0:           # cam0 = LEFT camera, optical frame: z out of lens, x right, y down
  translation_m: [x, y, z]
  rotation_rpy_deg: [roll, pitch, yaw]
stereo_baseline_mm: 75           # nominal, from the camera datasheet
camera_tilt_down_deg: 0          # if the camera looks down at an angle
height_above_ground_on_skids_m: 0.0
source: "CAD rev X / measured with calipers"
uncertainty: "translation ±2 mm, rotation ±1 deg"
```

Why the numbers matter even though calibration will refine them:
- Calibration estimates are only trusted if they agree with CAD to within a few millimetres and a degree or so. For comparison, EuRoC's two published calibrations differ by 0.36° and 7.6 mm.
- PX4 needs the body-frame (FRD) pose. We convert the camera/IMU frame to FRD explicitly. In SITL, a sign or axis mistake shows up directly as a wrong heading (§4.4).
- The lever arm from IMU to CG is a PX4 parameter (`EKF2_EV_POS_X/Y/Z`).

Mechanical asks:
- Cameras and IMU on **one stiff bracket**. No flex under arm vibration or during hard landings.
- Vibration isolation for the IMU/compute stack. Please share the isolator part number. PX4's vibration metric (Flight Review) gives the green/orange/red thresholds: < 4.9, 4.9–9.8, > 9.8 m/s².
- Re-mounting invalidates the calibration. Please mark the screws, or tell us whenever something is moved.

## 4. Time synchronization (Chelsea)

### 4.1 One clock for camera and IMU
- Camera frames and IMU samples must be stamped by **the same device clock** (e.g., the OAK device clock) and then mapped to the Jetson's monotonic clock.
- Stamps should mark mid-exposure if the device supports it; otherwise, tell us which point the stamp refers to.
- A constant camera–IMU offset is calibrated (§5). What breaks VIO is an offset that **changes**: stamps taken at USB arrival, buffering jitter, dropped IMU packets.
- The supervisor watches for this. It flags a camera–IMU offset above 20 ms. On EuRoC, the measured offset is within ±3 ms.

### 4.2 Latency budget
- Measured on the development GPU, not on Orin:
  - capture → pose is dominated by image decode plus `track()`;
  - `track()` p50 is 0.7–6 ms on an RTX 4060 Ti.
- The supervisor treats a pose older than **200 ms** as stale and no data for **400 ms** as an outage. These are the same values as PX4 EKF2 (`EV_MAX_INTERVAL` and 2×).

### 4.3 Jetson ↔ flight controller: TIMESYNC is mandatory
This was verified in PX4 v1.17.0 source and reproduced in SITL:
- PX4 maps ODOMETRY `time_usec` onto its own clock **only after MAVLink TIMESYNC has converged**.
- Convergence needs 500 accepted round trips, each with RTT < 10 ms. In our SITL run that took about **70 s**.
- **Until then, PX4 silently replaces our capture time with the arrival time.**
  - In SITL, a 150 ms capture latency became 0 ms without TIMESYNC.
  - With TIMESYNC converged, PX4 kept a latency of about 100 ms. The remaining gap is a SITL clock-rate artifact (see the validation note).
- Consequences for integration:
  - the companion must answer PX4's TIMESYNC requests;
  - the link must have low, stable RTT (UART at ≥ 921600 baud or Ethernet; not a congested USB hub);
  - allow about a minute after boot before vision data is trusted.
- Please tell us which serial port / MAVLink instance is reserved for the companion computer.

### 4.4 Frames we will send (if fusion is ever enabled)

| Field | Value |
|---|---|
| Message | MAVLink `ODOMETRY` (#331) |
| `frame_id` | `MAV_FRAME_LOCAL_FRD` |
| `child_frame_id` | `MAV_FRAME_BODY_FRD` |
| `estimator_type` | `MAV_ESTIMATOR_TYPE_VIO` |
| Rate | 20–30 Hz |

- **Not** ENU/FLU. PX4 converts ENU positions but **does not convert the quaternion**. In SITL, sending a correct ENU attitude made PX4 read the heading as 50° instead of 40°.

## 5. Calibration data (together)

| What | How | Output |
|------|-----|--------|
| Camera intrinsics + stereo extrinsics | Kalibr (or OpenCV) with an AprilGrid; slow motion; cover the whole image | Kalibr `camchain.yaml` |
| Camera–IMU extrinsics + time offset | Kalibr `kalibr_calibrate_imu_camera`; excite all axes; 60–90 s | `camchain-imucam.yaml` |
| IMU noise | datasheet, or Allan variance from a ≥ 2 h static recording | `imu.yaml` |

- Camera model: tell us whether the lens is closer to pinhole or fisheye. EuRoC uses an equidistant/fisheye model, and cuVSLAM supports both.
- Please keep the **raw calibration recordings**, not only the YAML. We re-run calibration when results look wrong.

## 6. Recording / log format

Record in the **EuRoC/ASL layout**. All our tools run on it unchanged.

```
<recording>/mav0/
  cam0/data.csv            # timestamp_ns,filename      (left)
  cam0/data/<ts>.png       # 8-bit grayscale, lossless
  cam1/data.csv, cam1/data/ (right)
  imu0/data.csv            # timestamp_ns,wx,wy,wz[rad/s],ax,ay,az[m/s^2]
  cam0/sensor.yaml, cam1/sensor.yaml, imu0/sensor.yaml   # from §5
  exposure.csv             # optional: timestamp_ns,exposure_us,gain
notes.md                   # date, operator, hardware rev, what was done
```

- All timestamps are integer nanoseconds on **one** clock (§4.1).
- Do not resample, undistort or compress lossily.
- Record the **PX4 ulog** of the same session. Set `SDLOG_PROFILE` to include "estimator replay", so that `vehicle_visual_odometry`, `sensor_combined` and `estimator_aid_src_ev_*` are logged.
- Our outputs live in a new directory per run: `trajectory_body.tum`, `frames.jsonl`, `run.json` and `SHA256SUMS`. Old runs are never overwritten.

## 7. Open questions for you

1. Which exact camera module and IMU, and is the IMU inside the camera housing or separate? (E1)
2. Does the camera stamp frames on its own clock, and is the IMU on the same clock? (E2)
3. Which flight-controller port is free for the companion, and what is its baud rate? (E3)
4. Can the camera be mounted on the same plate as the flight controller IMU, or will it be on a separate bracket? (M1, M3)
5. Is anything (propellers, legs) visible in the camera image at the planned mount position? (M4)
