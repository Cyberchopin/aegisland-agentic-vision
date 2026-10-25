# Stage 3 draft: `aegisland.localization.v1` and shadow supervision

Code: `src/aegisland/localization_message.py`, `localization_supervisor.py`,
`localization_faults.py`, replay tool `benchmarks/gps_denied/fault_replay.py`.
Shadow only: every assessment carries `control_output="DISABLED"`,
`command_executed=False`. Nothing here sends MAVLink or touches PX4.

## Message fields

| Field | Meaning | Rule |
| --- | --- | --- |
| `schema` | `aegisland.localization.v1` | exact match; `from_dict` rejects others |
| `source` | producer + version, e.g. `cuvslam-17.0.1` | free text |
| `sequence` | producer frame counter | non-negative; gaps = dropped frames |
| `frame_id` | world frame: `odom_frd`, `odom_ned`, or `odom_dataset` | `*_dataset` = recording's own frame, offline only, never for PX4 |
| `child_frame_id` | body frame: `body_frd` or `body_dataset` | |
| `clock_domain` | `host_monotonic`, `device_monotonic`, `px4_boot`, `dataset` | both timestamps share it |
| `capture_time_ns` | sensor exposure time (what PX4 calls `timestamp_sample`) | must strictly increase |
| `publish_time_ns` | when the pose was produced | `>= capture_time_ns`; difference = processing latency |
| `position_m` | world_from_body translation, meters | finite |
| `orientation_xyzw` | world_from_body unit quaternion (Hamilton, xyzw) | norm within 1e-3 |
| `velocity_mps` | optional, body or world per frame names | finite or null |
| `position_variance_m2` | optional diagonal variance | non-negative or null; never invented |
| `quality` | -1 failed, 0 unknown, 1..100 ordinal health | same semantics as MAVLink ODOMETRY `quality`; **not a probability** |
| `valid` | producer believes pose usable | `valid` is incompatible with quality -1 or state lost/initializing |
| `tracking_state` | `initializing`, `tracking`, `lost`, `relocalized` | |
| `reset_counter` | increments on every re-init / relocalization jump | same semantics as ODOMETRY `reset_counter` |

Mapping to PX4 later (not implemented): `estimator_type=VIO`, `frame_id=MAV_FRAME_LOCAL_FRD`,
`child_frame_id=MAV_FRAME_BODY_FRD` (BODY_FLU velocity is silently dropped by PX4 main),
`quality` with `EKF2_EV_QMIN >= 1` (QMIN 0 fuses even quality -1). See `hardware-research.md` §3.

## Data expiry and fault rules (defaults are starting points, not tuned)

| Reason | Status | Default | Rationale |
| --- | --- | --- | --- |
| `contract:*` | fault | — | message cannot be interpreted |
| `stale` | fault | now - capture > 200 ms | mirrors PX4 `EV_MAX_INTERVAL` |
| `no_messages` (tick) | fault | no message for 400 ms | mirrors PX4 2 x `EV_MAX_INTERVAL` stop |
| `capture_time_not_increasing` | fault | — | clock or ordering bug |
| `tracking_lost` | fault | invalid / quality -1 / lost | |
| `pose_jump` | fault | > 5 m/s or > 10 rad/s implied, no reset | indoor 7" drone limits; tune per arena |
| `imu_time_offset` | fault | |offset| > 20 ms, corr >= 0.6, motion std >= 0.05 rad/s, 5 s window | |gyro| vs pose angular speed cross-correlation |
| `dropped_frames` | degraded | sequence gap > 1 | |
| `frame_interval_gap` | degraded | capture gap > 1.5 periods | |
| `relocalized` | degraded | reset_counter changed | jump expected; not a fault |

## Fault replay (synthetic, measured this session)

`fault_replay.py` on `cuvslam-stereo-20261024-synthetic-s7-run1` (200 frames, no IMU):
every fault detected, baseline 0 false alarms, 0 alarms outside fault windows.
Latency from fault start: pose_jump 0.8 ms, out_of_order 51 ms, dropped_frames 251 ms
(first frame after the gap), stale_delivery 301 ms (= injected delay), camera_outage 375 ms.
EuRoC results (with real IMU for the IMU-offset case) are recorded in the overnight report.

## Known limits

- Thresholds are not tuned on flight data.
- `imu_time_offset` needs rotation; in hover it cannot be observed (by design it stays silent).
- Pose jump uses a constant limit; a slow drift is invisible to it (needs ground truth or loop closure).
- Supervisor trusts the producer's `sequence`; a producer that renumbers after drops hides them.

## Update 2026-10-24: supervisor v3 and the MAVLink ODOMETRY mapping

- **v3 adds two degraded-level reasons**:
  - `rotation_inconsistent`: visual angular velocity minus gyro, bias-tracked, 0.5 s mean > 0.014 rad/s;
  - `low_texture`: VDR feature matches, 0.5 s mean < 597, passed through `assess(..., feature_matches=)`.
- Both thresholds were fitted on EuRoC MH_01/03/04 and checked on MH_02/05. Evidence and caveats are in `degradation-study.md`.
- Fault logic is unchanged: v2 and v3 fault scoring is identical on all 10 replays.
- **Mapping to ODOMETRY** (`src/aegisland/mavlink_odometry.py`):
  - Frames: `odom_frd`/`odom_ned` map to LOCAL_FRD/LOCAL_NED; `body_frd` maps to BODY_FRD; estimator type is VIO.
  - Verdict "degraded": the position variance is raised to at least 1 m².
  - Verdict "untrusted": quality 0. This needs `EKF2_EV_QMIN >= 1` on the vehicle.
  - Tracking lost: quality -1.
- SITL evidence for each of these choices is in `sitl-ekf2-ev-validation.md`. SITL only; nothing is sent to hardware.
