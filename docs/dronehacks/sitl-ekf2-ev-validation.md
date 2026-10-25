# PX4 SITL：把 AegisLand 定位消息转成 MAVLink ODOMETRY 喂给 EKF2（只回放）

日期：2026-10-24。**只在 PX4 SITL 中进行，飞控是仿真进程，所有 MAVLink 只走 127.0.0.1。** 载具从不解锁；我们只发送 HEARTBEAT（companion 身份）、TIMESYNC 应答和 ODOMETRY；没有发送任何命令、setpoint 或模式切换。CONTROL OUTPUT = DISABLED。

## 环境（为什么选这个）

- **PX4 v1.17.0**：最新稳定 tag；tag object `a5eb12d2`，commit `d6f12ad1c4f7`。
  - 使用 shallow clone 加 shallow submodules，占 1.8 GB，放在 `~/work/px4/`，不进 git。
- **SIH（Simulation-In-Hardware）**，机型 `10040_sihsim_quadx`：
  - 物理模型在 PX4 进程内运行，**不需要安装 Gazebo 或任何仿真器**；
  - 编译依赖都用 pip 装进 `~/.venvs/px4`（300 MB，cmake<4 + ninja + empy 3.3.4 等），不需要 sudo；
  - `make px4_sitl_default` 在 28 核上编译 28 s。
- WSL2 上可行。启动日志显示 "MAVLink only on localhost"，网卡上没有广播。
- 工具：
  - `benchmarks/gps_denied/sitl_ev_replay.py`，每个场景启动一个全新的 PX4（独立 rootfs、参数和 ulog）；
  - 转换逻辑在 `src/aegisland/mavlink_odometry.py`（纯函数，有单元测试）。
- 结果：
  - 摘要在 `benchmarks/results/gps_denied/sitl-ev-replay-px4-v1.17.0-20261024-103714/summary.json`；
  - 40 MB 的 ulog 留在 `~/work/px4/runs/` 下，SHA256 已记录。
- 公共参数：
  - `EKF2_GPS_CTRL=0`（不用 GPS）、`EKF2_MAG_TYPE=5`（不用磁力计）、`EKF2_EV_CTRL=15`、`EKF2_HGT_REF=3`（vision）；
  - `EKF2_EV_DELAY=0`、`EKF2_EV_NOISE_MD=0`、`EKF2_EV_QMIN=0`；
  - 每个场景都从 ulog 读回参数，确认生效。

**最重要的限制：** SIH 里的载具静止在地面，它的 IMU 说"没动"。所以只能喂"静止一致"的位姿：
- 能验证坐标系、时间戳、质量和方差字段的语义，以及门控和重置的行为；
- **不能**验证运动中的融合精度。
- 回放 EuRoC 的运动段只会测到"IMU 和视觉不一致"，所以没有做。
- 下一步如果要做运动融合验证，需要构造带 EuRoC IMU 的 ulog，走 EKF2 replay。

## 结果

ulog 时间里流在约 8 s 时开始（前 8 s 是 EKF 倾角对准）。

### 1. 坐标系转换（FRD/NED）✅

**`frames_euroc_static`：** cuVSLAM MH_01 Stereo-Inertial 在真值静止段（20.5–27.5 s）的位姿，经过 `dataset_to_frd` 转换后发出。
- 转换方法：
  - 体轴 FRD：z 取静止时 IMU 比力的反方向（向下），x 取 cam0 光轴的水平投影；
  - 世界 FRD：再绕向下轴转一个已知的 40°。
- 期望：EKF2 航向 = 40°。
- 实测：`vehicle_local_position.heading` 在 40.0–40.1° 之间。
- 位置跟随 EV，误差为毫米级；`cs_ev_pos`、`cs_ev_yaw` 都是 true；EV 位置 test ratio 为 0。

**`enu_pitfall`（反面测试）：** 同样的位姿，按 ROS 习惯发成 `MAV_FRAME_LOCAL_ENU`，位置和四元数都正确地写成 ENU/FLU。
- 实测：EKF2 航向是 **50.0°**（应为 40°）；位置正确。
- 这证实了源码里看到的行为：PX4 对 ENU **只转换位置，四元数原样拷贝**（`mavlink_receiver.cpp`）。
- 结论：我们只发 FRD/NED。`to_odometry_fields` 拒绝 ENU/FLU 和 `*_dataset` 帧。

### 2. 时间戳 ✅（重要）

| 场景 | 我们发送的采集延迟 | `vehicle_visual_odometry.timestamp − timestamp_sample` |
|---|---|---|
| `time_no_timesync`（不应答 TIMESYNC） | 150 ms | **0.0 ms**（p5 0，p95 0） |
| `time_timesync`（应答 TIMESYNC，先预热 70 s） | 150 ms | 中位 **99.9 ms**（p5 72，p95 127） |

- 源码（`Timesync.cpp`）的逻辑：`sync_stamp()` 在收敛之前直接返回 `hrt_absolute_time()`，也就是**用到达时间替换我们的采集时间**，不报任何错。
- 收敛条件是 `CONVERGENCE_WINDOW = 500` 个 RTT < 10 ms 的样本。PX4 发 TIMESYNC 请求的速率约 7–10 Hz，所以 SITL 里大约要 70 s。
- 收敛之后测到约 100 ms，而不是 150 ms。原因：
  - SITL 的 PX4 时钟和墙钟之间有约 3.7% 的速率差，这是由 `timesync_status.estimated_offset` 每 10 s 漂移约 0.37 s 推算出来的；
  - 收敛后滤波增益只有 0.003，跟不上这个漂移，所以留下约 50 ms 的偏差。
  - 这是 SITL 的伪影（💭 推断，原因未逐行确认），真机时钟漂移是 ppm 级。
- **结论：**
  - companion 必须应答 TIMESYNC；
  - 开机后大约 1 分钟内，时间戳不可信（会被当成"刚采集"）；
  - 如果需要补偿延迟，在收敛前用 `EKF2_EV_DELAY`，或者干脆等到收敛之后再发 EV。

### 3. 质量（quality）字段，以及 EKF2_EV_QMIN=0 的行为 ✅（已在 SITL 复现）

- **`quality_qmin0`：** `EKF2_EV_QMIN=0`，全程发 quality = **−1（"failed"）**。
  - 结果：EV 照常融合（`cs_ev_pos` true，`fused` true），位置跟随 EV。**复现了源码里的结论：QMIN=0 时 −1 也会被融合。**
- **`quality_qmin1`：** `EKF2_EV_QMIN=1`，quality 依次为 80 → 0 → 80 → −1 → 80，每段 8 s。

  | 段 | ulog 时间 | EKF2 |
  |---|---|---|
  | 80 | 7.9 s 起 | 开始融合 |
  | 0 | 约 15.4 s 起 | 停止融合（`fused` false），0.36 s 后 `cs_ev_pos` 变 false；约 4.6 s 后 `xy_valid` 变 0 |
  | 80 | 23.2 s | 重新开始融合 |
  | −1 | 31.0 s | 同样停止；约 5 s 后 `xy_valid` 变 0 |
  | 80 | 38.7 s | 重新开始 |

### 4. 方差字段 ✅

**`variance`：** `EKF2_EV_NOISE_MD=0`。

| 发送的 `pose_covariance[0]` | EKF2 实际使用的观测方差（`observation_variance[0]`） |
|---|---|
| NaN（"未知"） | 0.01（= `EKF2_EVP_NOISE` 的平方，0.1²） |
| 1e-4 | **0.01**（被参数下限抬高） |
| 1.0 | 1.0 |
| 25 | 25 |

→ 消息里的方差**只能把噪声调大，不能低于参数**（`fmaxf`）。

### 5. 门控、重置，以及渐进漂移 ✅（与渐进退化研究直接相关）

| 场景 | 发送的内容 | EKF2 的反应 |
|---|---|---|
| `jump` | 第 8 s 起位置 +1 m，保持 8 s | 先拒收（test ratio 3.6），**约 1 s 后重置到跳变后的位置**并继续融合；跳回时也一样 |
| `reset_jump` | +1 m，同时 `reset_counter` +1 | 立即接受新位置（只拒收了 1 个样本） |
| `slow_drift` | 第 8 s 起每秒 +5 cm，持续 20 s（总共 1 m） | **全程融合，test ratio ≈0，没有任何拒收**；EKF2 位置完全跟随（1.23 → 2.0 m） |
| `degraded_drift` | 同样的漂移，但从漂移开始就把方差设为 1.0 m²（v3 的 "degraded" 标记） | 仍然跟随，只是略滞后（例如 EV 1.20 时 EKF 1.11，EV 2.0 时 EKF 1.91） |
| `untrusted_drift` | 同样的漂移，`EKF2_EV_QMIN=1`，从漂移开始发 quality 0（"untrusted" 标记） | **EV 立即停止融合，EKF2 位置停在 1.0 m，不跟随漂移**；约 5.8 s 后 `xy_valid=0`（进入 dead reckoning，本地位置失效） |

- `jump` 场景的源码依据：`ev_pos_control.cpp`，`is_fusion_failing = isTimedOut(time_last_fuse, no_aid_timeout_max) // 1 second`。只要 quality 足够，就会 reset 到 EV（`_nb_ev_pos_reset_available = 5`）。
- **含义：**
  - EKF2 的 innovation 门控只防"突然的大跳变"，而且在 EV 是唯一位置源时只防 1 s；
  - **渐进漂移和 IMU 一致，门控根本看不见。** 这正是我们在 EuRoC 上看到的 cuVSLAM 的失效方式；
  - 所以"可信度"必须由 companion 侧的 supervisor 判断，并且用 EKF2 能执行的方式传过去。

## 6. supervisor 判定"不可信"时应该怎样标记

| supervisor 判定 | MAVLink 标记（`TRUST_POLICY`） | 飞控侧前提 | SITL 证据 |
|---|---|---|---|
| ok | 原样 | — | `frames_euroc_static` |
| degraded（v3：`rotation_inconsistent` / `low_texture`） | 位置方差下限 1.0 m² | `EKF2_EV_NOISE_MD=0` | 降权：漂移被跟随得慢一些，**不能阻止** |
| untrusted / fault（跟丢、跳变、时钟错误、持续 degraded） | **quality 0** | **`EKF2_EV_QMIN ≥ 1`**（默认 0 时无效） | `quality_qmin1`、`untrusted_drift`：0.4 s 内停止融合 |
| 重定位 | `reset_counter` + 1 | — | `reset_jump`：立即接受新位置 |
| 停止发送 | — | — | 0.4 s（2 × `EV_MAX_INTERVAL`）后 EV 停止（PX4 源码；`quality_qmin1` 中观察到 0.36 s） |

补充：
- quality 0 和 −1 在 `QMIN=1` 下效果相同。MAVLink 定义里 0 是"unknown"、−1 是"failed"，我们对"不可信"用 0，对 tracker 自己报告的跟丢用 −1，这样日志里能区分来源。
- **不可信之后飞控会怎样：** 没有其他位置源时，约 5 s 后 `xy_valid` 变 0。在真实飞行中这会触发位置失效的 failsafe（Position → Altitude，或者自动模式下 Land）。所以"标记不可信"本身是一个有后果的决定，**阈值必须先在真实数据上调好，而且需要团队同意**。v3 目前只把 degraded 映射到降权，没有自动升级成不可信。
- **飞控参数建议**（只在 SITL 里验证过，真机由团队决定）：
  - `EKF2_EV_QMIN=1`、`EKF2_EV_NOISE_MD=0`；
  - companion 必须应答 TIMESYNC；
  - 只发 FRD/NED 帧；
  - 正确设置 `EKF2_EV_POS_X/Y/Z` 杠杆臂。

## 7. Update the same evening: re-checked with real IMU data (HIL)

All the conclusions above were measured with a stationary, ideal SIH IMU. They were re-checked with `sitl_hil_replay.py`, which feeds the real EuRoC MH_01 IMU into EKF2. The injected drift was 5 cm/s for 20 s, and the "detected" verdict was supplied as an oracle. Results are in `demo-drift-marking-20261024-121340`.

| Marking | Actual error at end of drift | EKF2 self-reported eph | Position invalid |
|---|---|---|---|
| ok | 0.98 m | 0.05 m | no |
| variance 1 m² | 1.04 m | 0.33 m | no |
| variance 25 m² | 1.31 m | 1.28 m | no |
| variance 400 m² | 2.96 m | 3.96 m | no |
| quality 0 | **10.0 m** | 17.9 m | after 4.6 s |

- **"quality 0 stops it from following the drift" still holds**, but with a real IMU, 20 s of pure inertial dead reckoning is far worse (10 m) than the drift itself (1 m). SIH hid this because its IMU has no error.
- **Raising the variance does not reduce the error; it makes it larger** (EKF2 leans more on the IMU). Its value is elsewhere: it brings the self-reported eph close to the actual error. With 25 m², eph is 1.28 m against an actual 1.31 m, so downstream logic (e.g. `COM_POS_FS_EPH`) sees a truthful uncertainty.
- See §8 of the day report for the full discussion.
