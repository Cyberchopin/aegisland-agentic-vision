# 三个可重复的 SITL 演示

日期：2026-10-24。所有演示都是：**EuRoC 离线数据 → AegisLand → MAVLink ODOMETRY → PX4 v1.17.0 SITL 的 EKF2**。只走 localhost，载具从不解锁，没有发送任何命令。CONTROL OUTPUT = DISABLED。

**和昨天 SIH 演示的区别：** 这里 AegisLand 扮演 PX4 `none_iris` 机型的外部仿真器（`simulator_mavlink`，TCP 4560，lockstep），把 **EuRoC 真实的 IMU** 作为 HIL_SENSOR 喂给 PX4。
- 所以 EKF2 是在**运动中**真正融合视觉位置和 IMU，不是静止台架。
- 陀螺零偏用起飞前静止段的平均值去掉，相当于 PX4 在地面做的陀螺校准。
- 工具：`benchmarks/gps_denied/sitl_hil_replay.py`。图由 `demos/demo_plot.py` 生成。
- 限制：这条仿真器链路不走 TIMESYNC，ODOMETRY 按到达时间打戳，通过 `EKF2_EV_DELAY` = 30 ms 补偿。TIMESYNC 的行为已经在 `sitl-ekf2-ev-validation.md` 里用真实的 MAVLink 接收路径单独验证过。

| 演示 | 一条命令 | 运行时间（4 倍速） | 结果目录 |
|---|---|---|---|
| A 定位 | `benchmarks/gps_denied/demos/demo_a.sh` | 约 45 s | `benchmarks/results/gps_denied/demo-a-20261024-115807/` |
| B 渐进退化 | `benchmarks/gps_denied/demos/demo_b.sh` | 约 60 s | `demo-b-20261024-115848/` |
| C 故障 | `benchmarks/gps_denied/demos/demo_c.sh` | 约 60 s | `demo-c-20261024-115940/` |

每个结果目录里都有：图（`demo_*.png`）、`metrics.json`、每次回放的 `summary.json` 和 `sent.jsonl`，以及 SHA256SUMS。ulog 留在 `~/work/demo-runs/`，SHA 写在 metrics 里。

前提：
- PX4 已经编译在 `~/work/px4`；
- 已有 `~/.venvs/px4`（pymavlink、pyulog、matplotlib）；
- EuRoC 数据在 `~/aegisland-data/euroc`；
- 已有 `*-20261024-102659-signals` 的 cuVSLAM 运行。

## A 定位：MH_01，Stereo-Inertial

**数字**（`demo-a-20261024-115807/metrics.json`）：
- 回放 155 s，发出 3099 条 ODOMETRY；
- cuVSLAM 输出相对真值的 RMS 是 0.036 m；
- **EKF2（EV + EuRoC IMU）的 RMS 是 0.054 m，最大 0.135 m**；
- 两者都只做 SE(3) 对齐，不做尺度校正。

**讲解词（中文）：**
这是 EuRoC 数据集的一段 155 秒飞行：cuVSLAM 算出位姿，经过我们的 shadow supervisor 和 MAVLink 桥，以 ODOMETRY 消息送进 PX4 仿真里的 EKF2，同时喂给它的是同一段飞行里真实的 IMU。
左图里 EKF2 的轨迹和真值几乎重合，右图是误差：EKF2 的均方根误差 5.4 厘米，最大 13.5 厘米。
这证明坐标系转换、时间延迟补偿和消息字段都接对了；但它是数据集回放加仿真飞控，不代表我们的相机或真实飞行。

**Narration (English):**
This is a 155-second EuRoC flight: cuVSLAM poses go through our shadow supervisor and MAVLink bridge into PX4's EKF2 in simulation, together with the real IMU recorded on the same flight.
On the left, the EKF2 track lies on top of ground truth; on the right, its RMS error is 5.4 cm and the maximum is 13.5 cm.
This shows the frame conversion, latency compensation and message fields are wired correctly — it is a dataset replay into a simulated autopilot, not our camera and not a real flight.

## B 渐进退化：MH_05 暗段，cuVSLAM 不报错

**数字**（`demo-b-20261024-115848/metrics.json`）：
- 暗段在回放的 45–60 s，平均亮度最低约 8/255；
- cuVSLAM 跟丢帧数为 **0**；
- 误差峰值约 0.3 m，出现在暗段里；
- supervisor 的告警：
  - `low_texture` 占 10.3% 的帧，**覆盖了整个暗段**；
  - `rotation_inconsistent` 占 5.0% 的帧，在暗段之后误差第二次上升时（约 66–71 s）报警。
- EKF2 的 RMS：原始桥接 0.173 m，"degraded → 方差 ≥1 m²" 桥接 0.180 m。

**与预期不同，如实说明：**
- 演示原来的设想是"陀螺一致性检查抓到暗段"。实际回放里，**暗段期间陀螺残差一直低于阈值**；抓到暗段的是 VDR 匹配数（`low_texture`）。
- 陀螺检查报警的是暗段之后那次误差上升。
- 降权没有降低 EKF2 的误差：没有其他位置源时，降权只是让 EKF2 更依赖 IMU，而 IMU 的积分误差在这里并不更小。原因见报告里的"概念"一节。

**讲解词（中文）：**
这段飞行中间有 15 秒几乎全黑，cuVSLAM 自始至终报告"正在跟踪"，一帧都没丢，但它的位置误差涨到了约 30 厘米——这就是渐进退化：系统还在输出，但已经不可信。
我们的 supervisor 在暗段里用独立前端的特征匹配数标出了"低纹理"，在暗段之后误差再次上升时，用陀螺和视觉角速度不一致标出了第二次退化。
但图里也能看到，单纯把这些帧降权并不能让 EKF2 的误差变小，所以"发现退化"和"让飞控不受影响"是两个问题，后者还没有解决。

**Narration (English):**
For 15 seconds in the middle of this flight the scene is nearly black; cuVSLAM reports "tracking" the whole time and never loses a frame, yet its position error grows to about 30 cm — gradual degradation: still outputting, no longer trustworthy.
Our supervisor flags the dark segment as low texture using an independent front end's feature matches, and flags a second degradation afterwards when the camera's angular rate stops agreeing with the gyro.
The chart also shows that merely down-weighting those frames does not reduce the EKF2 error, so detecting degradation and protecting the autopilot from it are two different problems, and the second one is not solved yet.

## C 故障：VIO 输出冻结 2 s

在回放的第 60 s 注入故障：视觉管线卡住，2 s 内不断重发卡住前的最后一帧位姿，采集时间戳也不变。这是常见的管线卡死形式。

**数字**（`demo-c-20261024-115940/metrics.json`）：

| | 原始桥接 | supervised 桥接 |
|---|---|---|
| 故障前 EKF2 RMS | 0.041 m | 0.045 m |
| 故障后 10 s 内的最大误差 | **1.04 m** | **0.15 m** |
| 发出的 quality | 一直是 80 | 80 → **0**（整个故障期间）→ 80 |
| EKF2 的反应 | 把冻结的位姿当成"刚测到的"照收约 2.3 s，误差一路涨到 1.04 m；之后拒收约 1 s，再**突然重置**回真实位置 | 0.2 s 内停止融合，靠 IMU 推算 2 s，误差只涨到 0.15 m；恢复后平滑接上 |

- supervisor 的判定原因：`stale`、`capture_time_not_increasing`、`dropped_frames`（fault 级）。
- 为什么原始桥接会照收冻结位姿：这条链路按到达时间打戳。真实的 MAVLink 接收路径在 TIMESYNC 未收敛时也是这样（昨天已验证）。

**讲解词（中文）：**
我们在第 60 秒让视觉管线"卡住"两秒，它不停重发同一个旧位姿，这在真实系统里很常见。
不加 supervisor 时，EKF2 把这个旧位姿当成新测量照单全收，位置误差冲到 1 米，然后又突然跳回来；加了 supervisor 后，它识别出时间戳不再前进、数据过期，把这两秒的消息标成 quality 0，EKF2 立刻停止使用视觉，误差只到 15 厘米。
这是硬故障的情况：判定清楚、动作明确；代价是如果持续超过约 5 秒，飞控会认为位置失效并进入 failsafe。

**Narration (English):**
At 60 s we make the vision pipeline hang for two seconds: it keeps re-sending the same stale pose, a common real-world failure.
Without the supervisor, EKF2 accepts the stale pose as a fresh measurement, the position error climbs to 1 m and then snaps back; with the supervisor, the stalled timestamps are detected, those messages go out with quality 0, EKF2 stops using vision at once, and the error stays at 15 cm.
That is the hard-fault case — a clear verdict and a clear action; the price is that if it lasts more than about 5 s, the autopilot declares position invalid and enters failsafe.
