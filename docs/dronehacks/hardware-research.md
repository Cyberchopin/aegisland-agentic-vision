# 比赛硬件调研：Orin Nano + Luxonis OAK + Pixhawk 6C（GPS-denied）

日期 2026-10-24。检索由 3 个子 agent 并行完成；**标 ✅ 的结论由主会话重新打开原文核对过**，
标 ◻ 的只来自子 agent 报告（附原链接，尚未逐字复核），标 💭 的是推断。
已确认硬件只有：7" 机架、Pixhawk 6C、"Luxonis depth cameras"、"NVIDIA Orin Nanos"
（官网 <https://dronehacks.groups.stanford.edu/>，✅ 本日复查）。其余全部是 **候选，未确认**。

## 0. 候选配置矩阵（全部：候选，未确认）

| 未知项 | 候选 | 对方案的影响 |
| --- | --- | --- |
| 相机型号 | OAK-D Pro W / OAK-D Pro / OAK-D W / OAK-D S2 / OAK-D Lite / OAK 4 D (Pro) | 见表 2；Lite 双目分辨率最低（640×480），OAK4 功耗 10–15 W |
| 相机 IMU | BNO086 / BMI270 / OAK4 上 ICM-42688-P 或 LSM6DSV（官方两页矛盾）/ 无（早期 Lite） | 决定 Stereo-Inertial 是否可行 |
| Orin Nano 内存 | 4 GB / 8 GB（是否 "Super"） | 深度网络是否放得下 |
| JetPack | 5.x（Ubuntu 20.04, CUDA 11）/ 6.x（22.04, CUDA 12）/ 7.x | **cuVSLAM aarch64 wheel 只有 cu12 cp310 manylinux_2_35**（✅ release 资产列表）→ 实际要求 JetPack 6；JP5 只能源码编译（💭） |
| 飞控固件 | PX4 v1.14 / 1.15 / 1.16 / 1.18 或 ArduPilot | 参数名（EKF2_EV_CTRL 自 1.14 起取代 EKF2_AID_MASK ◻） |

## 1. Orin Nano 上的 VIO/SLAM 方案

| 方案 | 许可 | Jetson 上的数据（区分来源） | 相机/IMU 要求 | 集成难度 |
| --- | --- | --- | --- | --- |
| **cuVSLAM v17.0.1**（pip wheel） | NVIDIA Community License，仅限 NVIDIA 硬件 ✅ | **作者论文** arXiv 2506.04359 Table 1（AGX Orin, RealSense 640×480）：Stereo 1.8 ms/调用，CPU 5.5%，GPU 1.7%；Stereo-Inertial 3.8 ms；Mono-Depth 15.1 ms ✅。Orin Nano Super 25 W、640×360 30 FPS multi-stereo "substantial resource headroom" ✅；Mono-Depth 在 Nano 上 31 ms、丢约 30% 帧 ✅。EuRoC Stereo odom RMSE APE 0.13 m、SLAM 0.054 m、ORB-SLAM3 0.068 m；Stereo-Inertial odom 0.19 m ✅。**独立 Orin Nano 实测：未找到。** **本机实测**（RTX 4060 Ti，合成序列）warm track p50 0.82–0.89 ms | 官方 OAK 示例："Global shutter is a fundamental requirement for cuVSLAM." ✅；仅在 OAK-D W Pro 上验证、仅 stereo，无 IMU 示例 ✅ | 低（JP6）：pip + Python，无需 ROS |
| Isaac ROS Visual SLAM | NVIDIA | 当前 Isaac ROS 5.0 面向 JetPack 7.2 ◻；JP6 对应 3.2 线 ◻；时间要求 stereo 帧间偏移 ±100 µs、帧率抖动 ±2 ms ◻ | 同上 | 中高（ROS 2 + 版本锁） |
| DepthAI v3 BasaltVIO | depthai MIT ✅（PyPI 3.11.0）；Basalt BSD-3 ◻ | 无 Jetson 数据；是 **host node**（跑在 Jetson CPU 上）◻；"early access and may be less stable" ◻；开放 issue：原地 360° 旋转闭合误差 9 m（#1931）◻ | 用 OAK 自身 IMU；示例 640×400@60 FPS、IMU 200 Hz ◻ | 低：`pip install depthai`，有 aarch64 wheel（manylinux_2_28）✅ |
| OpenVINS | GPL-3 ◻ | 独立 Orin NX 研究（arXiv 2507.11241）：原参数"not reproducible"，EuRoC MH_02/MH_04/V2_02 初始化失败 ◻ | **必须 IMU**；需 Kalibr 标定 | 高（构建 + 调参） |
| VINS-Fusion (+GPU fork) | GPL-3 ◻ | 上游 2021 停更 ◻；Orin NX 研究中 APE 最差 ◻ | 全局快门 + 硬同步推荐 | 高（OpenCV3 / CUDA 编译） |
| Basalt（原版） | BSD-3 ◻ | 无 Jetson 数据；无 aarch64 二进制 ◻ | stereo+IMU | 中高 |
| ORB-SLAM3 | GPL-3 ◻ | Jetson-ORB-SLAM3（arXiv 2608.17874，2026-08）称 Orin Nano 上 mono-inertial 平均 32 FPS ✅（摘要原文）；**无代码链接** ✅ | IMU 初始化需激励运动 | 高 |
| Spectacular AI | "Free for non-commercial use" ✅ | **PyPI 1.54.0 只有 x86_64 / win_amd64 wheel，无 aarch64** ✅ → Orin 上不可用 | — | 排除 |
| RTAB-Map + OAK | BSD-3 ◻ | Orin NX 研究：EuRoC V1_01/V1_02 未初始化 ◻ | — | 适合建图，不适合作主里程计 |

## 2. OAK 系列双目 + IMU（候选型号，官方文档 ◻，链接见子 agent 原始 URL）

| 型号 | 双目传感器/快门 | 基线 | 双目 HFOV | IMU | IR | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| OAK-D Pro W | OV9282 全局 1280×800 | 75 mm | 127° | BNO086 | 点阵 + 泛光 | cuVSLAM 官方验证型号 ✅ |
| OAK-D Pro | OV9282 全局 | 75 mm | 80° | BNO086 | 点阵 + 泛光 | |
| OAK-D W | OV9282 全局 | 75 mm | 127° | BNO086（2021–23 批次可能 BMI270） | 无 | |
| OAK-D S2 | OV9282 全局 | 75 mm | 80° | 同上 | 无 | |
| OAK-D Lite | OV7251 全局 640×480 | 75 mm | 73° | BMI270；Kickstarter 批次无 IMU | 无 | 左右目与彩色未硬同步 |
| OAK 4 D (Pro) | OV9282 全局 | 未找到 | 77°（W 版 127°） | 页面写 ICM-42688-P，节点页写 LSM6DSV（矛盾） | Pro 版有 | RVC4，平均 10–15 W，启动约 15 s |

时间戳机制 ◻：所有帧/IMU 在设备时钟打戳，并与 host 同步（USB "below 200µs accuracy ... at 1σ"）；
左右目通过 FSYNC 硬同步（"only OV9282 and OV9782 can output FSYNC"）；**IMU 与相机之间未找到官方硬同步说明**
（只靠共享设备时钟，💭 需现场测偏移）。BNO086 陀螺 >400 Hz 会抖动 ◻。

片上深度 ◻：Pro/S2 中位绝对误差 <2%（<4 m）、<4%（4–7 m）、<6%（7–10 m）；
extended disparity 与 subpixel 在 v3 文档中标为不兼容 ◻。
点阵投影随相机运动 → 💭 会制造假特征，VIO 时应关闭投影（默认关闭 ◻）。

## 3. PX4 外部视觉融合（EKF2 EV）

**以下 4 条已在 PX4 `main`（提交 cf0797e4440e）源码中逐行核对 ✅：**

1. MAVLink `ODOMETRY` 按 **`estimator_type`** 分流：UNKNOWN/NAIVE/VISION/VIO → `vehicle_visual_odometry`（EKF2 订阅）；
   MOCAP → `vehicle_mocap_odometry`；GPS/LIDAR 等直接拒绝（`mavlink_receiver.cpp` 1770–1795）。
   → 发送时用 `MAV_ESTIMATOR_TYPE_VIO`。
2. `child_frame_id` 只接受 LOCAL_NED/ENU/FRD/FLU 与 BODY_FRD（及弃用别名）；**`MAV_FRAME_BODY_FLU` 落入 default，速度被丢弃**。
3. `EV_MAX_INTERVAL = 200e3` µs（`common.h:71`）；样本间隔 <200 ms 才启动，`2*EV_MAX_INTERVAL` 无新样本即停止（`ev_control.cpp:104`）。
4. 质量门限：`quality_sufficient = (ekf2_ev_qmin <= 0) || (quality >= ekf2_ev_qmin)`（`ev_control.cpp:57`）。
   → **默认 QMIN=0 时，quality=-1（"failed"）也会被融合**。应设 `EKF2_EV_QMIN ≥ 1`，VIO 失效时发 -1。

其余 ◻（子 agent 读源码/文档，未复核）：ENU/FLU 位置会转换但四元数不转换 → 在 Jetson 侧自行转成 FRD/NED；
`EKF2_EV_NOISE_MD=0` 时用消息方差，参数作下限；`EKF2_HGT_REF` 默认 GPS，室内需改；
`EKF2_NOAID_TOUT` 默认 5 s；`COM_POS_FS_EPH` 默认 5 m；`reset_counter` 变化触发 EV 重置，空中最多 5 次；
推荐发送频率 30–50 Hz；`EKF2_EV_DELAY` 通过 IMU 角速度与 EV 角速度互相关 + 扫参数看 innovation 来调。

**降级路径（◻ + 💭）：** EV 中断 0.4 s → 停止 EV；若光流 + 测距有效则继续速度辅助（漂移）；
无任何辅助 → `cs_inertial_dead_reckoning`，5 s 后本地位置失效 → 多旋翼 Position 模式退到 Altitude，
自主模式 Return → Land → Descend。**这条链本身就是 AegisLand shadow supervisor 应该监控和提前预警的对象。**

## 4. GPS-denied 失效模式与健康度信号（文献 ◻，链接见子 agent 报告原文已转录到附录）

| 失效模式 | 代表文献 | 机载可监控信号 | 阈值来源 |
| --- | --- | --- | --- |
| 弱纹理 | Zhang/Kaess/Singh ICRA 2016 退化因子；PL-VIO 2018 | 跟踪特征数、空间分布、信息矩阵最小特征值、协方差增长 | 无公开数值，需自标定 |
| 快速旋转/运动模糊 | Delmerico & Scaramuzza ICRA 2018 基准；Ultimate SLAM RA-L 2018 | 陀螺角速度、Laplacian 方差、track length 崩塌 | 💭 曝光×角速度×焦距 估算模糊像素 |
| 光照突变/自动曝光 | Zhang/Forster/Scaramuzza ICRA 2017 | 帧间亮度跳变、梯度质量、曝光/增益变化 | 未找到 |
| 振动 → IMU 混叠/削顶 | PX4 EKF 调参文档、振动隔离文档 | `vehicle_imu_status` 振动指标、`accel_clipping` 计数、`fs_bad_acc_*` | Flight Review：<4.9 绿 / 4.9–9.8 橙 / >9.8 红 m/s² |
| 动态物体 | VIODE RA-L 2021 | RANSAC 内点率、残差 | 未找到 |
| 卷帘快门 | Schubert IROS 2019 | 设计上用全局快门 | — |
| 估计器过度自信 | Hesch TRO 2014 | 在线用 innovation test ratio（PX4 <0.5 正常，>1 拒绝） | PX4 文档 |
| 跳变/偏置爆炸 | VINS-Mono TRO 2018 失效判据 | 帧间位姿跳变 vs IMU 预测、偏置变化率、`reset_counter` | 未给数值 |

AegisLand 现有 `PerceptionQualityMonitor` / `PerceptionTrustGate` / `SensorHealthMonitor` 已覆盖
"特征数/内点率/亮度/纹理"一类信号；**新增价值在于把 VIO 侧信号映射为 ODOMETRY `quality`，并监控 PX4 侧
`cs_ev_*`、`cs_inertial_dead_reckoning`、EV test ratio 与振动指标**（只读）。

## 5. 推荐

**主方案（候选，前提：JetPack 6 + 全局快门 OAK）：** cuVSLAM Stereo（pip wheel），OAK 原生双目 640×400/30 FPS，
点阵投影关闭；输出 ODOMETRY(VIO, LOCAL_FRD/BODY_FRD, quality, reset_counter)。
风险：无 OAK Stereo-Inertial 示例（IMU 外参/时间要自己做）；仅 NVIDIA 硬件许可；无独立 Orin Nano 实测；
需要先在 EuRoC 上确认我们的坐标/时间链（本仓库阶段 1 正在做）。

**备选 1：** DepthAI v3 BasaltVIO（host node）。风险：early access、开放的旋转漂移 issue、Jetson 无性能数据；优点是 JP5/JP6 都可能可用、使用 OAK 自带 IMU。

**备选 2：** 光流 + 测距由 PX4 自身融合（不依赖 Jetson），作为 VIO 失效时的悬停/降落保底。风险：需要硬件里有光流/测距模块（未确认）。

**不推荐（48 h 内）：** OpenVINS / VINS-Fusion / ORB-SLAM3（构建与调参成本高，独立测试中初始化/发散问题）；Spectacular AI（无 aarch64）。

## 6. 拿到硬件后第一个小时

1. `cat /etc/nv_tegra_release`、`free -h`、`nvidia-smi` 或 `tegrastats` → JetPack、内存、功耗模式。
2. `python3 -c "import depthai as dai; d=dai.Device(); print(d.getProductName(), d.getConnectedCameraFeatures(), d.getConnectedIMU(), d.getUsbSpeed())"` → 型号、传感器（OV9282？）、IMU 芯片、USB 3？
3. 读出出厂标定（`readCalibration()`），保存 JSON；检查 IMU→相机外参是否存在。
4. 录 60 s 静止 + 60 s 手持晃动的原始左右目 + IMU（带设备与 host 时间戳），存盘；这就是我们自己的"EuRoC 格式"回放。
5. 检查：左右目 sequence/timestamp 一致；IMU 实际频率；帧间隔抖动；用本仓库 `estimate_time_offset` 估 IMU-相机偏移。
6. 在 Orin 上 `pip install` cuVSLAM cu12 wheel → 用录制数据离线跑 Stereo，看 `track()` 耗时和跟踪率（**这才是 Orin 实测**）。
7. 在 Pixhawk 上 `param show EKF2_EV*`、`ver all` → 固件版本与参数名；确认 MAVLink 链路与 TIMESYNC。
8. 称重相机 + 线 + 支架；检查 USB 供电下是否掉线。

## 附录：子 agent 报告中引用的主要原始链接

- cuVSLAM: <https://github.com/nvidia-isaac/cuVSLAM>，OAK 示例 <https://github.com/nvidia-isaac/cuVSLAM/blob/main/examples/oak-d/README.md>，论文 <https://arxiv.org/html/2506.04359>
- Isaac ROS releases: <https://nvidia-isaac-ros.github.io/releases/index.html>
- Luxonis VIO: <https://docs.luxonis.com/overview/toplevel-features/vio_and_slam/>，Basalt 示例 <https://docs.luxonis.com/software-v3/depthai/examples/vslam/basalt_vio.md>，IMU 节点 <https://docs.luxonis.com/software/depthai-components/nodes/imu>，设备时钟 <https://docs.luxonis.com/software-v3/depthai/depthai-components/device.md>，硬件同步 <https://docs.luxonis.com/hardware/platform/deploy/data-sync/hardware-sync>
- Basalt issue: <https://github.com/luxonis/depthai-core/issues/1931>
- Orin NX VIO 对比: <https://arxiv.org/html/2507.11241>；Jeon et al. 2021: <https://arxiv.org/abs/2103.01655>；Jetson-ORB-SLAM3: <https://arxiv.org/abs/2608.17874>
- PX4: <https://docs.px4.io/main/en/ros/external_position_estimation>，源码 `src/modules/mavlink/mavlink_receiver.cpp`、`src/modules/ekf2/EKF/aid_sources/external_vision/ev_control.cpp`、`params_external_vision.yaml`，安全 <https://docs.px4.io/main/en/config/safety>，EKF 调参 <https://docs.px4.io/main/en/advanced_config/tuning_the_ecl_ekf>
- 失效模式: Zhang ICRA 2016 <https://frc.ri.cmu.edu/~zhangji/publications/ICRA_2016.pdf>，VINS-Mono <https://arxiv.org/abs/1708.03852>，Hesch TRO 2014，VIODE <https://arxiv.org/abs/2102.05965>，Introspective Perception <https://arxiv.org/abs/1607.08665>，IV-SLAM <https://arxiv.org/abs/2008.02760>
