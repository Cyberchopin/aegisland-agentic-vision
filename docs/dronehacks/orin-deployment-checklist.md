# Orin 部署清单（cuVSLAM + AegisLand shadow supervisor）

日期：2026-10-24。状态：**全部基于开发机（RTX 4060 Ti，WSL2）与官方发布页，未在 Orin 上运行过任何东西。**
标记：✅ 已对照原始来源核对；📏 开发机实测（不是 Orin）；💭 推断，待 Orin 验证。

## 1. cuVSLAM 是否有 aarch64 / JetPack 版本 ✅

来源：GitHub releases API `nvidia-isaac/cuVSLAM`，v17.0.1（2026-10-13）资产列表，以及该 tag 下的 README 安装表。

| 目标 | Ubuntu / JetPack | wheel | 大小 |
|---|---|---|---|
| **Jetson Orin** | 22.04 / **JetPack 6.x** | `cuvslam-17.0.1+cu12-cp310-cp310-manylinux_2_35_aarch64.whl` | 113 MB |
| Jetson Thor | 24.04 / JetPack 7.x | `cuvslam-17.0.1+cu13-cp312-abi3-manylinux_2_39_aarch64.whl` | 84 MB |
| 开发机（我们现在用的） | 24.04 | `cuvslam-17.0.1+cu13-cp312-abi3-manylinux_2_39_x86_64.whl` | 90 MB |

此外，C++ 包有 `cuvslam-cpp-17.0.1-orin-cuda12.6.3-ubuntu22.04.tar.gz`。

部署要点：
- README 原文：Orin 用 **cu12 / cp310**。"Only the combinations listed above are provided as pre-built wheels."
- CUDA Toolkit 不随 wheel 附带，major 版本必须与 `cu12` 匹配。
- JetPack 只带 runtime，需要时要另装 dev 包（README "Build on Jetson"）。
- JetPack 5.x（Ubuntu 20.04）**不在支持表里** → 如果比赛的 Orin Nano 刷的是 JP5，需要重刷 JP6 或从源码构建（💭 风险高）。
- **开发机用的是 cu13 + Python 3.12，Orin 会是 cu12 + Python 3.10**：
  - 这是不同的构建，开发机上的精度和确定性结论不能直接继承。
  - 建议先在开发机装 `+cu12-cp310 x86_64` wheel，重跑 MH_01，确认结果一致（便宜，不需要 Orin）。

## 2. 内存预算

Orin Nano 是 CPU/GPU **统一内存**（4 GB 或 8 GB，型号未确认），所以 host RSS 和 GPU 显存要加在一起算。

| 组件 | 开发机实测 📏 | 测量方法 | Orin 上的预期 💭 |
|---|---|---|---|
| cuVSLAM Stereo，MH_01 全序列，Python 进程 | 最大 RSS **530 MiB** | `/usr/bin/time -v` | 与 CUDA context 大小有关，Jetson 上通常更小，待测 |
| cuVSLAM Stereo-Inertial | 最大 RSS **539 MiB** | 同上 | 同上 |
| cuVSLAM GPU 显存增量 | 峰值比运行前多约 **+275 MiB**（3446→3721 MiB） | `nvidia-smi` 每 250 ms 采样 | 统一内存里与 RSS 有重叠，单独计没有意义，要在 Orin 上用 `tegrastats` 测 |
| VDR（AegisLand 核心，OpenCV ORB） | 最大 RSS **74 MiB**（600 帧） | `/usr/bin/time -v` | 相近 |
| supervisor + MAVLink bridge | 未单独测（纯 Python/numpy，窗口 ≤5 s） | — | < 100 MiB 💭 |
| OS + JetPack 桌面 | — | — | 关掉桌面（`multi-user.target`）可省数百 MB 💭 |
| FFS 深度网络 | 不进比赛路径（已决定） | — | 0 |

粗预算 💭：cuVSLAM 约 0.6 GB，VDR 约 0.1 GB，supervisor/bridge 约 0.1 GB，系统约 1–1.5 GB。8 GB 版本有余量；**4 GB 版本可能偏紧，必须实测**。

结果文件：`benchmarks/results/gps_denied/cuvslam-*-MH_01_easy-*-memory-resources/`。

## 3. 开发机上的结论，到 Orin 可能不成立的

| 开发机结论 | 为什么到 Orin 可能不成立 | Orin 上怎么验证 |
|---|---|---|
| `track()` p50 0.7–0.9 ms（批量回放）📏 | **今天查明：这是 GPU 处在 boost（P2，2535 MHz）时的数字**。<br>回放约 7 s 后 GPU 降到 P5（210 MHz），`track()` 变成 5–6 ms。<br>按实时 20 fps 节流时，从一开始就是 4–8 ms。<br>Orin 有自己的 DVFS / nvpmodel。 | 固定 nvpmodel 并执行 `jetson_clocks`；**按真实帧率**回放（`--throttle-ms`）测 p50/p95；同时记录 `tegrastats` |
| 两次运行逐字节相同（部分配置）📏 | 今天在开发机上就已经不成立：MH_01 Inertial 在同一时段连续两次相同，但不同时段分别得到 0.031、0.040、0.0335 m。再加上 GPU 不同、CUDA 12 与 13、wheel 不同 | 在 Orin 上每个配置**在不同时段**各跑 ≥3 次，报告分布，不报单次 |
| Stereo-Inertial 在 MH_04/05 比 Stereo 差 📏 | 数据集结论，与平台无关；但比赛相机的 IMU 质量不同 | 用我们自己的录像重跑 |
| supervisor v3 的阈值（陀螺残差 0.014 rad/s，VDR 匹配数 597）📏 | 是在 EuRoC 相机（752×480）和 IMU 上拟合的；OAK 的分辨率、噪声、ORB 匹配数分布都不同 | 先录 3–5 段正常飞行或手持数据，**只用来重新拟合阈值**，再用另外的数据评估 |
| VDR p50 12–13 ms（1 线程）📏 | Orin Nano 的 CPU 是 6 核 A78AE，单核比开发机慢 💭 | 实测，必要时降分辨率 |
| 帧到达和时间戳 | EuRoC 时间戳是离线对齐好的 | 用 OAK 设备时钟；检查 left/right ≤1 ms；用 supervisor 的 `imu_time_offset` 实测 |
| PX4 TIMESYNC 约 70 s 收敛（SITL）📏 | SITL 的仿真时钟和墙钟有约 3.7% 的速率差，真机没有这个问题；但 500 个样本的门槛不变 | 真机 bench 上看 `timesync_status` |

## 4. 第一小时清单（拿到 Orin 时）

1. 确认系统版本、内存和功耗模式：`cat /etc/nv_tegra_release`、`free -h`、`nvpmodel -q`、`tegrastats`。
2. 是 JP6 就装 cu12/cp310 wheel；是 JP5 就停下来，把升级与否交给团队决定。
3. 先离线回放 EuRoC MH_01（从开发机拷 1.6 GB 数据），使用同一 runner：
   - 用 `--throttle-ms 40` 模拟 20 fps；
   - 对比 trajectory SHA256 和 ATE。
4. 测 `track()` p50/p95，以及 `tegrastats` 里的 RAM 和 GPU 频率；分别在 `jetson_clocks` 打开和关闭时各测一次。
5. 接 OAK：录 60 s 静止 + 60 s 手持，按 `teammate-interface-spec.md` §6 的格式。离线回放，不接飞控。
6. 只有以上都通过后，才讨论 bench 上的 TIMESYNC / ODOMETRY。**CONTROL OUTPUT 仍然是 DISABLED**；是否融合由团队决定。

## 5. OAK 深度与 cuVSLAM 同时运行的资源问题（2026-10-24 晚补充；全部**未在 Orin 上验证**）

### USB 带宽（按像素数据量估算，不含协议开销）

| 流组合 | 数据量 | 结论 |
|---|---|---|
| 2 路单色 640×400 GRAY8 @20（只给 VIO） | 10.2 MB/s（0.08 Gbit/s） | USB2 也可以（USB2 实际约 30–40 MB/s 💭） |
| 2 路单色 640×400 @30 + 深度 640×400 uint16 @30（`nav` / `near` 预设） | 30.7 MB/s（0.25 Gbit/s） | **USB2 已经接近上限**，需要 USB3 |
| 2 路单色 1280×800 @30 + 深度 @30 | 122.9 MB/s（0.98 Gbit/s） | 只能走 USB3 |
| 2 路单色 1280×800 @15 + 深度 @15（`accuracy` 预设） | 61.4 MB/s（0.49 Gbit/s） | 只能走 USB3 |

- OAK 的深度是在设备端算的，所以**深度不占 Orin 的 GPU**，只占 USB 带宽和 host 端的 CPU 拷贝。
- 线材或接口降级到 USB2 时，DepthAI 会照常运行但掉帧。`oak_record.py` 会记录被拒收和未配对的帧；还需要再记录 `device.getUsbSpeed()`（待加，未验证）。
- 可以用来降带宽的办法：只传 VIO 需要的单色图加上降采样后的深度（decimation 2），或者深度改用 15 fps。

### Orin Nano 上的 CPU、GPU 和内存（开发机数字仅供数量级参考）

| 组件 | 开发机实测 📏 | 跑在 Orin 的哪里 | 风险 |
|---|---|---|---|
| cuVSLAM 里程计 | RSS 约 530 MiB；实时 20 fps 下 `track()` p50 约 6–7 ms（GPU 降频状态下） | GPU 加一个 CPU 线程 | 见第 3 节 |
| cuVSLAM **SLAM**（回环） | RSS 峰值 525–587 MiB；`track()` p95 从 29 ms 升到 50–55 ms（MH_05）；MH_01 完整建出的地图保存后 67 MB | 另有一个 SLAM 线程（异步模式），或者在跟踪线程里（同步模式） | 详见 `slam-loop-closure.md` |
| DepthAI host 端（取帧、拷贝） | 未测 | CPU | **必须实测**：在 Orin 上同时开 3 路流时的 CPU 占用 |
| 避障扇区（`depth_sectors`） | 7.7 ms/帧（752×480，开发机单线程） | CPU | Orin 的 A78AE 单核更慢 💭；可以降采样或只用部分帧 |
| VDR（ORB，`low_texture` 用的匹配数） | 12–13 ms/帧 | CPU | 同上 |
| supervisor + bridge | 很小 | CPU | — |

### 拿到 Orin 后必须实测的项目

1. `tegrastats`：分别测只开 VIO、VIO + 深度（`nav` 预设）、VIO + 深度 + SLAM 三种情况下的 CPU 各核占用、GPU 频率和 RAM。
2. `device.getUsbSpeed()`：确认连接是 SUPER（USB3），并测掉帧率。
3. `track()` 的 p50/p95：在 `jetson_clocks` 开启和关闭两种状态下，按 20 fps 实时节流运行，同时记录频率。
4. 深度延迟：用 `oak_depth_validation.py` 的延迟统计，比较 Orin 负载很重时和空载时的差别。
5. 内存：把 SLAM 的 `max_map_size` 设为 300（默认值）时，跑 10 分钟看 RSS 是否稳定。
