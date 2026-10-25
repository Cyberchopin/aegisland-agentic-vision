# 夜间工作汇报 2026-10-24（02:20–03:45 PDT）

分支 `work/dronehacks-gps-denied-c`，本地提交，未 push。所有 GPU 数字来自**开发机 RTX 4060 Ti（WSL2）**，不是 Orin。
结果目录均在 `benchmarks/results/gps_denied/`，下文用目录名指代"来自哪次运行"。

## 0. 总览

| 线 | 状态 | 一句话 |
| --- | --- | --- |
| B 真实 EuRoC | ✅ MH_01–MH_05 全部跑完 | Stereo ATE 0.036–0.155 m，Stereo-Inertial 0.031–0.176 m，全程 0 帧跟丢；困难序列表现为"渐进退化"而不是"跟丢" |
| A 硬件调研 | ✅ 初稿 `hardware-research.md` | 主方案 cuVSLAM Stereo（前提 JetPack 6 + 全局快门 OAK）；PX4 EV 关键行为已读源码核对 |
| C 性能/接口 | ✅ | ORB 特征缓存：`compensate()` p50 18.87→13.96 ms（1.35×），599 对 EuRoC 帧输出逐字节一致；阶段 3 消息 + shadow supervisor + 故障注入在 5 段序列上 7 种故障全部检出 |
| D 双目深度 | ✅ 最小起步 | C-FFS 模型 p50 110 ms（iters=8）/ 91 ms（iters=4）；代理指标优于 SGBM；**代码许可为非商用研究用途**，需你决定 |
| E 方法笔记 | ✅ 一页 | `inference-optimization-method-notes.md` |

测试：全量 **263 collected，262 passed，1 skipped**（可选 CUDA 集成测试），exit 0（03:31 开始的最后一轮）。

## 1. 线 B：真实 EuRoC

### 数据完整性
- `machine_hall.zip` = 12,683,729,426 B = **12096.15 MiB**，与 ETH 记录页一致；官方未给校验和。
- 每段：外层条目 `unzip -t` 无错 → 内层 ZIP `unzip -tq` 无错 → 解压到 `~/aegisland-data/euroc/`（不进 git）。ZIP 内还有 `.bag`，未使用。

### 难度 vs 结果（`summary-MH_01-05-with-MH_04-20261024.json`）

| 序列 | 模式 | ATE SE(3) RMSE / max (m) | Sim(3) scale | RPE20 m / deg | 跟丢帧 | run1==run2 |
| --- | --- | --- | --- | --- | --- | --- |
| MH_01_easy | Stereo | 0.036 / 0.093 | 1.0016 | 0.0108 / 0.084 | 0 | ✔ |
| | Stereo-Inertial | **0.031** / 0.077 | 1.0017 | 0.0118 / 0.085 | 0 | ✔ |
| MH_02_easy | Stereo | 0.047 / 0.117 | 0.9978 | 0.0087 / 0.122 | 0 | ✔ |
| | Stereo-Inertial | 0.045–0.049 / 0.20 | 0.995 | 0.0157 / 0.364 | 0 | ✘（不确定性） |
| MH_03_medium | Stereo | 0.121–0.126 / 0.37 | 0.9989 | 0.030 / 0.20 | 0 | ✘ |
| | Stereo-Inertial | **0.072** / 0.136 | 0.9932 | 0.031 / 0.118 | 0 | ✔ |
| MH_04_difficult | Stereo | **0.097** / 0.275 | 0.9964 | 0.033 / 0.154 | 0 | ✔ |
| | Stereo-Inertial | 0.168 / 0.279 | 0.9894 | 0.034 / 0.118 | 0 | ✔ |
| MH_05_difficult | Stereo | **0.155** / 0.336 | 0.9856 | 0.028 / 0.111 | 0 | ✔ |
| | Stereo-Inertial | 0.160–0.176 / 0.30 | 0.985–0.987 | 0.032 / 0.113 | 0 | ✘ |

论文对照（作者数字，arXiv 2506.04359 Table 2，EuRoC 全部序列平均，非本机）：Stereo odom 0.13 m，Stereo-Inertial odom 0.19 m。

**困难序列在哪里、怎么失效：**
- 没有一次"跟丢"（`untracked_frames` 全为 0）。失效形式是**误差渐进增长 + 尺度漂移**（MH_05 scale 0.985 ≈ 1.5%）。
- MH_05 分段（stereo run1，15 s 窗口）：60–75 s 图像平均亮度降到 **14.2/255**，随后 75–90 s 误差达到最高 **0.211 m**，之后维持 0.18 m——暗段积累的漂移在纯里程计下无法回收。MH_01 全程 0.02–0.05 m。
- MH_02 Stereo-Inertial 在第 369–370 帧出现一次**真实跳变**：隐含速度 5.62、4.28 m/s，而真值 0.69 m/s、Stereo 0.70 m/s（约 0.5 m / 100 ms）。shadow supervisor 的 `pose_jump` 正确检出。
- **对比赛的含义**：现场最可能的不是"VIO 报失败"，而是"VIO 继续报 tracking 但越来越不准"。所以安全监督不能只看 tracking 状态（见 §8 概念 3）。

### Stereo-Inertial 为什么在 MH_04/05 更差：按要求先排查外参与时间
| 检查 | 结果 | 结论 |
| --- | --- | --- |
| 外参链 | 重标定 `inv(cam0_from_imu)` vs 原始 cam0 `T_BS`：0.36°、7.6 mm（所有序列同一标定） | 一致 |
| 相机–IMU 时间偏移 | cuVSLAM 角速度 vs \|gyro\|：MH_01 +1.5、MH_02 +1.5、MH_03 −0.5、MH_04 +1.5、MH_05 −1.5 ms（r=0.92–0.98） | 对齐 |
| IMU 连续性 | 5 段均无 >7.5 ms 的缺口；唯一不足 10 个样本的区间是第一帧（IMU 比相机晚开 20–25 ms） | 正常 |
- **结论**：我们的坐标/时间链不是原因。作者论文本身也是 Inertial 平均更差。没有为此调算法参数。

### 一次只改一个参数
- `async_sba`（`cuvslam-*-MH_01_easy-20261024-030838-asyncsba-run{1,2}`）：开启后**两次运行不一致**；Stereo ATE 0.034 / **0.066** m（关闭时恒为 0.036），Inertial 0.041 / 0.040 m（关闭时 0.031）；Stereo 最大延迟 111→27–38 ms，但 p50 不变。→ 保持官方默认 `async_sba=False`。

### 确定性
- `async_sba=False` 时，MH_01/MH_04 两模式、MH_02/MH_05 Stereo、MH_03 Inertial 两次逐字节相同；**MH_02-Inertial、MH_03-Stereo、MH_05-Inertial 不相同**（ATE 差最多 0.016 m）。→ cuVSLAM 不是完全确定的，报告必须给多次运行。

### track() 耗时（开发机）
- smoke 前 200 帧 p50 0.72 ms；全序列 p50 0.7–8 ms，p95 5–32 ms，max 35–171 ms。
- MH_01 Stereo 在 ~1500 帧处从 0.8 ms 阶跃到 ~8 ms，并在**不同负载的 3 次运行中都在同一位置**出现 → 由工作负载/内部状态引起，不是 CPU 争用；WSL 下 nsys 采不到 kernel 时间，原因未定。
- 部分运行与 1 核 profiling 任务重叠（已注明）；Orin 上必须重测。

### 数据问题：MH_04 左右目帧数不同
- 官方 `dataset_utils` 直接拒绝（"Number of frames differs"）。cam0 多一帧 `1403638127245096960`。
- 新增 `--pair-by-timestamp`：只用两路共有时间戳并把未配对帧写入 `run.json`；在 MH_01 上与官方结果**完全相同**（3682 帧 / 40502 条含 IMU）。失败的空目录保留为 `*-FAILED-unpaired-stereo`。
- 这正是现场 OAK 掉帧时会遇到的情况。

### VDR 与 cuVSLAM 的公平对照（图像空间，非米制）
| 序列 | VDR 有效率 | VDR 方向 vs 真值 cos 中位数（∞ / 5 m 假设） | 5 m 同半平面 / 45° 内 | VDR p50 ms |
| --- | --- | --- | --- | --- |
| MH_01 | 3681/3682 | 0.968 / 0.999 | 0.990 / 0.979 | 13.0 |
| MH_02 | 3039/3040 | 0.953 / 0.998 | 0.994 / 0.969 | 12.9 |
| MH_03 | 2699/2700 | 0.965 / 0.999 | 0.994 / 0.979 | 13.2 |
| MH_04 | 2032/2033 | 0.981 / 0.999 | 0.995 / 0.990 | 12.1 |
| MH_05 | 2272/2273 | 0.964 / 0.998 | 0.996 / 0.989 | 12.0 |
- cuVSLAM 用同一方法打分为 ~1.000（指标上限）。VDR 在 EuRoC 上**从不失效**（唯一"无效"帧是第一帧），角速度到 2 rad/s 也没有。
- **局限**：方向一致 ≠ 位置准；VDR 的 x/y 依旧是归一化图像单位；EuRoC 的运动对 VDR 来说偏温和。

### 故障注入回放（supervisor v2，`fault-replay-*-supervisor-v2-*`）
| 故障 | 5 段全部检出 | 检出延迟（ms） |
| --- | --- | --- |
| dropped_frames（5 帧） | ✔ | 251–256（下一帧到来时） |
| camera_outage（1 s） | ✔ | 375（400 ms 超时） |
| pose_jump（+1 m） | ✔ | 8–12 |
| stale_delivery（+300 ms） | ✔ | 301–332 |
| out_of_order | ✔ | 51–53 |
| imu_time_offset（+50 ms） | ✔ | 1951–5951 |
- 未注入故障的基线：MH_01/03/04/05 **0 次告警**；MH_02 **1 次**，就是上面那次真实跳变（真阳性）。

## 2. 夜里发现并修正的错误（都保留了旧结果目录）
1. **时间偏移估计的混叠偏差**：200 Hz \|gyro\| 点采样 vs 20 Hz 位姿差分（区间平均）在 MH_01 上给出虚假的 −12.5 ms（相关曲线呈 5 ms 锯齿）。改为对 IMU 做同区间平均后 +1.5 ms。加了能复现的测试。旧结果 `compare-*-MH_01_easy-20261024-030019`（无后缀）作废。
2. **回放延迟模型错误**：publish = capture + 本帧 track 时间，会让慢帧之后的帧"先发布"，造成 4 次假告警。改为单线程排队模型 `max(capture, 上一帧完成) + track`。旧结果 `-fix1` 作废。
3. **supervisor IMU 偏移告警"粘住"**：估计不可用（悬停）时沿用上次结论，MH_02 产生 60 次假告警；低相关窗口（<0.85）给出 −27.5/−75 ms 的错误估计。修正：不可用→清除；相关阈值 0.6→0.9（依据 MH_02 实测）；连续 2 次超阈才告警。代价：检出延迟变为 2–6 s。旧结果 `fault-replay-*-20261024-0311..0324`（无 supervisor-v2）作废。
4. **profiling 基线被我中途改源码破坏**：`before-syn40-retry` 不完整，已改名 `-INCOMPLETE`，用旧提交的 worktree 重跑（`-r2`），源码哈希核对通过。教训：基线运行期间不改被测代码。
5. FFS：`cv2.fisheye.stereoRectify` 要求连续数组；C-FFS checkpoint 的 `model.args` 缺 `normalize`（按上游 `make_plugin_onnx.py` 的默认 True 补上并记录）。失败目录标记 `-FAILED-*`。

## 3. 线 A：硬件调研（详见 `hardware-research.md`）
- 主方案：cuVSLAM Stereo（pip wheel，JetPack 6；aarch64 只有 cu12 cp310 wheel）。备选：DepthAI v3 BasaltVIO（host node，early access）；保底：PX4 光流+测距。
- 已读 PX4 `main` 源码核对：ODOMETRY 按 `estimator_type` 分流；`BODY_FLU` 速度被丢弃；EV 200 ms 间隔/400 ms 超时；**`EKF2_EV_QMIN=0` 时 quality=-1 也会被融合**。
- Spectacular AI 无 aarch64 wheel（PyPI 核对）→ 排除。
- 第一小时清单见该文档 §6。

## 4. 线 C：性能与接口
- **证据先行**：cProfile（合成序列）`detectAndCompute` 2×7.4 ms（64%），`knnMatch` 6.6 ms（28%），每帧对上一帧重算 ORB。
- **优化**：按字节相等复用上一帧 ORB 结果（2 项缓存，私有拷贝防外部改写）。
  - 正确性：`profiling.py --reference` 4 种配置（默认/752×480 × base/retry）逐字节等价全部通过；EuRoC 599 对 A/B 输出完全一致；3 个新单测。
  - 计时（无 profiler、交错 A/B，`orb-cache-ab-MH_01_easy-20261024.json`）：p50 18.87→**13.96 ms**，p95 23.9→17.9 ms。
  - 剩余主要开销是 `knnMatch`；改成 crossCheck/LSH 会改变结果，未做。
- **Nsight Systems**：从 Windows 安装复制 `target-linux-x64` 后可在 WSL 运行（2024.6.2）；只有 CUDA API trace，**无 GPU kernel 时间**（WSL 限制）。cuVSLAM：冷启动 31 次 `cudaHostAlloc` 共 347 ms；每帧 ~20 次 kernel launch + 2 次上传 + 2 次同步。
- **阶段 3 接口**：`localization-interface.md`、`localization_message.py`、`localization_supervisor.py`、`localization_faults.py`、`fault_replay.py`。

## 5. 线 D：Fast-FoundationStereo（`ffs-c-ffs-*`）
- 版本：仓库 `476f424`；checkpoint C-FFS（HF `53a84e2`，71,105,659 B，sha256 `7aee8594…`）。PyTorch 2.6.0+cu124 独立 venv（5.5 GB，单个下载均 <3 GB）。
- 输入：EuRoC 鱼眼标定 → `cv2.fisheye.stereoRectify` 校正，752×480，基线 0.110 m；人工检查可视化，视差图结构合理。
- 计时（40 帧，跳 50 帧采样，首帧单列）：

| 配置 | 校正 | 上传+pad | 模型 p50 / p95 | 后处理 | 首次调用 | 峰值显存 |
| --- | --- | --- | --- | --- | --- | --- |
| iters=8，MH_01 | 0.51 ms | 1.51 ms | 109.8 / 117.1 ms | 1.68 ms | — | 778 MB |
| iters=8，MH_05 | 0.64 ms | 1.58 ms | 109.6 / 115.4 ms | 1.66 ms | 1318 ms | 780 MB |
| iters=4，MH_05 | — | — | **90.8** / 93.9 ms | — | 1433 ms | — |

- 代理指标（**不是精度**，EuRoC 无深度真值）：

| | FFS 无效像素 p50 | SGBM 无效像素 p50 | FFS LR 一致 | SGBM LR 一致 | FFS–SGBM 视差差中位 |
| --- | --- | --- | --- | --- | --- |
| MH_01 | 3.1% | 29.1% | 0.90 | 0.77 | 0.30 px |
| MH_05 | 1.3% | 21.0% | 0.95 | 0.77 | 0.31 px |
| MH_05 iters=4 | 1.3% | 21.0% | 0.94 | 0.77 | 0.31 px |

- OAK 原生深度基线需要真机，未测。TensorRT 路径未做（需 Docker/额外安装与 engine 构建）。
- **许可**：FFS 代码 `LICENSE.txt` §3.3 "only may be used ... non-commercially ... research purposes only"；C-FFS 权重为 NVIDIA Open Model Agreement。

## 6. 需要你决定
1. **FFS 代码许可**：比赛（有奖金）是否算"research purposes only"？在确认前它只能作为离线研究基线，不应进入比赛代码。
2. **opencv-2026 许可**：仍未复制任何文本（按你的要求）。
3. **是否把阶段 3 消息桥接到 MAVLink ODOMETRY（只在 SITL）**：目前完全不发送任何东西。
4. **结果入库体积**：git 里的结果合计约 19 MB（单文件 ≤1.2 MB）；fault-replay 逐帧日志（每段 ~20 MB）已 git-ignore，只提交摘要与 SHA256SUMS。是否接受？
5. **磁盘**：`~/aegisland-data/euroc` 约 9 GB，`~/.venvs/ffs` 5.5 GB，ZIP 仍在 Downloads（12.1 GB）。可删 ZIP 或保留。
6. 是否给组委会发 `organizer-questions.md` 里的问题。

## 7. 测试
- 基线（`eda23d3`，昨天）：216 collected，215 passed，1 skipped。
- 最后一轮（含全部新代码）：**263 collected，262 passed，1 skipped**，exit 0。新增 47 个测试：trajectory_eval 11、localization_diagnostics 10、localization_message 7、localization_supervisor 9、localization_faults 7、motion_compensation 缓存 3。
- 新文件 `ruff check` 全部通过；仓库原有文件中存在的超长行（`report.py`、`perception.py` 等）未改动。

## 8. 赛前最该弄懂的 3 个概念（用今晚的真实数据）

### 概念 1：SE(3) 与 Sim(3) 对齐——"0.036 m" 到底是什么意思
- cuVSLAM 的世界坐标是"开机那一刻的相机"，EuRoC 真值的世界坐标是动捕系统。两者之间有一个未知的刚体变换，所以算 ATE 前先用 Umeyama 求一个**最佳旋转+平移（SE(3)）**把估计贴到真值上，剩下的残差 RMSE 才叫 ATE。MH_01 Stereo：0.036 m。
- 如果再允许一个**尺度**（Sim(3)），尺度误差也会被"吸收掉"。MH_01 的 Sim(3) 尺度是 1.0016（只差 0.16%），说明双目基线给了真正的米制尺度；MH_05 是 0.9856，即走 10 m 会少算约 14 cm。
- 所以：**只有 SE(3) 的 ATE 才是米制精度**；Sim(3) 的 RMSE 只用来诊断尺度。VDR 的输出连尺度都没有，所以只能在图像空间比方向（cos 0.95–0.98），绝不能写成"VDR 误差 xx 米"。
- 坐标链本身：`world_from_body = world_from_rig · rig_from_imu`。重标定文件给的是 cam0_from_imu，原始 EuRoC `T_BS` 给的是 body_from_cam0；两者互逆，实测差 0.36° / 7.6 mm——这就是"外参链是否接对"的检验。

### 概念 2：时间对齐——为什么"−12.5 ms"是假的
- 相机 20 Hz，IMU 200 Hz。用相邻两帧位姿差分得到的角速度，是这 50 ms 区间里的**平均**角速度；IMU 给的是 5 ms 一个的**瞬时**值，还带振动。
- 拿"平均值"去和"某一时刻的瞬时值"做互相关，会产生周期为 IMU 采样间隔的锯齿，峰值落在 −12.5 ms（相关 0.943）。把 IMU 也在同一个 50 ms 窗口内取平均后，峰值落在 +1.5 ms（相关 0.956），曲线变平滑。
- 教训：**比较两个信号前先确认它们代表的是同一个物理量**（区间平均 vs 瞬时）。到了真机上，OAK IMU 与相机没有官方硬件同步说明，`EKF2_EV_DELAY` 也要靠同样的方法估——这个工具（`estimate_time_offset(..., average_window_s=帧间隔)`）就是为那一步准备的。

### 概念 3：失效是渐进的——安全监督该看什么
- MH_04/MH_05 里 cuVSLAM **从未报告跟丢**，但 MH_05 在亮度跌到 14/255 的暗段之后，误差从 ~0.08 m 涨到 0.21 m 并且不再回来。MH_02 Inertial 甚至在"跟踪正常"状态下瞬间跳了约 0.5 m。
- 如果监督层只看 `tracking == true`，这些全部会被放过。所以 supervisor 同时看：时间（stale/outage/顺序）、物理合理性（隐含速度 >5 m/s 或角速度 >10 rad/s = pose_jump）、相机–IMU 时间一致性，以及（真机上）PX4 的 EV innovation test ratio。
- 对 PX4 的直接含义：`quality` 字段要真实地反映健康度，VIO 失效时发 −1，并且把 `EKF2_EV_QMIN` 设为 ≥1——否则（源码已核对）PX4 会把"失败"的数据照样融合进去。

## 9. 推荐下一步
1. 拿到硬件前：把 OAK 录制格式定成和 EuRoC 相同的目录结构（cam0/cam1/imu0 + data.csv），这样今晚的全部工具可以直接用在现场数据上。
2. 在 SITL 中把 `aegisland.localization.v1` → MAVLink ODOMETRY 桥接（只在 SITL，需你同意），验证 `EKF2_EV_QMIN`、`reset_counter` 与失效降级。
3. 在 Orin 上第一件事：按 `hardware-research.md` §6 测 cuVSLAM 的真实 track() 耗时（今晚看到的 ~8 ms 阶跃与 100+ ms 尖峰需要在 Orin 上确认）。
4. 若 FFS 许可允许：试 TensorRT FP16（一次只改这一项），并在 Orin 上测；否则以 OAK 原生深度为准。
