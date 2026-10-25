# 下午报告 2026-10-24：为真实硬件做准备、SITL 演示、cu12 一致性

## 0. 边界与决定执行情况

| 你的决定 | 执行情况 |
|---|---|
| 1. degraded 只降权，不升级为 quality 0 | `TRUST_POLICY` 保持不变：degraded 映射为方差下限 1 m²，untrusted 映射为 quality 0。赛前自检的第 4 项会检查这一条 |
| 2. 在独立 venv 里验证 cu12 wheel | `~/.venvs/cuvslam-17.0.1-cu12`：Python 3.10，CUDA 12.6 数学库，1.5 GB。结果见 §1 |
| 3. 接受 19 MB 入库 | 已按此提交。本轮新增 11.7 MB（159 个文件，单个文件最大 0.8 MB） |
| 4. 保留 `~/work/px4` | 保留。演示和自检的运行目录也放在 `~/work/` 下 |
| 清理过期的 monitor | 4 个（bc616ar27、bkt5b8wb6、bhdtbyi4z、bpp60s1yd）都已过期，`TaskStop` 查不到它们；没有残留进程 |

- 没有 push，没有付费，没有发送任何消息。
- PX4 只在本机 SITL 里运行，载具从不解锁。CONTROL OUTPUT = DISABLED。
- 数字都来自开发机 RTX 4060 Ti 上的 EuRoC 回放和 PX4 v1.17.0 SITL，**不是 Orin，也不是我们的相机**。
- 测试：见 §8。

## 1. cu12 wheel 与当前 cu13 版本在 MH_01 上是否一致

运行条件：
- 两个版本交替运行，每个版本 × 每种模式各跑 2 次；
- 全部按新规则 `--realtime`（20 fps）运行，并记录 GPU 频率；
- 结果目录：`cuvslam-*-MH_01_easy-20261024-114757-{cu12,cu13}-realtime-run{1,2}`。

| 模式 | cu13（开发机现用，Python 3.12） | cu12（与 Orin 同一个构建族，Python 3.10，x86_64） | 结论 |
|---|---|---|---|
| Stereo-Inertial | ATE 0.0335 m，SHA `c4da193c` ×2 | ATE 0.0335 m，SHA `c4da193c` ×2 | **两个版本逐字节相同** |
| Stereo | ATE 0.0356 m，SHA `c13dd481` ×2 | ATE **0.0284 m**，SHA `c86f32f5` ×2 | **每个版本自身可复现，但两个版本之间不同** |
| `track()` p50 / p95（实时） | 6.3–7.1 / 12–17 ms | 5.9–7.0 / 12–17 ms（一次是 1.4 / 10 ms，那次 SM 频率中位数为 555 MHz） | 相近，由 GPU 频率决定 |

- 含义：Orin 用的 cu12 构建在 Stereo 模式下会给出与开发机不同的结果，量级约为 7 mm ATE。
- 开发机上的 Stereo 精度数字不能直接当成 Orin 的数字。
- Inertial 模式的轨迹在两个构建之间完全一致。
- 注意：这里的 cu12 是 x86_64 版本，不是 aarch64，Orin 上仍需实测。

## 2. 主线：为拿到真实硬件做准备

### 2.1 DepthAI 录制工具（`docs/dronehacks/oak-recording.md`）

- `src/aegisland/euroc_writer.py`：与设备无关的 EuRoC writer，6 个单元测试。主要行为：
  - 左右帧按时间戳配对，相差 >1 ms 的拒收；
  - 乱序或重复的样本拒收并计数；
  - IMU 用 `repr()` 无损写出；
  - 深度写成 16 bit 毫米 PNG；
  - 生成 manifest 和 SHA256SUMS。
- `benchmarks/gps_denied/oak_record.py`：DepthAI v3 前端（depthai 3.11.0，环境 `~/.venvs/depthai`）。录制内容：
  - 双目原图和设备端深度；
  - IMU：gyro 和 accel 合并，accel 插值到 gyro 的时间戳上；
  - EEPROM 标定、曝光时间和 ISO；
  - **所有时间戳都用设备时钟。**
  - 没有设备时报错退出，什么都不写（已验证）。
- **格式验证：** MH_01 前 400 帧经过 writer 重写后，用 `run_cuvslam_euroc.py` 原样运行，Stereo 和 Inertial 的 trajectory 都和原数据**逐字节相同**（`*-writercheck-*`）。
- 依赖具体型号的部分已在文档表格中逐项列出：
  - 是否有 IMU；
  - 分辨率：OV7251 是 640×480，OV9282 是 1280×800 或 640×400；
  - 快门：只用 global shutter 的双目；
  - IR 投射器：有就强制关闭；
  - 镜头模型与文件名；
  - 外参默认单位是**厘米**。
- 核对 API 时发现两个容易踩的坑：
  1. **cuVSLAM 的 EuRoC loader 按文件名选择畸变模型**：`sensor_recalibrated.yaml` 当作 fisheye、外参相对 cam0；`sensor.yaml` 当作 Brown、外参相对 IMU。writer 会按镜头模型写出对应的文件名。
  2. DepthAI 返回外参时默认单位是**厘米**。

### 2.2 相机–IMU 标定流程（英文，`docs/dronehacks/camera-imu-calibration.md`）

- 工具：Kalibr（Docker）和 Allan variance。
- 标定板：AprilGrid 6×6，tag 0.088 m，要哑光、刚性，并用卡尺实测 tag 尺寸。
- 三段录制：
  - 只录相机，60–90 s；
  - 相机 + IMU，60–120 s，先后激励 6 个轴，首尾各静止 5 s；
  - 静止 IMU，≥2 h。
- 验收门限：
  - 相机重投影 RMS < 0.3 px，相机–IMU < 0.5 px；
  - 与 CAD 相差 < 5 mm / 1°；
  - |时间偏移| < 5 ms，两次标定之间相差 < 1 ms。
- 转换工具 `kalibr_tools.py`：
  - `to-kalibr`：转成 bagcreater 需要的目录结构；
  - `from-kalibr`：写出 cuVSLAM 需要的 yaml，并按时间偏移修正 IMU 时间戳。输出是新目录，图像用硬链接，不修改原录制。
- 核心映射在 `kalibr_import.py`，有 2 个单元测试。两个方向都用合成的 Kalibr yaml 端到端跑通了：1 ms 的偏移正确作用到了 IMU 时间戳上。
- **还没有真正运行过 Kalibr**，因为还没有标定数据。

### 2.3 赛前自检脚本：四个陷阱，每项给出 PASS/FAIL（`preflight_sitl_check.py`）

| 检查项 | 正常配置（`sitl-preflight-20261024-120137`） | 故意配错 `EKF2_EV_QMIN=0`（`sitl-preflight-negative-qmin0-20261024-120533`） |
|---|---|---|
| 1 ENU/NED | **PASS**：航向误差 0.02°；bridge 拒绝 ENU 帧和 dataset 帧 | — |
| 2 TIMESYNC | **PASS**：发送 150 ms 延迟，PX4 保留了 160 ms | — |
| 3 慢漂移照收 | **PASS**：漂移 1 m，EKF2 只跟了 0.01 m | **FAIL**：EKF2 跟了 0.997 m |
| 4 quality 0 → failsafe | **PASS**：degraded 时发 80、untrusted 时发 0；4.4 s 后位置失效 | **FAIL**：位置一直没有失效 |

- 一条命令运行；全部 PASS 时退出码为 0。可以用 `--only` 只跑部分检查，用 `--ekf2-ev-qmin` 指定被测的配置。
- **限制，晚些时候才发现：** 第 3 项用的是 SIH 理想 IMU。在真实 IMU 上，quality 0 的代价是很大的惯性漂移（见 §7）。脚本的 docstring 和 SITL 文档里都已写明这一点。

### 2.4 性能测试规则（已写进 `gps-denied-benchmark-plan.md` §0）

- runner 新增 `--realtime`：按数据集时间戳节流，EuRoC 即 20 fps。
- runner 自动记录 `gpu_clocks.csv`（P-state、SM 和显存频率、利用率、功耗，每 0.5 s 一次）。
- `run.json` 新增 `valid_for_performance_claims`：只有 realtime 运行才为 true。
- 规则：
  - 能锁频就锁频（Orin 上用 `jetson_clocks`）；锁不了就必须记录频率；
  - 报告 p50/p95 时同时给出 SM 频率范围；
  - 确定性要用不同时段的 ≥3 次运行来报告。

## 3. 演示准备（`docs/dronehacks/sitl-demos.md`，讲解词中英文各一份）

**新工具 `sitl_hil_replay.py`：** AegisLand 充当 PX4 `none_iris` 机型的外部仿真器（TCP 4560，lockstep），把 **EuRoC 真实的 IMU** 作为 HIL_SENSOR 喂给 EKF2。
- 这样 EKF2 是在运动中真正融合视觉和 IMU，弥补了昨天 SIH "只能静止" 的限制。
- 陀螺零偏用起飞前静止段估计，约 0.083 rad/s，相当于 PX4 在地面做的陀螺校准。

| 演示 | 一条命令 | 关键数字 |
|---|---|---|
| A 定位（MH_01，155 s） | `demos/demo_a.sh` | cuVSLAM 输出 RMS 0.036 m；**EKF2 RMS 0.054 m，最大 0.135 m** |
| B 渐进退化（MH_05 暗段） | `demos/demo_b.sh` | 暗段 45–60 s；cuVSLAM 0 帧跟丢；误差峰值约 0.3 m；`low_texture` 覆盖整个暗段；`rotation_inconsistent` 在暗段之后的 66–71 s 报警 |
| C 故障（VIO 冻结 2 s） | `demos/demo_c.sh` | 原始桥接：EKF2 最大误差 **1.04 m**；supervised 桥接：**0.15 m**，quality 80→0→80 |

**演示 B 与预期不同，如实说明：**
- 原来的设想是"陀螺一致性检查抓到暗段"。实际上**暗段期间陀螺残差一直低于阈值**，抓到暗段的是 VDR 匹配数（`low_texture`）。
- 讲解词按实际情况写了，没有改数据去迎合原来的设想。
- 降权也没有降低 EKF2 的误差：RMS 原始 0.173 m，降权后 0.180 m。

**做演示时发现并修复的 v3 缺陷：** 起步时陀螺零偏还没估计出来，`rotation_inconsistent` 就已经在报警。
- 实测：EuRoC 的零偏约 0.08 rad/s，所以 MH_05 演示开头有 2 s 误报。
- 修复：在积累到 20 帧零偏历史之前不做这项检查，并加了单元测试。
- 用修复后的代码重跑了 10 个故障回放（`fault-replay-*-signals-v3.1-*`）：
  - fault 级评分与 v3 完全相同，6/6；
  - degraded 误报率在 MH_02 上略有下降（10.0%→9.6%、6.2%→5.8%）；
  - 对真值事件的检出数不变。

## 4. 失败与订正

1. 下午的第一张演示 A 图里，cuVSLAM 曲线是按"送达时间"画的，应该按采集时间画（两者差 30 ms 的延迟）。已修正；修正后 cuVSLAM RMS 从 0.039 m 变成 0.036 m。
2. HIL 工具第一次试跑时，我把 EKF2 和发送记录的时间轴错开了 1 s，误以为"EKF 没有跟随"。检查后发现 PX4 收到的位置与发送的完全一致，问题出在我比较时的时间轴上。
3. 演示 C 图的 y 轴刻度 −1 和 0 重叠了，已修正。
4. 演示 B 的结果与设想不符，见 §3。
5. 昨天的自检"慢漂移被挡住"只在理想 IMU 下成立，见 §2.3 和 §7。

## 5. 需要你决定

1. **Orin 上的 Stereo 精度：** cu12 和 cu13 构建在 Stereo 下不一致（0.028 对 0.036 m）。是否把"Orin 构建重新做精度基线"列为拿到 Orin 后的必做项？我建议列为必做。
2. **长时间退化时该怎么处理（§7 的中间方案）：** 目前所有方案都有明显代价。是否需要硬件里的第二个独立位置或速度源，比如光流加测距？这需要和 Chelsea 确认硬件。
3. 本轮新增入库 11.7 MB。

## 6. 推荐的下一步

1. 拿到 OAK 后按 `oak-recording.md` 先录 60 s，接着跑 benchmark，再按标定流程录 R1–R3。
2. 用自己相机的数据重新拟合 v3 阈值。演示 B 也说明，EuRoC 上的陀螺阈值在暗段不起作用。
3. 用 HIL 工具回放我们自己录的数据。

## 7. 概念：为什么只调大方差挡不住慢漂移，而 quality 0 能挡住却会触发 failsafe；两者之间有没有中间方案

**先看今天在 HIL（真实 IMU）里测到的数据**（`demo-drift-marking-20261024-121340`）：
- 在 MH_01 回放中，从第 40 s 起，往视觉位置里注入每秒 5 cm 的漂移，持续 20 s，共 1 m。
- 假设 supervisor 已经正确判定（用"预言机"直接给出判定），只比较不同标记方式的效果：

| 标记方式 | 漂移结束时 EKF2 的真实误差 | EKF2 自己报告的不确定度 eph | 位置是否失效 |
|---|---|---|---|
| 不标记 | 0.98 m | 0.05 m | 否 |
| 方差 1 m² | 1.04 m | 0.33 m | 否 |
| 方差 25 m² | 1.31 m | 1.28 m | 否 |
| 方差 400 m² | 2.96 m | 3.96 m | 否 |
| quality 0 | **10.0 m** | 17.9 m | 4.6 s 后失效 |

**为什么调大方差挡不住慢漂移：**
- 没有其他位置源时，EKF2 判断视觉位置对不对，唯一的参照是 IMU 推算出来的位置。
- 每秒 5 cm 的漂移，每帧只差 2.5 mm，远小于 IMU 推算本身的不确定度，所以 innovation 检验永远通过，test ratio 约等于 0。
- 调大方差只是让 EKF2 "跟得慢一点"，相当于降低了跟随带宽；只要漂移持续得比这个时间常数长，最终还是会全部跟上。
- 更糟的是，跟得慢意味着更依赖 IMU。而这块 IMU（EuRoC 的 ADIS16448，零偏约 0.08 rad/s）在 20 s 里的惯性推算误差比漂移本身还大，所以方差越大，误差反而越大：1 m² 时 1.04 m，400 m² 时 2.96 m。

**为什么 quality 0 能挡住，却会触发 failsafe：**
- quality 0 加上 `EKF2_EV_QMIN≥1`，相当于把视觉的权重直接设为 0，漂移一点都进不去。
- 但这同时也拿走了唯一的位置约束：
  - EKF2 只能靠 IMU 积分，20 s 里误差到了 10 m；
  - 它知道自己失去了位置，所以 eph 一路涨到 17.9 m；
  - 4.6 s 后 `xy_valid` 变成 false，在真实飞行中这会触发位置失效的 failsafe。
- 昨天 SIH 自检显示"quality 0 不跟随漂移、误差很小"，是因为 SIH 的 IMU 没有任何误差。换成真实 IMU，这个结论只剩前一半成立。

**有没有中间方案？有几个，但都不是免费的：**
1. **把方差调到"诚实"的程度，而不是"挡住"的程度。**
   - 表里方差 25 m² 那一行，EKF2 报告的 eph（1.28 m）和真实误差（1.31 m）基本一致；不标记时，eph 是 0.05 m 而真实误差是 0.98 m，自信地错了 20 倍。
   - 降权的真正价值不在于减小误差，而在于让飞控知道自己有多不确定，从而让 `COM_POS_FS_EPH`（默认 5 m）这类下游逻辑按真实的不确定度来决定。
   - 代价：误差本身略增（1.31 m 对 0.98 m）。方差值需要按 supervisor 估计的漂移率来设定，目前还没有这样的估计器。
2. **短时 quality 0，用于硬故障。**
   - 只在故障持续时间小于 EKF2 失效时限（约 4.5–5 s）时用。
   - 演示 C 就是这种情况：冻结 2 s，quality 0 让误差只到 0.15 m，不会触发 failsafe。
   - 对持续 20 s 的慢漂移不适用。
3. **第二个独立的位置或速度源**，例如 PX4 原生支持的光流加测距。
   - 这样把视觉标成 quality 0 后，EKF2 仍然有速度约束，不会进入纯惯性推算，也就不会马上 failsafe。
   - 这是真正的中间方案，但依赖硬件（尚未确认），而且光流本身也会漂。
4. **确认重定位后用 `reset_counter` 重置。** 如果之后能确认某个位置是对的（例如回到已知地标），就用 `reset_counter+1` 让 EKF2 干净地跳过去，不让滤波器和观测"打架"。目前我们没有地图或地标，所以这只是一个方向。
5. **任务层面的应对：** 例如退化时减速、悬停或降落。这属于控制决策，超出了我们 read-only、CONTROL OUTPUT = DISABLED 的边界，需要团队决定。

**一句话总结：**
- 调大方差改变的是 EKF2 "有多相信自己"，改变不了"它跟谁走"；
- quality 0 改变的是"它跟谁走"，但手里只剩一块会漂的 IMU；
- 真正的中间路线要么是让不确定度诚实地传下去，交给下游决定；要么是给 EKF2 另一个可以依靠的独立信息源。

## 8. 测试

- 共收集 289 个测试（上一轮 280），新增：
  - `test_euroc_writer` 6 个；
  - `test_kalibr_import` 2 个；
  - supervisor 零偏预热 1 个。
- 改动过的模块的定向测试共 70 个，全部通过；ruff 全部通过。
- **全量测试：289 个，288 passed、1 skipped（可选的 CUDA 测试）、0 failures、0 errors，exit 0，耗时 1235 s。** 来源是 junit xml。该项目的 pytest 配置不打印汇总行，所以第一次运行只看到了进度点，我没有据此下结论，而是加上 junit 输出重跑了一次。
