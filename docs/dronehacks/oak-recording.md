# OAK 录制工具：从拿到硬件到进入 benchmark

日期：2026-10-24。**还没有连过任何 OAK 设备**，下面的格式验证全部基于 EuRoC 重写和合成数据。

## 第一个小时的命令

```bash
cd ~/aegisland-agentic-vision/benchmarks/gps_denied
# 0. 设备在吗、是什么型号、有没有 IMU（录制工具也会自动检查并写进 recording.json）
~/.venvs/depthai/bin/python -c "import depthai as d; print(d.Device.getAllAvailableDevices())"
# 1. 录 60 s：静止 10 s，手持走动 50 s（IR 投射器会被强制关闭）
PYTHONPATH=../../src ~/.venvs/depthai/bin/python oak_record.py --output ~/aegisland-data/oak/first-60s --duration-s 60 --depth
# 2. 直接接入现有 benchmark（与 EuRoC 完全相同的命令）
LD_LIBRARY_PATH=... ~/.venvs/cuvslam-17.0.1/bin/python run_cuvslam_euroc.py \
  --sequence ~/aegisland-data/oak/first-60s/mav0 --cuvslam-examples <pinned examples> \
  --mode stereo --realtime --pair-by-timestamp --output ../results/gps_denied/cuvslam-stereo-oak-first-60s-<tag>
```

没有设备时，工具会报 `no OAK device found; nothing written` 并退出（已验证）。

## 组成

| 文件 | 作用 | 测试 |
|---|---|---|
| `src/aegisland/euroc_writer.py` | 与设备无关的 EuRoC writer：双目对、IMU、深度、标定 yaml、manifest、SHA256SUMS | `tests/test_euroc_writer.py`（6 个） |
| `benchmarks/gps_denied/oak_record.py` | DepthAI v3 前端（depthai 3.11.0，环境 `~/.venvs/depthai`） | 无设备，只验证了无设备路径；API 名称对照真实安装的包核对过 |
| `src/aegisland/kalibr_import.py`、`benchmarks/gps_denied/kalibr_tools.py` | Kalibr ↔ EuRoC 的双向转换 | `tests/test_kalibr_import.py`（2 个），外加一次端到端试跑 |
| `benchmarks/gps_denied/euroc_rewrite_check.py` | 格式一致性检查 | 见下 |

**格式验证（最关键的一条）：**
- 把 EuRoC MH_01 的前 400 帧（图像、IMU、标定）经过 writer 重写一遍；
- 再用 `run_cuvslam_euroc.py` 在原数据和重写数据上各跑一次；
- 结果：Stereo 和 Stereo-Inertial 的 trajectory 都**逐字节相同**（SHA `73344849…` 和 `2f8ed28d…`）。
- 结果目录：`cuvslam-*-MH_01_easy-*-writercheck-{original,rewritten}`。
- 这说明 writer 的输出能被现有 runner 原样读取，而且读出来的东西完全一样。

## 录下来的东西

- `cam0/`、`cam1/`：左右原始单色图，8 bit PNG（无损），时间戳是**设备时钟**的 ns。
  - 左右按 sequence number 配对，|Δt| > 1 ms 的拒收并计数；
  - 实测的左右时间差写在 `cam_pairs.csv`。
- `imu0/data.csv`：gyro 和 accel 的报告各有自己的时间戳，accel 线性插值到 gyro 时刻上（不外推）。数值用 `repr()` 写出，是精确的往返表示。
- `depth0/`（可选，`--depth`）：设备端 StereoDepth 的输出，16 bit PNG，单位 mm。这一项**不在** EuRoC 标准里，现有工具会忽略它。
- `host_timestamps.csv`：设备时间戳和 host 同步后时间戳的对照，用来分析延迟。
- `exposure.csv`：每帧的曝光时间（µs）和 ISO。昨天的研究里亮度是重要信号，所以把曝光也录下来。
- 标定 yaml：来自 EEPROM 出厂标定。
- `recording.json`：设备型号、IMU 类型、投射器状态、计数、被拒收的样本，以及完整的出厂畸变系数。

## 哪些部分依赖具体型号 ⚠️

| 项目 | 依赖什么 | 工具怎么处理 |
|---|---|---|
| 有没有 IMU | OAK-D Lite 有些批次没有 IMU；Pro / S2 有 BMI270 或 BNO086 | 运行时 `getConnectedIMU()`；没有 IMU 就不写 imu0，只能跑 Stereo。`--no-imu` 可以强制关闭 |
| 分辨率 | OV7251（Lite）640×480；OV9282（D / Pro / S2 / W）1280×800 或 640×400 | `--resolution`，默认 640 400；**不同分辨率要分别标定** |
| 快门 | 双目是 global shutter（OV7251 / OV9282）；彩色相机通常是 rolling shutter | 只用 CAM_B/CAM_C 双目，不用彩色 |
| IR 点阵投射器 | 只有 Pro 型号有 | 有就强制关闭（`setIrLaserDotProjectorIntensity(0)`），状态写进 manifest |
| 镜头模型 | 普通 OAK 是 perspective（最多 14 个系数）；W 型号是 fisheye | perspective 只保留 k1,k2,p1,p2，写成 `sensor.yaml`（Brown）；fisheye 保留 k1..k4，写成 `sensor_recalibrated.yaml`。**cuVSLAM 的 loader 按文件名选模型** |
| 外参单位与方向 | DepthAI 默认返回**厘米** | 显式请求 METER；IMU↔相机变换的方向待用 Kalibr 确认 |
| IMU 噪声参数 | EEPROM 里可能没有 | 先用数据表量级的默认值，并在 manifest 里写明 "NOT measured"；之后用 Allan variance 替换 |
| 双目基线 | 型号不同（例如 75 mm） | 来自 EEPROM，标定时核对 |

## 还没有验证的

- 真实设备上的 pipeline 构建、帧率、丢帧、USB 带宽，以及左右 sequence number 是否总能配上。
- IMU 报告的打包方式：每个 packet 里 gyro 和 accel 的时间戳关系。
- 出厂标定里 IMU 外参的方向。
