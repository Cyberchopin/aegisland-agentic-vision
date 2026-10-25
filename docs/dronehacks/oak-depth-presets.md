# OAK StereoDepth：参数取舍、候选预设、验证流程

日期：2026-10-24。**没有实机，下面所有预设都没有在硬件上验证。**

数字来源：
- **[v3]** Luxonis StereoDepth 节点文档（DepthAI v3）：<https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes/stereo_depth>
- **[cfg]** Luxonis "Configuring Stereo Depth"：<https://docs.luxonis.com/hardware/platform/depth/configuring-stereo-depth>
- **[v2]** DepthAI v2 API 的 StereoDepth 页：<https://docs.luxonis.com/projects/api/en/latest/components/nodes/stereo_depth/>
- **[pkg]** 本机安装的 `depthai==3.11.0`，直接 `dir()` 核对的 API 名称

关于引用：只参考了官方文档，没有复制任何示例代码进仓库。DepthAI 本身是 MIT 许可（PyPI 元数据）；文档页面的许可没有逐页核对，所以这里只做转述并附链接。

## 1. v2 和 v3 API 的区别（以免照着旧教程写错）

| | v2（depthai 2.x） | v3（depthai 3.x，本机 3.11.0） |
|---|---|---|
| 相机节点 | `MonoCamera` + `setResolution` | `Camera.build(socket)` + `requestOutput(size, type, fps)` [pkg] |
| 连线 | `left.out.link(stereo.left)`，输出要接 `XLinkOut`，再用 `Device(pipeline)` 和 `getOutputQueue()` | `StereoDepth.build(left_out, right_out, presetMode)`；`stereo.depth.createOutputQueue()`；`pipeline.start()` [pkg] |
| 预设 | `HIGH_ACCURACY` / `HIGH_DENSITY`（已标记弃用），以及 `DEFAULT`、`FACE`、`HIGH_DETAIL`、`ROBOTICS` [v2] | `DEFAULT`、`FACE`、`HIGH_DETAIL`、`ROBOTICS`、`FAST_ACCURACY`、`FAST_DENSITY`，另有 RVC4 用的 `ACCURACY` / `DENSITY` [pkg][v3] |
| 扩展视差与亚像素同时开 | v2 页写着 "Currently incompatible"（两者不能同时开）[v2] | v3 的 `FACE` / `HIGH_DETAIL` 预设本身就是"扩展视差 + 5 bit 亚像素"[cfg] → **新版本已经可以同时开**。v2 页面的说法过时了（以设备固件为准，未验证） |
| 时间戳 | `getTimestampDevice()` / `getTimestamp()` | 相同，另有 `dai.Clock.now()` 可以和 `getTimestamp()` 相减得到延迟 [pkg] |

比赛用的 Orin 和 OAK 型号确认之前，代码一律按 v3 写（`oak_record.py` 和 `oak_depth_validation.py` 都是）。

## 2. 每个参数的取舍（数字都是 RVC2，也就是 OAK-D / Pro / Lite 这一代）

| 参数 | 作用 | 精度 | 近距离盲区 | 延迟和算力 |
|---|---|---|---|---|
| 分辨率 400P 对 800P | 输入尺寸 | 800P 远处更准 | **400P 的 MinZ 是 800P 的一半** [cfg]（焦距像素数减半） | 校正 1.5 对 6 ms；立体匹配（LR+Ext+Sub）20.4 对 62 ms [v2] |
| 左右一致性检查（LR check） | 去掉遮挡边缘的错误匹配 | 边缘更干净 | — | 所有预设都开；RVC2 上在硬件里做，阈值默认 10（HIGH_DETAIL 是 5）[cfg] |
| 扩展视差 | 视差搜索 0–95 扩到 0–190 [v2] | — | **MinZ 减半**：OAK-D 800P 约 70 cm → 约 35 cm [cfg] | 整条流水线要跑两遍，增加延迟和占用 [cfg] |
| 亚像素（3/4/5 bit） | 小数视差 | 远处更准（754/1506/3010 个深度级）[cfg] | — | 有硬件开销；800P 要搬约 98 MB 的 cost 数据；"位数不影响延迟" [v2]；**4/5 bit 时中值滤波失效** [v2] |
| 置信度阈值（0–255，越低越严） | 拒掉低置信度像素 | 越低越准、空洞越多 | — | 几乎无开销；预设值：DEFAULT / ROBOTICS 是 15，HIGH_DETAIL 是 55 [cfg] |
| 中值滤波 3/5/7 | 去椒盐噪声 | 会模糊边缘 | — | 约 0.8–1.8 ms [v2]；亚像素 ≥4 bit 时被禁用 |
| 空间滤波（hole filling） | 补洞、平滑 | 可能"补"出假表面 | — | 1 MP 时可能成为瓶颈 [cfg] |
| 时间滤波 | 帧间平滑 | 静态场景更好 | — | **运动时会产生拖影**，ROBOTICS 预设关闭了它 [cfg] |
| 视差偏移（disparity shift） | 把搜索窗口平移到近处 | 近处更准 | 让 MinZ 更小，但 MaxZ 下降得更快 [v2] | **不增加计算和延迟** [cfg] |
| 阈值滤波（min/max range） | 截断距离范围 | — | — | 很小 |
| IR 点阵投射器（仅 Pro） | 主动纹理 | 弱纹理面上更密 | — | 无；**和 VIO 冲突**，因为点阵随相机移动，会产生假特征 |

MinZ 的计算公式：MinZ = f_px × B / d_max。以 OAK-D 800P 为例，f = 882.5 px，B = 7.5 cm，d_max = 95，得到约 0.70 m [cfg]。这个公式已写进 `depth_sectors.min_depth_m`，并用这组数字做了单元测试。

## 3. 候选预设（`oak_depth_validation.py` 里的 `PRESETS`）

| 名字 | 基础预设 | 分辨率 / fps | 主要设置 | 适合什么场景 | 预期代价（来自文档，未验证） |
|---|---|---|---|---|---|
| **nav** | ROBOTICS | 640×400 @30 | LR 开，扩展关，亚像素 3 bit，时间滤波关（预设默认），最远 10 m | 飞行中的前向避障：低延迟、无拖影 | MinZ ≈ 70 cm 的一半，也就是 35 cm 左右 💭（按 400P 推算）；立体匹配约 3–20 ms |
| **near** | ROBOTICS | 640×400 @30 | 同上，但**扩展视差开**，最远 4 m | 起降、穿门、靠近墙面 | MinZ 再减半，约 17–20 cm（受对焦限制，约 19.6 cm [cfg]）；延迟增加 |
| **accuracy** | HIGH_DETAIL | 1280×800 @15 | LR 开，亚像素 5 bit，扩展关，最远 15 m | 静态测量、标定检查、建图；**不适合避障** | 立体匹配最多约 62 ms，大约 16 fps 级别 [v2] |

三套预设都**关闭了 IR 投射器**。理由：要和 cuVSLAM 同时运行（见 `hardware-research.md`）。如果以后只用深度、不跑 VIO，可以再单独评估打开投射器的效果。

## 4. 拿到硬件后的深度验证流程（一条命令）

```bash
cd ~/aegisland-agentic-vision/benchmarks/gps_denied
PYTHONPATH=../../src ~/.venvs/depthai/bin/python oak_depth_validation.py \
  --distances 0.3 0.5 1 2 3 5 --presets nav near accuracy --frames 60 \
  --output ../results/gps_denied/oak-depth-validation-<型号>-$(date +%Y%m%d-%H%M%S)
```

准备工作：
- 一块平整、有纹理的板子（A1 打印的随机纹理，贴在硬板上），或者一面有纹理的墙；
- 卷尺，从相机前玻璃沿光轴量距离；
- 室内正常照明。

每个距离、每套预设会输出以下指标（`src/aegisland/depth_validation.py`，3 个单元测试）：
- **偏差与相对误差：** 中心 ROI 的中位数减去卷尺距离。参考：Luxonis 说 OV9282 在约 70 cm、纹理好时误差 <2%（±1.5 cm）[cfg]。
- **时间噪声：** ROI 中位数在帧之间的标准差。
- **平面残差 RMS：** 反投影成三维点后做最小二乘平面拟合，看起伏程度。
- **无效像素比例：** ROI 内和整帧分别统计。
- **延迟：** `dai.Clock.now() − getTimestamp()`，取 p50/p95。
- **实际 fps。**

结果直接写进 `benchmarks/results/gps_denied/`，带 SHA256SUMS。没有设备时脚本报错退出，不写任何东西（已验证）。

**通过门限（暂定，2026-10-24 确认按此执行；拿到真机后根据实测再改）：**
- nav：1–3 m 时 |相对误差| < 3%，ROI 无效 < 10%，延迟 p95 < 60 ms；
- near：0.3–1 m 时 |误差| < 3 cm；
- 任何预设：0.3 m 处要么给出正确距离，要么判为无效。**不允许给出远于真实距离的数**，因为那会被误当成"前方没有障碍"。

## 5. 避障输入模块（与设备无关）

- 文件：`src/aegisland/depth_sectors.py`，5 个单元测试。
- 输入：深度（米）、有效掩码、时间戳、内参。OAK 的 uint16 毫米深度用 `depth_from_oak_mm` 转换。
- 输出：每个方位扇区的最近距离（取有效距离的 5 分位数，抗散点噪声）、有效像素比例、判定和原因、数据年龄。
- **只输出判定，不输出任何控制量。**
- 判定规则：
  - "盲区内有回波" 和 "有效像素太少" 都判为**无效**，而不是 "没有障碍"；
  - 数据超过 200 ms 判为不新鲜；
  - 距离按沿射线的长度算，不用 z。
- **EuRoC 替身测试**（`depth_sectors_euroc.py`，SGBM 替代 OAK 深度，结果在 `depth-sectors-sgbm-*`）：
  - 开发机耗时：校正 3.3 ms，SGBM 46 ms，扇区计算 7.7 ms。扇区计算原来要 19 ms，缓存射线网格后降到 7.7 ms，输出逐项相同。
  - 整帧有效像素比例的中位数：MH_01 0.93，MH_05 0.94。
  - 发现并修正了一个问题：视差落在搜索上限（≥94 px）的匹配，原先被当成 0.49 m 处的障碍物。现在把它们算作 "盲区内有回波"，因为它们的含义是 "这么近或者更近、无法测量"。修正之前的结果目录已保留，并标记为 SUPERSEDED。
  - 未解释：两侧扇区（±25–35°）的 p10 约 0.5 m，可能是校正图像边缘的伪影，**未验证**。
  - EuRoC 没有深度真值，以上都只是一致性指标。
