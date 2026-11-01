# DroneHacks perception-only 交接

当前工作目录是唯一写入的 perception worktree，分支 `work/dronehacks-perception`，起点
`e13b6984b2479691e9178223fbbaa5a3460585e8`。本次改动没有提交、推送或部署。没有接入真实
飞控、发送电机/解锁/飞行命令，也没有新增飞行执行入口。

| 产物 | 位置 |
| --- | --- |
| 感知模块及输入契约 | [course_perception.py](../../src/aegisland/course_perception.py) |
| 集中阈值及 provenance/hash | [course_config.py](../../src/aegisland/course_config.py) |
| 离线合成场景与失败案例 | [course_synthetic.py](../../src/aegisland/course_synthetic.py) |
| 本地 synthetic / NPZ 回放 | [course_replay.py](../../src/aegisland/course_replay.py) |
| 单元/合成/边界/回放测试 | [test_course_perception.py](../../tests/test_course_perception.py) |
| 圆弧/阴影/墙沿/片段、gap 范围与执行边界测试 | [test_course_adversarial.py](../../tests/test_course_adversarial.py) |
| 设计和字段解释 | [course-perception-design.md](../../docs/course-perception-design.md) |
| 通用性证据、实测边界和残留误检 | [generality-defense.md](../../docs/generality-defense.md) |
| 当前 v2，36 帧硬化 trace/report/config | [report.md](../../examples/course-perception-hardened/report.md)、[report.json](../../examples/course-perception-hardened/report.json)、[trace.jsonl](../../examples/course-perception-hardened/trace.jsonl)、[config.json](../../examples/course-perception-hardened/config.json) |
| v1 历史快照，保留初版误检证据 | [report.md](../../examples/course-perception/report.md)；其中源码哈希不代表当前代码 |

实现优先级依次是走廊/迷宫线、Gate A 开口候选、tunnel 低光/深度退化、Gate B/chicane 间隙、
pad 外圈与下降几何条件。没有实现 window 或 return 路线逻辑。所有输出仅为 advisory evidence；
低光重试即使成功，状态仍为 degraded、权限变化 NONE。`ZoneCandidate.safe` 始终 false。

运行使用已有环境的解释器，没有修改其依赖，也没有往其他 checkout 写入。系统 `python3`
缺少 cv2，因此精确复现命令如下；代码通过 `PYTHONPATH=src` 指向当前 worktree。

```bash
PYTHONPATH=src <main-checkout>/.venv/bin/python -B \
  -m aegisland.course_replay --adversarial --output examples/course-perception-hardened

<main-checkout>/.venv/bin/python -B -m pytest \
  tests/test_course_adversarial.py tests/test_course_perception.py tests/test_depth_sectors.py \
  tests/test_perception_quality.py tests/test_perception_trust.py \
  tests/test_recovery.py tests/test_profiling.py tests/test_perception_scene_benchmark.py \
  tests/test_gps_denied.py

<main-checkout>/.venv/bin/ruff check \
  src/aegisland/course_config.py src/aegisland/course_perception.py \
  src/aegisland/course_synthetic.py src/aegisland/course_replay.py \
  src/aegisland/perception.py src/aegisland/trace.py \
  tests/test_course_perception.py tests/test_course_adversarial.py --output-format concise

git diff --check
```

以上是硬化后的实际执行命令：170 passed in 11.17s；Ruff 输出 `All checks passed!`，exit 0；
`git diff --check` 无输出，exit 0；回放生成 36 帧，executed=false。回放命令中的目录现在已存在，
复跑时请换一个新输出目录。此前初版运行通过 122 项，不能替代此次硬化回归。
没有运行全仓测试、真实硬件或真实飞行验证。NPZ roundtrip 测试用的是保存后的合成帧，没有
把它算作真实相机数据。运行输出目录必须是新目录，以免覆盖上次证据；可用
`--config examples/course-perception-hardened/config.json` 复用完整参数，或在 JSON 中修改并声明新的
provenance。录制输入格式见设计文档，必须提供时间戳、视角以及深度组的实际内参/标定 ID。

本次只在原结构中增加线路几何检查、gap 合约及 RGB/depth 冲突代理，没有改写架构。
所有新增阈值仍在 CourseConfig，provenance 为 synthetic-v2-adversarial 启发值，未现场标定。

线路硬化前，partial_ring、curved_shadow（dark polarity）、wall_edge 和 wall_highlight
均出现 1.000 的误检，旧 clipped_pad 为约 0.615。现在这些图像全部拒绝；新测试改变图像尺寸、
比例、位置和圆弧尺度，另有 short_fragment 和没有近场端点的长片段反例。几何检查包括拟合
误差、宽度一致性、足够行数、近场支持、侧边裁切和上部墙沿连续性。几何失败或多候选竞争
时 confidence=0、points_normalized=[]，保留 candidate_confidence 和理由供诊断。
干净斜线/直线/L 形转角正例通过；shadow 的对抗测试显式使用 dark polarity，CLI 示例统一
使用默认 bright 配置。硬化初测曾误拒绝两个 L 形转角正例，排除宽转角行后恢复，原测试保留。

每个 gap（包括 opening 内嵌 gap）现在显式 scope=local_camera_z_probe_plane，
local_plane_supported 只描述局部宽度；supported 是其兼容别名。full_3d_traversability 永远
UNKNOWN，supports_3d_traversal=false、authority_change=NONE、executed=false。实际采样
ROI 及未评估体积随结果保存。探测平面外/ROI 外存在障碍的反例中局部支持可以仍为 true，
但不会成为全三维通行或执行许可。过期深度不输出宽度或列区间；缺标定在输入检查阶段抛出
ValueError，回放不写出任何事件。RGB 紧凑阻挡结构与平坦远深度冲突时，整体及全部内嵌
候选均输出 rgb_depth_conflict_suspected、局部支持 false、宽度 unknown。视觉开口对应近距
深度墙时也拒绝支持。

其余回归观察：高对比线和单转角通过；平行线、分叉、断线拒绝；有纹理低光
线路约 0.475→0.560 恢复，近黑图拒绝；正常开口有独立深度支持，堵塞/未知开口拒绝；
chicane 投影宽度 1.0625 m，窄缝 0.2857 m。正常亮/暗 pad 的静态几何条件通过，侵入物、
倾斜、小尺寸、实心圆、明显椭圆和裁切外圈拒绝。CPU observe p50=4.788 ms、p95=8.578 ms、
max=17.562 ms，单次 36 帧含冷启动，不是 Orin 实测，也不作为相对初版的加速结论。

附加边界验证：capabilities 始终将 window/return navigation 标为 unimplemented、metric
VIO/SLAM 标为 not_provided。改变 frame 编号、全部时间戳平移、穿插其他场景，不改变线路
或 gap 结果；无固定 course route、waypoint、比赛像素坐标或比赛时刻切换。fixture 中的绘图
坐标和采集时间是测试输入。现有 VisualDeadReckoner 输出仍是归一化、无量纲图像运动，
不是米制 VIO/SLAM；未更改该模块。证据、嵌套 gap、trace 和 report 的 executed/command_executed
递归检查均为 false；不产生 action 或 command 对象。CONTROL OUTPUT 仍为 DISABLED。

最重要的使用限制：开口身份仍为 unverified；间隙只是相机 z=2 m 探测平面的局部投影；
`descent_geometry_suitable` 只覆盖可见静态几何/深度，不代表可以下降。动态障碍、重力对齐、
完整三维机体足迹和接地动力学未评估。已测裁切圆弧误检已拒绝，但未见的长半径近直圆弧、
局部墙上条纹和与真实线同形的阴影仍缺乏语义区分；强弯真实路径、侧边进入和遮挡路径
可能被保守拒绝。RGB/depth 冲突检查只是代理：可能拒绝地面涂装，也可能漏掉透明障碍、细杆、
无 RGB 纹理结构的障碍或系统性配准错误。校准 ID 只记录调用者声明，不能证明标定正确。
局部平面以外、遮挡体积、整机扫掠体积以及时序稳定性仍未知。所有 confidence 是启发得分，
不能当作正确概率。

下一步应采集/提供已有基本路线录制和标签，先验证地面线、出口候选、tunnel 恢复/伪恢复、
局部深度、间隙误差和落点拒绝分布，再根据实际相机标定与整机尺寸更新 provenance。当前
交付不授权将这些结果转换成真实飞控命令。
