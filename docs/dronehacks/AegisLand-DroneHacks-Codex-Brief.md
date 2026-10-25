# AegisLand · DroneHacks 两周执行说明

给执行任务的 Codex：请先完整阅读本文件，再检查当前工作目录与仓库。目标是为真实 GPS-denied 无人机任务做好准备，同时让 Shiori 读懂自己的感知、定位和性能数据。按下列阶段推进；每阶段先做最小可运行实验，再扩大验证。常规、可逆的本地工作直接执行，出现真实缺失信息时继续推进不依赖它的工作。

## 1. 比赛、设备与目标

截至 2026-10-23，官方主页确认：

- 比赛：Stanford Horizons DroneHacks，2026-11-06 至 11-08，Stanford Robotics Center，48 小时。
- 方向：GPS-denied flight，包括 perception、navigation、obstacle avoidance、autonomous decision-making。
- 官方提供：7 英寸无人机套件、Pixhawk 6C、Luxonis 深度相机、NVIDIA Orin Nano。
- 来源：https://dronehacks.groups.stanford.edu/

尚未确认：具体评分表与赛场任务、可复用的赛前代码范围、相机具体型号/IMU、Orin 内存与 JetPack、相机/飞控驱动与固件、地面站和网络限制。不能从产品系列名推断这些配置，也不能把上一场 OpenCV 比赛的规则当作本场规则。

目标：先复现一条具有尺度和明确输入输出的定位链路，再做一次与避障有关的双目深度推理对照。最终交付可重复运行的命令、原始数据、接口约定、失败案例和使用说明。优先完成稳定、可解释的最小任务。

## 2. 三名已知队员的暂定分工

以下来自用户粘贴的个人简介，不代表已经协商或验证的能力；其他队员信息待补。

| 人员 | 已知背景 | 建议牵头工作 | 对接产物 |
| --- | --- | --- | --- |
| Darion Lian | Penn 机械工程一年级；自述擅长 CAD、快速原型和迭代测试 | 机架与安装结构、传感器固定/朝向、机械与振动问题、装配和测试夹具 | 安装图、机械限制、传感器实际位置、装配记录 |
| Chelsea Yan | Penn 电气工程与统计；简介提到 GRASP 无人机、MIT Lincoln Laboratory 蜂群/路由、雷达及软硬件设计经历 | 电子/嵌入式、设备驱动与采集、飞控通信、系统集成；参与总体方案 | 设备配置、原始传感器流、飞控接口与集成测试 |
| Shiyue / Shiori Wang | UCLA Mathematics of Computation junior；AegisLand 的 CV、安全策略、profiling 与 PX4 shadow supervision 工作 | 图像/深度处理、定位基线复现、感知有效性与安全监督、性能/正确性验证 | 可调用感知模块、定位结果、延迟记录、有效性状态和回放工具 |

职责有交叉：Chelsea 与 Shiori共同确认数据接口；Darion提供传感器实际安装几何，Chelsea落实设备配置，Shiori检查计算所用的标定和坐标约定。不要把“所有算法”自动交给某一位。

接口至少写明：左右图/深度/IMU格式、图像分辨率、采集时间戳和时钟域、处理完成时间、frame_id、相机内外参、位姿方向、米/毫米/弧度单位、更新频率、有效标记、数据过期和定位失效行为。位姿置信度如未经校准，不能标成真实正确概率。

个人资料来源：用户提供的简介；LinkedIn 本次未能读取，不额外推断经历或熟练程度。

## 3. 现有项目与执行边界

仓库：https://github.com/Cyberchopin/aegisland-agentic-vision

先读取本地 AGENTS.md、README、当前分支/提交、未提交改动，以及用户现有学习/交接文档，再决定修改位置。保留现有工作，独立组织复现环境与输出，避免改变已有依赖环境来满足新模型。

本次已检查的公开代码显示：

- `src/aegisland/gps_denied.py` 的 VisualDeadReckoner 累计归一化图像运动；x/y 不是米，不能直接作为飞控位置。
- `docs/From Simulation Safety to PX4 Shadow Supervision.md` 描述只读 PX4 监督，控制输出禁用。
- `docs/cpu-profiling.md`、`docs/first-cuda-kernel.md` 已有 profiling 协议和灰度 kernel 测量。已有合成数据诊断提示光流与运动补偿耗时较多，但需要在代表性数据上重新确认。

上述是公开主分支快照，可能落后于本地工作。以实际检查结果为准，不重复实现已完成的模块，不照搬文档中的成熟度百分比。

保持 `CONTROL OUTPUT=DISABLED`、`COMMAND EXECUTED=FALSE`；当前任务涵盖离线回放与只读/SITL接口验证，不授权真实电机、解锁或飞行。不要自动推送、合并、部署或启动付费资源。

用户已提出预算压力。先用现有已授权机器；盘点GPU、显存、磁盘、CUDA/驱动和软件版本。下载仅选必要序列/权重，先查询大小；先短 smoke run，再做正式实验。不要默认租卡、调用付费API、下载完整Kimi K3、训练新模型或反复运行耗时的无关全量测试。需要额外费用时，先列出具体用途和费用上限供用户决定。

## 4. 按顺序执行

### 阶段 0：仓库与环境预检（建议第 1 天）

1. 确认工作目录确实是目标仓库，记录提交和已有改动。
2. 盘点实际开发机器；区分“以前测过的 RTX 4060 Ti”和“现在可以使用的 GPU”，也区分开发机器与比赛 Orin。
3. 画出实际调用链，说明已有输入、输出和真实连接范围。
4. 查官方任务资料，列出本场仍未知的信息及它会阻塞哪个步骤。
5. 给出最小运行环境、资源需求与本次实验范围。不要升级系统驱动或替换工作中的环境来碰运气。

交付：环境记录、现有能力/缺口表、第一条可复现命令。缺硬件时继续做数据集回放与接口准备。

### 阶段 1：cuVSLAM 定位复现（建议第 2–5 天，主任务）

1. 核对官方当前版本及硬件支持，锁定依赖、提交或发布版。
2. 选一段最小 EuRoC 序列，先复现 Stereo，再在输入与标定满足条件时尝试 Stereo-Inertial。沿用官方示例。
3. 保存估计轨迹、时间戳、运行配置、日志；核对尺度、坐标系和真值对齐方式。
4. 记录ATE/RPE等适用指标、跟踪失效/恢复、样本数与处理耗时。不要把图像内运动或无尺度结果写成米制位置。
5. 用相同输入重复运行，检查初始化、丢帧和时间问题。若遇兼容性阻塞，保存命令、错误和版本，优先解决最小原因；不要顺手扩展成多个SLAM框架评测。

完成标准：全新终端按记录命令能重跑同一序列，并生成轨迹和评估；Shiori能解释图像/IMU、标定、位姿、尺度和失效信号的关系。数据集复现不代表已经接入现场相机或可以飞行。

### 阶段 2：双目深度推理对照（建议第 6–9 天）

1. 研究 Fast-FoundationStereo 的输入、视差/深度转换、相机参数和部署条件。只选一个checkpoint。
2. 固定相同左右图、标定、分辨率、批大小和迭代次数；先跑通原始配置。
3. 分别测预处理、模型、后处理，以及同步完成的一次完整调用。分开记录冷启动和预热后结果；报告分布及样本数。
4. 先与一种改动对照，例如同配置下的TensorRT FP16；若不支持，则只改变迭代次数。一次只改变一个因素。
5. 有匹配深度真值时报告误差和有效像素范围；没有真值时只能报告一致性、无效比例或已测距离等代理检查，不得写成准确率验证。
6. 有相机后先保留 OAK/Luxonis 自带双目深度基线，再评估新模型是否值得占用 Orin 算力。RTX上的速度不能当成Orin实测。

完成标准：相同输入上的可复现对照，含质量变化、完整耗时、峰值内存及失败样例。新模型不优于现有方案时保留负结果。

### 阶段 3：只读接口与故障验证（建议第 10–12 天）

把定位结果封装为独立、可回放的消息，先接 shadow supervisor 或 SITL 的验证路径。定义frame、单位、时间戳、有效性和重定位状态；验证过期数据、相机中断、IMU错时、定位跳变的处理。安全决策记录与控制执行严格分开。

交付给Chelsea/Darion：接口样例、配置、采集需求、安装/标定需求、日志和可重复的回放。没有真实设备，明确哪些只在数据集或SITL验证。

### 阶段 4：冻结与演练（建议第 13–14 天）

固定工作版本，重复运行核心用例，整理启动、恢复、已知限制和硬件接入步骤。准备一个定位演示、一个深度/性能对照，以及一个故障处理演示。最后阶段只修阻碍演示或集成的实际问题。

## 5. 推理优化学习支线

Kimi K3 / vLLM文章作为方法参考：先识别workload，再采集trace，再选择优化。不能把K3多卡attention/MoE优化直接视为OpenCV光流的加速方案。

如有余量，在显存能容纳且已获授权的模型上，只选vLLM或SGLang其中一个框架练习。固定模型/精度、输入输出长度、并发、缓存条件；正常运行测TTFT、逐token耗时、总延迟、吞吐和显存，再单独采集profiling trace。8k输入/1k输出只是一个测试条件，不能代替无人机实际工作负载。

Kernel fusion必须有trace证据、正确性检查及未启用profiler的对照。先弄清预处理、拷贝、同步、计算或排队中的主要开销。不要把kernel-only时间和完整pipeline比较，也不要预设FP8必然适用于Orin。

Rust可用于一个小的日志/回放工具；仅在上述核心任务不受影响时开展，不重写现有主链路。

## 6. 教学与每次汇报

每完成一个小实验，用简短中文说明：做了什么、原理、实际输入输出、结果/失败、下一步。引用真实文件和命令，保留关键英文术语。教Shiori读懂代码，尤其是时间、坐标、尺度和延迟边界；不要只交一份她无法解释的工程。

每次汇报给出：实际改动、运行命令、数据/配置、产物路径、通过和失败项、资源使用、剩余阻塞。所有数字来自本次日志或明确标注的历史记录，测试计划、论文指标和本次实测分别标明。

如果文件或引用的slides不在当前环境，直接说明缺失并继续不依赖它的工作。不要声称已经查看slides、完成硬件集成或重跑了论文。

## 7. 已筛选的资料

| 优先级 | 资料 | 阅读目的 |
| --- | --- | --- |
| 必须 | [DroneHacks官网](https://dronehacks.groups.stanford.edu/) | 核对本场任务和硬件；完整评分规则仍待补 |
| 必须 | [cuVSLAM](https://github.com/nvidia-isaac/cuVSLAM)；[EuRoC示例](https://github.com/nvidia-isaac/cuVSLAM/blob/main/examples/euroc/README.md) | 复现定位、版本/硬件支持、标定和输入输出 |
| 必须 | [Fast-FoundationStereo，CVPR 2026](https://github.com/NVlabs/Fast-FoundationStereo) | 双目深度、迭代次数、部署后端和速度/质量取舍 |
| 必须 | [PX4外部视觉定位](https://docs.px4.io/main/en/ros/external_position_estimation) | 坐标系、时间、外部位姿输入；对照现场实际固件版本 |
| 方法 | [Kimi K3的vLLM优化](https://vllm.ai/blog/2026-07-27-k3) | workload、缓存、融合与测量方法；不默认部署完整模型 |
| 方法 | [vLLM profiling](https://docs.vllm.ai/en/stable/contributing/profiling/)；[SGLang profiling](https://github.com/sgl-project/sglang/blob/main/docs/docs/developer_guide/benchmark_and_profiling.mdx) | 捕获trace；具体CLI以锁定版本为准 |
| 后续 | [Swift，Nature 2023](https://www.nature.com/articles/s41586-023-06419-4) | 理解感知、状态估计与控制的系统分工 |
| 后续 | [UZH FPV比赛](https://fpv.ifi.uzh.ch/rss-2024-fpv-drone-racing-vio-competition/) | 高速无人机状态估计的测试设计；页面本次完整读取超时 |

开始时请先执行阶段0，并在环境允许时接着做阶段1的最小复现。不要先展开完整Kimi部署、训练、全仓重构或新的宣传页面。
