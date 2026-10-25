# 推理优化方法论笔记（vLLM / Kimi K3 文章 → 感知链路）

参考：<https://vllm.ai/blog/2026-07-27-k3>（本次未重新逐段阅读，只取 brief 中列出的方法论要点；不部署任何 LLM）。

## 可迁移的方法

| 方法 | 在 LLM serving 里 | 在我们的感知/定位链路里 | 本次实际做法 |
| --- | --- | --- | --- |
| 先识别 workload | prefill vs decode、批大小、上下文长度 | 每帧：解码 → ORB ×2 → knnMatch → RANSAC → warp；cuVSLAM track()；输入是 20 Hz 752×480 单目/双目 | cProfile 显示 detectAndCompute 占 64%、knnMatch 28% |
| 先 trace 再优化 | Nsight / torch profiler | cProfile（CPU）、Nsight Systems（GPU 侧，cuVSLAM） | 先有 trace，发现"上一帧 ORB 每帧重算" |
| 融合/缓存要有证据 | kernel fusion、KV cache 复用 | 特征缓存：上一帧的 ORB 结果直接复用 | 用 `profiling.py --reference` 做逐字节等价对照 |
| 区分冷启动/预热 | 首 token vs 稳态 | cuVSLAM 首次 track() 18 ms vs 稳态 <1.5 ms | run.json 分开记录 |
| 无 profiler 的计时 | profiler 开销会扭曲结果 | 速度结论只用 baseline 模式（不开 cProfile）的样本 | profiling.py 的 `mode=baseline` |
| 端到端 vs kernel-only | 不能拿 kernel 时间比 pipeline | 不能拿 track() 时间比"整帧延迟"（缺解码、拷贝、排队） | 文档中分别标注边界 |

## 不能迁移的东西

- MoE / 多卡 attention / KV cache 分页：我们的链路没有这些结构。
- 吞吐优先：飞行需要的是**单帧延迟和抖动上界**，不是批量吞吐；批处理反而增加延迟。
- FP8 / 量化：对 ORB/RANSAC 这类整数/几何计算无意义；对 Fast-FoundationStereo 才可能相关，且 Orin（Ampere）不支持 FP8 tensor core（💭 需在硬件确认）。
- RTX 上的加速比不能当成 Orin 的加速比。
