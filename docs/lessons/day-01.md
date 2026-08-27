# Day 1 教学：让视觉结果真正改变动作

今天先掌握整套项目最重要的底层逻辑：**OpenCV 不负责“决定”，它负责“提供可测量证据”；安全状态机把证据变成动作。**

## 你已经学过的知识如何升级

你之前用 HSV mask、morphology 和 contour 找颜色区域。AegisLand 把同一个思维扩展成四类证据：

1. `Canny` 找结构边缘：区域边缘太密，通常不适合降落。
2. `Laplacian` 测纹理变化：粗糙、复杂表面得到更高风险。
3. `Farneback optical flow` 测连续两帧像素运动：有人或物体进入区域时，motion occupancy 上升。
4. contour 只包围“运动区域”，再算候选区到它的 clearance。

这与 HSV 最大的区别是：HSV 回答“这个像素是什么颜色”，光流回答“这个区域正在怎么变化”。无人机安全更关心后者。

## 先读这三个文件

1. `src/aegisland/perception.py`：看 `observe()` 怎样把一帧变成 `VisionEvidence`。
2. `src/aegisland/planner.py`：看规则的优先级，尤其 collision 为什么在 critical battery 前面。
3. `tests/test_planner.py`：看同样 6% 电量，安全 zone 得到 `LAND`，不安全 zone 得到 `REQUEST_HUMAN_APPROVAL`。

## 你今天必须亲手做的实验

运行：

```powershell
python -m aegisland demo --scenario low_battery_intrusion --output runs\day01
python -m aegisland serve --directory runs\day01
```

然后回答并写进你的学习日志：

1. `edge_density`、`motion_occupancy`、`clearance` 各自会在哪一种场景失败？
2. 为什么 2% 电量时不能永远等待人工批准？
3. 为什么“检测到人”不是当前代码能诚实声称的能力，而“检测到运动区域”可以？
4. 在 `test_open_cv_result_changes_the_action` 中，把安全 zone 的分数从 `0.84` 改成 `0.50`，动作为什么改变？

## 今日小改动任务

给 `SafetyPolicy` 增加一个 `maximum_landing_motion` 阈值，并写一个测试：候选区总分很高，但 `motion_occupancy` 超标时依然不得落地。不要先让 AI 写；你先独立尝试 20 分钟，再对照现有测试结构修改。

完成标准：`pytest` 全绿，而且你能不用看代码解释 perception → decision → action 的闭环。

