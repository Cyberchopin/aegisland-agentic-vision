# GPS-denied 上游证据与许可证审计交接

完成日期：2026-10-31 America/Los_Angeles（取回日期 2026-11-01 UTC）；母仓库起点 `e13b698`。工作范围已完成：官方版本、许可/NOTICE、小规模源码/接口、哈希与使用决定。没有复制第三方实现到母仓库，没有重写第三方算法，没有下载大型数据、模型、二进制或容器，没有调用 REA。

交付文件：

- [provenance/third_party.csv](../../provenance/third_party.csv)：60 行资产记录，含 44 份实测小文件、14 个仅官方摘要的 cuVSLAM 发布资产、2 个未下载的大资产占位。
- [docs/upstream-method-matrix.md](../../docs/upstream-method-matrix.md)：固定版本、方法接口、NVIDIA/Jetson 要求及覆盖边界。
- [docs/license-and-redistribution-review.md](../../docs/license-and-redistribution-review.md)：分层许可、NOTICE 与再分发审计。
- [docs/integration-decision.md](../../docs/integration-decision.md)：选择、阻塞门槛、发行范围与 fallback 触发条件。
- [provenance/audit-evidence.json](../../provenance/audit-evidence.json)：派生证据元数据，无第三方源码正文。
- [provenance/verify_consistency.py](../../provenance/verify_consistency.py)：本地一致性复核程序，不执行网络请求。
- [provenance/verification-report.json](../../provenance/verification-report.json)：复核结果、逐文件绑定和跨文件决定对照。

当前结论：

| 候选 | 结论 | 官方依据 |
| --- | --- | --- |
| cuVSLAM v17.0.1 | **CONDITIONAL**，主候选；NVIDIA 平台限制必须保留 | [LICENSE](https://github.com/nvidia-isaac/cuVSLAM/blob/51631ba358ea51262342f9ae649820d639363eb9/LICENSE) |
| ROS `isaac_ros_cuvslam` v5.0-0 | **CONDITIONAL**；wrapper Apache 与库 NCL 分开；gitlink 不是 v17.0.1 | [固定仓库](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam/tree/9058506efd0b967ac894a717bdbe98df017637e5)、[依赖许可](https://github.com/nvidia-isaac/cuVSLAM/blob/33da60662b27c0c5b63e0597bd778066af2731ed/LICENSE) |
| EuRoC MAV | **CONDITIONAL**；非商业范围、比赛媒体和实际数据哈希待核 | [ETH 官方记录](https://www.research-collection.ethz.ch/items/bcaf173e-5dac-484b-bc37-faf97a594f1f) |
| DepthAI / OAK-D 官方示例 | **CONDITIONAL**；MIT 接口参考 **GO**，实际固件和发行包另审 | [MIT](https://github.com/luxonis/depthai-core/blob/e9ead6536e1bf9842d91211fcf80f0b4ef119954/LICENSE)、[固件](https://github.com/luxonis/depthai-core/blob/e9ead6536e1bf9842d91211fcf80f0b4ef119954/notices/depthai-device-RVC2-LICENSE) |
| Fast-FoundationStereo | **NO-GO** 进入比赛交付；继续按 research/evaluation-only 处理 | [许可证](https://github.com/NVlabs/Fast-FoundationStereo/blob/476f4249561f7c79ca707326954f9255643412a6/LICENSE.txt)、[模型卡](https://github.com/NVlabs/Fast-FoundationStereo/blob/476f4249561f7c79ca707326954f9255643412a6/model_card.md) |
| Kimera-VIO | **CONDITIONAL**，fallback 未启动；内置 Triangle 有额外分发限制 | [构建](https://github.com/MIT-SPARK/Kimera-VIO/blob/ce8c59b7b273ab5ac29db7e5572e1623760e19c7/CMakeLists.txt)、[Triangle](https://github.com/MIT-SPARK/Kimera-VIO/blob/ce8c59b7b273ab5ac29db7e5572e1623760e19c7/third_party/triangle/src/triangle.c) |

接手顺序：先关闭 `CUV-01/02/03` 与 `OAK-01`，再按需要处理 `ROS-01`；离线评估/公开媒体先关闭 `EUR-01`。`FFS-01` 未关闭前不得将相关依赖、权重、派生 engine 或媒体加入比赛发行物。只有记录了 cuVSLAM 的明确阻塞才触发 `KIM-01`，之后仍须先关闭 `KIM-02`，不启动多框架比较。

证据验证：44 份文件共 1,244,609 字节，逐一匹配固定 Git 树的 blob SHA-1/长度并计算 SHA-256；CSV 中发布资产摘要为上游报告、未本地核验，dataset/checkpoint 哈希显式未知。ETH 记录直接访问 HTTP 403，权利标签来自官方域名搜索索引；细节见审计文档。在线 ROS release-5.0 文档为审计日观察，不能替代不可变包版本证据。

临时缓存 `/tmp/gps-denied-provenance-audit/` 含本次取回的小文件与本地生成/核验脚本，可能被系统清理；持久记录只保留固定 URL、元数据和摘要。可按 JSON 的 `raw_url` 逐个重新获取，核对长度、Git blob SHA-1 和 SHA-256；不要取回 release assets 或大资产来重现本次审计。历史运行结果与既有源码来源不在此次认证范围内。

本次仅执行证据一致性与文档链接/范围检查；未运行 GPU、OAK、ROS、飞行或完整应用测试，未安装任何候选。许可条件尚未关闭不影响本次审计完成，但仍阻止相应集成/发布放行。

ROS cuVSLAM gitlink `33da60662b27c0c5b63e0597bd778066af2731ed`：**CONDITIONAL**，仅作为 ROS 封装的独立依赖审计；不能以 wrapper 的 Apache-2.0 覆盖，也不视为 standalone v17.0.1 的替代。[固定依赖许可证](https://github.com/nvidia-isaac/cuVSLAM/blob/33da60662b27c0c5b63e0597bd778066af2731ed/LICENSE)。

CSV 的 `decision` 是候选比赛集成/重分发的总体决定；限定的文档/API 参考 GO 或研究评估用途不提升总体状态。内置组件继承所属候选的未放行门槛；未逐项审阅的其他依赖不视为获批。

## Verification-and-repair pass

Final verification: **PASS (exit 0)**. This is a consistency result; the CONDITIONAL / NO-GO integration decisions and open license gates remain in force. The verification command below is rerun after the handoff edits. Machine evidence is in [verification-report.json](../../provenance/verification-report.json); no previous report is trusted as a verification input.

### Historical exit 1 and the additional repair failure

The exact trailing command in the original combined check was:

```bash
git diff --check --no-index /dev/null docs/integration-decision.md
```

Original combined tool exit: **1**. The preceding Python assertions printed `result: PASS`; there was no assertion traceback. Reproduction of that exact Git command before repairs: **exit 1**, stdout `""`, stderr `""`, failed assertion **none**. The document differs from `/dev/null`; `git diff --no-index` uses a difference exit status. This was not evidence of a failed CSV/hash assertion, but the combined command did not satisfy a required exit-0 success contract. The previous broad PASS marker is now explicitly superseded in `audit-evidence.json`.

The first run of the new local verifier also exited **1**. Its exact failed assertion was:

```text
Broken local Markdown link docs/upstream-method-matrix.md: ../provenance/verification-report.json
```

Stdout was empty. Stderr was:

```json
{"status": "BLOCKED", "exit_code": 1, "failed_assertion": "Broken local Markdown link docs/upstream-method-matrix.md: ../provenance/verification-report.json", "command": "python3 provenance/verify_consistency.py --cache-root /tmp/gps-denied-provenance-audit --report provenance/verification-report.json", "downloads_performed": 0}
```

Cause: the first-run verifier checked a link to its generated report before materializing the report path. Repair: create the designated output path before link checks; do not read its content as evidence. The failure and stderr are retained in `audit-evidence.json` under `verification_repair_history` and copied into the final report.

### Repairs and exact counts

- Replaced the ambiguous diff-based final check with a persistent local verifier that returns 0 only when all assertions pass. Markdown whitespace is checked directly, including untracked files; no exit 1 is silently normalized to success.
- Kept **60 CSV data rows excluding the header**: **44** small source/document evidence rows + **14** release metadata rows + **2** undownloaded large-asset placeholders = **60**.
- Independent project counts: cuVSLAM **22**, DepthAI **11**, EuRoC **2**, FFS **6**, Kimera **11**, ROS **7**, ROS cuVSLAM dependency **1**; sum **60**. Decision totals: CONDITIONAL **54**, NO-GO **6**; no blanket candidate GO.
- Bound every local digest to one canonical relative cache path `by-sha256/<digest>` and exact `size_bytes`, with a distinct `artifact_id` for each pinned source identity. The 44 source rows represent **43 distinct content hashes/paths**; the two cuVSLAM LICENSE revisions have identical bytes and intentionally share the one canonical object. Both original relative source paths, pins and URLs remain explicit. Source-record byte total: **1,244,609**; this total counts both source identities.
- Separated download status from hash verification and publication status. **44** rows have locally verified SHA-256; **14** have official upstream SHA-256 not locally verified. **16** artifact payloads were not downloaded (the 14 release assets plus the 2 placeholders). For **44** source-file rows no published SHA-256 is present in the reviewed metadata; this does not deny publication elsewhere. For **2** placeholders publication/asset identity remains `NOT_ESTABLISHED`; **0** rows prove universal absence of a published hash. The unknown hash sentinel is now `UNKNOWN_NOT_ESTABLISHED`, independently of download status.
- Made the ROS gitlink dependency's CONDITIONAL decision explicit in all four Markdown documents. Cross-checked all **7 dependency subjects** across the CSV and four documents. API-reference GO and research-only permissions do not change the overall competition integration/redistribution decision. FFS remains NO-GO; Kimera fallback remains inactive.
- Confirmed the real regular filename is `docs/license-and-redistribution-review.md`; no filename contains a literal backslash and no escaped-name duplicate exists. Verified local document links, fixed source URLs, UTF-8/newlines, absence of third-party source bodies in durable evidence, and the exact permitted worktree status.

### Final commands and exit codes

Run from the provenance worktree root (`<worktrees>/provenance`):

| Command | Exit code | Result |
| --- | --- | --- |
| `python3 /tmp/gps-denied-provenance-audit/repair_consistency.py` | 0 | Applied the narrow metadata, path-binding and scope repairs; no upstream source edits. |
| `python3 provenance/verify_consistency.py --cache-root /tmp/gps-denied-provenance-audit --report provenance/verification-report.json` | 0 | Final consistency verification; 60 rows, 44 local source bindings / 43 canonical objects, 7 subjects across 5 deliverables, filename and status checks. |
| `git status --short --untracked-files=all` | 0 | Exact stdout below; stderr empty. |

The historical reproduction and first-run bootstrap failure above are failure evidence, not successful final checks. No application, GPU/OAK/ROS, model, flight, build or package tests were run in this pass. No downloads, vendoring, push, merge, dependency installation or changes to another worktree were performed. The persistent verifier is the authoritative recheck command; the older temporary deliverable generator is not a repair replay procedure.

### Remaining unknowns

The 14 upstream release digests have not been independently checked against payload bytes; wheel/C++ archive and actual firmware contents/NOTICE remain uninspected. The 2 dataset/checkpoint placeholders lack fixed downloaded bytes, confirmed upstream hash publication and sizes; the model revision itself remains unselected. ETH record access previously returned 403, so archive-specific rights and competition media permissions remain open. Actual target hardware, ROS/standalone compatibility, complete transitive dependencies, optional DepthAI components, Triangle/ORB-vocabulary authorization, and historical local asset/source provenance remain unverified. No such unknown has been converted into permission or a GO decision. The temporary cache is needed to repeat local byte checks; if missing, the verifier must fail explicitly rather than trust the recorded digest.

### Exact git status

Command: `git status --short --untracked-files=all` (exit 0, stderr empty). All eight changes are untracked audit artifacts; tracked application files and the staging index are unchanged.

<!-- verification-git-status:start -->
```text
?? docs/handoffs/provenance-HANDOFF.md
?? docs/integration-decision.md
?? docs/license-and-redistribution-review.md
?? docs/upstream-method-matrix.md
?? provenance/audit-evidence.json
?? provenance/third_party.csv
?? provenance/verification-report.json
?? provenance/verify_consistency.py
```
<!-- verification-git-status:end -->
