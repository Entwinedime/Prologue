# 多拓扑、复杂调度与真实流量大范围验证及局部补采泛化提速状态

**更新日期**　2026-11-27

**进度**　**已关闭（阶段完成提前收口，6/14 完整验收 + S07 阶段基础）**。L0（S01–S02）、L1（S03–S05）与 S06 验收完成，S07 完成流绑定修复、TP2 完整拓扑对照与 TP4 单卡来源核验。2026-11-27 经 AGY 深度审查与用户决策，将已验收的 55 个改动文件与 6 个新增文件分层提交落袋，剩余多配置预测验证（TP2/4/8、Mistral、PD、PP、MoE-EP、Azure 真实流量与 8 域总表）转入新窗口 `2026-11-27_multi_config_prediction_validation` 分 M1–M3 短里程碑推进。分支 `large-scale-validation`，窗口起点 `e7903ce`，源码基线 `8d10807`，基础行为基线 `b500cbe`。

**当前交接**　本窗口已收口归档，全部品质与隔离门禁经 AGY 独立复核通过（219 个 Python 模块 AST 0 环 0 缺失、>550 NE 文件 = 0、函数体 >25 行代码墙 = 0、四维隔离 0 越层、`ruff` 0 错、活动源码 0 `.pyc`）。后续多配置模拟预测验证由新窗口 `2026-11-27_multi_config_prediction_validation` 接管。[TP2 结构](../../../data/large-scale-validation/dev/l2/s07/qwen8-tp2/targets/page-crossing/structure-full-peer/result.json) · [TP4 单卡来源](../../../data/large-scale-validation/dev/l2/s07/qwen8-tp4/sources/single-source-evidence/source-evidence.json)

| 层级 | 阶段 | 主题 | 状态 |
| --- | --- | --- | --- |
| L0　底座清债 | S01–S02 | PD 825 差异清零、TP8 来源/条件恢复、CANN 原生整数端点与设备 ABI 归口 | 完成 |
| L1　复杂调度与提速 | S03–S05 | 同会话合批提速（34.1% < 50%）+ Cache/Chunk/Mixed 局部配方 + TP1 复杂调度留出 5/5（1.00%–3.96% MAPE） | 完成 |
| L2　多卡 TP 与跨模型 | S06–S08 | 多卡 TP 局部补采与物理通信分组 + Qwen TP2/4/8 & Mistral TP2 多配置预测 | S06 完成，S07 阶段基础完成，余项转新窗 M1 |
| L3　异构高级并行 | S09–S11 | PD / PP3 / TP2×PP3 / MoE-EP 预测与固定图归因 | 转新窗 M2 |
| L4　真实流量与总终验 | S12–S14 | Azure 2024 真实流量 32→256 压测 + 8 域总验收与文档收口 | 转新窗 M3 |

**已知费用与输入**　截至窗口收口累计已结算 **86,201.81 卡秒**（防挂死上限 180,000 卡秒、240,000 秒后处理、512 GiB 均未触顶）。已核验官方 Azure 会话 CSV。Qwen3-8B 有完整权重文件，另两款模型具备官方配置，本窗口使用 dummy 权重路径。[TP2 账本](../../../data/large-scale-validation/dev/l2/s07/qwen8-tp2/native-accounting.json) · [TP4 单卡账本](../../../data/large-scale-validation/dev/l2/s07/qwen8-tp4/sources/native-accounting.json)

**已核验结果**　PD 端口差异从 825 降到 0，按现行端点重投影后 5,355 个动作仍全部对齐。固定 TP1 的 211,290 条轨迹、16 次发布、678 样本拟合和 43,276 项固定图成本与基线一致。TP8 来源核验覆盖 8,090 个动作，八设备时钟联合实现，受检工作结束点缺测为 0。历史三组设备任务的重叠用整数端点复算均由 1 ns 降为 0。[PD 回归](../../../data/large-scale-validation/dev/l0/baseline/pd-current-clock/structure/result.json) [TP1 基线](../../../data/large-scale-validation/dev/l0/baseline/comparison.json) [TP8 来源](../../../data/large-scale-validation/dev/l0/tp8/source-evidence/source-evidence.json) [TP8 实测条件](../../../data/large-scale-validation/dev/l0/tp8/timing/execution-inputs-report.json) [历史端点复算](../../../data/large-scale-validation/dev/l0/cann/historical-rounding-proof.json)

**S03 证据**　两个历史几何复测占完整服务的 53.3%。八个几何含两个并发跨页切点，局部 71.45 卡秒、同点服务 209.61 卡秒，占 34.1%。32 个成本查询通过，每个切点的任务序列、操作数和 KV 回读合格。S03 累计新增 1,015.26 卡秒，含失败 327.36 卡秒，L0–S03 共 2,380.74 卡秒。[提速对照与账本](../../../data/large-scale-validation/dev/l1/speedup-benchmark.json)

**S04 证据**　四份完整服务参考均通过原始来源核验。局部采集覆盖续块、混批、缓存未命中及命中、容量驱逐与两类页边界。两组分配会话各回答两项独立查询。缺少可复用结构时，四项续块宿主查询由配对采集规则一次补齐。S04 共 1,974.53 卡秒，L0–S04 累计 4,355.27 卡秒。[覆盖与账本](../../../data/large-scale-validation/dev/l1/s04/coverage.json)

**S05 证据边界**　第五轮小池的两次隔离自动补采耗 331.17 卡秒，最终 TPOT MAPE 仍为 6.59%。同会话扩包络在开发集由真实同步拷贝和 Decode Attention 探针证实。第六轮 5/5 均命中热库，自主 TTFT MAPE 为 1.00%–3.69%，TPOT 为 2.16%–3.96%。详细固定图替换全部成本后的 TPOT MAPE 为 22.22%–63.44%，恢复其宿主实测后降至 0.04% 以下，留作宿主时序归因。[第五轮](../../../data/large-scale-validation/holdout/rounds/s05-r5/results/s05/acceptance-summary.json) · [第六轮](../../../data/large-scale-validation/holdout/rounds/s05-r6/results/s05/acceptance-summary.json)

**S06 本地设备**　单卡定向执行 TP2 rank 1、TP4 rank 2、TP8 rank 7 的原生 Decode Attention，两个实测坐标均通过 KV 读回和成本发布。切口定义来自完整物理采集，单卡全来源建库及并发结构对照由 S07 验证。[TP2 单卡复核](../../../data/large-scale-validation/dev/l2/s06/attention/tp2-clean/result.json) · [TP4/TP8 证据](../../../data/large-scale-validation/dev/l2/s06/attention/wider-evidence.json)

**S06 物理通信**　TP2/4/8 各一次会话覆盖全部 14 个 rank。较宽的物理开发录制在 C++ 中覆盖全部 13,468 个 AllReduce/AllGather 区间。TP2 扩容至 8,192 token 和 8 槽后，原样本覆盖 148 次独立通信观测。目标操作数超界、归约方式或参与者变化均被拒绝。[通信覆盖](../../../data/large-scale-validation/dev/l2/s06/physical-peers/communication-coverage.json)

**S06 宿主**　14 个 rank 各匹配 13 批，独立 RAW 记录拟合后覆盖原图 647,006 个宿主区间。完整预测覆盖、双来源结构及性能精度继续由 S07/S08 验证。[宿主覆盖](../../../data/large-scale-validation/dev/l2/s06/host-timing/host-fit-coverage.json)
