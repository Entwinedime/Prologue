# 多配置模拟预测大范围验证与独立采集收益对照状态

**更新日期**　2026-11-28

**进度**　1/3 里程碑完成。**M1 Step A、B2、C 全部通过并经 AGY 独立审查与代码精修后分层提交落袋**。分支 `multi-config-prediction-validation`，窗口起点 `3029b0d`，等待用户确认放行 M2（启动 M2 前须按最新代码重建双镜像）。

**当前交接**　M1 已通过 AGY 独立终审（正式留出 4/4 通过，三路来源 100% 隔离，盲测 SHA-256 与落盘时序 4/4 验真，Qwen 三种 TP 吞吐/P95 TTFT/平均 TPOT 排序 100% 一致），并完成代码精修与分层提交。下一步待用户确认后启动 M2。 [验收总表](../../../data/multi-config-prediction/holdout/m1/acceptance-summary.json) · [工作区统计](../../../data/multi-config-prediction/holdout/m1/review/git-diff-stat.txt)

- **Step A 已验收通过**：Ascend 动态 NUMA 绑核与 `MPOL_BIND` 在全部实采模式生效；TP4 Rank 0 流同步挂起 `85.46 → 0.10 ms`、TTFT MAPE `10.92% → 5.11%`；同 NUMA 内 rank 差距仅 `0.03%`（NUMA 6）与 `0.79%`（NUMA 4）；跨 NUMA 6 vs NUMA 4 的 `5.01%` 差距已实锤为本机物理 DRAM 写带宽差（`+8.8%`），由按 rank 成本库忠实建模，放行不再卡关。
- **Step B2 已通过**：TP4 单轮宿主复验沿用绑核详细图及纯时延参考，各 rank 配对 11/11 批。3,212 个 all-reduce 调用均有连续的前后排除区间。TTFT／TPOT MAPE 由 5.11%／6.02% 变为 **4.68%／4.72%**。 [复算结果](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp4/continuous-auxiliary/dev-autonomous/result.json) · [区间核验](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp4/continuous-auxiliary/interval-check.json)
- **Step C 独立对照**　四份预测封存后录制十二份独立参考。四例结构全匹配，三路训练来源 100% 隔离，自主 TTFT／TPOT MAPE 为 0.71%–4.61%／1.04%–3.43%。初始热覆盖 0/4，三种 Qwen TP 各补采一次，Mistral 补采 18 次。首批不足 4/4 的盲测保留原证据。 [首批归档](../../../data/multi-config-prediction/dev/m1/retired-holdout-beb164e4/results/m1/blind.json) · [封存记录](../../../data/multi-config-prediction/holdout/m1/results/m1/blind.json)

**可依赖的事实**　既有 TP2、TP4、TP8 错峰结构和 Mistral TP2 开发证据继续有效；TP2 单卡装配独立对照匹配 `108/108` 个动作（已有兼容探针时单卡边际省 `46.52%` 卡时）；本轮绑核后的 TP4 `detailed` 与 `latency-only` 可直接复用。 [Mistral 来源](../../../data/multi-config-prediction/dev/m1/cases/mistral7-tp2/clean-host/replay-fit/source-evidence/selected-source.json) · [单卡装配](../../../data/multi-config-prediction/dev/m1/single-device-assembly-report.json)

| 里程碑 | 主题 | 预算上限 | 状态 |
| :--- | :--- | :---: | :--- |
| **M1** | Qwen3-8B TP2/TP4/TP8 & Mistral-7B TP2 多配置预测 + TP2 单卡装配对照 + Ascend NUMA 绑核 | ≤ 60,000 卡秒 | 完成（AGY 验收落袋） |
| **M2** | PD 分离 + Mistral PP3 / TP2×PP3 + Qwen3-30B-A3B TP4×EP4 异构多卡预测矩阵 | ≤ 60,000 卡秒 | 待做（需 M1 停机放行） |
| **M3** | Azure 2024 真实流量 32→128→256 压测 + 8 域总表与公开文档收口 | ≤ 60,000 卡秒 | 待做（需 M2 停机放行） |

**已知费用**　累计 **40,709.00 / 60,000 卡秒**，宿主计时合计 26,056.44 秒，存储采样峰值 45.39 GiB。失败录制和两批补采均计费。 [资源账本](../../../data/multi-config-prediction/dev/m1/resource-accounting.json)
