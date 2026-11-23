# 窗口状态

截至 2026-11-23，**已交付 52/57 项（91.2%）**，按交付项等权计数。前六阶段和第 01–13 轮已交付。下一阶段从第 14 轮的分块预填充开始。

| 阶段 | 已交付 | 当前结论 |
| --- | ---: | --- |
| R1、R2、F1、R3、F2、R4 | 39/39 | 重建业务目录和任务入口，交付两种采集模式、按缺口取证、按工作绑定成本以及三条独立评价路径。[单卡证据](../../../data/architecture-acquisition-performance/f1/configuration-evidence.json) · [按需采集](../../../data/architecture-acquisition-performance/f2/handoff.json) · [第四轮重构](../../../data/architecture-acquisition-performance/r4/handoff.json) |
| 01–04 三条评价路径 | 4/4 | 在 Qwen3-8B TP1 上核对固定图实测回算、KV 页边界、模型更换和到达改变批次的首处差异。[固定图](../../../data/architecture-acquisition-performance/rounds/01/result.json) · [页边界](../../../data/architecture-acquisition-performance/rounds/02/confirmation/assessment.json) · [模型与成本域](../../../data/architecture-acquisition-performance/rounds/03/independent/assessment.json) · [时序反馈](../../../data/architecture-acquisition-performance/rounds/04/independent/assessment.json) |
| 05–07 模型、单卡来源与 TP 宽度 | 3/3 | Qwen、Mistral、Llama 的共同 TP1 结构对照全部匹配。Qwen 与 Mistral 在 TP4、TP8 的完整来源匹配受检结构，单卡观察 rank 0、1 后核对其余等价角色的定义、张量端口和归约条件。更宽 TP 的单卡整图仍有通信资源布局差异。[三模型](../../../data/architecture-acquisition-performance/rounds/05/independent/assessment.json) · [TP2](../../../data/architecture-acquisition-performance/rounds/06/independent/assessment.json) · [TP4 / TP8](../../../data/architecture-acquisition-performance/rounds/07/independent-recovery/case-evidence.json) |
| 08–13 多角色交接 | 6/6 | PP、TP × PP、EP 开关与变长路由、PD 单请求和多成员交接均有独立主判据证据。五批最新留出合计 19 个物理目标、38 份原生参考、114 项评价。两条性能路径各 0/150 个可评分请求。 |
| 14–17 调度机制与持续流量 | 0/4 | 待启动。依次检查分块预填充、混合批次、前缀缓存、持续流量。 |
| 18 同版本跨轮验收 | 0/1 | 待启动。使用新的留出输入，分别核对三条路径及两项 MAPE 指标。 |

## 第 08–13 轮的当前证据

| 轮次 | 独立结果 | 收尾范围 |
| --- | --- | --- |
| 08 PP3 | Qwen 与 Mistral 的新单卡对照匹配 3,852、3,762 个动作和 756、728 条通信依赖，保留实机工作布局差异。两份同次实测回算匹配 270,583 个端点。此前的物理留出中，两模型完整来源均已匹配主结构。 | 新 Mistral 完整来源缺少单页 Prefill 的问题已修正，复用同一目标的开发回归匹配 3,762 个动作。新 PP3 实测参考含首次加载，保留性能评分限制。[流水留出](../../../data/architecture-acquisition-performance/rounds/09/independent-pipeline-current/assessment.json) · [完整来源早期留出](../../../data/architecture-acquisition-performance/rounds/09/independent/assessment.json) · [Mistral 开发回归](../../../data/architecture-acquisition-performance/rounds/09/repair/current-pipeline-source-coverage/revealed-target-regression/result.json) |
| 09 TP2 × PP3 | 两模型完整来源分别匹配 7,600、8,508 个动作。两种来源均逐边匹配 9,700、10,236 条通信依赖，单卡保留资源与端口布局差异。 | 同一批 PP 留出收齐 24 项，两条性能路径各 0/72 个可评分请求。修正宿主唤醒和活动读取后，离线开发回归的两份六卡参考匹配 849,962 个受检端点及 98 次 token 发布。[冻结结果](../../../data/architecture-acquisition-performance/rounds/09/independent-pipeline-current/assessment.json) · [Qwen 实测开发回归](../../../data/architecture-acquisition-performance/m3-early-wake-regression/qwen-tp2-pp3/result.json) · [Mistral 实测开发回归](../../../data/architecture-acquisition-performance/m3-active-read-preemption/fixed-graph/result.json) |
| 10 同宽 TP 开关 EP | Qwen1.5、Qwen3 在 TP2 下开、关 EP 的四项完整来源均匹配全结构。单卡配对同一组动作，保留工作布局差异。两模型使用相同的实际逻辑批次比较开关，核对了 1,440 次 DeepEP 张量交换。早期 Qwen3 完整 TP4 开关对照另有记录。 | 新 TP2 留出 24 项评价收齐，两条性能路径各 0/24 个可评分请求。固定图各四项成本缺口和实测回算差异，自主均缺成本。[新 TP2 开关证据](../../../data/architecture-acquisition-performance/rounds/10/independent-balanced-tp2-current/assessment.json) · [实际批次与交换几何](../../../data/architecture-acquisition-performance/rounds/10/development/current-balanced-tp2-sources/paired-native-geometry.json) · [早期 TP4 结果](../../../data/architecture-acquisition-performance/rounds/10/independent-balanced-ep-current/assessment.json) |
| 11 固定 EP 后改变路由 | 两模型 TP2 的完整来源分别匹配 1,000、838 个动作。两种宽度合计核对 2,784 次原生交换，实际发送行与对端接收行一致，覆盖均衡、偏斜及单向路由。18 项评价收齐。 | 原 TP4 留出缺双成员沿用原页的 Decode。修后完整来源在开发回归匹配 1,498 个动作。两种来源均匹配 37,984 条通信边和 160,844 条受检顺序，单卡保留工作及端口差异。新来源的 TP4 展开留待第 18 轮重新独立确认。另从原始记录恢复 720 段只收不发的可计费服务。[独立结果](../../../data/architecture-acquisition-performance/rounds/11/independent-route-geometry/assessment.json) · [TP4 两套开发回归](../../../data/architecture-acquisition-performance/rounds/11/development/route-aware-sources/overlapping-decode-coverage/scheduled-source-coverage) · [实测通信开发回归](../../../data/architecture-acquisition-performance/m3-receive-only-ep) |
| 12 PD 单请求 | 两模型各有两种 prompt 长度，完整和单卡均匹配两端 KV 与元数据交接及受检动作顺序。单卡保留工作布局差异。固定图四份实测参考匹配 267,178 个边界。 | 24 项评价已收齐。固定图与自主预测仍有目标成本缺口。[独立结果](../../../data/architecture-acquisition-performance/rounds/12/independent-protocol-reconfirmation/assessment.json) |
| 13 PD 多成员预建 | 两款模型均在两个 Decode rank 实测同一对请求进入同一预建批次，隔离负载只触发单成员。完整和单卡均匹配 47,110 个动作及 258,996 条受检顺序，单卡保留工作布局差异。 | 24 项评价已收齐，两条性能路径各 0/28 个可评分请求。原冻结结果原地保留，补入真实宿主等待后的开发回归已推进至成本查询。[独立结果](../../../data/architecture-acquisition-performance/rounds/13/independent-active-decode-confirmation/assessment.json) · [真实批次](../../../data/architecture-acquisition-performance/rounds/13/independent-active-decode-confirmation/native-coverage.json) · [等待归属回归](../../../data/architecture-acquisition-performance/m3-early-wake-regression/assessment.json) |

## 接续条件

阶段审查覆盖共同执行依赖、框架观测、设备计时、成本特征和独立评价的交接。C++23 核心及正式采集、控制镜像已构建。第 14 轮先在 Qwen 与 Mistral 上隔离 chunk 续接，使用页与 chunk 边界两侧的请求，再检查混批和缓存。[核心构建](../../../data/architecture-acquisition-performance/m3-delivery-build.json) · [镜像](../../../data/architecture-acquisition-performance/m3-handoff-images.txt)

单卡来源给出实际分片工作和约定范围的对端依赖，外接通信成本仍依赖物理采样及其适用范围。整体结构匹配、固定图可评分和自主精度分别取证。最终 TTFT 与 TPOT 的 6% MAPE 门槛待同版本的新留出验收。
