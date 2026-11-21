# 窗口状态

截至 2026-11-22，R1、R2、F1 及 R3 的编号片段均已完成。按片段等权计为 **27/45（60%）**。开发分支为 `architecture-acquisition-performance`。

| 阶段 | 已完成 / 计划 | 当前交付 |
|---|---:|---|
| R1 全仓职责划分 | 7/7 | 物理部署、框架观测、成本拟合与独立验证各有维护入口。完整模式 TP1、TP2 已走三路开发对照。 |
| R2 正式任务工作流 | 5/5 | 公开入口支持冷建库、已覆盖负载零补采复用、原生观测结束后的离线恢复，逐目标发布所用资产。 |
| F1 双模式采集 | 11/11 | 七种拓扑已有单卡本地分片和连接证据。TP2、PP3 两种来源的三路对照已收齐 12 项结果。 |
| R3 两种来源的共同边界 | 4/4 | 三种拓扑、两种采集模式、三条验证路径共 18 项开发报告。使用相同插桩的独立目标又做了六项结构复核。完整模式三项对齐，单卡三项均有可定位的通信提交差异。[交付及原始结论](../../../data/architecture-acquisition-performance/r3/handoff.json) |
| F2 按需取证与性能模型绑定 | 0/7 | 起点是从目标 attention 和搬运工作得到可采样的成本需求。 |
| R4 全仓收口 | 0/5 | 待 F2 交付。 |
| V 独立验证 | 0/6 | 最终留出验收独立计数。 |

## 单卡结果的适用范围

物理卡 4 独立取得了 Qwen3-8B TP1、TP2、TP8，Mistral-7B PP3、TP2 × PP3，Qwen3-30B-A3B TP4 × EP4，以及 Qwen3-8B Prefill TP1 / Decode TP2 的具名本地证据。TP8 在八个分片执行时仅详细观察两个代表。组合 PP 核对了六个角色上的载荷，EP 覆盖两种实际路由，PD 追到 Decode 两端的 72 次逐层首次读取。各项均有独立物理目标。[七项检查与来源](../../../data/architecture-acquisition-performance/f1/configuration-evidence.json)

| F1 交付时的开发来源 | 注入实测时间检查结构 | 在真实图上估计成本 | 自主闭环 |
|---|---|---|---|
| TP2 完整 | 200 个动作对齐，达到本例全节点条件 | 取得可比结果 | 取得可比结果 |
| TP2 单卡 | 比较了 71 个动作，首处分歧在 rank 0 的 `Scheduler.run_batch` | 444 项设备通信中 6 项 all-gather 可估值，438 项 all-reduce 缺少可绑定的目标载荷形状 | 首个成本缺口为 all-gather 调用范围内的主机工作 |
| PP3 完整 | 454 个动作在已测范围内对齐 | 42 项设备通信均可估值，676 个外部端点缺少重算时刻 | CPU 发送需要 1,696 字节，样本只覆盖 1,968 字节 |
| PP3 单卡 | 比较了 75 个动作，首处分歧在 rank 0 的 `isend` | 外接样本覆盖全部 42 项设备通信，同一参考的 676 个外部端点仍限制请求级重算 | 同一 CPU 发送大小超出范围 |

两份独立物理训练来源提供通信成本，留出的开发目标仅用于对照。PP3 的设备传输可以独立估值，其 CPU 提交和外部端点仍各有证据要求。单卡端更换通信成本资产时复用了现有观测，TP2 和 PP3 的新增原生尝试均为零。[12 项结果](../../../data/architecture-acquisition-performance/f1/handoff.json) · [外接资格检查](../../../data/architecture-acquisition-performance/f1/stage-comparison/physical-profiles/binding-check.json)

**R3.2 已核对的边界**　TP2 的单卡、双卡模板各自展开同一份独立物理目标，438 次求和与 6 次收集的类型和元素数分别与 SDK 一致。给定统一耗时时，两套模板各自走完 serving，444 个端点全部经过载荷检查。TP8 的代表 rank 1 可为 rank 2 生成连接，PP3 的首、中、尾阶段生成 21 对载荷匹配的发送和接收。两种来源仍保留各自真实的 CPU 提交工作，严格整图与请求级精度沿后续路径验证。[单卡闭环](../../../data/architecture-acquisition-performance/r3/reduction-operands/single_device/given-time/result.json) · [双卡闭环](../../../data/architecture-acquisition-performance/r3/reduction-operands/full/given-time/result.json)

**R3.3 的成本边界**　TP2 的 1,332 个物理通信端点中，rank 0 有 520 个、rank 1 有 216 个能确定自身服务时长范围。其余 596 个保留时钟缺测。只提供 rank 0 的通信资产时，rank 1 的 666 项通信查询全部报告缺口。两种模板各自带独立 CPU 和设备样本运行到第一个缺失的 `model.embed_tokens` 主机工作，已执行的首项计时工作在三个查询入口取得相同值。[测量资格](../../../data/architecture-acquisition-performance/r3/service-evidence/qualification.json) · [逐项查询对照](../../../data/architecture-acquisition-performance/r3/service-evidence/query-comparison.json)

**R3.4 的范围**　用与训练相同的插桩另做物理参照，完整来源的 TP8、TP4 × EP4 分别在该案例达到全节点条件，覆盖 604、356 个动作。TP2 × PP3 在已有测量范围对齐 910 个动作。单卡来源保留本地计算和通信端口，但部分物理提交的 CPU 工作在暂存执行中没有发生，三项严格对照各自保留了首处分歧。六项自主预测的原始结果均在成本缺口处停止。本轮训练请求和各自的开发目标长度一致，独立泛化和请求级误差仍待验收。[首处分歧](../../../data/architecture-acquisition-performance/r3/stage-comparison/layouts/comparison.json) · [阶段对照](../../../data/architecture-acquisition-performance/r3/handoff.json)
