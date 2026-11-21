# 窗口状态

截至 2026-11-21，R1、R2、F1 已交付。按计划片段等权计为 **23/45（51.1%）**。开发分支为 `architecture-acquisition-performance`，开窗基线为 `ascend-baseline`（`e152aeb`）。

| 阶段 | 已完成 / 计划 | 当前交付 |
|---|---:|---|
| R1 全仓职责划分 | 7/7 | 物理部署、框架观测、成本拟合与独立验证各有维护入口。完整模式 TP1、TP2 已走三路开发对照。 |
| R2 正式任务工作流 | 5/5 | 公开入口支持冷建库、已覆盖负载零补采复用、原生观测结束后的离线恢复，逐目标发布所用资产。 |
| F1 双模式采集 | 11/11 | 七种拓扑已有单卡本地分片和连接证据。TP2、PP3 两种来源的三路对照已收齐 12 项结果。 |
| R3 两种来源的共同边界 | 0/4 | 从角色证据、模板与端口、成本查询三条职责链开始整理。 |
| F2 按需取证与性能模型绑定 | 0/7 | 待 R3 交付。 |
| R4 全仓收口 | 0/5 | 待 F2 交付。 |
| V 独立验证 | 0/6 | 最终留出验收独立计数。 |

## 单卡结果的适用范围

物理卡 4 独立取得了 Qwen3-8B TP1、TP2、TP8，Mistral-7B PP3、TP2 × PP3，Qwen3-30B-A3B TP4 × EP4，以及 Qwen3-8B Prefill TP1 / Decode TP2 的具名本地证据。TP8 在八个分片执行时仅详细观察两个代表。组合 PP 核对了六个角色上的载荷，EP 覆盖两种实际路由，PD 追到 Decode 两端的 72 次逐层首次读取。各项均有独立物理目标。[七项检查与来源](../../../data/architecture-acquisition-performance/f1/configuration-evidence.json)

| 开发来源 | 注入实测时间检查结构 | 在真实图上估计成本 | 自主闭环 |
|---|---|---|---|
| TP2 完整 | 200 个动作对齐，达到本例全节点条件 | 取得可比结果 | 取得可比结果 |
| TP2 单卡 | 比较了 71 个动作，首处分歧在 rank 0 的 `Scheduler.run_batch` | 444 项设备通信中 6 项 all-gather 可估值，438 项 all-reduce 缺少可绑定的目标载荷形状 | 首个成本缺口为 all-gather 调用范围内的主机工作 |
| PP3 完整 | 454 个动作在已测范围内对齐 | 42 项设备通信均可估值，676 个外部端点缺少重算时刻 | CPU 发送需要 1,696 字节，样本只覆盖 1,968 字节 |
| PP3 单卡 | 比较了 75 个动作，首处分歧在 rank 0 的 `isend` | 外接样本覆盖全部 42 项设备通信，同一参考的 676 个外部端点仍限制请求级重算 | 同一 CPU 发送大小超出范围 |

两份独立物理训练来源提供通信成本，留出的开发目标仅用于对照。PP3 的设备传输可以独立估值，其 CPU 提交和外部端点仍各有证据要求。单卡端更换通信成本资产时复用了现有观测，TP2 和 PP3 的新增原生尝试均为零。[12 项结果](../../../data/architecture-acquisition-performance/f1/handoff.json) · [外接资格检查](../../../data/architecture-acquisition-performance/f1/stage-comparison/physical-profiles/binding-check.json)

**R3 起点**　先统一表达实际参与者、被观察者和每次观测窗口。随后用同一套对外端口比较两种来源，并保留控制线程提交、调用方返回、设备通信完成三个不同的作用。TP2 首次 Prefill 的 151 个调用范围一一对应，内部工作布局的差异集中在 73 次求和与 1 次收集。all-reduce 的目标载荷需要可追溯的生产者或张量端口。PP3 的 `isend` 提供第二种通信边界的对照。局部通信估值、整图结构正确和请求级精度分别验收。[TP2 首处分歧](../../../data/architecture-acquisition-performance/f1/stage-comparison/tp2-action-layout.json)
