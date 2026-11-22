# 窗口状态

截至 2026-11-22，**当前窗口已交付 43/57 项（75.4%）**。前六阶段 39/39，第 01–17 轮已交付 4/17，第 18 轮最终验收待启动。按交付项等权计数，各项工时不同。第一个验证里程碑已完成，第 05 轮开始三模型同配置对照。

第 02 轮以 Qwen3-8B TP1 独立对照 31 与 32 个 token 的页边界。在两个目标、两种来源的四组对照中，全部匹配 283/283 个动作、返回值、资源效果及所查依赖。两份详细参考用实测耗时复现 38,866 个端点和 16 次 token 发布，四份物理参考均无首次加载。三路 12 项评价已收齐。两条性能路径仍缺三请求 Prefill 的批次准备成本，本轮验收依据执行结构。[独立确认](../../../data/architecture-acquisition-performance/rounds/02/confirmation/assessment.json) · [首次独立结果](../../../data/architecture-acquisition-performance/rounds/02/independent/assessment.json)

第 03 轮在原训练样本上为两套来源分别选出 7 个和 6 个 attention 模型绑定。两个新目标均通过结构对照。44 个 token 的目标中，两种来源均可求值 288 段未见过该特征组合的 Decode attention 和一次 184 字节的 KV 释放读回。100 个 token 的目标中，实际测得的 288 段 Decode attention 和 408 字节的释放读回被明确记为超出训练范围。整图与自主耗时仍缺其他工作成本。[独立对照](../../../data/architecture-acquisition-performance/rounds/03/independent/assessment.json) · [开发回归](../../../data/architecture-acquisition-performance/rounds/03/development/assessment.json)

第 04 轮的两组新到达间隔，各用两套来源完成结构和自主预测。930 毫秒案例的首个分歧发生在第二个批次。详细运行尚未结束第一个 Prefill，预测已结束该 Prefill 并选择 Decode；注入同次原生时间后，两套来源各匹配 148 个动作。2,750 毫秒案例的原生和自主预测均形成四个相同批次，固定图也可评分。两种来源的四项自主预测均取得独立轻量评分。[逐项结果](../../../data/architecture-acquisition-performance/rounds/04/independent/assessment.json)

第 01 轮在 Qwen3-8B TP1 新单请求及错峰六请求上取得 4 份稳态物理参考。填回同次详细采集的实测服务耗时后，固定图重现 51,653 个可核对端点和 22 次 token 发布。两种来源共用这两个物理目标。三条路径的 12 项评价全部有终态，替换为预测成本及自主闭环均未取得合格的请求误差。[第 01 轮逐项结果](../../../data/architecture-acquisition-performance/rounds/01/result.json)

## 既有交付与首轮证据

截至原方案 14:20 的记录，前六个阶段完成 39/39 个片段，加上旧 V.1–V.3 共 **42/45（93.3%）**。该比例只描述原计划。旧 V.4–V.6 的固定图、闭环与返修证据随相关机制进入新轮次。原来预登记的 16 组复验负载只有取值说明，尚无复验结果。[原单批预登记](../../../data/architecture-acquisition-performance/v-revalidation/inputs/specification.json)

原冻结实验的 96 项已收齐。固定真实图的 32 项中，22 项缺少时序重算证据，6 项参考含首次加载，4 项在已知真实图上误触发预测模板选择。自主闭环的 32 项中，23 项因参考含首次加载在预测前结束，6 项缺合格模板来源，3 项缺成本数据。两条性能路径可独立评分的请求数均为零。[首轮 96 项独立报告](../../../data/architecture-acquisition-performance/v/results/independent-assessment.json)

| 原阶段 | 已完成 / 原计划 | 交付时的证据 |
|---|---:|---|
| R1 全仓职责划分 | 7/7 | 物理部署、框架观测、成本拟合与独立验证各有维护入口。完整模式 TP1、TP2 已走三路开发对照。 |
| R2 正式任务工作流 | 5/5 | 公开入口支持冷建库、已覆盖负载零补采复用、原生观测结束后的离线恢复，逐目标发布所用资产。 |
| F1 双模式采集 | 11/11 | 七种拓扑已有单卡本地分片和连接证据。TP2、PP3 两种来源的三路对照已收齐 12 项结果。 |
| R3 两种来源的共同边界 | 4/4 | 三种拓扑、两种采集模式、三条验证路径共 18 项开发报告。使用相同插桩的独立目标又做了六项结构复核。完整模式三项对齐，单卡三项均有可定位的通信提交差异。[交付及原始结论](../../../data/architecture-acquisition-performance/r3/handoff.json) |
| F2 按需取证与性能模型绑定 | 7/7 | 两种来源各核对五种资产起点及三条开发对照路径，充分复用时新增原生启动和资产后处理均为零。双卡来源对齐 200 个结构动作、40,630 个固定图端点。单卡来源配上 444 段物理通信成本，自主预测产出 3 个 token。单卡与物理目标的主机提交仍有差异，由 R4.4 追查。[阶段结果与适用边界](../../../data/architecture-acquisition-performance/f2/handoff.json) |
| R4 全仓收口 | 5/5 | 已交付模型绑定、共同求值、产物发布与恢复、独立参考及评分四条职责链。已审查 249 个业务源文件、4 个示例源文件和 20 个相关文件。三种拓扑各用两种来源走完三路开发对照。[18 项结果与边界](../../../data/architecture-acquisition-performance/r4/handoff.json) |
| V 首轮独立验证 | 3/6 | 32 份观测覆盖 16 组目标、59 个请求和 204 个输出 token。结构对照 32 项全部归因。缓存配置的同一个 TP1 短请求在两种来源下均完整匹配。其余 18 项缺适用子图，6 项有实际子图布局差异，6 项缺合格来源。[原始结构对照](../../../data/architecture-acquisition-performance/v/results/frozen/comparisons.json) · [真实参考](../../../data/architecture-acquisition-performance/v/physical) · [各负载预热证据](../../../data/architecture-acquisition-performance/v/results/native-warmup-evidence.json) |

13:55 的修复回归中，Qwen3-8B 缓存短请求已能自主预测全部 4 个 token。一次额外的同输入轻量采集在目标窗口中观察到零次首次加载，用它核对同一份预测，TTFT 和 TPOT 误差为 9.12% 和 6.74%。这两个数只用于开发回归。[开发回归结果](../../../data/architecture-acquisition-performance/v-repair/representative-warmup/result.json)

14:20 的开发回归使用原训练库，为 Qwen3-8B TP1 和 TP2 两个短请求各自选出了原先缺失的一次模型调用，实际工作和局部依赖均匹配。一次新的 TP1 训练启动取得 1 次申请新页和 9 次沿用原页的 Decode，已建库。MoE 完整训练中四个 rank 的逻辑单请求均可识别。待核对的范围包括完整请求的跨调用依赖、单卡通信、PD 批次准备及两条性能路径。[模型调用](../../../data/architecture-acquisition-performance/v-repair/template-choice/results.json) · [分页训练](../../../data/architecture-acquisition-performance/v-repair/allocation-source/results.json) · [来源发布](../../../data/architecture-acquisition-performance/v-repair/qualification/results.json)

首轮冻结前的模板训练共启动 28 次原生作业。25 个作业发布了可追溯的模板，3 个主库建库失败。32 种“留出负载与来源模式”组合中，26 种能定位到声明支持该目标的模板库，余下 6 种为 MoE 的两种模式及 PD 单卡各两个负载。实际展开还逐动作检查库中能否选出合适的子图。定向成本采集共启动 65 次。16 套成本资产中有 11 套覆盖各自已发布模板里的全部训练工作，其中含只发布局部库的 PD 单卡。[逐配置资产、适用范围和缺口](../../../data/architecture-acquisition-performance/v/freeze.json)

前六种配置中有四种覆盖训练成本需求。Mistral-7B TP2 × PP3 的完整部署仍缺 246 个成本特征点，单卡仍缺 150 个。Qwen3-30B-A3B TP4 × EP4 两种模式都只发布了补充库，两个留出输入均选不到主库。完整部署把已运行的单请求 Decode 判为缺失，单卡则把同一专家回传编成两种服务，两处已分别定位。MoE 单卡库另缺 23 个成本点，涉及 1,449 段工作。其中 1,448 段已有可对应的四卡服务测量，现有成本表示尚未接入这些数据。剩下 1 段普通归约，用正式的逐对端计算无法定时。离线合并四卡原生通知后，其服务时长可限定在 15.29–21.17 微秒。这项结果来自原训练，尚未进入正式成本库。[通信组校时证据](../../../data/architecture-acquisition-performance/v/diagnosis/development-joint-notify-bounds.json) · [逐类成本去向](../../../data/architecture-acquisition-performance/v/diagnosis/training-expert-staged-cost-coverage.json) · [完整部署的发布条件](../../../data/architecture-acquisition-performance/v/diagnosis/training-expert-batch-qualification.json) · [单卡的模板歧义](../../../data/architecture-acquisition-performance/v/diagnosis/training-expert-staged-service-kind.json)

第七种 PD 配置的完整部署已发布主库和局部库，单卡仅发布局部库。原生 Decode 对每个请求各做一次缓存索引拷贝。单卡主库将一请求和三请求编成相同的适用条件，建库时就报告歧义。完整主库在训练中只见过单请求。拿它展开单卡训练中的三请求批次，输出张量尺寸正确，却只生成 1 次拷贝，实测有 3 次。完整部署成本仍缺 8,022 段主机工作，其中 5,256 段归约提交已有可用的局部 CPU 计时。单卡成本覆盖已发布局部库的 10,800 段工作。[候选条件与实际展开](../../../data/architecture-acquisition-performance/v/diagnosis/training-pd-preparation-cardinality.json) · [主机时延证据](../../../data/architecture-acquisition-performance/v/diagnosis/training-pd-host-operands.json)

最后一种 Qwen3-8B TP1 配置启用了分块预填充、混合组批和一级前缀缓存。两种模式均已发布主库和局部库。两份主库的训练各实际出现 6 个混合批次、3 次跨请求前缀命中和 4 次驱逐。3 次命中复用同一段 32 个 token 的前缀，原生记录可追到两次各写入 16 个 token 的操作。[训练行为及实际写入位置](../../../data/architecture-acquisition-performance/v/diagnosis/training-cache-behavior.json)

## 单卡结果的适用范围

物理卡 4 独立取得了 Qwen3-8B TP1、TP2、TP8，Mistral-7B PP3、TP2 × PP3，Qwen3-30B-A3B TP4 × EP4，以及 Qwen3-8B Prefill TP1 / Decode TP2 的具名本地证据。TP8 在八个分片执行时仅详细观察两个代表。组合 PP 核对了六个角色上的载荷，EP 覆盖两种实际路由，PD 追到 Decode 两端的 72 次逐层首次读取。各项均有独立物理目标。[七项检查与来源](../../../data/architecture-acquisition-performance/f1/configuration-evidence.json)

## R4 交付时的开发对照

每种拓扑使用一个独立开发请求，输入 17 个 token，生成 3 个 token。两种来源共用同一份详细物理参考和同一份轻量计时参考。表中的 TTFT、TPOT 分别为首个 token 时延和之后每个 token 的平均时延。百分数是此次请求各指标的绝对百分比误差。[逐项记录](../../../data/architecture-acquisition-performance/r4/handoff.json)

| 拓扑与来源 | 注入实测时间后的结构 | 固定真实图后的成本评价 | 自主闭环 |
|---|---|---|---|
| TP2 完整 | 200 个动作全部匹配 | 实测基线通过 40,630 个端点，TTFT / TPOT 误差 15.95% / 15.20% | 1.98% / 8.75% |
| TP2 单卡 | 比较 59 个动作后，在 rank 0 提交批次时出现差异 | 同一实测基线通过，单卡样本缺少物理提交线程的工作 | 9.52% / 4.94% |
| PP3 完整 | 454 个动作在已测范围内匹配；控制器转发请求的消息字节数仍有差异 | 676 个主机协议端点均有实测时刻，固定图重算尚未使用 | 原生转发 1,968 字节，成本库有两次该尺寸实测。控制器生成 1,696 字节，成本查询因此停止 |
| PP3 单卡 | 比较 108 个动作后，张量发送的完成端口及执行流出现差异 | 受同一重算缺口影响，另有主机成本缺口 | CPU 请求转发同样从 1,968 变为 1,696 字节，两套来源共享控制器 |
| PD 完整 | 5,464 个动作全部匹配 | 已评价 296 段设备通信。另有 280 段 CPU 集合通信具备同机实测边界，固定图验证器仍只查询设备记录 | 16.31% / 13.28% |
| PD 单卡 | 比较 5,335 个动作后，在 Decode rank 2 提交批次时出现差异 | 受同一 CPU 通信验证缺口影响，现有成本覆盖 296 段设备通信 | 单元素 CPU 集合通信未进入成本库。已有单卡计时含同次运行的两端记录，预热后的采样可对应模板内全部 165 段主机归约 |

单卡实际发生的本地工作、外接的跨卡服务和真实通信提交线程各有证据来源。其余四种拓扑保留前期实际覆盖的范围。TP8 与 TP4 × EP4 的完整来源曾在开发案例达到全节点结构条件，TP2 × PP3 在已测范围匹配 910 个动作。独立泛化由 V 检查。[F1 角色采集](../../../data/architecture-acquisition-performance/f1/handoff.json) · [R3 结构与成本](../../../data/architecture-acquisition-performance/r3/handoff.json)
