# 多配置模拟预测大范围验证与独立采集收益对照日志

## 2026-11-27 — 建窗：开启 M1–M3 短里程碑多配置模拟预测验证窗口

**范围**　承接上一窗口阶段收口提交 `53f1464`，切出新分支 `multi-config-prediction-validation`，建立只读纲领 `outline.md`、三段短里程碑计划 `plan.md` 与初始状态 `status.md`。

**改动**　经用户、AGY 与 Codex 三方复盘达成共识，正式确立本窗口四条执行铁律：① 主线反转，完整拓扑短流优先出多配置模拟预测结果（M1 TP2/4/8 & Mistral → M2 PD/PP/MoE-EP → M3 Azure 真实流量与 8 域总表），单卡建库与局部算子提速独立取证不阻塞主线；② 采集反特例红线，局部探针仅限可由明确输入复现的弱状态算子，冻结 `recipes/scheduler.py` 并严禁新增逆向调度状态构造剧本；③ 冻结前事前预登记轻量参考重复次数 `K` 并按请求级中位数聚合，严禁事后补测挑拣；④ 强制 M1/M2/M3 三处停机门控，单点超过 2 轮修复或 3,000 卡秒强制止损汇报。

**验证**　基线 `53f1464` 已通过 AGY 全量代码品质与四维隔离审计（219 模块 0 环 0 缺失、>550 NE = 0、>25L 代码墙 = 0、`ruff` 0 错、0 `.pyc`），本批仅建窗未触发真机采集（累计 0 卡秒）。

**结论与限制**　窗口就绪，交由 Codex 在干净工作区上启动 M1，完成 M1 交付表后停机待审。

## 2026-11-27 — M1：冻结评分接入与首份 TP2 开发来源

**范围**　重建运行资产，接入冻结前预登记的轻量参考，并启动完整拓扑短流的建库流程。

**改动**　矩阵将每例两或三份独立轻量参考的路径固定在冻结清单中。验收核对每份录制的训练隔离、逐次误差和盲测输出摘要，再按请求计算 TTFT、TPOT 中位参考。

**验证**　双镜像已核对安装哈希，C++ Release 编译通过，`ruff` 与 225 个 Python 模块的静态检查通过。用历史 S05 两份独立录制复算 32 请求，逐次 TPOT MAPE 为 6.59% 和 1.55%，按请求聚合后为 3.94%。新镜像取得 Qwen3-8B TP2 两 rank 原生来源，初始化 27.08 秒、预热 66.98 秒、有效观测 25.51 秒，按容器运行时间结算 321.46 卡秒，原始录制约 320 MiB。 [静态证据](../../../data/multi-config-prediction/dev/m1/bootstrap/static-round-1.json) · [真机收据](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp2/source-native/execution.json)

**结论与限制**　复算属于评分开发回归。首份来源已封存，成功录制后的投影、建库和独立验收仍在推进。

## 2026-11-27 — M1：TP2 开发图成本覆盖与错峰合批缺口

**范围**　核对 Qwen3-8B TP2 完整拓扑开发来源和固定图基线，推进 Qwen TP4、TP8 与 Mistral TP2 完整拓扑录制。

**改动**　同负载轻量宿主计时提供整轮服务样本，两个 rank 的同步拷贝按实际物理卡以明确字节数合批采样。第二段开发流声明错峰到达，使用正常请求提交观测原生合批。

**验证**　TP2 来源核验通过，配对宿主计时在每个 rank 对齐 20 批并消除 2,485 个宿主坐标缺口，两个同步拷贝探针补齐剩余 32 个坐标。首份开发图 171,028 个工作实例全部可定价。已有独立 S07 开发录制的固定实测基线通过，结构比对 314 动作后缺少合批，固定图仍有 58 个工作实例缺预测成本。TP4 和 Mistral TP2 录制通过，累计结算 1,594.14 卡秒。 [成本需求](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp2/costs-source.json) · [固定图](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp2/dev-fixed/result.json) · [结构](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp2/dev-structure/result.json)

**结论与限制**　首份分组短流覆盖已有工作。跨到达轮次的合批需要实际原生声明。独立留出尚未冻结，单卡短通信录制在已有七请求短流中仅命中每 rank 76/962 个通信叶。

## 2026-11-27 — M1：原生错峰合批复验与单卡装配候选

**范围**　对照 TP2 首轮分组和原生错峰流，核对 TP4 实测基线并准备独立留出声明。

**改动**　将错峰流中实测的队列合批模板并入 TP2 开发库。单卡装配使用一条 8-token、单输出请求的分片录制，按通信类型、参与者和张量端口匹配独立物理片段。留出声明固定每例两份轻量参考，Qwen 各 TP 共用同一随机请求序列。

**验证**　两个 rank 均观测到两次 `2+2` 和两次 `3+1` 合批，之前停在 314 个动作的开发目标复验达到 `full_structure_matched`，546 个动作匹配且训练来源隔离。两份 TP2 开发图的合并成本仅剩 4 个同步拷贝坐标。TP4 固定图的实测基线通过。单卡候选的每 rank 74 个通信片段均匹配物理探针并完成编译。累计结算 3,993.48 卡秒。 [原生合批](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp2/native-batch-transitions.json) · [结构](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp2/dev-structure-staggered/result.json) · [装配](../../../data/multi-config-prediction/dev/m1/assembly/composed/composition.json)

**结论与限制**　错峰开发流解决了已知 TP2 合批缺口。跨 TP 模板引用、单卡装配整图和四例独立留出仍待验证。当前留出文件只含抽样范围和参考路径。

## 2026-11-27 — M1：单卡装配完整对照及跨 TP 开发复验

**范围**　完成单卡装配独立对照，并核对 TP2 固定图时延与 TP4 模板复用。

**改动**　单卡装配以新录制的完整双卡 TP2 为目标。TP4 保持自己的角色和资源声明，通过既有模板引用复用 TP2 实测合批。TP2 自主预测增加逐批顺序对比。

**验证**　8-token 单输出装配用例全部 108 个动作相符，训练来源隔离。首次计入物理探针需 356.49 卡秒，完整双卡需 222.39 卡秒。兼容探针已具备时，单卡新增采集需 118.93 卡秒，降低 46.52%。TP4 开发结构 1,064 个动作全匹配。TP2 固定图 53,556 个工作实例均有成本，替换全部成本的 TTFT／TPOT MAPE 为 6.35%／3.01%，仅恢复实测宿主成本后为 0.43%／0.36%。旧轻量录制上的自主预测为 1.68%／15.52%，两 rank 的 6 个批次各自同序。 [装配报告](../../../data/multi-config-prediction/dev/m1/single-device-assembly-report.json) · [跨 TP 结构](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp4/dev-structure-cross-tp/result.json) · [固定图](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp2/dev-fixed-combined/result.json) · [自主批次](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp2/dev-autonomous-schedule.json)

**结论与限制**　独立双卡对照确认了本例全部装配关系。其他通信形状仍依赖各自证据。TP2 自主结果需要进一步对照轻量逐批时钟，固定图结果无法直接代表无 profiler 的时延。四例独立留出尚未验收。

## 2026-11-27 — M1：多卡错峰录制与 TP2 逐批时延归因

**范围**　取得 TP4、TP8 完整拓扑的错峰来源，核对 Mistral TP2 固定图基础与 Qwen3-8B TP2 自主时延差异。

**改动**　用现有逐批探针对齐两个 rank 的请求成员和上下文长度，比较原生与预测的 `run_batch` 边界。合并各已知同步拷贝坐标，使用已有的明确输入探针准备同卡批量采样。

**验证**　TP4、TP8 本批分别录制 4、8 个 rank。Mistral 的固定图通过 135,156 个实测边界核对，首份模板库在第 326 个动作后遇到缺失的合批调用，两份配对成本裁减至每 rank 19 个拷贝坐标。TP2 的四个 Decode 批次在两个 rank 均完全同序，`run_batch` 预测每批平均高 22.69 和 21.54 毫秒；单独替换设备通信的固定图 TTFT、TPOT 变化均低于 0.001%。本批四次已结束录制合计 2,538.57 卡秒，累计 8,915.03。四次录制的初始化、预热、有效观察分别合计 102.71、280.34、49.92 墙秒；本批已完成的对照、提样和拟合累计 202.12 墙秒，另有两项构图在运行。本批原生目录约 1.37 GiB，窗口存储采样峰值约 10.41 GiB。 [逐批配对](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp2/dev-batch-timing-comparison.json) · [固定图分组](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp2/dev-fixed-attribution/result.json) · [Mistral 成本](../../../data/multi-config-prediction/dev/m1/cases/mistral7-tp2/costs-staggered-paired.json)

**结论与限制**　同配置、同请求、同卡和同镜像的复核中，两种观测各自匹配每 rank 11 批，Decode 主线程时间的逐批中位差为 20.13 和 22.25 毫秒。该录制新增 281.50 卡秒，初始化、预热、有效观察分别为 23.80、73.02、4.25 墙秒，累计 9,196.53 卡秒。现有证据把差异定位到宿主采样口径，尚未定位到具体探针或工作区间。TP4 和 TP8 错峰构图分别用 317.60、675.38 墙秒；开发对照运行中。 [同负载复核](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp2/dev-paired-host-probe-overhead.json)

## 2026-11-27 — M1：多卡开发结构与同步拷贝收口

**范围**　独立对照 TP4、TP8 错峰图，补齐各配置已识别且可用明确输入复现的同步拷贝成本。

**改动**　按物理卡合批执行既有拷贝探针，仅将执行域相符的样本放入各配置的拟合。两张卡的首次预检缺少原拟合包络，补入已有真实边界后提交；预检未启动真机。

**验证**　TP4、TP8 错峰图分别匹配 2,046、4,030 个动作。TP8 固定图通过 630,880 个实测边界核验；TP4 独立固定图的 107,084 项成本全部覆盖。八个拷贝会话逐项答复 274 个查询，用 113.47 卡秒；合计准备、预热、观测、读回均为探针墙钟，分别为 8.74、0.01、0.10、0.06 秒。TP4 配对宿主录制用 600.66 卡秒，初始化、预热、有效观察分别为 25.38、74.45、10.30 墙秒。累计结算 9,910.65 卡秒，存储采样峰值 11.72 GiB。 [TP4 结构](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp4/dev-structure-staggered/result.json) · [TP8 结构](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp8/dev-structure-staggered/result.json) · [预算](../../../data/multi-config-prediction/dev/m1/budget-snapshot.json)

**结论与限制**　三份开发库覆盖全部已查工作；TP8 当前未覆盖项集中在一次 KV 成组释放的两个宿主调用，每 rank 14 个工作区间。TP2 八请求开发复验的 TTFT、TPOT MAPE 为 3.11%、14.42%，说明采样偏差仍影响闭环。正式留出尚未启动。 [TP8 成本覆盖](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp8/costs-settled.json) · [TP2 开发时延](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp2/dev-staggered-publication-score.json)

## 2026-11-27 — M1：通信探针准备时间归属修正

**范围**　定位 TP2 宿主采样中残留的观察器工作，复验修正并确认 TP8 自主预测的首个实际缺口。

**改动**　通信观察器把参数绑定、张量检查及记录准备显式登记为辅助区间，宿主计时保留实际原生调用的起止。共用插桩继续为详细图采集相同的通信语义。官方脚本通过宿主构建网络复用依赖缓存，重建推理和控制镜像并核对实装哈希。

**验证**　CPU 调用栈采样在既有剔除区间内捕获到队列记录器调用。修正前后两个 rank 的 11 批及 4,591 个宿主调用全部配对，Decode 主线程相对轻量计时的差值分别从 20.13、22.25 毫秒降至 8.96、7.01 毫秒。两次已结束录制新增 578.33 卡秒，初始化、预热和观测墙钟合计分别为 49.63、142.69、18.00 秒，原生记录与诊断采样保留约 183 MiB；正式双镜像构建 7.17 秒。累计结算 10,488.98 卡秒。 [同负载复测](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp2/dev-observer-correction-comparison.json) · [镜像](../../../data/multi-config-prediction/dev/m1/bootstrap/observer-correction/images.txt) · [静态检查](../../../data/multi-config-prediction/dev/m1/bootstrap/observer-correction/static.json)

**结论与限制**　该修正减少了已测到的采样差异，修正后的 TPOT 精度仍待独立请求发布对照。TP8 实际运行在 `release_calls=3` 处触及样本上界，当前请求的缺口记录缺少通用调度回放上下文。四例正式留出尚未运行。 [TP8 首个缺口](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp8/dev-autonomous-staggered/cost-gap.json)

## 2026-11-27 — M1：修正后 TP2 成本通过开发时延对照

**范围**　保持预测模板和设备样本不变，将计时修正落实到实际宿主成本，比较同一输入的请求结果。

**改动**　用新镜像取得两 rank 的详细图并核验原生来源，再提取同镜像配对宿主样本。开发拟合仅在规范化工作、执行域及完整特征相同的位置优先使用新测量，保留其余旧样本的真实来源。 [样本选用记录](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp2/observer-correction-samples/selection.json)

**验证**　两 rank 各配对 11 批，宿主区间缺测为零，重新拟合覆盖三个开发图中的 377,892 项工作。四请求独立轻量录制与八请求逐批诊断的 TPOT MAPE 由 15.52%、14.42% 分别降至 5.05%、5.06%，TTFT MAPE 为 1.11%、1.01%。本批一次详细录制新增 309.62 卡秒，初始化、预热和观察分别 25.73、71.42、16.03 墙秒；图投影 145.38 墙秒，累计结算 10,798.60 卡秒。 [四请求对照](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp2/dev-observer-corrected-page-crossing/result.json) · [八请求对照](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp2/dev-observer-corrected-score.json)

**结论与限制**　TP2 的两个开发结果均低于 6%。TP8 的原生工作预检确认现有局部配方会用推导的入队分组代替实际回放，预检未上卡。剩余 7,201.40 卡秒接近正式参考录制估算，其他拓扑的宿主采样修正、TP8 成本缺口与留出验收仍待闭合。 [回放预检](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp8/dev-release-replay-preflight.json) · [费用口径](../../../data/multi-config-prediction/dev/m1/reference-budget-forecast.json)

## 2026-11-27 — M1：TP8 原生补齐、编译复用及留出预检

**范围**　补齐原生错峰流的 TP8 宿主成本，测量编译产物复用对真机预算的影响，并固定四例独立留出的输入与参考次数。

**改动**　沿用原详细图的采集镜像和八请求流，配对录制一次宿主时间。Ascend 部署允许显式指定 Triton 编译缓存，各份参考继续独立预热与计时。验收要求所有预登记共享请求组均报告完整的吞吐、P95 TTFT 和平均 TPOT 排序。

**验证**　TP8 每 rank 的 11 个批次全部配对，实测覆盖单批三次 KV 释放。两份开发图共 1,091,630 项工作可定价，八请求自主预测完成。TP4 轻量录制用 92.42 墙秒，预热 26.01 秒，观测窗口中新加载编译项为零。本批两次录制共用 1,583.90 卡秒，初始化、预热和有效观测分别合计 50.12、102.77、14.90 墙秒，八 rank 提样与拟合合计约 222.5 宿主秒；录制保留约 179 MiB，宿主样本导出约 13.4 MiB，编译缓存约 76 MiB。 [TP8 拟合](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp8/costs-native-staggered.json) · [TP8 预测](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp8/dev-native-staggered-prediction/result.json) · [TP4 轻量记录](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp4/dev-warmed-light/result.json) · [预算](../../../data/multi-config-prediction/dev/m1/budget-snapshot.json)

**结论与限制**　TP4 单次轻量对照的 TTFT／TPOT MAPE 为 8.26%／9.50%。首份留出冻结四案，每案例两份轻量参考、Qwen 三案共用同一组八请求。四案初始覆盖预检均止于成本包络，三例 Qwen 缺单请求 17-token Prefill，Mistral 缺 `1+3` 合批，现行配方未取得该合批的真实回放历史。预检不占用设备，正式参考尚未录制。 [TP4 对照](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp4/dev-staggered-warmed-score.json) · [首轮冻结归档](../../../data/multi-config-prediction/dev/m1/retired-holdout-9b5fec66/freeze/m1/source-manifest.json) · [配方预检](../../../data/multi-config-prediction/dev/m1/retired-holdout-9b5fec66/results/m1/initial-coverage/recipe-preflight.json)

## 2026-11-27 — M1：普通 dense 合批的原生回放

**范围**　首份冻结因真实调度回放缺失退役进 Dev。复用其请求验证修复，保留冻结清单、预检和中止的来源核验记录，正式参考录制数为零。 [退役记录](../../../data/multi-config-prediction/dev/m1/retired-holdout-9b5fec66/retirement.json)

**改动**　普通 dense 调度亦使用现有入队回放合同，保留所有 TP 共用的条件。局部采集保留编译缓存配置，关闭该局部进程的 PD 传输环境。首个合批样本发布后，配对宿主采集复用现有多 rank 观察，在同一次物理运行中独立核对每个 rank 的样本。

**验证**　Mistral 的开发缺口保留 6 个选择步和 8 个请求的入队历史。两遍原生回放的等待、运行、前一批及选批相符，rank 0 的目标成本成功发布。本次单 rank 配对用 290.01 卡秒，两遍选定 worker 的初始化合计 15.44 秒、池准备与读回 18.60 秒、捕获与导出 11.80 秒，另计资格处理 1.80 秒和发布 7.59 秒，保留录制及样本约 1.43 MiB。 [回放绑定](../../../data/multi-config-prediction/dev/m1/replay-regression/binding-proof.json) · [真机资格](../../../data/multi-config-prediction/dev/m1/replay-regression/mistral-native/result.json) · [镜像校验](../../../data/multi-config-prediction/dev/m1/bootstrap/ordinary-replay/installed-flow.sha256)

**结论与限制**　首次发布后的预测越过了 rank 0，在 rank 1 的同一合批点终止。现有探针已经启动两张卡，单 rank 观察使同一物理批次的样本需要重复启动。 [rank 1 缺口](../../../data/multi-config-prediction/dev/m1/replay-regression/mistral7-tp2-published/cost-gap.json)

## 2026-11-27 — M1：同一原生回放发布各 rank 实测

**范围**　用已知 rank 1 缺口验证一次物理配对能否同时覆盖两个 rank，核对当前正式参考预算和 Qwen 可复用的样本。

**改动**　按照已声明的物理参与角色采集。每个 rank 单独核对两遍原生切点和产生的样本，同一发布保留各自的身份与耗时。单卡模式仍用当前选定角色。

**验证**　同一录制的两个 rank 各有 122 项合格宿主样本（含目标合批各 22 项），两份表都已发布。一次配对结算 263.98 卡秒，按各遍最慢 rank 计的初始化、准备与读回、采集与导出合计分别为 17.36、6.50、9.24 秒，资格处理和发布合计约 10.93 秒，保留约 2.46 MiB，样本表约 0.31 MiB。累计结算 12,936.49 卡秒。 [同轮资格](../../../data/multi-config-prediction/dev/m1/replay-regression/mistral-peer-native/result.json) · [发布清单](../../../data/multi-config-prediction/dev/m1/cases/mistral7-tp2/replayed-release/current.json) · [资源账本](../../../data/multi-config-prediction/dev/m1/resource-accounting.json)

**结论与限制**　Mistral 回归越过两 rank 的合批，下一处缺口是 32-token 单请求 Prefill 的宿主工作。Qwen 三种 TP 的对应样本（17 token）复核后仍低于已匹配的 31-token 下界。余额 5,063.51 卡秒，四案参考暂估 4,880 卡秒，补采费用尚未完整进入估算。正式留出仍无物理参考。 [Mistral 新缺口](../../../data/multi-config-prediction/dev/m1/replay-regression/mistral7-tp2-peer-published/cost-gap.json) · [Qwen 样本核对](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp8/dev-single-prefill-sample-audit.json) · [参考预算](../../../data/multi-config-prediction/dev/m1/reference-budget-warmed.json)

## 2026-11-27 — M1：刷新多卡宿主训练录制

**范围**　将 M1 额度更新为 23,000 卡秒。在正式冻结前为 TP4、TP8 和 Mistral TP2 用通信观察修复后的镜像重录。沿用原生错峰请求和分组提交，增加独立提交的 16、17、32、33 token 请求，与已有 15、31 token 同轮录制。

**改动**　固定原生镜像并使用 ABI 兼容的 Triton 缓存。每种配置分别录制详细图和宿主纯计时，两遍使用同一请求及拓扑。保留旧设备与物理通信资产的独立来源，筛除修复前宿主表。 [镜像源码哈希](../../../data/multi-config-prediction/dev/m1/bootstrap/ordinary-replay/installed-clean-observers.sha256) · [TP4 工作负载](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp4/clean-host/input.json)

**验证**　TP4 的两次录制共用 1,122.06 卡秒。初始化、预热和观测分别累计 52.47、69.39 和 81.63 墙秒。四个 rank 均有 43/43 个按请求身份及批次形状对齐的批次，包含 15、16、17、31、32、33 token 单请求 Prefill。此前 TP2 单卡的 17-token 补采用 143.94 卡秒，分片模板的子调用与物理模板不同，正式资格检查没有发布该条宿主成本。 [TP4 批次核对](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp4/clean-host/paired-batch-geometry.json) · [单卡缺口资格](../../../data/multi-config-prediction/dev/m1/prefill-shard-regression/native/result.json)

**结论与限制**　本段 TP4 证明了批次及切点可重录。配对宿主样本、所需覆盖和轻量参照误差仍须核对。账本累计结算 14,202.49 卡秒，余额 8,797.51 卡秒。正式留出结果仍为 0/4。 [资源账本](../../../data/multi-config-prediction/dev/m1/resource-accounting.json)

## 2026-11-27 — M1：新宿主成本复验与 TP8 止损

**范围**　核验 TP4 与 Mistral TP2 的配对宿主成本和短 Prefill，使用原八请求开发流复算 TP4。TP8 缩短至八请求并改用设备详细和独立宿主计时配对，验证能否在剩余额度内取得样本。

**改动**　TP4 与 Mistral TP2 的新成本库筛除旧宿主表，保留已核对的设备、通信及拷贝样本。Mistral 增补同一修正镜像下已合格的原生回放证据。短请求实测与退役开发流原缺口逐项核对，匹配域、模板工作区间和全部特征。 [TP4 训练来源](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp4/clean-host/sample-provenance.json) · [Mistral 训练来源](../../../data/multi-config-prediction/dev/m1/cases/mistral7-tp2/clean-host/sample-provenance-complete.json)

**验证**　TP4 四 rank 与 Mistral 两 rank 均为每 rank 43/43 批次配对，来源核验分别通过 6,908 和 3,562 个动作。TP4 的 17-token 模型调用区间与 Mistral 的 32-token 原生批次区间取得直接实测。Mistral 的原八请求流可预测，退役流另有 rank 0 的 `1+3` 合批设备成本缺口。TP4 的同一轻量参照上 TTFT／TPOT MAPE 由 8.26%／9.50% 变为 10.92%／5.39%。首批 TTFT 低估约 125 毫秒。另一个已隔离的四请求固定图为 0.96%／3.06%，恢复实测宿主时间后为 0.26%／0.24%，设备或集合通信的单独替换影响很小。该固定图尚不能定位八请求首批差距对应的原生边界。 [Mistral 开发预测](../../../data/multi-config-prediction/dev/m1/cases/mistral7-tp2/clean-host/dev-prediction/result.json) · [TP4 开发复验](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp4/clean-host/dev-score.json) · [TP4 固定图归因](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp4/clean-host/dev-independent-fixed/result.json)

**结论与限制**　Mistral 配对新增 529.13 卡秒，初始化、预热和有效观测合计分别为 52.28、62.62、67.02 墙秒。TP8 两次均完成请求观测后被超时终止，Docker 退出码均为 137，累计 2,419.32 卡秒。短流已有原始采集清单，正式录制未发布，仅保留作诊断。单点余 580.68 卡秒，即约 72.58 秒八卡运行。加上 TP4 TTFT 门槛未通过，四例正式清单未冻结，正式验收为 0/4。累计 17,150.94 卡秒，宿主处理与协调约 10,914 秒，存储采样峰值 22.59 GiB。 [TP8 首次退出](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp8/clean-host/service-reference/container-exit.json) · [TP8 短流退出](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp8/clean-host/short-recovery/resource-trace/container-exit.json) · [资源账本](../../../data/multi-config-prediction/dev/m1/resource-accounting.json)

## 2026-11-27 — M1：TP4 首批离线归因与 Mistral 拟合输入修正

**范围**　用已有开发录制定位 TP4 首批 TTFT，并复查 Mistral 端到端预测为何仍缺 `1+3` 合批设备成本。

**改动**　上一轮 Mistral 拟合传入了旧的样本清单，漏读来自已合格 rank 0 原生回放的设备区间。重新拟合时按完整清单核对路径和原生录制，所需区间有 1.52 微秒实测。TP4 和 Mistral 的固定图各自排除本次参考录制派生的全部样本。 [Mistral 修正后拟合](../../../data/multi-config-prediction/dev/m1/cases/mistral7-tp2/clean-host/replay-fit/costs.json) · [TP4 样本隔离](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp4/clean-host/staggered-fixed-isolation/source-selection.json)

**验证**　Mistral 的原开发流与退役流各完成 8 个请求、发布 25 个 token。独立固定图基线与实测边界一致，87,744 项工作全部可定价，TTFT／TPOT MAPE 为 3.72%／5.88%。TP4 的同八请求独立固定图覆盖 206,802 项工作，MAPE 为 0.57%／6.59%，恢复实测宿主时间后为 0.01%／0.06%。在配对宿主录制和仿真之间，首批 `run_batch` 墙钟相差 42.80 毫秒，批前相差 9.40 毫秒，合计 TTFT 相差 50.64 毫秒。rank 0 批内 2,047 项主线程工作全部配对，合计 CPU 服务拟合比该次实测少 7.13 毫秒。 [Mistral 退役流开发回归](../../../data/multi-config-prediction/dev/m1/cases/mistral7-tp2/clean-host/replay-fit/retired-regression/result.json) · [TP4 八请求固定图](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp4/clean-host/staggered-fixed-isolation/verification/result.json) · [TP4 批内 CPU 服务](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp4/clean-host/first-prefill-host-service.json)

**结论与限制**　TP4 的纯时延录制使用八请求预热，只记录提交、原生接收及首 token 发布。配对宿主录制使用 23 请求预热，二者首批 TTFT 相差 74.44 毫秒，现有纯时延记录无法定位这段差异。固定图保留实测等待条件，仍以自主开发流的 10.92% TTFT MAPE 判断冻结准入。TP8 单点预算仍受限，本轮上卡增量为零，宿主累计约 11,409 秒。 [TP4 首批边界对照](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp4/clean-host/first-prefill-boundaries.json) · [资源账本](../../../data/multi-config-prediction/dev/m1/resource-accounting.json)

## 2026-11-27 — M1：正式参考启动门禁与模板来源核验

**范围**　检查正式采集准入及 Mistral 最终开发库的来源证据，保持四案正式输入隔离。

**改动**　原入口以盲测终止报告存在为准，未成功的案例也可能触发完整参考采集。详细和轻量参考入口现在先检查每个冻结案例的成功状态和预测文件哈希，诊断仍可读取失败结果。 [留出门禁](../../../example/coarse_graph/holdout.py) · [矩阵入口](../../../example/coarse_graph/matrix.py)

**验证**　离线预检覆盖两种正式采集入口。盲测未完成或封存预测发生改动时，入口在写出采集任务及卡时预约前报错。封存结果完整时通过准入检查。Mistral 最终开发库的两份完整拓扑来源和两份已合格的原生回放均通过模板核验，现有矩阵选源入口确认四个录制均有证据。 [Mistral 来源清单](../../../data/multi-config-prediction/dev/m1/cases/mistral7-tp2/clean-host/replay-fit/source-evidence/selected-source.json)

**结论与限制**　本轮上卡增量为零。按 4,880 卡秒正式参考估算，M1 约余 969 卡秒供开发准备。若 TP8 详细录制按最近一次 756 卡秒实耗计，约余 213 卡秒供独立宿主录制和 TP4 诊断，需同时复核窗口余额及单点额度。当前修正镜像下查到的两份 TP8 录制均未取得成功退出凭证。 [参考预算及 TP8 账本](../../../data/multi-config-prediction/dev/m1/reference-budget-warmed.json)

## 2026-11-27 — AGY 审查与放行：实锤 TP4 跨 NUMA 慢卡根因并放宽预算

**范围**　针对 Codex 在 M1 冻结前的停机报告，离线审计 TP4 首批 `off_cpu` 与 4-rank 逐层宿主耗时、核验宿主机 CPU/NUMA/NPU 物理拓扑，并同步更新窗口预算与 M1 收口路线。

**改动**　经用户批准，更新 `outline.md`、`plan.md` 与 `status.md`：将 M1/M2/M3 预算统一放宽至各 `60,000 卡秒`（全窗 `180,000 卡秒`、宿主后处理 `240,000 秒`、存储 `512 GiB`），单点止损线调整为 3 轮不同根因修复或 `12,000 卡秒`，解除 TP8 短超时误杀（`exit 137`）触发的旧熔断，硬性要求多卡容器超时 `>= 900s`，并在 `plan.md` 中写入 Ascend 按物理 NPU 动态绑定本地 NUMA 核与 `MPOL_BIND` 本地内存的 M1 两步收口计划。

**验证**　宿主机拓扑实测确认本机为 192 核 8-NUMA Kunpeng-920（无 `cpufreq` 动态降频，NUMA 本地距离 `10` vs 跨片 `24–32`），物理映射为 `NPU0/1 → NUMA 6 (CPU 144-167)`、`NPU2/3 → NUMA 4 (CPU 96-119)`、`NPU4/5 → NUMA 0 (CPU 0-23)`、`NPU6/7 → NUMA 2 (CPU 48-71)`。直接拆解 TP4 首批 Prefill（`clean-host/host-timing` scope 70）证明：Rank 0 的 `93.63 ms` `off_cpu` 中 `85.46 ms (91.3%)` 来自 `aclrtSynchronizeStream` 自愿等待、`6.84 ms (7.3%)` 来自 `drvMemcpy`、OS 抢占仅 `1.20 ms (1.3%)`；同批次 4 rank 非同步宿主计算分别为 Rank 0 `618.43 ms`（`17.0 ms/层`）、Rank 2 `680.53 ms`（`+10.0%`）、Rank 1 `701.41 ms`（`+13.4%`）、Rank 3 `702.73 ms`（`+13.6%`），证实未绑核导致跨 NUMA 慢卡拖累每层 AllReduce 同步并制造跨运行 `74.44 ms` 漂移。本轮离线取证零上卡（累计仍为 `17,150.94 卡秒`）。

**结论与限制**　代码实现与真机复验交由 Codex 执行；待 Codex 完成 Ascend 每卡 NUMA 绑核、TP4 复验、全四配置同口径干净宿主刷新及 M1 四例正式留出后，须在 M1 强制停机并保持零提交交回 AGY 与用户审查。

## 2026-11-27 — M1 Step A：Ascend 本地核段与内存绑定

**范围**　Ascend 容器部署、原生 serving worker 及局部探针入口。

**改动**　从驱动 PCI sysfs 读取物理卡和 NUMA 节点，切分每节点的在线 CPU，部署层限制容器 CPU 和内存节点并传入相同的每卡放置数据。原生 worker 在调度器初始化前绑定专属核段与 `MPOL_BIND`，回读内核状态并记录。

**验证**　双镜像重建耗时 8.70 秒，实装 130 个 Python 文件与工作区 SHA-256 一致。无设备容器验证单进程专属核段和内存策略。全仓 Ruff、AST、规模和四维隔离检查通过，复用的 Release 二进制哈希一致。 [双镜像](../../../data/multi-config-prediction/dev/m1/bootstrap/numa-placement/images.txt) · [静态质量](../../../data/multi-config-prediction/dev/m1/bootstrap/numa-placement/static.json)

**结论与限制**　详细采集、纯宿主计时和关闭 `LD_PRELOAD` 的纯时延路径在 TP4 复验中核验。局部探针已接通同一绑定入口，本批无单独上卡。

## 2026-11-27 — M1 Step B：TP4 定向复验未通过放行门槛

**范围**　Qwen3-8B TP4 的原八请求错峰流。详细、宿主和轻量三次录制使用相同预热输入，每次超时 900 秒。

**改动**　复用已核验的模板和设备、通信、拷贝样本，四个 rank 的宿主成本全部来自本批配对录制。纯时延参考与两份模板来源及 24 份样本来源隔离。

**验证**　四个 rank 各配对 11/11 个批次，自主预测完成 8 个请求并查询 206,802 项成本。首批 Prefill 的 Attention 墙钟扣除流同步后，rank 差距由 13.63% 降至 5.01%。Rank 0 的 off-CPU 时间由 93.63 降至 7.93 毫秒，流同步挂起由 85.46 降至 0.10 毫秒，rank 3 仍有 24.05 毫秒流同步挂起。宿主与轻量首 token 差为 65.53 毫秒，八请求 TTFT／TPOT MAPE 为 5.11%／6.02%。本批三次录制新增 1,028.28 卡秒，初始化、预热、观测各为 75.58、39.78、31.05 墙秒，已计时离线处理 375.54 宿主秒。原始录制约 487 MiB，SDK 导出约 322 MiB，累计存储采样峰值 23.70 GiB。 [首批报告](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp4/numa-placement/first-prefill-report.json) · [开发评分](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp4/numa-placement/dev-autonomous/result.json) · [账本](../../../data/multi-config-prediction/dev/m1/resource-accounting.json)

**结论与限制**　rank 差距与 TPOT 均未达到放行线。历史宿主录制预热 23 请求，本轮两个对照模式均预热 8 请求，每个条件各一对录制，仅能比较本次首 token 差。M1 停在 Step B，正式留出为 0/4，交回 AGY 与用户审查。工作区零提交、零推送。

## 2026-11-27 — AGY Step B 离线定案：确认硬件物理带宽差并定位分段 `auxiliary` 漏扣

**范围**　针对 Codex 在 M1 Step B 的停机汇报，离线核验 TP4 绑核后 4-rank 差距（`5.01%`）与 TPOT（`6.02%`）残余系统性正偏。

**改动**　经用户批准，更新 `plan.md` 与 `status.md`：正式确认 Step A 验收通过并调整 Step B 判据（以同 NUMA 内 rank 差距与 TTFT/TPOT `< 6%` 为准），追加 Step B2（合并 `communications.py` / `host_timing.py` 外层连续 `auxiliary` 区间并简化通信热路径），批准 Codex 跳过额外 `batch-timing` 诊断、TP4 复验达标后不停顿直推 Step C 收口 M1。

**验证**　宿主机独立子进程 `4 MiB × 40` 本地 `sched_setaffinity + MPOL_BIND` 写入微基准测得：`NUMA 0` `146.3 ms`、`NUMA 2` `143.5 ms`、`NUMA 4` `143.0 ms`、`NUMA 6` `155.6 ms`（物理慢 `+8.8%`），与 TP4 Prefill `attention` 4 MiB mask 区间（R0/R1 `17.5 ms` vs R2/R3 `16.3 ms`；同 NUMA 差距仅 `0.03% / 0.79%`）完全吻合；对比 TP1 S05 R6（TPOT 正偏 `+2.2%–3.9%`）与 TP2/TP4 纯 Decode 正偏（`+7.2–8.4 ms` / `+7.5%–9.6%`），微秒级拆解 TP4 `B02 Layer 5` 实锤单次 `GroupCoordinator.all_reduce` 的 5 段零散 `auxiliary`/scope 间存在 `8.1 + 9.4 + 10.9 + ~10 ≈ 38 us` 未扣缝隙（×74 次 ≈ `2.81 ms`，加 `layer`/`attention` 外层包装 ≈ `0.7 ms`，合计 `~3.5 ms/批`）。本轮离线取证零上卡（累计仍为 `18,179.22 卡秒`）。

**结论与限制**　交由 Codex 执行 Step B2 纯 Python 修复、重建双镜像并仅重录 TP4 `host-timing` 验证双 `< 6%`，通过后直推 Step C 完成 M1 四例正式留出并强停待审；全流程继续保持零提交。

## 2026-11-27 — M1 Step B：离线拆解绑核后的残余差距

**范围**　复用四 rank 的八请求宿主录制、纯时延参考、已封存预测和未绑核基线。零新增卡时。

**改动**　按宿主样本的记账口径，从线程实际在 CPU 上运行的区间扣除已记录的同步等待、同步拷贝和探针区间。按请求及 token 序号核对发布组和间隔，保留每次录制原始口径。

**验证**　首批剩余 CPU 服务差距为 5.19%。五个 Prefill 批次中，位于 NUMA 6 的两 rank 平均 CPU 服务比 NUMA 4 长 3.40%–6.11%，六个 Decode 批次的差值为 −2.65% 至 +0.37%。预测与宿主录制 11/11 组发布同序同成员，纯时延参考的 10 个可见组与之相符。五个可直接对齐的 Decode 发布间隔中，宿主较纯时延长 27.50–31.02 毫秒，预测较纯时延长 4.09–7.95 毫秒。新增离线处理 21.81 宿主秒。 [逐批归因数据](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp4/numa-placement/residual-attribution.json)

**结论与限制**　Prefill 残余已定位到线程在 CPU 上的服务区间。纯时延录制只能核对首末 token 和由此可见的发布组，现有数据尚不能分辨观察开销、宿主运行条件和成本模型各自的贡献。Step B 放行条件及正式留出范围照旧。

## 2026-11-27 — M1 Step B：逐批启动预检与留出草案超时校正

**范围**　轻量逐批采集的启动参数和四例尚未冻结的留出草案。

**改动**　草案沿用的详细／轻量参考超时为 360／330 秒。两类参考超时统一更新为 TP8 1,500 秒、其余配置 900 秒；逐例两次轻量参考及请求抽样定义沿用原草案。

**验证**　用当前镜像解析 TP4 `batch-timing` 原生启动参数，确认关闭 `LD_PRELOAD`，设备和 CPU／NUMA 放置与上一轮纯时延录制一致。实际超时为 900 秒的启动配置，需要预留 3,600 卡秒，预检耗时 0.67 宿主秒。 [逐批启动预检](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp4/numa-placement/batch-timing-preflight.json) · [留出草案](../../../data/multi-config-prediction/holdout/m1/freeze/m1/specification.json)

**结论与限制**　草案仍未冻结，零新增卡时。轻量逐批入口记录选批、执行与结果处理的线程时钟，可用于下一次 Step B 开发归因。放行仍依既定门槛。

## 2026-11-28 — M1 Step B2：观察器连续区间复验通过

**范围**　仅重录绑定在原四张卡的八请求 TP4 宿主计时。复用上一轮详细图、设备／通信／拷贝训练资产和纯时延参考。

**改动**　宿主观察器将外层准备及收尾分别登记为紧贴原生调用边界的连续区间，常见单位置参数的 all-reduce 走直接绑定。跨镜像提样仅对明确核对的此组镜像启用。共用底座、运行配置、原生探针与入口哈希一致，实装文件差异限于两个观察器文件。双镜像经官方脚本重建，沿用原 C++ Release 二进制。

**验证**　四 rank 各配对 11/11 批，已核验 6,600 个目标调用的连续区间，其中 3,212 次 all-reduce。自主复算训练隔离，TTFT／TPOT MAPE 为 4.68%／4.72%。新增 332.38 卡秒，容器配置超时 900 秒。初始化、预热、观测为 24.25、13.59、8.17 秒，离线提样、拟合、发布、复算和区间检查合计 139.17 宿主秒。 [镜像核验](../../../data/multi-config-prediction/dev/m1/bootstrap/continuous-auxiliary/pair-compatibility.json) · [自主复算](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp4/continuous-auxiliary/dev-autonomous/result.json) · [静态检查](../../../data/multi-config-prediction/dev/m1/bootstrap/continuous-auxiliary/static.json)

**结论与限制**　TP4 通过开发放行线。已启动 Step C 的 TP8 配对刷新，正式留出资产仍待冻结。

## 2026-11-28 — M1 Step C：刷新四套成本库并冻结正式留出

**范围**　同一绑核和观察器版本下，完成 Qwen3-8B TP2、TP4、TP8 与 Mistral-7B-v0.3 TP2 的开发配对录制。继续使用已核验的模板与非宿主数据。

**改动**　扩充开发流的短 Prefill 和合批测点，按 rank 重新提取宿主服务时间。Qwen TP2 复用既有退役开发输入，经正式预测入口补齐 KV 释放相关的宿主、设备和拷贝缺口。冻结 344 个代码及构建文件、双镜像、二进制、配方和四套初始库，并一次性抽取四例正式输入。

**验证**　Qwen 三种 TP 的每个 rank 配对 43/43 个批次，Mistral 的两个 rank 各配对 54/54 个批次。Mistral 与完成按需补采的 Qwen TP2 跑通各自的退役开发流。TP4 与 TP8 的退役开发预测在 KV 释放相关成本处停住。Qwen 三例正式输入的请求序列一致，每例预登记两次轻量参考。四套配对录制及 Qwen TP2 的开发补采新增 4,913.91 卡秒，冻结前累计 23,425.51 卡秒。 [开发补采报告](../../../data/multi-config-prediction/dev/m1/cases/qwen8-tp2/refreshed-host/retired-dev-explore/result/result.json) · [首批冻结清单](../../../data/multi-config-prediction/dev/m1/retired-holdout-beb164e4/freeze/m1/source-manifest.json) · [账本](../../../data/multi-config-prediction/dev/m1/resource-accounting.json)

**结论与限制**　冻结和抽样时未创建正式参考目录。已启动四例盲测，独立结构、固定图及自主时延验收均待执行。

## 2026-11-28 — M1 Step C：记录第一批盲测失败并调整补采额度

**范围**　四例盲测预测与局部补采。两批均使用 Step B2 的镜像及编译二进制。

**改动**　归档第一批完整目录，保留已核验的局部样本作为开发资产。每例补采次数由 16 提高至 48，M1 卡时上限照旧。重新生成冻结清单，由其哈希抽取四例未见输入。

**验证**　第一批 TP4 与 Qwen TP2 封存成功。TP8 的 16 次探针均正常结束，仍缺 rank 2 的 32 字节拷贝。Mistral TP2 的宿主探针在 Gloo 初始化报连接重置，进程退出码 1，配置超时 1,500 秒。全批实际 8,622.64 卡秒，累计 32,048.15 卡秒。第一批无正式参考目录。第二批四例输入均与两轮退役请求不同，三种 Qwen TP 共享新请求序列。 [第一批结果](../../../data/multi-config-prediction/dev/m1/retired-holdout-beb164e4/results/m1/blind.json) · [第二批抽样](../../../data/multi-config-prediction/holdout/m1/results/m1/draw-integrity.json) · [账本](../../../data/multi-config-prediction/dev/m1/resource-accounting.json)

**结论与限制**　第一批没有形成 M1 验收结果。第二批盲测已启动，待四份预测封存后再录制每例完整详细参考与事先登记的两份轻量参考。严格覆盖、训练隔离与 <6% 门槛继续生效。

## 2026-11-28 — M1 正式验收及强制停机

**范围**　Qwen3-8B TP2、TP4、TP8 共用八请求抽样，另验 Mistral-7B-v0.3 TP2 八请求。每例使用一份详细和两份独立轻量参考。

**改动**　四份盲测预测先封存并复核 SHA-256，再启动参考采集。复用 Step B2 镜像和二进制，TP8 物理参考超时 1,500 秒，其余为 900 秒。归档正式总表、资源账本、源码统计与证据清单。

**验证**　16 个初始模板来源合格。四例结构全匹配，分别为 1,102、2,140、4,216、952 个动作。固定图复现 1,297,854 个实测边界，843,968 项工作均有实测和预测成本。三路训练来源全隔离。自主 TTFT／TPOT MAPE 依次为 **3.32%／3.33%、4.61%／3.43%、0.71%／1.04%、0.82%／2.86%**，Qwen 三种 TP 的吞吐、P95 TTFT、平均 TPOT 排序均一致。第二批按需补采 21 次，共 4,613.76 卡秒，十二份参考共 4,047.09 卡秒。M1 累计 40,709.00 卡秒，宿主计时 26,056.44 秒，存储采样峰值 45.39 GiB。双镜像实装哈希、Ruff、AST、代码规模和四维隔离复核通过。 [验收总表](../../../data/multi-config-prediction/holdout/m1/acceptance-summary.json) · [账本](../../../data/multi-config-prediction/dev/m1/resource-accounting.json) · [证据清单](../../../data/multi-config-prediction/holdout/m1/review/evidence.sha256)

**结论与限制**　M1 验收 4/4，停机待审。固定图使用模型耗时时，Qwen TP2、TP4 和 Mistral TP2 的 TPOT MAPE 分别为 6.17%、7.18% 和 8.28%。Mistral 的补采卡时为该例三份独立参考总卡时的 6.23 倍。TP2 单卡装配匹配 108/108 个动作，复用已有通信探针时边际省 46.52% 卡时，首次冷启动多耗 60.30%。 [单卡装配](../../../data/multi-config-prediction/dev/m1/single-device-assembly-report.json)

## 2026-11-28 — AGY 独立审查、代码精修与 M1 分层落袋

**范围**　独立复核 M1 四例正式留出证据、盲测防篡改与全仓 diff，按用户要求逐文件精修逻辑紧凑性后分层提交并推送分支 `multi-config-prediction-validation`。

**改动**　① 还原 `src/python/prologue/integrations/frameworks/sglang/acquisition/host_samples.py` 中 0 调用方的一次性 `compatible_images` 旁路参数，恢复严格同镜像约束；② 精修 `communications.py` 与 `host_timing.py`：用 `scope(..., entered=None)` 替代 `own_auxiliary` 布尔开关并保持 `finally` 异常安全，消除 `observe_exchange` 返回 `(result, scope)` 的多态破坏与 `@contextmanager def outer()` 每次通信的生成器分配，对被跳过的纯 CPU 通信亦记录观察器排除区间，去除 `all_reduce` 快捷路径的冗余条件并将 `observe_expert_exchange` 对齐 `timed_call`；③ 其余 20+ 文件逻辑与紧凑性审查通过，按职责分层提交并推送。

**验证**　独立脚本复核 M1 留出 4/4 盲测 SHA-256 与落盘时序（`盲测 < 参考`）全部为 `True`，三路 `training == "separate"` 100% 隔离，Qwen TP2/TP4/TP8 吞吐（`6.01 > 5.97 > 5.67` vs `5.81 > 5.76 > 5.65`）、P95 TTFT 与平均 TPOT 排序全部一致；精修后全仓 220 个源码模块 + 6 个示例脚本复检 `>550 NE = 0`、`>25L 代码墙 = 0`、AST 0 环 0 缺失、四维隔离 0 命中、`ruff` 0 错、0 `.pyc`，并执行观察器协议路径离线回归检查。本轮纯审查与精修零上卡（M1 累计终账 `40,709.00 卡秒`）。

**结论与限制**　M1 正式收口落袋；M1 留出证据绑定于精修前 Step B2 镜像快照，M2 启动前须先基于最新提交重建双镜像以生效上述精修。
