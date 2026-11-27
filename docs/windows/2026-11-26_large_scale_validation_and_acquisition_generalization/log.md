# 多拓扑、复杂调度与真实流量大范围验证及局部补采泛化提速日志

## 2026-11-26 — 窗口开启：制定五层大范围验证与局部补采泛化提速纲领及精细计划

**范围**　在已终审关闭的现代架构基线 `8d10807` 之上开启单一综合大窗口（建议分支 `large-scale-validation`），将路线二（局部按需补采跨机制/跨拓扑泛化与冷启提速）与路线一（8 域多拓扑、复杂调度、真实流量开发回归、固定图误差归因与冻结独立留出 < 6% MAPE 验收）整合进五层递进闭环。

**改动**　AGY 基于本机实测 8 卡 Ascend、3 款已验证模型路径（`Qwen3-8B`、`Mistral-7B-v0.3`、`Qwen3-30B-A3B`）、历史 PD 825 端口差异证据、CANN 1ns 舍入重叠诊断及现有 `recipes/`/`probes/`/`cost_families.json` 边界，起草只读战略纲领 `outline.md` 与精细分层的 5 层 14 阶段执行计划 `plan.md`（L0 清债 → L1 复杂调度与合批提速 → L2 多卡 TP 与跨模型 → L3 PD/PP/MoE 异构并行 → L4 Azure 真实流量 32→256 压测、误差驱动模型族演进与 8 域总冻结留出矩阵终验），明确开发/留出绝对物理隔离、插件化与代码品质零回退、72,000 卡秒宽松有界预算、每阶段 Bigger-Picture Overview 动态更新机制及全流程零 `git commit` 纪律。

**验证**　核验本机 8 张 NPU（`/dev/davinci0..7`）与 `example/coarse_graph/{prepare_matrix,matrix,azure_workload}.py` 就绪，复核留存 `pd-structure` 825 处端口差异集中在 11 次 `prefill/0 run_batch`；核验新建窗口文档 0 跨层引用与 0 断链。

**结论与限制**　窗口文档就绪，待用户确认分支起点（直接从 `8d10807` 切出或先合并 `main`）后交由 Codex 全窗口端到端自主实施，完成后保留干净工作区交回 AGY 终审。

## 2026-11-26 — S01 收尾　原始证据与环境核验

**范围**　核查窗口起点、三款模型、八卡、镜像和 Azure 输入。读取 PD、长窗 CANN、TP8 和 R09 的留存开发证据。

**改动**　归档资源与根因报告，下载并按官方发布的 SHA-256 核对 Azure 会话 CSV。全仓复盘将 S02 拆成模板再生产、设备端点、来源资格和回归四批。

**验证**　825 处端口差异均为分配归属。11 次批处理的 Prefill/Decode 分布为 1/5/5。同份原始训练图经当前编译器在三个 rank 分别恢复 300、228、228 个声明分配的帧。六行 CANN 记录的整数端点证明三个相邻任务对实际接触。Azure 转换器实读 32 条原始记录。新增真机用量为零。[诊断与原始证据定位](../../../data/large-scale-validation/dev/l0/diagnosis.json)

**结论与限制**　Qwen3-8B 有权重文件，Mistral 与 Qwen3 MoE 本机只有配置，现有适配明确按 dummy 权重启动。R09 两处缺口分别为两成员分配切口未覆盖和页边界 guard 不匹配。S02 以重新核验后的资产和原始任务端点判定关闭。

## 2026-11-26 — S02-A：恢复 PD 模板的分配归属

**范围**　复用原先独立的两份训练录制与同一份留存真实目标，核对全部 5,355 个动作。

**改动**　经现行投影和模板编译重新发布 Dev 双库，沿用原成本样本。全仓复盘将该问题归于模板生产与发布，执行、控制和独立对照继续使用现行规则。

**验证**　独立来源检查通过三个训练 rank 的模板核验。目标报告为 `full_structure_matched`，动作内容及依赖均为 5,355/5,355，端口差异 0，训练来源与目标分离。新增卡时 0，五项离线操作合计约 355.5 秒。[来源核验](../../../data/large-scale-validation/dev/l0/pd/source-evidence/source-evidence.json) [动作对照](../../../data/large-scale-validation/dev/l0/pd/structure/comparison/actions.json)

**结论与限制**　本例的历史端口差异已清零。这批对照使用原目标图的计时事实。S02-D 将用同一真实目标重新投影，核对 S02-B 接入对结构回归的影响。

## 2026-11-26 — S02-B：接入 CANN 原生任务端点

**范围**　沿 SDK 导出、来源解释、独立参考和局部设备采样核对设备区间的所有权。

**改动**　从 CANN `TASK` 表导出整数起止时间。生产端和独立参考端各自按设备流、任务、子任务和起点读取区间，保留原有时钟校准。首次读大表遇到同一记录被重复导出，已改为接受完全相同的时间值并拒绝互相矛盾的端点。[重复行原始证据](../../../data/large-scale-validation/dev/l0/cann/native-repeated-task-evidence.json)

**验证**　重新投影一份留存 Qwen3-8B TP1 真机录制，`verify source` 判定联合来源一致，检查 3,666 个设备测量点。用新路径读取历史整数表的 28,934,746 份唯一端点，确认已知三组相邻任务的重叠都由 1 ns 降到 0。两条正式 CLI 共约 24.2 秒，旧表全量读取和复算约 249.9 秒，新增卡时 0。[来源核验](../../../data/large-scale-validation/dev/l0/cann/real-tp1-qualification/source-evidence.json) [历史端点复算](../../../data/large-scale-validation/dev/l0/cann/historical-rounding-proof.json) [离线用时](../../../data/large-scale-validation/dev/l0/cann/post-monotonic.tsv)

**结论与限制**　原生整数端点已通过当前 TP1 生产和核验链，历史长窗的三处低层时间差已复算。TP1 录制未命中重叠，历史长窗使用旧录制格式。仍需完成命中重叠的正式来源回归。

## 2026-11-26 — S02-B：固化镜像并对照现行长提示录制

**范围**　核对现行 SDK 端点在真机服务、来源核验及成本提取链的交接，寻找同时命中重叠和现行来源合同的材料。

**改动**　重建采集和控制双镜像。成本补采复用由实际生产镜像标识，不同生产镜像的来源重新派生证据。定向运行沿用历史长窗前 12 个请求的提示长度与到达时刻，生成上限为 4 token。

**验证**　归档检索覆盖 1,357 份可定位原始录制的派生 trace，未发现匹配候选。单卡正式来源核验通过 13,668 个设备测量点。精确端点对照核查 8,795 个原生任务，457 个旧渲染结束时间晚 1 ns，638 个早 1 ns。本例没有物理相邻端点接触。真机一次净计 131.24 卡秒，其中初始化 26.38、预热 53.03、有效观测 11.70、其余运行与收尾 40.14 秒。原始记录约 68.97 MB，现场保留峰值约 69.12 MB，SDK 导出 42.83 MB，后处理与复算约 52.72 秒。[正式来源](../../../data/large-scale-validation/dev/l0/cann/long-prefill-dev/source-evidence/source-evidence.json) [端点对照](../../../data/large-scale-validation/dev/l0/cann/long-prefill-dev/endpoints.json) [搜索一](../../../data/large-scale-validation/dev/l0/cann/trace-source-search.json) [搜索二](../../../data/large-scale-validation/dev/l0/cann/trace-source-search-tail.json) [镜像](../../../data/large-scale-validation/dev/l0/cann/current-image-identities.txt)

**结论与限制**　现行流水线已复现原始结束时刻的 1 ns 舍入偏差并通过联合来源核验。历史三组重叠有真实端点复算，旧录制缺少现行 rank 记录和独立 RAW 校准事实。命中实际重叠的同案正式回归仍待证据。

## 2026-11-26 — S02-D：提前完成独立的 TP1 和 PD 对照

**范围**　用现行正式双镜像完成固定 TP1 行为回归，并对原 PD 目标重新派生精确端点图。

**改动**　沿用同一只读 TP1 基线双库。PD 对照固定独立目标与 S02-A 重建的训练库。

**验证**　TP1 四目标的 211,290 条执行轨迹与四份请求时延文件逐字节一致，发布 16 token。678 样本拟合、两份成本点输出和 43,276 项固定图成本一致。PD 重投影后 `full_structure_matched`，动作 5,355/5,355，端口差异为 0，来源隔离为 `separate`。[TP1 回归](../../../data/large-scale-validation/dev/l0/baseline/comparison.json) [PD 回归](../../../data/large-scale-validation/dev/l0/baseline/pd-current-clock/structure/result.json)

**结论与限制**　两条对照均保持其原验证口径。L0 汇总仍需纳入 TP8 条件提取与 CANN 同案结论。

## 2026-11-26 — S02-C：恢复 TP8 联合来源资格

**范围**　复用八卡同跑的原始录制，核对来源结构及实测条件。

**改动**　以现行 guard 与 CANN 原生端点重新投影八个 rank。全仓复盘确认该缺项属于过期派生资产，来源解释、图生成和独立核验的职责保持分离。

**验证**　八个 rank 的映射均为 `consistent`，时钟均为 `compatible`，联合来源通过，核查 8,090 个动作和 293,214 个设备测量点。实测条件提取为 `extracted`，八个设备时钟共同满足 1,700,774 条因果约束。受检工作结束点缺测为 0，保留 6,232 个带校准区间的边界。三项后处理共约 970.32 秒，新增卡时 0。[来源核验](../../../data/large-scale-validation/dev/l0/tp8/source-evidence/source-evidence.json) [实测条件](../../../data/large-scale-validation/dev/l0/tp8/timing/execution-inputs-report.json)

**结论与限制**　留存完整拓扑录制已具备独立参考资格。单卡分片的候选覆盖和 TP8 自主精度仍由 L2 验证。

## 2026-11-26 — S02-D：闭合四个设备 ABI 声明的所有权

**范围**　沿采集镜像、平台声明、SGLang ABI、控制器启动参数和离线回归检查装配。

**改动**　将四个设备专属符号从 SGLang 版本表移入平台 ABI 声明。复用 `PlatformDomain` 的既有传递机制，框架在装配时核对四项必需声明。重建采集与控制镜像，保留旧镜像核验的产物和身份。

**验证**　采集镜像里的四个符号迁移前后逐值相同，实装文件哈希与工作区一致。CPU-only 控制镜像的四目标输出仍逐字节一致。静态检查覆盖 217 个 Python 模块，导入环、缺失依赖、框架平台越层、超长函数代码块和字节码残留均为 0，`ruff` 通过。[镜像](../../../data/large-scale-validation/dev/l0/static/abi-image-identities.txt) [控制对照](../../../data/large-scale-validation/dev/l0/static/abi-control-comparison.json) [静态核对](../../../data/large-scale-validation/dev/l0/static/quality.json)

**结论与限制**　符号装配和控制回归通过。最终镜像已用于本层三卡 PD 定向录制。该录制的来源证据与 CANN 同案结论仍待产出。

## 2026-11-26 — S02-B：保留三卡 PD 超时证据并熔断

**范围**　用最终采集镜像进行一次三卡原拓扑复现，核对 CANN 的同案来源门槛。

**改动**　原样保留历史首观察批 12 个请求的到达与生成长度。执行采用 410 秒时限和 420 秒采集预算，保留失败后的原始录制目录与运行日志，核验清单绑定当前源码与各次所用镜像。

**验证**　两侧都完成预热并开始正式录制，随后超时。Prefill 已写出 `rank-0.json`，三个 profiler 目录仍处于暂存态，联合录制未发布，投影和 `verify source` 未启动。墙钟计 411.41 秒，含收尾，共 1,234.24 卡秒。失败任务没有按阶段结算的初始化、预热和有效观测收据，阶段分项保留为空。原始保留 2.278 GB，现场采样峰值约 2.279 GB，SDK 导出量与本次后处理均为 0。两次累计 1,365.48 卡秒，原采集容器已退出。[执行账本](../../../data/large-scale-validation/dev/l0/cann/full-pd-repro/native/execution.json) [运行日志](../../../data/large-scale-validation/dev/l0/cann/full-pd-repro/native/attempts/1/capture.log) [证据清单](../../../data/large-scale-validation/dev/l0/evidence-manifest.json)

**结论与限制**　历史整数端点复算和现行 TP1 正式来源核验各自成立。本次失败界限是规定时限内未封存完整 PD 录制。同一份现行录制同时命中重叠并通过独立来源核验的门槛未达。按熔断规则停在 L0，交 AGY 裁决。

## 2026-11-26 — AGY 裁决：L0 放行，升级防挂死预算与高阈值熔断策略

**范围**　复核 L0（S01–S02）全部产出及 S02-B 熔断原因，结合用户关于“不要轻易熔断、预算仅作防挂死保护网以保证全窗口连贯推进”的明确指示，修订 `outline.md` 与 `status.md` 并放行进入 L1。

**改动**　① 裁决 **L0（S01–S02）验收通过**：CANN 原生 `TASK` 整数端点修复已由历史 2,893 万行原表 3 对 `1ns → 0ns` 复算与现行单卡 8,795 任务精确端点（捕获 457 处 `+1ns` / 638 处 `-1ns` 旧浮点渲染偏差）双向证完，不再强求在 L0 专门为概率性物理 0ns 紧贴任务对反复跑 3 卡超时复现（留待 L3/L4 大流量自然顺带观察）；② 修订 `outline.md` 边界 4 与熔断规则：取消紧绷分层小预算，将防挂死总上限放宽至 **180,000 卡秒 / 36,000s 后处理 / 512 GiB**（仍如实记账），明确单次多卡/长流量运行须给足合理超时（如 900s–1,800s），非阻断主线的偶发边界或单案例原生限制留证单列后继续全速推进，仅遇核心主线 3 轮无进展、环境级故障或触及防挂死上限时才停机。

**验证**　复核 [historical-rounding-proof.json](../../../data/large-scale-validation/dev/l0/cann/historical-rounding-proof.json)、[endpoints.json](../../../data/large-scale-validation/dev/l0/cann/long-prefill-dev/endpoints.json) 与超时日志 `capture.log`（确认系 410s 保守超时在收尾前触发，非代码故障）；确认 PD 825 差异已清零、TP8 8,090 动作联合来源与实测条件合格、TP1 基线 211,290 轨迹一致、217 模块静态门禁全绿；核验修订后文档 0 跨层引用与 0 断链。

**结论与限制**　L0 放行，全窗口累计已用 1,365.48 / 180,000 卡秒。Codex 立即从 **L1（S03 同会话合批提速）** 连贯自主推进至 L4（S14）收官，非真死阻断不停机。

## 2026-11-26 — S03：定位冷启动开销与短采样导出失败

**范围**　Qwen3-8B TP1，复测 R08 单次服务中的两个 Decode 长度。历史局部采集已把两个长度放入同一 `ModelRunner`，预热用了 45.36 秒。

**改动**　批次、页分配与 Decode 元数据由原生接口生成。直接向真实 KV 池写入已分配的前缀，并在原生 Decode attention 中测量目标计算。SDK 导出显式请求文本和数据库，从合并库的 `TASK` 表保存整数端点。

**验证**　早期两次完整服务和一次局部尝试均完成录制，在读取 SDK 中间库时失败，合计 327.36 卡秒。用留存文件确认合并库与 CANN 独立库的九条 `TASK` 坐标逐项一致。修正后离线重导出两个局部切点和两个完整服务切点，设备任务身份与顺序逐项一致。证据见 `data/large-scale-validation/dev/l1/s03/recovered-sdk/final/device-sequences.json`。

**结论与限制**　SDK 取表根因已查明，失败用量保留。本批次设备任务对照来自录制后恢复的证据。按照同一镜像重新运行公开入口，以完整发布、真实 KV 回读和冷启账本作为 S03 判据。

## 2026-11-26 — S03：冷启动多点同会话通过

**范围**　沿用 Qwen3-8B TP1、同一物理卡和同一镜像对照。保留原 8/16 两个上下文长度，再验证八个几何，包括跨页的两个并发批次。两侧均只启动一次原生会话。

**改动**　空前缀从原生页池直接分配，按原生表和 KV 读回核对提交；正式 Decode 仍使用原生 `ModelRunner`。所有几何先各自预热和无 profiler 测量，再在共同 profiler 中按独立区间测量。沿用同一补采规则和发布流程。

**验证**　两点为 68.18 / 127.94 卡秒（53.3%），八点为 71.45 / 209.61 卡秒（34.1%）。分别通过 8、32 项查询，实际操作数、四段设备任务的身份与次序均与完整服务相同，所有实际 KV 读回通过。八点局部初始化 20.97 秒、预热 24.11 秒、SDK 后处理 7.38 秒。另对留存长录制重投影，9,313 行整数端点一致，粗图逐字节一致，局部与联合来源资格保留。证据在 `data/large-scale-validation/dev/l1/speedup-benchmark.json`。

**结论与限制**　达到八几何合批的 50% 门槛，两点仍高于 50%。本批全部实采累计 1,015.26 卡秒，含失败 327.36 卡秒。局部存储峰值按工作区采样，完整服务保留量可复算，历史及本批服务瞬时峰值未测。S04 接续调度切点、宿主工作及小池验证。

## 2026-11-26 — S04：续块、混批、前缀命中与驱逐的局部采集

**范围**　Qwen3-8B TP1。完整服务分别录制分块、混批及 192-token 小池的前缀缓存压力，取可定位的真实批次作为开发对照。

**改动**　局部 worker 复用原生调度器的选择、分配和结果处理。准备阶段按请求已提交 tokens 推进，实际选出的成员及 token 区间由采集端回读。缓存查找用原生请求准备所需前缀树，核对查询长度、匹配长度和拆分两侧；驱逐用池容量和请求的分配需求准备驻留页，核对原生释放。结构与成本缺项保留原生 continuation，模板选择与规则分发保持原合同。

**验证**　中间续块、收尾续块、首个混批全部通过 `library acquire` 并发布到开发库。缓存未命中、48-token 命中且拆分 16＋16、小池触发真实页释放皆通过局部资格。中间续块的四项宿主成本在独立计时进程中全通过。C++ Release 构建无告警与错误，Python 静态检查通过。原始证据在 `data/large-scale-validation/dev/l1/s04/`。

**结论与限制**　已验证这些局部操作能够进入双库。组合机制下的更多队列历史、`block_table_strided` 与多成员分配端口仍在进行开发验证。S05 的冻结留出与 6% MAPE 验收待启动。

## 2026-11-26 — S04：页边界、三成员混批与成本发布完成

**范围**　Qwen3-8B TP1。沿同一录制和发布流程核对分块、混批、前缀缓存和页分配，补齐严格成员对照、设备计时与同会话多切点证据。

**改动**　模型采集按真实输入 token 准备前缀 KV。混批回读区分续块、刚到的 Prefill 和现有 Decode，页分配用相同原生状态回答一组切点。独立宿主计时与设备 profiling 分别沿用已注册的规则。

**验证**　`block_table_strided=0/1` 均发布；单成员和双成员分配各在一次原生启动中通过两项查询。缓存命中模型切点和三成员混批分别获得 70、58 条设备样本记录，各验证一项原生设备工作进入持久成本库。四项续块宿主查询通过独立计时并发布。四份完整服务参考均通过原始来源核验。18 次真机采集共 1,848.14 卡秒，图生产计量 185.58 秒，工作区保留约 0.87 GiB。初始化、准备与回读、采集导出及采样峰值逐次记账。完整服务的瞬时峰值缺测。最近 C++ Release 构建、Python 静态检查及 `git diff --check` 通过。证据在 `data/large-scale-validation/dev/l1/s04/coverage.json`。

**结论与限制**　S04 的定向局部能力通过。早期三成员测试按形状成立，精确内容证据采用随后按原生 `WorkloadContent` 重建的 `mixed-exact`。多个活跃队列可形成同形状但进度不同的批次，后续 S05 用五配置端到端开发回归检查可构造性及预测覆盖，冻结精度另行验收。

## 2026-11-26 — S05：从空库定位首个准入缺项

**范围**　三机制全开的 Qwen3-8B TP1，先用一个跨页请求启动空库自动补采；已准备五组配置与 1、4、12 请求的 15 个开发输入。

**改动**　局部配方按原生调用实际需要的输入判断可采性。缓存查询明确要求可构造的 trie 操作数，其余切点沿用实际批次、分支和发布读回。独立矩阵可核对局部会话的发布片段，完整服务仍用独立原始记录认证联合结构和实测时序。

**验证**　首轮自动取得初始 worker 并发布，85.65 卡秒，0.77 秒后处理。随后首个请求准入时钟因原全局缓存开关屏蔽配方，返回 `no_matching_rule`；原生准入边界可由空闲队列重建。按切点判断后从同一开发任务续跑，保留原用量和发布。两份真实局部发布重新编译对齐，未发布的局部片段被拒绝；完整服务页边界参考仍通过联合资格。

**结论与限制**　开发库初始化及局部来源核对可分别复现，首例端到端与独立留出仍未验收。首轮证据在 `data/large-scale-validation/dev/l1/s05/cold-one/`，局部发布核对在同层 `source-qualification/`。

## 2026-11-26 — S04 补证：续块宿主成本的自动回退

**范围**　以四项已通过的续块宿主查询，验证缺少可复用切图时的规则切换。

**改动**　用另一个空闲物理设备运行正式局部采集入口，保留同一工作身份、特征和原生镜像。

**验证**　高优先级规则报告 `missing_local_structure_source`，未启动容器。配对规则随后在一次采集任务中完成结构与独立计时，四项均通过。新增 126.39 卡秒，S04 共 1,974.53 卡秒，L0–S04 累计 4,355.27 卡秒。证据已入 `data/large-scale-validation/dev/l1/s04/coverage.json`。

**结论与限制**　实证确认了规则优先级及没有原结构时的回退能力。读回覆盖本次声明的四项宿主工作。

## 2026-11-26 — S05：五配置原生对照与结果操作数

**范围**　继续空库单例，分别录制五组配置的代表性开发参考。独立检查结构对照、实测固定图基线和局部来源发布。

**改动**　共同入口在启动设备前解析完整前缀树，并在缺项绑定时复用解析结果。结果阶段的嵌套缓存调用以外层实际批次进度还原采集操作数，子调用仍保持自身参数。配对宿主规则可将已经读回的结构切点与独立计时样本一起发布。

**验证**　最初两个多请求输入把子前缀总长填成增量，启动后被原生拒绝，合计 19.91 卡秒。按累计 48 tokens 修正后，15 个开发输入通过入口校验，旧输入在启动前被拒绝。五配置参考均通过联合资格，小池 12 请求中观测到 11 个混批及 8 次驱逐进入实际页释放。首例实测固定图重放核对 27,733 个边界与 4 次 token 发布，均对齐；当时冻结的诊断库仍有成本缺项。早期结构对照核对 34 个动作后遇到首个模型模板缺项。原有配对录制离线重验后同时发布结构及四项宿主成本，获得局部来源资格。九份原生图的 915 个结果片段逐一确认派生批次成员、相位和 token 范围与原记录相同。

**结论与限制**　五份参考供后续开发回归，和发布库的训练来源分开。空库已经越过请求准入、首个模型切点并进入本轮结果路径。嵌套缓存的同会话真机发布仍在核对，多轮混批中更复杂的先前队列历史仍待闭合。证据在 `data/large-scale-validation/dev/l1/s05/diagnosis/`。

## 2026-11-26 — S05：缓存子调用实采与固定图基线核验

**范围**　核对首个案例收尾 Prefill 中的缓存更新，以及所有代表性开发参考的实测固定图基线。

**改动**　缓存查找按请求进度和调用所有者链核对。外部查找可准备已有前缀，更新过程中的查找由原生插入提供状态。结果进入时的批次几何只供采集计划，与缓存叶调用参数分开。

**验证**　固定图的五份实测基线合计 335,560 个边界对齐，五份核验的训练来源均独立。首例结构对照使用新的已发布切点，从 34 个推进到 116 个动作，下一处是 64-token 前缀加 1-token 收尾块在模型内部的分支缺项。真实收尾块的缓存插入和查找由一次局部启动回答并发布，88.81 卡秒。现行调用链条件在同一原始录制上重验通过，两份查询的模板资产摘要一致。Python lint、AST、规模、空行和字节码检查通过。

**结论与限制**　实测固定图可作后续成本归因，当前诊断库仍未覆盖全部成本。冷库自动补采继续运行，冻结独立验证尚未开始。两次结构对照与嵌套缓存读回保留在 `data/large-scale-validation/dev/l1/s05/diagnosis/`。

## 2026-11-26 — S05：并行开发回归与同录制发布

**范围**　保持三机制全开冷库单例自动补采，同时在独立设备上启动分块、混批、前缀缓存三个结构域的九个开发目标。用已录制单例检查矩阵的局部来源与轻量参考环节。

**改动**　局部切图携带同一原始录制上合格的设备成本，按需的宿主配方优先核验同一原生调用。矩阵按适配器选中的物理卡数预约和结算，从任务输入显式绑定验证路由。容器子命令关闭源码挂载上的字节码写入。

**验证**　同一宿主缺项选择同一录制，所需投影从四次降到一次。复用已有收尾块和缓存更新录制分别取得 65、27 条设备成本记录，均与原成本提取路径一致，额外真机次数为零；缓存更新含一项需要就绪操作数的复制，继续由独立路径提供。收尾块的模板和成本通过同次发布，C++ 成本查询命中。矩阵六份局部来源均核验通过，轻量参考实录 109.43 卡秒，按单卡原始账本一致。自主验证使用旧固定库，按已知模板缺项报告，训练来源隔离通过。新子命令在真实局部录制上重验通过，清理本轮初次矩阵试跑生成的字节码后，源码树无残留。

**结论与限制**　三机制全开冷库已越过收尾模型模板缺项，独立结构对照推进到 149 个动作，下一处缺项是首次 Decode 的准备片段。并行回归和轻量采集属于开发集，保持冻结留出未抽样。五个留出请求族已预先声明，待开发闭合后以资产清单的 SHA-256 一次抽取。实证在 `data/large-scale-validation/dev/l1/s05/`。

## 2026-11-26 — S05：单请求结构闭合与共享前缀回归

**范围**　复用五份独立开发参考，分别定位结构的首个缺项和固定图上的成本覆盖。定向采集混批分配与模型工作、缓存前缀分裂和请求收尾。

**改动**　SGLang 局部配方用原生前一批的真实请求与已提交区间判断缓存分支是否可由准备批生成。保留缓存命中、分裂长度和所有队列的原生核验。验收工具中的输入生成与选取参考步骤按职责分组。

**验证**　缓存预置版本让上一批 Prefill 从位置 48 开始，目标队列证据要求从 0 开始，原生采集保留拒绝及 95.77 卡秒。改用已有请求建立分支后，在同一物理卡和镜像实采 95.19 卡秒，查找读到命中 48、分裂后缀 16，片段发布和来源核验通过。混批四请求推进至 166 个动作，前缀缓存十二请求推进至 184 个动作。三机制全开单例独立结构对齐 230 个动作、27,578 个边界和 18,770 项工作，差异为 0，训练来源独立。固定图仅替换 Decode 注意力的宿主成本时，该单例 TPOT 相对详细实测基线下降 15.48%，设备注意力单组替换的请求指标一致。

**结论与限制**　结构对照已通过单请求。混批仍需验证上一批筛选，前缀缓存仍需验证多成员 Prefill。固定图结果给出详细参考上的分组误差，完整自主预测和冻结留出精度仍待验收。失败、补采和重复录制的物理卡时按原始容器身份记入 `data/large-scale-validation/dev/l1/s05/diagnosis/dev-structure-snapshot/native-budget.json`。

## 2026-11-26 — S05：首例三路开发验证

**范围**　复核三机制全开单请求的完整预测和三路证据，启动同库的全开多请求与小池开发用例。

**改动**　沿用完成结构对照的模板字节与同一 C++ 二进制。针对最终发布的成本资产，在详细实测图上替换全部工作，并在独立轻量录制上比较自主预测。

**验证**　固定图 18,770 项工作均有成本，完整替换的 TTFT 和 TPOT 误差为 6.84% 和 21.46%。只恢复实测宿主耗时后，两项降至 0.24% 和 0.50%。对独立轻量参考的自主预测误差为 2.81% 和 3.56%，来源隔离均通过。首例探索的协调器记录 30 次原生尝试，2,625.61 卡秒。用于对照的发布另包含一次并行定向收尾模板。离线分支仅用协调器的原始实采重发同一切点，热库复预测新增 0 次原生尝试，TTFT 与 TPOT 均相同。15 个开发目标均已启动。

**结论与限制**　三路验证已在一个开发用例上闭合。固定图使用详细录制，自主预测以轻量录制评分。同一热库通过 1,536→192 token 的单请求小池零卡复预测，多请求的跨容量复用和独立冻结留出仍待验收。[三路证据](../../../data/large-scale-validation/dev/l1/s05/diagnosis/combined-one-complete/) · [小池零卡账本](../../../data/large-scale-validation/dev/l1/s05/diagnosis/combined-one-automatic-lineage/hot-small-pool/result.json)

## 2026-11-26 — S05：回放原生选批与隔离辅助校准

**范围**　推进混批、前缀缓存的多请求结构。检查三机制全开和小池配置共用的缓存宿主缺口，并完成代表用例的轻量参考采集。

**改动**　从实际上一批恢复同批开始和收尾的请求。原生控制把请求进入等待队列的顺序和批进度写入缺口，由采集器重放。校准从自己的空池开始，逐个核对几何，合格主切点可以独立入库。

**验证**　混批首次筛选、执行与结果切点通过真机来源核验，独立对照从 166 推到 274 个动作。更新后自动生成的选批历史在真机重现三成员混批，100.37 卡秒。缓存多成员录制原先把计划中的 64 个新 token 变成 48 个已缓存加 16 个新 token。重用原录制并排除该辅助点后，主切点与其余符合计划的辅助点发布通过，额外真机次数为零，前缀缓存结构对照达到 200 个动作。全开和小池四个用例共享同一缺失成本坐标，核对真实上一批后，用一次 126.08 卡秒的配对采集补齐，两个回归均越过缺口。四份新增轻量参考共 552.84 卡秒，五份轻量参考窗口内均无新特化加载。Release 构建、安装镜像与 Python 静态检查通过。

**结论与限制**　五种配置的单请求已产出预测，多请求的结构和成本覆盖仍在核验。轻量参考用于开发对照，新增四份统一在物理卡 2 录制，详细参考保留原物理卡位置。留出等待开发闭合后抽取。[混批回放](../../../data/large-scale-validation/dev/l1/s05/diagnosis/scheduler-replay/session/result.json) · [缓存重验](../../../data/large-scale-validation/dev/l1/s05/diagnosis/next-native-cuts/cache-batch-forward/session/result.json)

## 2026-11-26 — S05：首个多请求结构闭合

**范围**　核对分块、混批、缓存和小池开发回归的原生调度准备，分别检查自动采集、独立结构和固定图成本覆盖。

**改动**　同步稠密 TP1 的分块和缓存配置统一记录实际选批入口及请求入队历史。局部 worker 重放至目标选批，空选批仍执行原生状态更新。宿主预热等待实际队列排空后确认 KV 和槽位恢复。批次读回不符以明确缺口和实测几何返回。

**验证**　上一组 Decode 分配结构在 99.14 卡秒实采及来源核对后，混批结构对照推进到 430 个动作。新的统一回放按七次原生选批还原三成员 Decode，109.15 卡秒。切点发布与来源核验通过，同一详细参考上 505 个动作、45,399 个边界和 30,714 项工作纳入对照，动作差异为零。四个自动回归会话各发布首个合格切点，共 496.13 卡秒。此前卡住的 12 请求宿主预热实际执行九批并恢复 1,536 个 KV token 和八个请求槽，命中样本在 136.08 卡秒内发布且来源核验通过。四份固定图按案例各有 1,958 至 4,491 个缺失坐标，主要集中在注意力和模型层的宿主工作。Python 静态检查通过，九个已安装模块与源码摘要一致。

分块四请求、缓存十二请求和小池十二请求分别对照至 387、465 和 429 个动作，后续缺口分别落在四成员 Decode 分配、八成员批次筛选与四成员模型 Decode。混批固定图在后续自动发布中已有 13,004/23,107 项宿主工作可求值，上一发布为 11,036 项。实测基线和来源隔离仍通过。

**结论与限制**　混批四请求的结构开发回归已闭合。一次定向成本诊断将模型字段 `0.0` 转为 `0`，改变了严格工作身份，140.91 卡秒实采未入库。以原类型重查时自动补采已覆盖该坐标，省去重复采集。多请求固定图仍有成本缺口，全部开发回归合格后再冻结留出。[结构对照](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/mixed-four/structure/result.json) · [缓存预热](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/cache-twelve/session/result.json) · [混批固定图](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/mixed-four/host-cost/typed/fixed/result.json) · [真机账本](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/native-budget.json)

## 2026-11-26 — S05：分块与缓存结构闭合、批量成本补采

**范围**　完成其余多请求结构对照中的定向采集，检查四份多请求固定图的成本缺口及已有原生录制的复用能力。

**改动**　复算同域已有样本的增量覆盖。将缺失坐标与已有原生切点的精确工作身份求交，按覆盖量选择独立宿主计时。对尚无切点的模型批次，按同步 TP1 下原生输入读取轮次与模型调用边界构造回放需求，另用物理单卡采集和核验。

**验证**　分块四请求 538 个动作、纯前缀缓存十二请求 975 个动作与小池十二请求 1,580 个动作完整匹配，差异为零。小池再次采集实际页释放切点耗费 105.75 卡秒。五个代表用例合计匹配 3,828 个动作，结构与固定图来源隔离均通过。合并现有同域样本无新增覆盖，混批有一项坐标失去覆盖，该试验资产未发布。

混批、缓存和小池各复用一个已采结构切点，独立计时耗费 104.11、111.96 和 107.11 卡秒。单表复算分别新增 2,546、2,958 和 9,091 项固定图宿主工作覆盖，原有坐标覆盖均保留。按真实读取轮次生成的四成员混批和 77-token Prefill 各通过一轮独立配对采集，耗费 136.20 和 140.11 卡秒。两张表分别新增 3,444 和 2,963 项宿主工作覆盖，混批后一发布共有 21,183 项宿主工作可求值。

混批固定图单独替换宿主混批模型层的 5,472 项工作后，TTFT 与 TPOT MAPE 分别为 6.83% 和 16.53%，读入就绪关系检查通过。替换已配对设备模型层或已配对 Prefill 注意力设备工作时，两项请求指标变化均为零。同输入混批详细与轻量参考的平均 TPOT 分别约 565.08 和 73.63 毫秒。录制方式和物理卡位置同时变化，该差距仅作口径证据。[分组归因](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/mixed-cost-attribution/result/result.json) · [参考口径对照](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/mixed-cost-attribution/reference-mode-comparison.json)

**结论与限制**　五个代表用例的结构开发核验已闭合。已采切点复用与按真实输入构造的新切点都可成组补成本。剩余形状正在核验，固定图成本与自主精度闭环待完成。[小池结构](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/small-pool-free/structure/result.json) · [旧样本复算](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/offline-sample-reuse/coverage.json) · [已采切点影响](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/host-donor-plan/) · [新切点覆盖](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/native-geometry-plan/mixed-four/impact.json) · [真机账本](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/native-budget.json)

## 2026-11-26 — S05：合并同步拷贝并按原生调用补成本

**范围**　分块四请求、混批四请求、缓存十二请求和小池十二请求的固定图成本及对应开发回归。

**改动**　从同一 C++ 求值器批量取得缺失坐标。同步张量拷贝按运行时和实际操作数组批，但仍按各配置的成本域发布。把其余宿主缺口归到原生选批、模型执行或结果处理，用独立参考记录的 RAW 读取边界和请求成员构造局部输入。完整服务的计时仅用于对照，成本样本由另一轮局部真机测量提供。

**验证**　一次单卡同步拷贝实采耗时 15.34 卡秒，答复全部 34 个查询并分别发布至四个开发库。隔离复算新增 50 项固定图传输工作覆盖，无原覆盖丢失。采集记录的准备、预热、观测、读回分别约 1.098 秒、1.75 毫秒、15.96 毫秒和 12.38 毫秒，采集峰值与保留量均约 168 KB。宿主发布脚本更正 Python 3.9 不支持的参数后用同一有效回执完成发布。

首批四个模型切点各通过 8 项查询。在同一旧成本资产上单独加入相应样本，四份固定图分别增加 756、832、8,436 和 2,561 项覆盖。第二批四个切点各通过 8 项查询，分别增加 756、40、2,865 和 5,122 项。混批最后一个结果处理切点通过 4 项查询，增加 5 项。所有复算保留原覆盖，四份参考的成本来源隔离均通过。九个切点共用 1,360.89 卡秒，独立记录各会话准备、采集与导出阶段。[同步拷贝实采与复算](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/remaining-costs/grouped-copy/impact.json) · [原生切点计划](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/remaining-costs/native-cuts/plan.json) · [切点收据与对照](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/remaining-costs/native-cuts/)

**结论与限制**　混批四请求的固定图首次达到 30,714 项工作全部可求值。使用同一发布的自主结果与独立轻量参考相比，TTFT 和 TPOT MAPE 为 18.05% 和 10.00%。它与独立详细参考的 10 个批次顺序和成员一致。固定图替换全部工作耗时的两项 MAPE 为 24.79% 和 57.42%，恢复详细实测宿主耗时后降为 0.10% 和 0.07%。轻量与详细参考的原始观测跨设备位置和镜像，同卡同镜像对照正在录制。分块、纯缓存和小池固定图尚缺 74、2,762 和 6,903 项，补采继续。冻结留出待开发回归闭合。[固定图归因](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/remaining-costs/verification/mixed-four/fixed/result.json) · [自主精度](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/remaining-costs/verification/mixed-four/autonomous/result.json) · [真机账本](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/native-budget.json)

## 2026-11-26 — S05 分块四请求闭合

**范围**　补齐分块四请求在上一发布留下的 74 项固定图工作，复用已封存的独立参考。

**改动**　对原生选批和结果处理切点分别取样。第三个候选开跑前由 C++ 重新求值，确认已覆盖，省去一次真机采样。

**验证**　两次独立切点采集各通过 8 项查询，耗时 134.20 和 127.28 卡秒，分项与存储计量见原生账本。新发布与实测基线一致，42,883 项固定图工作均可求值。独立轻量参考的自主 TTFT 和 TPOT MAPE 分别为 1.53% 和 2.38%。两条验证均通过来源隔离。账本刷新时窗口累计约 32,430.25 卡秒，存储约 12 GiB。[成本对照](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/remaining-costs/post-wave-plan/verification/chunk-four/fixed/result.json) · [自主对照](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/remaining-costs/post-wave-plan/verification/chunk-four/autonomous/result.json) · [真机账本](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/native-budget.json)

**结论与限制**　分块四请求达到两项开发精度门槛。混批的同一预测以第二份轻量录制复算，TTFT 和 TPOT MAPE 为 1.71% 和 8.90%。轻量参考两次录制的物理卡和镜像同时不同，TTFT 变化来源待同条件核实，TPOT 在两份轻量参考上均超出目标。其余多请求开发目标与冻结留出按原口径核验。[混批轻量复测](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/remaining-costs/verification/mixed-four/same-card-reference/light-repeat-metrics.json)

## 2026-11-26 — S05 混批参考口径对照

**范围**　同一个混批四请求输入和同一个已发布自主预测，比较两种真机参考录制。

**改动**　用相同镜像和物理卡顺次取得轻量与详细录制，复用已生成的自主发布结果作比较。

**验证**　两次录制各发布 19 个 token，耗费 130.82 和 142.39 卡秒。实测 p95 TTFT 为 2.47 和 6.60 秒，平均 TPOT 为 74.38 和 551.93 毫秒。窗口累计已结算约 32,977.03 卡秒，存储约 12 GiB。分项和录制回执随原生账本保存。[同条件录制对照](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/remaining-costs/verification/mixed-four/same-card-reference/mode-comparison.json) · [真机账本](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/native-budget.json)

**结论与限制**　固定设备位置与镜像后，详细与轻量录制仍给出明显不同的延迟。混批的自主 TPOT 误差在两份轻量参考上均超标，需要针对轻量时间边界继续归因。单对同条件录制尚无法分解具体观测开销。

## 2026-11-26 — S05 轻量批次诊断与宿主采样开销

**范围**　定位混批四请求的稳态误差，核对已有预测的 Dev 精度，继续补齐缓存和小池的固定图。

**改动**　在 SGLang 观察探针注册 `--batch-timing`，可选记录原生选批、模型执行和结果处理的 RAW 时钟边界，复用 CPU 中的原生批次操作数。正式评分保留独立的 `--latency-only`。宿主采样沿既有辅助区间扣除子范围的插桩建设与清理开销。更新记账时同时读取局部采集的设备清单和完整服务部署内的设备清单，并按容器身份去重。

**验证**　同卡同镜像轻量诊断和普通轻量录制分别耗费 126.68 和 124.93 卡秒，诊断和预测的 10 个批次在阶段、成员和 token 范围上均相符。稳态两个 Decode 周期的模型调用多估 8.30 和 7.00 毫秒，调用外多估 0.50 和 1.52 毫秒。正式评分入口按合同只接受普通轻量录制，其 TTFT、TPOT MAPE 为 1.34% 和 7.90%。[批次分段](../../../data/large-scale-validation/dev/l1/s05/diagnosis/lightweight-batch-timing/mixed-four/batch-comparison.json) · [正式评分](../../../data/large-scale-validation/dev/l1/s05/diagnosis/lightweight-batch-timing/mixed-four/score-latency-only/result.json)

Decode 同切点复采用 144.21 卡秒，全数答复 8 项查询。单录制对照检出插桩 CPU 时间 0.84 毫秒，扣除后 47 个原有坐标的中位数降低，原有坐标全数保留。该批次旧局部宿主录制为 76.51 毫秒，预测为 75.45 毫秒，轻量服务为 68.45 毫秒。缓存四请求通过两项指标，分块和缓存单请求 TTFT 仍超标。完整服务此前所缺的计费来自部署内设备字段，本次复算纳入 21 个已结算尝试及 2,534.14 卡秒。截至本次账本约 39,088.81 卡秒，存储约 12 GiB，录制阶段明细见账本。首次构建的默认网络无法连到本机代理，切换现有宿主网络选项后双镜像构建完成，实装观察模块逐文件核对通过。[同切点实测](../../../data/large-scale-validation/dev/l1/s05/diagnosis/host-scope-overhead/decode/impact.json) · [开发评分](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/remaining-costs/supplementary-dev/) · [修正账本](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/native-budget.json)

混批同切点用 140.02 卡秒，全数答复对应查询，实测插桩 CPU 开销 0.90 毫秒。候选发布只从基线替换两张经重新采样的宿主表，保留全部 30,714 项固定图成本。在相同独立轻量参考上，TTFT MAPE 从 1.34% 变为 1.38%，TPOT MAPE 从 7.90% 降至 7.31%。候选发布和旧发布分别留存。[混批切点](../../../data/large-scale-validation/dev/l1/s05/diagnosis/host-scope-overhead/mixed/impact.json) · [受控换表及覆盖](../../../data/large-scale-validation/dev/l1/s05/diagnosis/host-scope-overhead/revised-release-evidence.json) · [候选自主评分](../../../data/large-scale-validation/dev/l1/s05/diagnosis/host-scope-overhead/comparison-autonomous/result.json)

**结论与限制**　当前 8 个目标已评分，5 个两项指标均达标。实测插桩开销能解释一部分 Decode 多估，候选发布的 TPOT 仍超标。完整宿主录制对照与其余开发案例继续，冻结留出待验收。

## 2026-11-26 — S05 完整服务中的宿主采样影响

**范围**　同卡、同镜像、同输入比较完整服务的宿主采样和轻量批次诊断。

**改动**　复用两种正式采集入口，按阶段与原生请求 token 范围对齐批次，汇总主调度线程上已有时间证据的观察器活动。

**验证**　轻量与宿主采样各耗费 126.74 和 130.97 卡秒。10 个批次一一匹配。宿主录制的 Mixed 和 Decode `run_batch` 分别平均长 6.11 和 6.21 毫秒。两类已标记的 Python 辅助及运行时包装区间在主线程合计约 1.51 和 1.74 毫秒，保留原有重叠合并口径。最新窗口累计结算 40,662.71 卡秒，存储约 12 GiB。[完整服务对照](../../../data/large-scale-validation/dev/l1/s05/diagnosis/host-scope-overhead/serving-pair/comparison.json) · [已标记开销](../../../data/large-scale-validation/dev/l1/s05/diagnosis/host-scope-overhead/serving-pair/known-overhead.json) · [累计账本](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/native-budget.json)

**结论与限制**　宿主采样的耗时抬升在完整服务中同样复现。现有时间标记只覆盖其中部分工作，剩余差值的组成待逐项核实。缓存和小池的十二请求固定图复查分别剩余 56 和 230 项成本缺口。[固定图复查](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/remaining-costs/remaining-scheduler/verification/)

## 2026-11-26 — S05 多请求成本补采与原生批次预热

**范围**　缓存及小池的十二请求固定图，重点核查小池四成员 Decode 的宿主计时资格。复用原有详细参考和轻量参考。

**改动**　按当前成本发布筛选仍缺的原生切点。模型切点预热改为沿采集输入的原生选批准备执行一次目标批次。随后由原生结果处理排空并恢复空池，保留设备工作区，计时窗口内仍检查新特化。记录预热批次和实际耗时。

**验证**　六次原生局部会话耗费 854.68 卡秒，五次取得合格样本。发布后固定图实测基线与来源隔离均通过，缓存缺口由 56 项降到 5 项，小池由 230 项降到 99 项。小池首组失败耗费 149.32 卡秒，记录到新模型特化。使用新构建的源与控制镜像，在同一物理卡复测原 8 项查询，四成员批次与预热批次一致，空池恢复到 192 个 token 和 4 个槽位，预热耗时 8.89 秒。复采耗费 151.72 卡秒，窗口内新增特化从 1 次降到 0 次，8 项查询全部通过。静态检查和双镜像构建通过。[固定图复查](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/remaining-costs/closing-cuts/verification/) · [预热对照](../../../data/large-scale-validation/dev/l1/s05/diagnosis/native-cut-warmup/comparison.json) · [镜像摘要](../../../data/large-scale-validation/dev/l1/s05/diagnosis/native-cut-warmup/images.txt)

**结论与限制**　本次确认实际批次预热消除了该切点的冷加载。复测样本发布后补齐 38 项固定图工作，小池仍缺 61 项。缓存剩余 5 项落在同一个结果处理调用中的前缀插入。S05 的自主精度与冻结留出门禁仍开放。[发布后覆盖](../../../data/large-scale-validation/dev/l1/s05/diagnosis/native-cut-warmup/coverage.json) · [剩余坐标](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/remaining-costs/after-closing-wave/plan.json)

## 2026-11-26 — S05 缓存十二请求闭合与计时开销归因

**范围**　缓存、小池、分块与混批十二请求的开发回归。复查混批四请求、分块与缓存单请求的时间差异。

**改动**　缓存最后五项固定图工作归到同一次原生前缀插入。已采模型切点按当前发布裁掉已覆盖坐标。对混批用同一物理卡、相同请求和同一底图，比较普通宿主采样与单独停用 Ascend CPU profiler 的诊断镜像。对两个单请求用批次诊断划分首 token 前模型调用与其余时间，再用普通轻量录制核对结果。

**验证**　缓存新增一次 140.95 卡秒局部采集，4 个宿主查询和随附的设备样本补齐全部 52,220 项固定图工作，实测基线与来源隔离通过。自主 TTFT 与 TPOT MAPE 分别为 2.64% 和 4.02%。小池最新求值只余 4 项。分块与混批十二请求已取得独立详细和轻量录制，原始来源和联合参考资格均通过，从真实固定图一次查出 5,196 和 20 项成本缺口。

混批的两次录制耗费 130.70 和 128.68 卡秒，10 个批次准确配对。关闭 CPU profiler 后，Mixed 和 Decode 的模型调用墙时分别平均缩短 7.64 和 8.49 毫秒，调度线程在 CPU 上的运行时间有相近的变化。用于归因的镜像保留其他宿主观察器，诊断录制缺少队列证据。缓存与分块单请求的诊断分别匹配 4 和 6 个批次，首 token 差值主要落在模型调用。对同一预测，普通轻量复测 TTFT MAPE 为 3.58% 和 0.53%，原始对照为 8.60% 和 8.75%。这些用例的两轮普通参考 TTFT 分别相差 86.85 和 194.84 毫秒。[缓存固定图和评分](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/remaining-costs/after-closing-wave/verification/cache-twelve/) · [队列采样对照](../../../data/large-scale-validation/dev/l1/s05/diagnosis/queue-profiler-attribution/comparison.json) · [单请求对照](../../../data/large-scale-validation/dev/l1/s05/diagnosis/single-request-batch-latency/) · [十二请求成本坐标](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/remaining-costs/twelve-request-cuts/plan.json)

**结论与限制**　宿主计时的 profiler 开销已定位到运行时观测路径，进一步测量队列打点实际占用的 CPU 区间。两个单请求的首轮和复测均纳入证据，硬件与运行环境波动尚待进一步区分。冻结留出尚未抽取。

## 2026-11-26 — S05 队列记录器开销接入正式采样

**范围**　Ascend 宿主区间采样和成本提取，以及小池十二请求固定图与自主验证。

**改动**　SDK 队列标记保留原样，独立写出标记记录完成时的 RAW 时间。用两个 RAW 锚点和有理数计算标记起点上界，只扣除能够归属记录器的在核运行区间。原生队列配对继续按完整记录执行。宿主对齐保留原有命令列表参数，另外接收已观测到的记录器区间。

**验证**　诊断录制覆盖 24,768 个标记。相同批次的 Mixed 和 Decode 分别测到 2.28 和 1.75 毫秒记录器 CPU 开销。正式源与控制镜像构建成功。单卡实测 Decode 和 Mixed 分别用 144.03 和 138.24 卡秒，配对 416 和 599 个命令，保留 1,664 和 2,396 个记录器区间，合计 9 项原查询全部合格。三组在途资格检查曾遇到调用端与模块加载时的接口版本不一致，修正后复用原录制完成验证与发布，原生尝试次数均保持 1。[记录器计时诊断](../../../data/large-scale-validation/dev/l1/s05/diagnosis/queue-profiler-attribution/recorder-span/comparison.json) · [正式镜像](../../../data/large-scale-validation/dev/l1/s05/diagnosis/queue-profiler-attribution/production/images.txt) · [两组正式回执](../../../data/large-scale-validation/dev/l1/s05/diagnosis/queue-profiler-attribution/production/)

对相同新录制分别保留和扣除记录器区间，两个混批候选均覆盖 30,714 项固定图成本，来源隔离通过。同一个独立轻量参考上，TTFT 和 TPOT MAPE 从 1.27% 和 5.47% 变为 1.25% 和 5.05%。[同源复算](../../../data/large-scale-validation/dev/l1/s05/diagnosis/queue-profiler-attribution/production/sample-impact.json) · [扣除记录器评分](../../../data/large-scale-validation/dev/l1/s05/diagnosis/queue-profiler-attribution/production/corrected/autonomous/result.json) · [保留记录器评分](../../../data/large-scale-validation/dev/l1/s05/diagnosis/queue-profiler-attribution/production/retained-recorder/autonomous/result.json)

混批十二请求的两组同步拷贝合计耗费 28.66 卡秒，完成 77,477 项固定图成本覆盖。自主 TTFT 与 TPOT MAPE 为 0.89% 和 10.41%，来源隔离通过。[混批十二请求固定图和评分](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/remaining-costs/twelve-request-cuts/copy-tail-verification/verification/mixed-twelve/)

**结论与限制**　小池的 81,046 项固定图成本已全覆盖，实测基线与训练隔离通过，独立轻量 TTFT 与 TPOT MAPE 为 3.12% 和 10.36%。混批候选通过该开发用例，同口径合入最新混批库和两个十二请求用例的高 TPOT 归因继续。可扣除区间对应原生标记生成后的实测记录开销。[小池固定图和评分](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/remaining-costs/tail-conditions/verification/small-pool-twelve/)

## 2026-11-26 — S05 十二请求批次归因与同点宿主证据选择

**范围**　小池和混批十二请求的自主批次与发布时刻，以及组合配置的三份新开发对照。

**改动**　按请求从首 token 到末 token 的实际区间聚合对应批次时间，并沿模拟动作的原生调用归属拆解选批和结果处理。宿主采样产物保留已测记录器区间数，拟合在同域、同工作和同完整特征坐标优先选用已实测开销的宿主样本。缺少该证据的坐标沿用已有真实测量。

**验证**　小池与混批的 26 和 25 批完全匹配。混批模型调用贡献约 6.59 个百分点，其中 Mixed 占 5.77，选批再贡献 2.79。小池两轮轻量参考的差值贡献约 4.35 个百分点。按原有拟合追加新版录制，混批四和十二请求的 TPOT MAPE 分别为 6.38% 和 9.83%。标准池组合四、十二请求及小池四请求均已完成固定图与独立评分，来源隔离通过。三例的 TTFT 与 TPOT MAPE 分别为 1.77% 和 9.08%、1.61% 和 6.87%、4.33% 和 3.32%。新增分块批次合并的局部采集耗费 132.62 卡秒，七成员与一成员合并的原坐标通过并发布。[混批发布归因](../../../data/large-scale-validation/dev/l1/s05/diagnosis/mixed-twelve-batch-latency/mixed-twelve/publication-attribution.json) · [混批外围调用归因](../../../data/large-scale-validation/dev/l1/s05/diagnosis/mixed-twelve-batch-latency/mixed-twelve/host-cycle-attribution.json) · [小池发布归因](../../../data/large-scale-validation/dev/l1/s05/diagnosis/small-pool-batch-latency/small-pool-twelve/publication-attribution.json) · [新增评分](../../../data/large-scale-validation/dev/l1/s05/diagnosis/four-remaining-dev-cases/)

从两份既有原生录制恢复了 1,664 与 2,396 个开销区间的资格，样本数值与封存资产一致。新拟合保留两份固定图的全部 30,714 与 77,477 项工作覆盖。相对普通追加拟合，混批四和十二请求在同一独立参考上的 TPOT MAPE 分别从 6.38% 和 9.83% 改善至 3.84% 和 8.69%。新增四例结构对照全匹配，合计 4,144 个动作。[同点证据和覆盖](../../../data/large-scale-validation/dev/l1/s05/diagnosis/queue-profiler-attribution/production/preferred-observer-evidence/selection.json) · [结构与评分](../../../data/large-scale-validation/dev/l1/s05/diagnosis/four-remaining-dev-cases/)

**结论与限制**　当前序列对照确认两份十二请求的性能误差在相同组批上积累。继续核对同点样本选择、大批次实测校准及分块成本覆盖。统计的分项是相对同一参考的有符号百分点，原生参考间的变化单独保留。


## 2026-11-26 — S05 大批次宿主样本与分块成本闭合

**范围**　混批、小池十二请求的宿主成本和分块十二请求的最后一组固定图余项。

**改动**　各用一次详细服务和一次宿主计时录制，按原生批次配对提取宿主样本，两组仍以各自早先独立的轻量服务评分。同卡同任务诊断 PyTorch 线程局部 profiler 状态。隔离拷贝探针补采时漏传新增计时参数，已修正调用，本探针不消费的队列打点输出交给系统空设备处理。

**验证**　混批和小池分别匹配 25 和 26 批，得到 5,684 和 5,801 行宿主成本。两组实测分别保留 66,284 和 65,296 个打点区间，TPOT MAPE 从 10.41% 和 10.36% 降至 6.40% 和 6.35%。混批四请求 TPOT 为 3.54%。原拟合与现行拟合在相同的两组既有输入上复算一致。诊断维持 25 批及其生产者队列顺序，Mixed 平均墙时差 0.05 毫秒，Decode 约慢 1.09 毫秒，未证实该线程局部状态造成稳态开销。[混批](../../../data/large-scale-validation/dev/l1/s05/diagnosis/mixed-service-cost-evidence/assessment/) · [小池](../../../data/large-scale-validation/dev/l1/s05/diagnosis/small-pool-service-cost-evidence/assessment/) · [原口径复算](../../../data/large-scale-validation/dev/l1/s05/diagnosis/queue-profiler-attribution/production/preferred-observer-evidence/prior-evidence-equivalence.json) · [线程状态诊断](../../../data/large-scale-validation/dev/l1/s05/diagnosis/queue-profiler-attribution/torch-state-diagnostic/comparison.json)

五次宿主相关真机开发录制合计 732.38 卡秒。各次初始化 23.94–26.13 秒、预热 60.89–72.60 秒、有效观测 9.35–25.96 秒。混批和小池详细图后处理耗时 67.18 与 70.16 秒，各会话导出并保留的原生来源合计约 389 MiB。分块 C++ 批量求值找出四个拷贝坐标。拷贝探针首轮计时收尾失败 13.34 卡秒，峰值 7,349 字节。双镜像重建并核对采样模块的实装哈希后，复测用 14.24 卡秒在一次会话覆盖四个坐标，各获七个合格观测。工具记录的准备、预热、有效观测和回读分别为 1.070、0.001、0.003 和 0.002 秒，回执前保留量和采样峰值均为 32,795 字节。自主 TTFT / TPOT MAPE 为 4.00% / 4.66%，训练来源隔离通过。Ruff 与 Git 差异检查通过。[批量坐标、失败回执和复测](../../../data/large-scale-validation/dev/l1/s05/diagnosis/chunk-twelve-transfer-cuts/) · [采集账本](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/native-budget.json)

将小池实测样本追加到标准池两例使用的原发布。原生采样成本域一致，固定图保留全部 27,989 与 68,137 项成本覆盖，自主 TPOT MAPE 从 9.08% 和 6.87% 降为 4.99% 和 2.62%。双方训练来源隔离通过，未新增卡时。截至本批窗口已结算 50,373.16 卡秒。[同原发布复核](../../../data/large-scale-validation/dev/l1/s05/diagnosis/small-pool-service-cost-evidence/same-release-assessment/)

**结论与限制**　15 个开发目标均已出分，混批与小池的十二请求用例仍高于 6% TPOT 目标。线程状态诊断仅用于归因。冻结留出尚未启动。


## 2026-11-26 — S05 统一开发发布与宿主写盘扰动

**范围**　四份候选发布覆盖五组配置，复算 15 个开发用例。核对四个超出首份参考门槛的用例，并区分原生时延波动与持续偏差。

**改动**　固定发布后统一重跑评分。对混批、小池十二请求各在原物理卡及原镜像上重录普通轻量参考。Ascend 宿主计时复用 SDK 原队列缓冲，缓冲过半才启动中途写盘，余量在停止后排空。详细采集沿用原有排出条件。构建明确优先使用同版补丁头文件，修复首轮误选安装头文件导致的编译失败。

**验证**　四个成本域分别找到 53、55、56、59 项原生观察的采集回执。15 项预测均出分且来源隔离，首份参考有 11 项同时低于 6%。以相同预测比较复测，混批十二请求 TPOT MAPE 6.40% / 7.27%，小池十二请求 6.35% / 4.32%。分块单请求 TTFT MAPE 7.98% / 0.31%，缓存单请求 7.74% / 4.34%。按最新轨迹逐批核对，混批与小池分别对齐 25 和 26 批。混批偏差中混合批执行约占 4.30 个百分点，小池两次原生参考的差异约占 4.35 个百分点。[统一评分](../../../data/large-scale-validation/dev/l1/s05/diagnosis/current-cost-residual/selection/scoring/summary.json) · [用例复测](../../../data/large-scale-validation/dev/l1/s05/diagnosis/current-cost-residual/selection/reference-repeats/) · [混批轨迹归因](../../../data/large-scale-validation/dev/l1/s05/diagnosis/current-cost-residual/mixed-twelve/publication-attribution.json)

两次轻量复测使用 126.61 和 140.39 卡秒。初始化 22.21 和 22.87 秒，预热 61.78 和 72.86 秒，观测 4.95 和 7.13 秒。移出写盘诊断用 135.29 卡秒，初始化 25.37 秒、预热 61.67 秒、观测 9.38 秒。全部 25 批的队列生产者顺序一致，Mixed 平均墙时减少 1.36 毫秒。三份原生来源分别保留 48 KiB、48 KiB 和约 24.7 MiB，原生现场峰值未单独采样。正式双镜像构建及计时符号核对通过。此时窗口已结算 50,775.45 卡秒。[写盘诊断](../../../data/large-scale-validation/dev/l1/s05/diagnosis/queue-profiler-attribution/deferred-dump-diagnostic/comparison.json) · [正式镜像](../../../data/large-scale-validation/dev/l1/s05/diagnosis/queue-profiler-attribution/buffered-production/images.txt) · [账本](../../../data/large-scale-validation/dev/l1/s05/diagnosis/unified-selection-replay/native-budget.json)

正式宿主采样用 134.29 卡秒，初始化 24.27 秒、预热 61.15 秒、观测 9.41 秒。25 批及其队列生产者顺序均与原样本一致，Mixed 平均缩短 1.45 毫秒，Decode 慢 2.48 毫秒。原详细录制与新宿主录制的镜像摘要不同，由成本提取拒绝，保留拒绝及真实宿主记录。补充同镜像详细录制以完成匹配。[正式采样对照](../../../data/large-scale-validation/dev/l1/s05/diagnosis/queue-profiler-attribution/buffered-production/mixed-twelve/comparison.json)

**结论与限制**　同口径复测确认混批的 TPOT 偏差持续存在。正式采样正在核对队列与测量覆盖，随后将按同一发布和两次独立轻量参考复算。候选发布尚未冻结。


## 2026-11-26 — S05 同镜像宿主样本复算及资产冻结

**范围**　收口混批十二请求的持续 TPOT 偏差，核对混批候选发布、模板来源和跨池热库复用。随后启动五组冻结独立留出。

**改动**　将已完成的正式宿主计时与同镜像详细录制配对。仅更新混批候选的成本发布，逐文件核对其模板与既有候选完全相等。将四份经核对的库、源码、已构建 Release 二进制和镜像摘要固定到留出资产，以清单 SHA-256 执行一次声明的请求抽取。

**验证**　同镜像详细录制用 152.45 卡秒，初始化 26.69 秒、预热 61.18 秒、观测 23.65 秒，图后处理 69.28 秒。详细与宿主计时原生来源分别保留约 151.8 和 25.2 MiB，现场峰值未单独采样。新源通过独立联合资格检查，25 批全部配对并提取 5,684 行。混批十二请求全部 77,477 项固定图工作可求值，基线边界与训练隔离通过。两份现有轻量参考的 TTFT MAPE 为 3.59% 和 0.71%，TPOT MAPE 为 4.20% 和 5.07%。混批一、四请求两项指标均达标。[同镜像样本](../../../data/large-scale-validation/dev/l1/s05/diagnosis/queue-profiler-attribution/buffered-production/mixed-twelve/measured-host.json) · [评分及固定图](../../../data/large-scale-validation/dev/l1/s05/diagnosis/queue-profiler-attribution/buffered-production/mixed-twelve/assessment/)

正式来源核验通过 223 项观察、224 个片段。首轮编排提前创建了由工具独占的输出目录，未取得核验报告，改用日志与输出并列的路径后完成。用同一发布运行标准池和小池的 1、4、12 请求，六项预测均完成，零次原生尝试。核心 Release 构建产物摘要一致。冻结前窗口结算 51,062.19 卡秒，新增冻结文件和证据清单可直接复核。[来源核验](../../../data/large-scale-validation/dev/l1/s05/diagnosis/current-cost-residual/selection/qualified-sources/summary.json) · [热库账本](../../../data/large-scale-validation/dev/l1/s05/diagnosis/current-cost-residual/selection/hot-shared-matrix/exploration/result.json) · [冻结清单](../../../data/large-scale-validation/holdout/freeze/s05/source-manifest.json)

**结论与限制**　混批的同口径持续偏差已在两份开发参考上收口。15 项开发评分保留首轮和复测口径，首份参考 12 项双指标达标，三项原生参考波动仍明显。五组冻结留出已一次抽取，详细原生采集已启动。

## 2026-11-26 — S05 冻结留出详细参考封存

**范围**　五个冻结用例的单卡详细原生参考，覆盖分块、混合批、前缀缓存、标准池组合和小 KV 池组合。

**改动**　保存十份抽样输入的 SHA-256。按冻结矩阵完成五次详细录制，未变更冻结代码、镜像、二进制和双库。

**验证**　五次均一次完成。各用例的原生启动输入、设备部署和镜像均与冻结声明相同，五个录制身份与模板及成本来源无重合。初始化 26.19–26.75 秒、预热 60.00–98.74 秒，有效观测 20.26–61.91 秒。合计 912.69 卡秒，原生输出保留约 1.163 GiB。现场峰值未单独采样，图后处理由后续统一阶段执行。[输入摘要](../../../data/large-scale-validation/holdout/results/s05/draw-integrity.json) · [详细参考回执](../../../data/large-scale-validation/holdout/results/s05/detailed.json)

**结论与限制**　详细参考已收齐，五组轻量参考开始采集。结构、固定图和自主精度将在两类参考收齐后统一评估。

## 2026-11-26 — S05 冻结留出轻量参考收齐

**范围**　五个冻结用例的普通轻量原生参考及两类录制的隔离复核。

**改动**　按同一冻结用例完成轻量采集，摘取普通原生请求发布时刻。五组详细与轻量记录收齐后，矩阵自动交接图生成。

**验证**　五次轻量录制均一次完成。初始化 22.70–23.27 秒、预热 59.69–96.15 秒，有效观测 3.48–10.46 秒。轻量合计 716.05 卡秒，保留约 272 KiB 原生输出。两轮合计 1,628.74 卡秒。十份回执对应不同原生身份，均与模板和成本来源隔离。输入、镜像和部署逐项与冻结声明一致。现场峰值未单独采样。[轻量回执](../../../data/large-scale-validation/holdout/results/s05/light.json) · [详细回执](../../../data/large-scale-validation/holdout/results/s05/detailed.json)

**结论与限制**　两类原生参考完整封存，图生成已启动。执行结构、固定图基线及自主性能结果待统一评估。

## 2026-11-26 — S05 首轮冻结留出判定

**范围**　按同一冻结源码、镜像、二进制和双库评估五组单卡复杂调度案例，覆盖结构、固定图与自主预测三路。

**改动**　保留首轮冻结资产、抽样输入和报告。完整结算后将五组案例永久登记为开发用途。开发修复先核对共同的 Decode 分配续段，再根据完整真实图汇总成本坐标。重新留出时使用新冻结资产和新原生参考。[开发集归属](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05/classification.json)

**验证**　五组真实图均生成，十份原生参考与训练来源隔离，固定图实测重放 5/5 对齐。五组结构均缺少 Decode 页分配从驱逐检查到返回的候选模板。模型覆盖 414,419/578,102 项固定图工作，自主预测均遇到采样范围缺口。两轮录制各五次，共 1,628.74 卡秒，初始化、预热及有效观测分项见此前两条封存记录。详细和轻量原生输出分别保留约 1.163 GiB 和 272 KiB，峰值未采样。整轮矩阵后处理 1,744.73 秒，最终保留约 2.74 GiB。窗口累计 52,690.94 卡秒。[独立判定和账本](../../../data/large-scale-validation/holdout/results/s05/acceptance-summary.json)

**结论与限制**　首轮 0/5 通过，自主 TTFT 和 TPOT 尚无评分。本轮确立了真实参考和固定图实测重放的资格，待开发复现确定模板生产与样本覆盖根因。S05 保持实施中，L2 未启动。

## 2026-11-26 — S05 开发回归跨页模板和整图成本范围

**范围**　复用已转为开发集的五张图。核对跨页 Decode 的候选缺口、同步拷贝的实测资格及宿主特征范围。

**改动**　四份开发库沿现有原生调度配方各纳入一个经独立核验的跨页模板。以原图和 C++ 成本求值器一次枚举需求。从已有详细录制复用设备样本，对拷贝保留操作数就绪后的单独计时。宿主按工作身份和实测范围归并待采端点。[调度配方与模板缺口](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05/template/summary.json) · [整图覆盖](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05/coverage/summary.json) · [宿主切点](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05/host-plan/summary.json)

**验证**　四组结构完整匹配 7,532 个动作，小池在第 933 个已比较动作后仍需单成员分支。设备样本复用后缺口为零，余 122,931 项宿主工作和 43 项拷贝工作，后者的 13 个不同坐标已全部实采合格。四次跨页模板采集 395.84 卡秒，初始化 6.67–7.17 秒，准备与回读 43.72–45.51 秒，观测并导出 12.64–28.10 秒，单会话原生输出约 0.50–1.21 MiB、采样峰值约 0.99–1.73 MiB。四次拷贝合计 59.31 卡秒，设备准备 1.05–1.08 秒，观测及回读按回执记录，单会话原生输出约 10–25 KiB、采样峰值约 14–30 KiB。覆盖枚举、提样拟合、发布、来源与结构核验等已结束的后台命令墙时合计 887.27 秒。已结算窗口累计 53,146.09 卡秒。[源核验与结构对照](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05/template/) · [设备样本复用](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05/native-device-reuse/) · [拷贝回执](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05/local-copies/)

**结论与限制**　标准池实测模板的多成员守卫有效，小池单成员正在实采。混批的宿主需求最集中，正在用一次原生宿主录制测可覆盖的端点范围。以上为开发证据，下一轮独立验收仍需新的冻结资产与原生参考。

## 2026-11-26 — S05 单成员分配和混批宿主覆盖

**范围**　小池跨页单成员结构、五张开发图的设备与拷贝拟合，以及混批 16 请求的宿主覆盖与自主预测。

**改动**　发布小池单成员跨页观测。编译整张小池真实图提取动作条件，按库内实际绑定与声明找缺项。混批通过既有 `host-timing` 和 `cost extract --kind host` 入口配对采样。首次组装开发库的外层容器未接入标准输入，修正后复用原拟合结果，另存核验报告。

**验证**　小池来源核验通过，实测条件对照匹配 3,981 个动作。实际 51 个条件仅缺空页列表释放声明。加入合格拷贝样本后，五张图均只剩宿主缺项。混批一次录制对齐 39/39 批，宿主没有缺测，104,585 项固定图工作可定价。分离的轻量参考上，TTFT MAPE 为 2.68%，TPOT MAPE 为 5.21%。两次单卡实采分别耗费 98.01 与 140.96 卡秒。前者的初始化、准备、观测和导出见本地回执。后者初始化 24.48 秒、预热 68.81 秒、有效观测 9.88 秒，提样 30.91 秒，保留约 35 MiB，现场峰值未采样。窗口累计 53,385.06 卡秒。[小池声明检查](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05/template/small-pool-32/declared-coverage/summary.json) · [小池来源报告](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05/template/small-pool-32/source/source-evidence.json) · [混批宿主样本](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05/host-window/mixed-16/measurements.json) · [混批开发评分](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05/mixed-with-host/development-autonomous/result.json)

**结论与限制**　混批开发评分达到双指标门槛。原小池实测图用于确定最后一项缺声明的原生条件，完整结构仍待新本地来源核对。同一详细录制已经用作成本训练来源，覆盖结果属于开发证据，下一轮独立验收必须换真值。

## 2026-11-26 — S05 五组持续并发开发回归收口

**范围**　补齐小池释放到空页列表的原生分支，复用首轮已降级开发的五张图建立宿主成本，并用分离的实测参考检验预测。

**改动**　小池空页释放经已有局部配方发布。四个配置补充完整宿主观察，按原生批次和工作区间与原录制配对。保留端点规划供稀疏缺口使用。223 个已核验来源的观察体与定义逐字节复核，另加六项新来源随候选库保留。业务代码、模型求值和独立核验合同沿用首轮冻结版本。[模板来源预检](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05/round-two-source-preflight.json)

**验证**　五组结构 12,213 个动作全部匹配，五张开发图 578,102 项工作全覆盖。五份不同物理录制的旧开发参考隔离通过，固定图实测重放与 397,425 项成本覆盖均通过。分块、混批、缓存、标准池组合和小池的自主 MAPE（TTFT/TPOT）依次为 1.39%/2.07%、2.68%/5.21%、1.13%/2.72%、1.80%/2.72%、3.75%/4.72%。五份宿主录制累计配对 202 批，取得 34,243 个样本坐标。四次新增宿主录制用 631.93 卡秒，初始化 24.10–25.11 秒、预热 60.61–98.15 秒、观测 10.56–23.06 秒，提样合计 141.89 秒，原生导出约 159 MiB，峰值未采样。小池局部补采用 111.06 卡秒，分项、峰值和保留量见账本。窗口累计 54,128.05 卡秒。[小池结构](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05/template/small-pool-32/empty-release-regression/result.json) · [旧开发固定图](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05/held-development-fixed/) · [开发评分](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05/with-host/) · [实采账本](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05/native-budget.json)

**结论与限制**　已通过五组扩展开发案例的三路检查。新冻结留出前继续更新旧小规模案例与热库证据。首轮独立失败记录保留，S05 的独立验收尚未通过。

## 2026-11-26 — S05 第二轮独立冻结与采集启动

**范围**　原 15 组开发回归、共享库零卡时复预测及第二轮留出的冻结入场条件。

**改动**　用四份更新的双库统一复算开发矩阵，保留单请求的两份测量口径。以新的冻结清单 SHA-256 一次性抽取五组新请求。输入核对首次遇到旧开发请求的可选前缀字段，归一化后验证同一份请求抽取记录。

**验证**　六个热库目标全部预测成功，零新增原生尝试、零卡时。旧开发首份参考 14/15 达标，分块单请求首份 TTFT 误差 10.58%，复测 TTFT/TPOT 为 2.52%/1.28%，两份训练来源均隔离。新冻结复制并校验 340 个源码文件、三镜像、Release 二进制、四份双库及 229 份来源报告。新清单 SHA-256 为 `7f9578cf820897e72399531dfca7410dfb58e39e11e7d13c907080b04a4d3f2d`。十份输入摘要已确认，新请求清单与首轮及旧开发集均不同。本批冻结、抽样和热库证据零卡时，详细原生参考已启动，卡时随原生回执结算。[冻结入场](../../../data/large-scale-validation/holdout/rounds/s05-r2/freeze/s05/dev-readiness.json) · [冻结清单](../../../data/large-scale-validation/holdout/rounds/s05-r2/freeze/s05/source-manifest.json) · [抽样核对](../../../data/large-scale-validation/holdout/rounds/s05-r2/results/s05/draw-integrity.json)

**结论与限制**　第二轮按固定资产推进五组独立核验。L1 的 6% MAPE 与结构、固定图门槛待第二轮统一判定。

## 2026-11-26 — S05 第二轮冻结资产核对与详细参考进度

**范围**　第二轮固定资产、工作区静态边界和前三份详细原生参考。

**改动**　只读核对冻结源码、Release 二进制、三镜像及前三组实际录制的输入、部署与来源身份。保留工作区静态检查报告。[静态检查](../../../data/large-scale-validation/dev/l1/s05/l1-static/quality.json)

**验证**　工作区的 340 个文件与冻结清单逐字节一致。218 个 Python 模块的导入环、缺失导入、过长函数块、越层引用、Ruff 问题及活跃字节码均为零。三份详细录制一次通过，录制身份与首轮、模板和成本训练来源均分离。已结算 513.52 卡秒，初始化 26.00–26.39 秒、预热 67.74–87.93 秒、有效观测 22.80–33.61 秒。整轮导出、保留与后处理量随矩阵终验登记。[录制回执](../../../data/large-scale-validation/holdout/rounds/s05-r2/results/s05/detailed.json)

**结论与限制**　详细参考已完成 3/5，组合配置继续录制，第二轮结构与精度判定仍待执行。

## 2026-11-26 — S05 第二轮详细参考封存

**范围**　第二轮五组详细原生参考与冻结镜像实装内容。

**改动**　保存两张镜像的实装模块与对应 Docker `COPY` 清单逐文件 SHA-256 对照。矩阵封存五份详细录制后进入轻量录制。[镜像检查](../../../data/large-scale-validation/dev/l1/s05/l1-static/image-installation.json)

**验证**　两张镜像分别匹配采集侧 82 个、控制侧 47 个 Python 文件，观测库四个计时接口均已导出。详细录制 5/5 均首试封存，校验请求、部署、镜像和原生身份，未发现首轮或训练来源重用。初始化 25.55–26.39 秒、预热 67.74–102.08 秒、有效观测 22.80–70.50 秒，共 947.56 卡秒，保留原生输出约 1.26 GiB，峰值未分项采样。窗口累计 55,075.62 卡秒。[详细录制回执](../../../data/large-scale-validation/holdout/rounds/s05-r2/results/s05/detailed.json)

**结论与限制**　五份详细参考已具备后续对照输入，轻量参考及三路验收仍在矩阵中推进。

## 2026-11-26 — S05 第二轮轻量参考封存与构图启动

**范围**　五份轻量原生参考及两类录制的来源隔离。

**改动**　逐份核对实际请求、部署与镜像。十份原生录制身份互不重复，均与首轮、模板和拟合来源分离。

**验证**　轻量录制 5/5 均首试封存。初始化 21.90–22.82 秒、预热 67.24–103.23 秒、有效观测 4.14–12.81 秒，共 757.92 卡秒，保留原生输出 272 KiB。十份录制共 1,705.48 卡秒，窗口合计 55,833.54 卡秒，最终后处理和总保留量随验收结算。[轻量录制回执](../../../data/large-scale-validation/holdout/rounds/s05-r2/results/s05/light.json) · [详细录制回执](../../../data/large-scale-validation/holdout/rounds/s05-r2/results/s05/detailed.json)

**结论与限制**　两类参考已封存，矩阵已进入构图。结构、固定图和自主性能门槛待对照结果齐全后判定。

## 2026-11-26 — S05 第二轮独立判定与覆盖根因定位

**范围**　第二轮实测时间结构对照、固定图成本和自主闭环五组结果。

**改动**　完整保留冻结资产、十份原生录制及结果。判定结束后，五组已永久登记为开发案例。复用同一 C++ 求值器汇总整图成本坐标，另启动两项原生分支的局部采集与设备样本离线提取。[开发集登记](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/classification.json)

**验证**　分块、标准池组合与小池结构完整匹配 9,487 个动作。混批的已发布跨页候选要求输入成员数大于一，留出触发单成员。前缀缓存已发布候选要求存在非整页尾部，留出触发整页缓存写入。五份固定图的实测时间重放全部对齐，612,915/648,528 项成本工作有预测值。四组自主预测遇到成本范围缺口，一组遇到缓存模板守卫。训练来源隔离通过。整轮十次实采用 1,705.48 卡秒，后处理 2,030.52 秒，总保留约 3.32 GiB，初始化、预热与观测见上两条记录。[第二轮验收与账本](../../../data/large-scale-validation/holdout/rounds/s05-r2/results/s05/acceptance-summary.json)

**结论与限制**　第二轮整体验收 0/5，S05 保持实施中。下一批按已声明工作负载的几何边界补开发覆盖，保留原生模板守卫和现行成本适用范围校验。

## 2026-11-27 — S05 原生边界取证与开发矩阵启动

**范围**　两条待发布结构分支，第二轮全部成本缺口和已有设备样本的复用边界。

**改动**　构造五种既有配置的稀疏与集中到达开发场景，覆盖原计划中每种请求模式的全部提示长度和输出步数。用正式适配入口核验十份配置，核对其请求序列均与既有开发集及两轮留出不同。首次构造调用使用了错误的适配器导入位置，修正后十份输入全部生成。真实分支、批次和可采样范围由本批原生录制确认。[开发输入与范围](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/geometry-envelope/plan.json)

**验证**　混批单成员跨页和整页缓存写入均首试局部取证合格。两次共 189.51 卡秒，初始化 6.63–6.70 秒、准备与回读 43.76–44.40 秒、观测和导出 9.52–18.27 秒，采样峰值约 1.00–1.28 MiB，原生保留及分相明细见回执。此前五张真实图复用设备实测后新增覆盖 9,735 项，余 1,293 个独立宿主坐标和一个就绪拷贝坐标。窗口已结算 56,023.05 卡秒。[局部回执](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/template/) · [需求与覆盖](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/native-device-reuse/)

**结论与限制**　两项结构观测均已发布并通过原生来源核验，完整结构复核继续。开发边界已启动真机录制。下一次独立留出须以开发边界的实际覆盖和三路对照结果为前提。

## 2026-11-27 — S05 第二轮旧案例结构回归与边界投影

**范围**　复核混批和前缀缓存两例的完整结构，获取最后一项 16 字节就绪拷贝的实测，投影新增的开发边界。

**改动**　用现有调度配方发布前缀命中后只剩一个待算 token 的 Prefill 分支。拷贝由现有同步拷贝探针取得。开发边界首次核验传入封存任务目录，改为从成功回执定位含原始记录的目录，复用完成的图投影。

**验证**　混批与缓存分别匹配 2,254 和 1,459 个动作，来源隔离均通过。两项局部采集各一次，共 131.49 卡秒。Prefill 初始化 5.41 秒、准备与回读 44.74 秒、观测及导出 36.57 秒，原生保留约 7.73 MiB、峰值约 13.50 MiB。拷贝准备 0.91 秒，预热、观测与回读合计约 12 毫秒，原生保留约 9.27 KiB、峰值采样约 8.45 KiB。发布、来源及结构对照墙时共 127.28 秒。已结算窗口累计 56,154.54 卡秒，开发边界采集单列结算。[混批对照](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/template/mixed-16/development-structure/result.json) · [缓存对照](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/template/cache-16/one-token-regression/result.json) · [拷贝回执](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/local-copies/cache/local/result.json)

**结论与限制**　第二轮五例都有完整结构证据，新增复核属于开发回归。已通过来源核验的开发图继续交由正式提取器和拟合入口汇总整图覆盖，整批宿主覆盖与自主精度待验证。

## 2026-11-27 — S05 统一成本键的数值语义

**范围**　合并缓存设备成本与已实测拷贝，对齐 Python 拟合、模板绑定和 C++ 成本域选择。

**改动**　合并输入的模型配置只有 JSON 表示 `0.0` 和 `0` 的差别。Python 之前按序列化字面值建立两个成本域，C++ 按数值相等拒绝重复资产。公共成本键改为递归统一数值相同的整数和浮点整数，保留布尔值、非整数和原始证据。

**验证**　用同两份原生样本核对键一致性并复用生产 `cost fit` 和 C++ 求值器。初次拟合在重复域判定停止（12.52 秒）。修复后 14.95 秒完成三图成本审计，发布一个成本域，旧缓存图覆盖由 76,520 增至 76,521 项，剩余 7,727 项均为宿主工作。Ruff 和 diff 检查通过。[原始失败](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/geometry-envelope/coverage/cache/fit.log) · [修复后回执](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/geometry-envelope/coverage/cache/normalized-identity/costs.json)

**结论与限制**　缓存拷贝可进入成本发布。分块、缓存的稀疏和集中到达开发图开始配对测量宿主服务。代码变更待后续同批冻结核对双镜像。

## 2026-11-27 — S05 批量定位调度分支和宿主坐标

**范围**　十份开发边界录制、现行模板发布的入口覆盖，以及分块、缓存、混批宿主成本。局部采集的空闲路径在 SGLang 观察适配器内处理。

**改动**　原生选批结果为空时，局部观察继续调用原生 `on_idle`，保留实测容量读取。C++ 逐个查询观测入口，确认短序列连续块表、单成员翻页及空闲读取缺项，再按原录制还原局部输入。首次批查询复用了不可覆盖的输出路径，改为每次消费后移除临时输出。分块开发库从固定包转入可写存储时补回同模板字节版本的真实发布回执，复用已完成的原生测量继续发布。

**验证**　用已核验的本地基底按原构建脚本生成双镜像，变更文件与实装 SHA-256 一致。默认基底标签无本地对应，正确标签在隔离网络下遇到代理连接失败，使用既有宿主网络构建配置完成。两种空闲切点与两种短序列块表均通过来源核验。十份详细录制各一次，共 2,594.73 卡秒，初始化 25.59–27.62 秒、预热 60.44–284.54 秒、观测并导出 25.11–160.45 秒，保留原生输出约 3.97 GiB。八份配对宿主录制共 1,472.09 卡秒，初始化 24.07–26.29 秒、预热 58.63–147.51 秒、观测 11.29–89.77 秒，提样 298.29 秒，保留原生输出约 434.34 MiB。两类完整服务录制的瞬时峰值未单独采样。四次局部采集共 369.58 卡秒，初始化 5.40–7.37 秒、准备回读 43.92–46.01 秒、观测导出 8.00–13.09 秒，逐次保留量和采样峰值见原生回执。窗口已结算 60,590.94 卡秒。[详细录制账本](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/geometry-envelope/native-budget.json) · [宿主配对](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/geometry-envelope/host-timing/) · [镜像回执](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/build-idle/images.json)

**结论与限制**　分块三图 236,469 项全部覆盖。缓存和混批各余 22、5 个宿主坐标。已知结构分支正在批量发布、核验，独立留出继续等待开发闭环。[分块覆盖](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/geometry-envelope/coverage/chunk/with-complete-pairs/costs.json) · [缓存覆盖](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/geometry-envelope/coverage/cache/with-complete-pairs/costs.json) · [混批覆盖](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/geometry-envelope/coverage/mixed/with-host/costs.json)

## 2026-11-27 — S05 定点宿主测量与三域开发闭环

**范围**　分块、缓存和混批的新旧开发图、独立固定图、原两轮的五组请求中三组可评分留存参考，以及共享库的合法原生分支。

**改动**　用成本工作身份回查原生选批和结果处理边界，按收包证据重建两个配对切点。补采分块空闲、单成员翻页、组合短 Prefill 和小池空释放目标四条分支。前三个候选库保留旧版 167 个来源的观测内容和定义字节，新增八个来源均有独立核验回执。

**验证**　分块、混批和组合稀疏到达各匹配 1,221、3,427、4,742 个动作。分块、缓存、混批九图共 1,016,995 项工作可定价，三份独立固定图的 248,242 项均有实测与预测成本。六个自主开发参考的 TTFT MAPE 为 1.40%–5.26%，TPOT 为 1.93%–4.43%，来源隔离均通过。两次配对录制共 253.58 卡秒，结构与计时探针各自的初始化 5.12–6.86 秒、准备与回读 2.46–44.31 秒、观测和导出 1.55–12.40 秒，保留 2.11 MiB，单会话峰值 0.97–1.16 MiB。拟合和整图检查共 41.40 秒。四次局部模板录制共 428.67 卡秒，初始化 5.40–6.78 秒、准备与回读 43.95–44.59 秒、观测导出 7.88–48.78 秒，保留 11.04 MiB，峰值 0.46–14.71 MiB。窗口已结算 61,273.19 卡秒。[宿主回执](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/geometry-envelope/host-residual/) · [按源字节核对](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/geometry-envelope/development-candidates/source-byte-check/) · [评分及固定图](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/geometry-envelope/development-candidates/assessment/)

**结论与限制**　三域六个新增边界自主预测有五个成功。缓存稀疏到达的模拟进入了尚未覆盖的原生空闲检查，相关局部源正在采集。共享标准池和小池的配对测量及小池整图复核继续进行。[开发自主运行](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/geometry-envelope/development-candidates/warm/)

## 2026-11-27 — S05 十组开发结构结算

**范围**　小池稀疏和集中到达的整图结构对照、缓存自主预测走到的空闲读取，以及共享库宿主模型。

**改动**　合并十组独立结构回执并记录其摘要。缓存空闲调用发布后，复用同一次结构录制测量宿主耗时。组合的 27 个短 Prefill 宿主坐标也使用已独立核验的原生结构录制，测量仅重做计时部分。

**验证**　小池两组分别匹配 5,506 与 6,212 个动作。十组共匹配 30,899 个动作，来源隔离均通过。两个定点宿主会话和一个空闲模板会话共 298.95 卡秒，初始化 5.63–6.88 秒、准备回读 43.72–45.55 秒、观测导出 1.37–48.99 秒。标准池组合两份计时共耗 561.38 卡秒，初始化 24.01–24.52 秒、预热 70.12–221.03 秒、观测 15.06–127.41 秒，保留约 247.32 MiB。两份提样共 158.34 秒，初次共享拟合 61.65 秒。该批组合原生峰值未单独采样，局部会话保留量与峰值见回执。窗口已结算 62,133.51 卡秒。[结构结算](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/geometry-envelope/development-structure-summary.json) · [局部采样](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/geometry-envelope/host-residual/) · [共享覆盖](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/geometry-envelope/coverage/shared/with-combined-host/costs.json)

**结论与限制**　缓存自主运行已通过空闲读取，后续单成员短 Decode 的宿主 Attention 坐标超出现有实测，已启动原生配对。共享库在计入标准池组合计时后余 194 个宿主坐标，使用小池与短 Prefill 的配对样本继续验证覆盖。[短 Decode 缺口](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/geometry-envelope/development-candidates/cache-qualified/warm/dev-cache-16-sparse/prediction/cost-gap.json)

## 2026-11-27 — S05 复用短 Decode 设备样本与收口共享库宿主范围

**范围**　缓存短 Decode 的原生工作与标准池、小池共享成本库。

**改动**　在上下文 33 的已合格宿主配对中复用结构录制提取 46 个设备和搬运坐标。下一次用原生队列重现开发输入中最短的上下文 23，同一配对发布 171 个宿主与 46 个设备和搬运坐标。共享库将标准池稀疏图剩余 54 个坐标归到两个真实原生调用，分别覆盖 41 和 13 个坐标。

**验证**　缓存三图及四个已见缺口全部可定价。小池稀疏图 378,801 项工作全部覆盖，分开采到的计时按原生批次对齐 129 批。两次局部配对耗 273.67 卡秒，单探针初始化 3.84–5.47 秒、准备与回读 2.39–45.23 秒、观测和导出 6.99–19.61 秒，单会话保留约 6.89–6.90 MiB、峰值 8.26–8.85 MiB。小池稀疏配对计时耗 507.24 卡秒，初始化 23.72 秒、预热 284.38 秒、观测 159.21 秒，原生留存约 234.19 MiB，未单独采样采集峰值。该计时提样用时 114.83 秒，两轮共享覆盖拟合共 121.51 秒。窗口已结算 62,914.42 卡秒。[缓存拟合](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/geometry-envelope/coverage/cache/with-minimum-native-decode/costs.json) · [共享覆盖](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/geometry-envelope/coverage/shared/with-sparse-and-prefill-host/costs.json)

**结论与限制**　缓存最短端点的自主预测在复核。标准池两个局部切点与小池集中到达计时已启动。阶段完成仍需新冻结独立留出及其三路验证。

## 2026-11-27 — S05 小池整图定价与缓存自主跨页切点

**范围**　共享库的大批次、缓存域自主推进和两组集中到达的开发结构复核。

**改动**　以两组小池实际批次的宿主测量补全新图范围。将已归入开发集的小池旧图余下八个区间归并到同一次原生三成员 Decode 结果处理。缓存自主预测到上下文为 64 的单成员跨页分配，缓存域仅具备多成员版本的模板守卫。用正式 `predict explore` 在独立可写开发库中按原生队列准备重现该切点。

**验证**　小池集中到达的 117 批全部对齐，324,428 项工作得到成本覆盖。标准池第一个局部切点 41 项原生查询全部合格。分块与标准池集中到达的原始记录在现行模板库下重新完整匹配 903 和 3,727 个动作。缓存独立固定图 52,220 项实测与模型均覆盖，保留的两份自主参考 TTFT MAPE 为 1.79% 和 3.00%，TPOT MAPE 为 2.98% 和 3.95%。小池集中计时耗 185.37 卡秒、提样 93.46 秒，初始化 24.05 秒、预热 84.21 秒、观测 36.36 秒，原生留存 105.10 MiB，未单独采样采集峰值。标准池第一个配对耗 247.01 卡秒，单探针初始化 3.88–5.35 秒、准备回读 2.43–44.73 秒、观测导出 67.85–68.42 秒，留存 11.70 MiB、峰值 14.61 MiB。已结算 63,346.80 卡秒。[共享覆盖](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/geometry-envelope/coverage/shared/with-small-pool-burst/costs.json) · [缓存评分](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/geometry-envelope/development-candidates/cache-minimum-qualified/assessment/cache/) · [现行模板结构回执](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/geometry-envelope/development-candidates/current-template-structure/)

**结论与限制**　标准池第二个切点和小池旧图释放切点封存后合并拟合。缓存按需补采则按正式入口记录全部原生资格和消耗。两个分支均完成三路开发复核后才具备重新冻结条件。[小池释放切点](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/geometry-envelope/host-residual/shared/small-pool-archived-release/plan.json)

## 2026-11-27 — S05 开发回归完成与第三轮独立冻结

**范围**　五个配置的最新双库、三路开发复核和重新抽样。

**改动**　标准池混批与归入开发集的小池释放批次分别完成 13 和 8 项配对查询。缓存用两次局部会话取得单成员跨页模板及其宿主成本，复用同一份结构录制并通过独立来源核验。四份双库以 242 份模板来源回执绑定冻结内容。镜像和 Release 按内容标识固定，五组请求由新清单 SHA-256 唯一抽样。

**验证**　共享库六张图共 1,576,475 项工作全部可定价。十组新开发边界匹配 30,899 个结构动作，热库复跑十组均完成。五份独立固定图回放合格，实测和预测覆盖 397,425 项工作。十份留存自主参考的 TTFT MAPE 为 1.40%–5.26%，TPOT 为 1.93%–5.03%，来源均隔离。计入两次配对和两次自动补采，本批消耗 614.63 卡秒，单探针初始化 3.97–6.70 秒、准备回读 2.39–44.40 秒、观测导出 2.03–71.46 秒。合并拟合用时 62.54 秒，自动补采后处理 17.43 秒，窗口累计已结算 63,961.43 卡秒。包装命令的参数展开和预算字段读取已修正，入场核验直接读取同一发布的实际回执。[开发入场证据](../../../data/large-scale-validation/holdout/rounds/s05-r3/freeze/s05/dev-readiness.json) · [扣账与自动补采](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/geometry-envelope/development-candidates/cache-automatic-short-page/exploration/result.json)

**结论与限制**　第三轮清单 `25d22138716b7a735df60b93e00cd2e4d6ed2702c0e2bc9aa1a1f98e15683e6b` 已生效。五组请求指纹与两轮留出及存量开发输入均有区分。逐文件读回核对覆盖 340 个源码文件、1,045 个库文件、242 份模板来源和双镜像中的 3 处已改安装文件。逐条扣除已证实的设备运行开销后，本轮前至少剩余 11,154 秒窗口后处理预算，第三轮采用已冻结的 8,902 秒上限。物理参考和三路独立结果待结算。[抽样回执](../../../data/large-scale-validation/holdout/rounds/s05-r3/results/s05/draw-integrity.json) · [冻结资产核对](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/geometry-envelope/round-three-frozen-asset-audit.json) · [宿主费用复核](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r2/processing-budget-before-r3.json)

## 2026-11-27 — S05 第三轮物理参考封存

**范围**　五组未见负载的详细执行参考及轻量延迟参考。

**改动**　每个负载完成两次独立原生录制，分别供参考图和自主性能对照。

**验证**　十次录制全部成功，来源互异且与前两轮及训练录制无重合。详细录制耗 931.74 卡秒，单组初始化 25.69–26.38 秒、预热 67.40–102.11 秒、观测 24.90–75.03 秒。轻量录制耗 745.92 卡秒，单组初始化 22.08–23.27 秒、预热 68.41–102.19 秒、观测 3.77–13.42 秒。参考封存时第三轮目录占约 1.48 GiB，窗口累计结算 65,639.09 卡秒。[详细参考](../../../data/large-scale-validation/holdout/rounds/s05-r3/results/s05/detailed.json) · [轻量参考](../../../data/large-scale-validation/holdout/rounds/s05-r3/results/s05/light.json)

**结论与限制**　物理录制齐备，正在生成参考图。结构与性能门槛等待三路验收结果。

## 2026-11-27 — S05 第三轮参考图齐备与验证调度修正

**范围**　冻结参考图生成及三路离线验证任务调度。

**改动**　提前并行启动分块对照时，结构任务因 7,260 秒最高预约与其他任务叠加而被门禁拒绝。保留零费用的启动拒绝回执。等原构图任务正常退出后，按结构先行、固定图和自主对照随后运行的顺序接管。相关任务沿用相同冻结输入、校验器和预算。[拒绝回执](../../../data/large-scale-validation/holdout/rounds/s05-r3/logs/s05/scheduling/early-chunk-verify-reservation.json)

**验证**　五份参考图均成功生成。分块先行固定图的 70,002 项工作均有实测和模型覆盖，实测边界重放合格。先行自主 TTFT MAPE 为 1.25%，TPOT MAPE 为 2.44%，两路来源均核验隔离。结构任务已按串行顺序获得预约。[参考图](../../../data/large-scale-validation/holdout/rounds/s05-r3/results/s05/targets.json) · [分块固定图](../../../data/large-scale-validation/holdout/rounds/s05-r3/results/s05/fixed/chunk-8/result.json)

**结论与限制**　五组结构和余下性能结论待完成。未来调度前需按各类任务的预约上界选取可并行阶段，阶段总费用仍用实际结算。

## 2026-11-27 — S05 第三轮结构对照齐备

**范围**　五组原生参考的结构注入与自主预测。

**改动**　结构按既定冻结资产完成串行对照，随后进入性能对照。

**验证**　五组共匹配 12,841 个结构动作，来源隔离均通过，结构对照合计耗时 775.49 秒。分块自主 TTFT、TPOT MAPE 为 1.25%、2.44%，缓存为 0.85%、0.96%。混批、标准池组合和小池组合的自主运行终止于成本缺项。[结构汇总](../../../data/large-scale-validation/holdout/rounds/s05-r3/results/s05/comparisons.json) · [自主汇总](../../../data/large-scale-validation/holdout/rounds/s05-r3/results/s05/autonomous.json)

**结论与限制**　固定图工作覆盖尚待全部结算，之后统一归因固定图与自主两条成本路径的缺口。

## 2026-11-27 — S05 第三轮结算与原生批次成本归因

**范围**　五组冻结留出和三组失败用例的历史证据。

**改动**　封存第三轮验收结果，三组失败用例转为开发用途。对其真实图和自主停点一次枚举，共有 1,324 个去重成本坐标超出现有实测范围。三组自主停点均与相应固定图中的工作身份和特征相同。复用原始设备观测较快扩充成本证据，宿主 22 次原生调用先合并为三次同请求配对采集。[开发归属](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r3/classification.json) · [成本缺口](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r3/coverage/summary.json)

**验证**　五组结构及固定图实测基线均通过，615,585/628,321 项工作有预测成本，覆盖 97.97%。分块和缓存完整通过，其余三组的首个缺口分别为短上下文混批 Attention 宿主工作、七成员 Decode 准备、双成员短 Prefill 分配。第三轮共耗 1,677.67 卡秒、2,030.22 秒后处理，结算时留存约 3.36 GiB。窗口累计结算 65,639.09 卡秒。[第三轮验收](../../../data/large-scale-validation/holdout/rounds/s05-r3/results/s05/acceptance-summary.json)

**结论与限制**　第三轮总体 2/5，L1 性能门槛尚未达到。离线复用设备样本新增 2,994 项覆盖，余下 64 字节搬运点通过就绪缓冲区采样，花费 13.70 卡秒。混批与标准池计时分别耗 149.51 和 157.43 卡秒，对齐全部 41 和 42 批。混批新旧四图的 638,532 项工作均已覆盖，独立开发性能复核已启动。小池对齐和共享库合并拟合待结算。[设备覆盖复算](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r3/coverage/with-device/) · [拷贝回执](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r3/local-copy/local/result.json)

## 2026-11-27 — S05 失败例开发回归与槽位边界实测

**范围**　第三轮转开发的三组负载，以及混批和标准池的纯 Decode 槽位上限。

**改动**　归并新旧图的宿主、设备和拷贝样本，发布混批及共享开发候选。以原生可接收 8 个请求和声明的短提示生成两组边界负载。现行混批的纯 Decode 调度样本上界为 6，标准池新观测为 7。将 64 字节同运行时拷贝录制按混批域重新核验，取得 7 次合格观测，新增卡时为零。

**验证**　小池宿主计时耗 198.09 卡秒，对齐全部 92 个原生批次。共享库新旧八图的 1,953,033 项工作全部覆盖。九份留存自主参考的 TTFT MAPE 为 0.37%–3.16%，TPOT 为 1.20%–5.39%，来源隔离均通过；三份独立开发固定图共 226,660 项工作均通过实测重放与模型覆盖。两次槽位边界详细录制共耗 354.37 卡秒。边界任务首发因路由参数展开错误被 CLI 拒绝，已保留原回执并修正再发，拒绝前无原生运行。[候选回归](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r3/candidates/) · [拷贝复用](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r3/capacity-envelope/mixed-copy-reuse/receipt.json)

**结论与限制**　开发回归覆盖第三轮的三个失败用例。混批原生时序观测到 14 次八成员纯 Decode。标准池遇到分片的后续 Prefill，在最早请求结束后才进入纯 Decode，实到七成员，已启动合法前缀命中校准。四份边界原生录制完成，来源核验入口的路径问题已修正，沿原录制离线恢复。新开发请求有两组触发同一批量 KV 释放调用的成本缺口，已启动正式按需采集。新的独立留出尚未开始。[标准池校准](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r3/capacity-envelope/combined-cache-hit/design.json) · [批量释放回归](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r3/grouped-release-dev/)

## 2026-11-27 — S05 批量 KV 释放回归及八成员 Decode 计时定位

**范围**　分块与缓存新增开发请求，及混批、标准池的纯 Decode 槽位边界。

**改动**　正式按需入口补采两种原生批量释放和各自的就绪搬运。标准池在自然前缀命中后取得两批八成员纯 Decode。对混批边界图的 359 个宿主缺口按工作定义和拟合区间归并，用原生输入读取顺序定位五个定点切口。完整图仍用于拟合后覆盖复核。[混批切点计划](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r3/capacity-envelope/mixed/host-cuts/plan.json)

**验证**　两种释放的来源均合格，开发结构匹配 903 和 1,386 个动作，两份固定图的 170,765 项工作全部覆盖，四个热预测零新增卡时。四次按需采集共 300.81 卡秒，原生保留 3.56 MiB，回执记录单次峰值 1.80–5.37 MiB，配对探针初始化 5.10–7.39 秒、准备及回读 2.38–44.32 秒、观测及导出 1.54–26.15 秒，采集后处理 30.64 秒。标准池前缀命中详细和宿主录制共 297.65 卡秒，初始化 24.71–26.34 秒、预热 67.43–69.07 秒、观测 9.27–18.72 秒，原生保留 138.68 MiB，未单独采峰值，投影和有效提取墙时共 106.67 秒。累计已结算 67,412.44 卡秒。[分块回归](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r3/grouped-release-dev/chunk/final-checks/) · [缓存回归](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r3/grouped-release-dev/cache/final-checks/)

**结论与限制**　标准池两次八成员批次只出现在详细录制。宿主配对只匹配 11 个批次，八成员仍缺计时资格。保留详细图的原生输入记录，用确定的批次切口采计时，避免再次依赖独立完整服务碰到相同批次。冻结留出等待开发回归和来源隔离齐备。[标准池来源](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r3/capacity-envelope/combined-cache-hit/qualification/source-evidence.json) · [宿主配对回执](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r3/capacity-envelope/combined-cache-hit/capacity-receipt.json)

## 2026-11-27 — S05 定点采集结算与短 Prefill 归因

**范围**　两条容量边界开发线，以及按独立验收声明生成的两批新开发请求。

**改动**　用九个局部切口补测未配对的宿主区间，按完整开发图重新拟合。标准池与小池的三个自主停点共享同一工作定义，均为双成员、总新增 3–6 token 的 Prefill 分配，原样本覆盖 14–30 token。沿该定义使用两个单 token 缓存尾段测量下界，并定点采集小池 Attention 的 48 token 缓存上下文加 29 token 尾段。[八成员计时回执](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r3/capacity-envelope/eight-decode-qualified.json) · [开发停点](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r3/capacity-envelope/verified/readiness/)

**验证**　九次原生会话共 478 个查询全部合格。混批五张完整图和共享库十张完整图共 2,888,930 项工作全部覆盖。新增原生费用 1,233.18 卡秒，单探针初始化 3.85–7.37 秒、准备及回读 2.37–44.47 秒、观测及导出 1.60–35.71 秒。原生保留 27.01 MiB，单会话采样峰值 11.42 MiB，采集外墙时不超过 29.11 秒。短 Prefill 与小池 Attention 首发被编排到同一物理卡，停止后者于观测前，保留其 22.53 卡秒账目，安排同卡顺序复试。累计已结算 68,668.15 卡秒。[定点采集账本](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r3/capacity-envelope/targeted-cuts-ledger.json) · [整图覆盖](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r3/coverage/with-targeted-cuts/)

**结论与限制**　混批两批声明范围开发请求均完成自主预测。混批槽位校准负载的自主路径还出现了真实图之外的低 Attention 工作量坐标，已按其调用定点采集。共享库的四个新开发预测停点共归到两个原生调用，补样资格和回归仍需确认。第四轮独立留出尚未启动。

## 2026-11-27 — S05 短批次补采回归与前向切口

**范围**　缓存命中后的双成员 Prefill 分配、单成员长尾 Prefill Attention、混批低工作量 Attention。

**改动**　三处定点补采均用带原生输入绑定的查询，继承已有样本来源隔离。两成员各新增一个 token 的分配及结果处理切口已实测，后续显露的 `Scheduler.run_batch` 缺口在同一几何下改用模型前向切口，不再重跑完整服务。[短批次账本](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r3/capacity-envelope/short-batches/ledger.json)

**验证**　三份配对实测合格，混批两批新请求和槽位校准均完成。混批三份留存自主对照 TTFT MAPE 为 0.81%–2.88%，TPOT 为 2.09%–5.12%，固定图 77,477 项全部覆盖。小池第二批也完成。三份合格采样加一次启动中止耗 478.39 卡秒，初始化 3.93–6.78 秒、准备和回读 2.40–43.60 秒、观测及导出 4.63–45.88 秒，原生保留 18.86 MiB、峰值 14.78 MiB，计费外墙时不超过 11.21 秒。混批与共享拟合分别完成，最新已结算 69,124.01 卡秒。[开发回归](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r3/capacity-envelope/after-short-batches/) · [拟合](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r3/coverage/with-short-batches/)

**结论与限制**　剩下两个标准池开发负载和一个小池开发负载的首个停点是同一前向工作定义。实际前向采样和更新后的共享库回归已调度。第三轮独立验收维持 2/5，第四轮等待共享库开发回归结算。

## 2026-11-27 — S05 短 Prefill 前向结算与标准池按需回归

**范围**　双成员短 Prefill 的宿主与设备阶段，及标准池剩余开发用例。

**改动**　将合格的前向切口配对宿主计时加入共享库。回归定位到同切口的搬运成本后，按采集回执校验并复用同次采集的设备样本。标准池剩余两个停点合用同一个 `predict explore` 增量发布任务，保留迭代中每份原生取证及发布身份。[前向回执](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r3/capacity-envelope/short-batches/shared-min-forward/acquisition/result.json) · [按需任务输入](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r3/capacity-envelope/complete-combined/exploration.json)

**验证**　前向切口取得 235 个宿主样本和 69 个设备侧样本，其中搬运 12 个。两探针初始化 3.86–5.43 秒、准备及回读 2.41–42.74 秒、观测及导出 14.90–19.88 秒，原生保存约 10.39 MiB，采集峰值约 13.86 MiB。采集一次耗 142.83 卡秒，采集外墙时不超过 4.15 秒，累计已结算 69,266.84 卡秒。十张完整开发图共 2,131,304 项工作全部可定价，标准池与小池的固定图基线各自合格。两份留存自主开发对照均满足双指标 6% 以内，来源隔离通过。[固定图与评分](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r3/capacity-envelope/after-forward-transport/shared/final-checks/) · [拟合覆盖](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r3/coverage/with-forward-transport/shared/costs.json)

**结论与限制**　两组小池与一组标准池新开发请求均完成。另一组标准池在 `RadixCache.match_prefix` 需要 48 token 的分裂样本，八成员校准负载需要总上下文 435 token 的 Decode Attention 宿主样本。开发证据中未找到相同工作定义与坐标的现成样本，已启动受预算约束的自动补采。第三轮独立结果保持原判，第四轮待共享库回归完成。[未覆盖坐标](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r3/capacity-envelope/after-forward-transport/readiness/)

## 2026-11-27 — S05 开发回归收口与第四轮冻结

**范围**　标准池剩余的开发负载、八成员校准，以及第四轮独立留出的入场核验。

**改动**　使用正式按需入口顺序补采前缀匹配、混批和 Decode Attention、两种批量 KV 释放。用最终共享发布复跑两组目标，并重验另一组标准池与两组小池开发请求。五份新增模板来源均独立核验。正式冻结五域覆盖的四份发布、双镜像、340 个源码文件和 C++ 二进制。[开发发布与验证](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r3/capacity-envelope/complete-combined/)

**验证**　自动补采五次均合格，共 756.84 卡秒，内部后处理 94.30 秒，外层总墙时 1,031.93 秒，探索保留约 44.20 MiB。五次初始化、准备、观测及峰值均在原生回执中留存。最终发布的 1,591 个原有成本绑定均保留覆盖范围，四个绑定改变拟合族。留存的标准池、小池自主对照 TTFT MAPE 分别为 1.82%、0.24%，TPOT 分别为 3.02%、1.12%，来源隔离通过。五组第四轮请求指纹与三轮已知输入和开发集不同。[开发结果](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r3/capacity-envelope/complete-combined/exploration/result.json) · [冻结证据](../../../data/large-scale-validation/holdout/rounds/s05-r4/freeze/s05/dev-readiness.json) · [抽样核验](../../../data/large-scale-validation/holdout/rounds/s05-r4/results/s05/draw-integrity.json)

**结论与限制**　第四轮已进入物理录制，尚无评分。窗口已结算 70,023.68 卡秒。冻结前保守剩余后处理 1,910 秒，投影扣账后默认结构对照的 1,660 秒预约可能提前拒绝正常任务。已保留录制和投影，暂控制三条评分入口，依据真实回执复核账目并安排按实际预算逐项执行。冻结资产及抽样身份继续有效。[冻结环境](../../../data/large-scale-validation/holdout/rounds/s05-r4/freeze/s05/environment.json) · [详细录制日志](../../../data/large-scale-validation/holdout/rounds/s05-r4/logs/s05/detailed.log)

后处理核账已用原生执行和合格回执确认，旧单 token 补采的 121.36 秒外层墙时中，116.79 秒为已计卡时的原生运行，先前未从宿主上界中扣除。保守可用量由 1,910 秒纠正为 2,026 秒。补充记录绑定原冻结摘要，仍以原清单抽样。[核账补充记录](../../../data/large-scale-validation/holdout/rounds/s05-r4/logs/s05/processing-audit.json)

继续核对旧窗口的缓存、混批模板及缓存搬运三次录制。外层计时、完成日志、原生执行和合格回执绑定同一会话，另有 204.21 秒已计卡时的原生运行仍在宿主上界中。保守可用后处理为 2,231 秒。第四轮评分接管程序已排队，在真实参考和投影全量结算后调用冻结矩阵，仅按实际余量收紧每条命令的外层预约和超时。标签、来源资格、固定图实际时长和评分阈值沿用冻结声明。[三次旧录制核账](../../../data/large-scale-validation/holdout/rounds/s05-r4/logs/s05/processing-audit-sessions.json) · [评分接管程序](../../../data/large-scale-validation/holdout/rounds/s05-r4/logs/s05/serial-scoring.py)

## 2026-11-27 — S05 第四轮详细参考封存

**范围**　第四轮五组冻结留出的完整服务详细录制。

**改动**　封存五份独立单卡参考，接续轻量时延录制。[详细录制回执](../../../data/large-scale-validation/holdout/rounds/s05-r4/results/s05/detailed.json)

**验证**　逐组核对模型、实例配置和请求与冻结输入一致。五个录制实例各一次成功，与 249 份已校验摘要的冻结来源证明分离。共 927.71 卡秒，含初始化 130.56 秒、预热 410.50 秒、有效观测 179.76 秒，其余为录制外围开销。[冻结清单](../../../data/large-scale-validation/holdout/rounds/s05-r4/freeze/s05/source-manifest.json)

**结论与限制**　累计已结算 70,951.38 卡秒。轻量参考、目标投影及评分仍待完成。[轻量录制日志](../../../data/large-scale-validation/holdout/rounds/s05-r4/logs/s05/light.log)

## 2026-11-27 — S05 第四轮完成双参考录制

**范围**　五组轻量时延参考、两类参考的来源隔离及录制核账。

**改动**　五组轻量录制一次成功。按录制身份和时间回执归档双参考，启动详细记录投影。[录制账本](../../../data/large-scale-validation/holdout/rounds/s05-r4/logs/s05/reference-accounting.json) · [投影日志](../../../data/large-scale-validation/holdout/rounds/s05-r4/logs/s05/targets.log)

**验证**　十次实采的模型、请求、部署及采集模式与冻结声明一致。轻量记录覆盖全部 96 个请求的首末 token 和输出长度。录制身份与 249 个模板来源、403 个拟合来源及前三轮 30 个录制身份无交集。工作区的 340 个运行源码文件均与冻结摘要一致，Ruff 通过。

**结论与限制**　轻量参考耗 744.60 卡秒，双参考共耗 1,672.30 卡秒和 10.31 秒录制外围宿主处理，持久化原生目录约 1.14 GiB。累计已结算 71,695.98 卡秒。目标投影与三路评分尚未结算。

## 2026-11-27 — S05 第四轮目标图完成并进入评分

**范围**　五组详细参考的图投影及三路评分交接。

**改动**　五组均完成 SDK 导出、解释和图投影。评分已按结构、固定图、自主预测的顺序接续。[目标图账本](../../../data/large-scale-validation/holdout/rounds/s05-r4/logs/s05/projection-accounting.json)

**验证**　逐组核对原始录制身份、图路径、冻结镜像和二进制摘要。评分交接回执与已完成的双参考、目标图和预算核账摘要一致，核对首条结构命令输入与对应目标一致。[评分交接](../../../data/large-scale-validation/holdout/rounds/s05-r4/logs/s05/serial-scoring-handoff.json)

**结论与限制**　投影结算 503.76 秒宿主时间，含录制外围处理后，评分起始余量为 1,716.93 秒。结构及计时结果须等三路评分结算。[评分日志](../../../data/large-scale-validation/holdout/rounds/s05-r4/logs/s05/serial-scoring.log)

## 2026-11-27 — S05 第四轮结构对照通过

**范围**　五组冻结用例的独立结构对照，以及已封存时延参考的指标覆盖。

**改动**　完成五组实测条件下的执行结构对照，接续固定图验证。[结构结果](../../../data/large-scale-validation/holdout/rounds/s05-r4/results/s05/comparisons.json)

**验证**　五组均达到 `full_structure_matched`，共匹配 12,235 个动作，条件覆盖 585,834 项图上工作。逐组核对原生来源资格、模板来源隔离、实测条件消费及联合时钟。工作结束点缺测为 0，实测条件保留校准范围。轻量参考的 96 个请求在 TTFT 与 TPOT 上均有正分母。[结构核验账本](../../../data/large-scale-validation/holdout/rounds/s05-r4/logs/s05/structure-accounting.json)

**结论与限制**　结构对照结算 732.04 秒宿主处理，进入固定图时余量为 984.88 秒。固定图及自主评分仍待达标。[评分日志](../../../data/large-scale-validation/holdout/rounds/s05-r4/logs/s05/serial-scoring.log)

## 2026-11-27 — S05 第四轮独立验收结算

**范围**　五组冻结用例的三路结果、正式结算、失败用例归属。

**改动**　封存整轮原结果，将混批和前缀缓存两例永久转入开发集。按原生队列记录核对混批实测重放首个差异，发现当前图的首个提交操作提前 5.06799 毫秒，SDK 中被用作前提的前序 dequeue 已省略。[开发集分类](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r4/classification.json) · [原生队列证据](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r4/queue-boundary/native-queue-evidence.json)

**验证**　独立验收为 3/5，结构全部通过，固定图实测重放 4/5。成本覆盖 585,806/585,834 项，混批缺一项，前缀缓存缺二十七项。两例自主停点均是 `alloc_for_extend` 的双成员宿主工作，分别需要 25 和 8 个 query token。三组自主评分覆盖 64 个请求，TTFT/TPOT MAPE 依次为 0.41%/1.15%、2.50%/0.84%、1.45%/2.40%。三路来源隔离全部通过。[独立验收](../../../data/large-scale-validation/holdout/rounds/s05-r4/results/s05/acceptance-summary.json) · [性能核验账本](../../../data/large-scale-validation/holdout/rounds/s05-r4/logs/s05/performance-accounting.json)

**结论与限制**　S05 门槛尚未达到。三组完整固定图在替换预测成本后的 TPOT MAPE 为 24.33%–61.99%，保留作归因证据。第四轮用量为 1,672.30 卡秒、1,969.88 秒宿主处理，约 3.18 GiB 保留量。第四轮结算时全窗口保守宿主余量为 261.12 秒，后续开发实际用量另计。

## 2026-11-27 — S05 开发回归修复队列重放

**范围**　归为开发集的混批原生录制和两张成本不全的开发图。

**改动**　队列交接以投影图保留的同线程前序为准。无前序时将相同队列关联号的实测提交到取出间隔保留在依赖上。用原录制、冻结二进制和相同成本库运行正式固定图入口。[原生队列证据](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r4/queue-boundary/native-queue-evidence.json)

**验证**　混批的 151,054 个实测边界全部重放匹配，首个 token 与主动读取差异均归零。Ruff 通过。两张图由现有 C++ 求值器整体枚举，28 个未覆盖坐标归于两批。混批为双成员 25 token 的分配宿主工作。前缀缓存为双成员 8 token 的分配工作和同批 `Scheduler.run_batch` 的 26 个工作区间。[混批开发重放](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r4/queue-boundary/assessment/result.json) · [成本坐标清单](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r4/cost-inventory.json)

**结论与限制**　新增真机用量为 0，已计时的原生证据检查、开发重放及两次成本清单命令共耗 129.87 秒宿主处理。保守账面余量为 131.25 秒，另为未独立计时的诊断预留 10 秒。原独立结果保持 3/5，成本补齐和新冻结验收待完成。

## 2026-11-27 — S05 两个批次成本闭合

**范围**　混批双成员 25 token 和前缀缓存双成员 8 token 的开发成本覆盖。

**改动**　合并检查 663 份带原生身份的去重样本表，其中 296 份在两目标域内。两批缺口没有可直接复用的测量，使用正式配对局部配方采集混批分配、缓存模型执行和缓存分配。沿用原模板与既有训练表构建两份开发候选。[样本复用核对](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r4/sample-reuse-audit.json) · [局部采样核验](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r4/qualified-local-coverage.json)

**验证**　28 个坐标均有同域实测观测。正式 C++ 求值器确认两张图的 160,073 项工作均可定价。原 2,747 个成本绑定均保留，已声明的采样区间无收窄，253 个模板文件及 268 个训练样本文件逐项核对摘要。前缀缓存 16 请求自主回归达到 2.11%/4.98% TTFT/TPOT MAPE，来源隔离通过。[自主开发回归](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r4/cache-autonomous-regression/assessment/result.json) · [候选与账本](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r4/targeted-qualification-accounting.json)

**结论与限制**　3 次采集共 406.59 卡秒、9.04 秒外围宿主处理。样本检查 19.63 秒，拟合及覆盖求值 44.05 秒，缓存自主回归 29.57 秒。累计 72,102.57 卡秒，保守宿主余量为 13.96 秒。第四轮原分数保留为 3/5。混批自主回归及五组新冻结验收仍缺实测结果。

## 2026-11-27 — S05 混批自主回归完成与宿主账目复核

**范围**　归入开发集的混批自主回归及窗口宿主处理余额。

**改动**　逐条复核旧账中三次 Docker 构建的日志，原计时 52.66 秒记入环境准备。给核账预留 2 秒，回归前保守可用后处理为 64.62 秒。用新的混批库和同一份留存轻量参考走正式自主验证入口。[构建账目](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r4/host-accounting-build-correction.json)

**验证**　混批 16 个请求均进入两个指标统计，TTFT/TPOT MAPE 为 0.96%/1.77%，来源隔离通过。混批和前缀缓存的实测重放、全图定价和自主回归均有对应开发证据。第四轮的独立分数仍为 3/5。[两例开发回归](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r4/development-regression-evidence.json)

**结论与限制**　混批回归消耗 32.24 秒宿主处理和 0 卡秒。当前保守余量为 32.38 秒。此前同规模五组独立验收实际消耗 1,969.88 秒宿主处理，其中投影为 503.76 秒。新一轮独立验收尚未开始。

## 2026-11-27 — S05 第五轮资产预核对

**范围**　四份候选库、模板来源、运行资产和两张既有开发图的成本覆盖。

**改动**　确认分块和共享库继续使用已核验发布，混批和前缀缓存使用配对补采后的新发布。保留原五组 8、16、16、24、32 请求的抽样规范。[候选与抽样规范摘要](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r4/round-five-preflight/readiness.json)

**验证**　核对 249 份模板来源证明、三份镜像身份、C++ 二进制和 340 个运行文件。相对第四轮的源码差异为队列交接修正。用两份新成本库复算先前已通过的两张 12 请求开发图，共 129,697 项工作均有预测成本。

**结论与限制**　预核对耗 8.74 秒宿主处理、0 卡秒，保守余量为 23.64 秒。上一轮五组中最短的单例投影耗 56.33 秒。第五轮尚未冻结、抽样或实采，完整独立验收仍需宿主额度。[预算核对](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r4/round-five-preflight/accounting.json)

## 2026-11-27 — S05 独立验收等待额度授权

**范围**　第五轮五组独立验收的宿主后处理额度。

**改动**　记录窗口阻断条件和可用于下一次冻结的四份候选。[候选证据](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r4/round-five-preflight/readiness.json)

**验证**　窗口上限仍为 36,000 秒，保守余量 23.64 秒。上一轮同规模实际消耗 1,969.88 秒。第五轮尚未启动，当前无运行中的本窗口实验。[额度账本](../../../data/large-scale-validation/dev/l1/s05/retired-holdout-s05-r4/round-five-preflight/accounting.json)

**结论与限制**　等待新的宿主后处理额度授权。接续时仍按五组新请求和三条验证路径验收，完成后才进入 S06。

## 2026-11-27 — S05 同会话端点与第五轮盲测启动

**范围**　同会话局部端点补采、冻结盲测以及第五轮独立验收交接。

**改动**　同步拷贝和 Decode Attention 从真实超界轴选择配置允许的端点，原生读取成功后才把时延加入样本表。验证矩阵增加逐例独立的正式盲测发布，先预测及补采，再开放完整参考录制，三路核验使用相同的最终发布。

**验证**　沿用开发图的 16 字节拷贝缺项，一次同步拷贝启动实测 16 和 64 字节点，每点读回核对，消耗 14.50 卡秒。第五轮核对三份镜像、二进制、343 个运行文件、四份初始库和 249 份模板证明，并从冻结清单抽取五组与旧留出不同的请求。盲测五例均完成，四例首次热命中。小池场景的双成员 31-token 分配和前向宿主时延超出初始采样范围，正式探索自动补采两次，消耗 331.17 卡秒和 39.51 秒采集后处理，发布新库后完成预测。C++ 对新增拷贝样本覆盖 64 字节，仍拒绝 72 字节。[端点实采](../../../data/large-scale-validation/dev/l1/s05/endpoint-envelope/copy/acquisition/sessions/sample/session/result.json) · [请求抽取凭证](../../../data/large-scale-validation/holdout/rounds/s05-r5/results/s05/draw-integrity.json)

**结论与限制**　启用 240,000 秒宿主总上限，冻结时可用约 204,023 秒。完整参考录制在五例盲测封存后启动，结构、固定图和自主评分待出。小池真实缺口属于调度器宿主切点，本轮按现有配对探针精确补采。同步拷贝端点已完成真机与 C++ 范围核验，Decode Attention 的新跨度仍需真机证据。[盲测发布及账本](../../../data/large-scale-validation/holdout/rounds/s05-r5/results/s05/blind.json) · [C++ 范围核验](../../../data/large-scale-validation/dev/l1/s05/endpoint-envelope/copy/bounded-cost-check.json)

## 2026-11-27 — S05 第五轮独立留出结算与小池归因

**范围**　五例冻结独立留出、盲测增量来源和小池性能差异。

**改动**　冻结期沿用五份独立发布。小池两个新增本地来源分别通过核验，三路统一使用盲测最终发布。小池 32 请求正式转入开发集，原评分保留。

**验证**　5/5 结构匹配、实测固定图基线对齐，三路训练隔离 15/15，通过 4/5。小池 TTFT/TPOT MAPE 2.45%/6.59%，32 个请求的 TPOT 均高估。小池局部补采 331.17 卡秒，详细与轻量完整参考合计 436.84 卡秒，占比 75.8%。同步拷贝与 Attention 的端点在开发集通过真机和 C++ 范围核验。第五轮共 2,009.51 卡秒、2,373.21 秒后处理，另有 182.62 卡秒轻量批次诊断和 202.15 卡秒宿主耗时诊断。[正式结果](../../../data/large-scale-validation/holdout/rounds/s05-r5/results/s05/acceptance-summary.json) · [证据账本](../../../data/large-scale-validation/dev/l1/s05/round-five-diagnostics/round-five-record.json)

**结论与限制**　89 批的原生与预测调用顺序完全一致。Decode 和 Mixed 前向中位高估 1.88、2.31 ms，批次间隔高估 0.82、1.13 ms。按会话汇总重复样本或追加同卡现行宿主样本仍分别为 6.45%、6.47%。宿主观察相同批次测得约 5 ms 附加墙时，Python 与队列标记已知观察区间约 2.4–2.7 ms。剩余墙时包含观察器扰动和独立运行波动，仍需用低扰动采样区分。S05 整轮门禁待关闭。[批次对照](../../../data/large-scale-validation/dev/l1/s05/round-five-diagnostics/small-pool/batch-attribution.json) · [宿主观察归因](../../../data/large-scale-validation/dev/l1/s05/round-five-diagnostics/small-pool/host-observer-diagnostic.json)

## 2026-11-27 — S05 宿主采样及拟合复核

**范围**　小池 32 请求的稳定时延偏差及三路验收之后的开发对照。

**改动**　分别检查重复样本权重、同卡现行宿主样本、同容量原始样本和采样器跨线程争用。线程独立缓冲用正式双镜像构建并实采复核，结束后撤回该试验改动，保留补丁及证据。

**验证**　同池 10 张训练表提供 19,329 个宿主坐标。选用相同容量的实测行后，小池开发 TPOT MAPE 为 6.57%。线程独立缓冲保持 89 批同序、208,160 条队列标记、76,694 次 API 记录及零线程活动缺测。扣除已测观察区间后的墙时在 Mixed 和 Decode 分别高 2.58、2.68 ms，该试验消耗 199.31 卡秒。窗口累计已结算 74,780.96 卡秒。[复核与证据摘要](../../../data/large-scale-validation/dev/l1/s05/round-five-diagnostics/repair-decision.json)

**结论与限制**　样本聚合为 6.45%，新增同卡宿主证据为 6.47%，同池选择为 6.57%，均未形成达标候选。以上数字属于开发对照，第五轮冻结分数仍为 4/5。对照定位了前向与批次间的性能偏差，现有证据还无法把各次运行的差异精确归到具体低层服务。S05 保留该门禁，L2 待放行。

## 2026-11-27 — S05 批次 CPU 归因与观测边界修正

**范围**　第五轮已退役的小池 32 请求案例及原生调度分段的宿主成本取证。

**改动**　现有轻量批次探针增加调度线程与进程 CPU 时钟。共享调用分段在父段关闭前、子段开启后显式记录适配器构造上下文、维护分段和编码状态查询的区间，由已有宿主对齐算法扣除这些区间。

**验证**　保留录制与新批次录制的 269 次调用完全同序。混批选批和结果处理的预测墙钟中位高估为 1.09 和 0.43 毫秒。纯时延复测的预测请求数据逐项一致，两次原生平均 TPOT 分别为 72.76 和 76.47 毫秒。两次诊断合计 366.95 卡秒，初始化 45.72 秒、预热 216.03 秒、有效观测 30.89 秒。三项批次分析耗 19.02 秒，另行执行正式自主评分。推理与控制镜像均重建，安装探针哈希与源码吻合。[调度归因](../../../data/large-scale-validation/dev/l1/s05/round-five-diagnostics/small-pool/cpu-intervals/scheduler-fragments.json) · [两份轻量参考](../../../data/large-scale-validation/dev/l1/s05/round-five-diagnostics/small-pool/cpu-intervals/independent-light-repeat.json)

**结论与限制**　同一预测在两份纯时延参考上的 TPOT MAPE 为 6.59% 和 1.55%。第二份属于开发参考，保留首份冻结门禁结果。进程 CPU 时间覆盖全部线程，用于诊断。插桩边界修正仍需以同镜像的独立结构与 CPU 录制核对可定价覆盖和成本。

## 2026-11-27 — S05 配对原生计时覆盖同步拷贝

**范围**　小池的宿主采样归属和同步拷贝成本，沿用两份纯时延参考及两项既有开发场景。

**改动**　同一原生调用的成对起止、字节数和方向用来提取同步拷贝成本。对同工作坐标和物理操作数优先采用较直接的测量。状态读取进一步在原生操作边界计时，以独立验证读取本身和编排开销的归属。

**验证**　同镜像结构与 CPU 录制匹配 89 批，独立来源核验通过，结构全匹配 5,143 个动作。原成本对两份宿主录制的 Decode 同步拷贝分别高估约 0.99、1.11 毫秒。复用录制配得 6,254 次拷贝，另有 737 项线程传输缺少本次配对证据。联合重拟合在首份和复测参考上的 TPOT MAPE 分别为 6.16% 和 1.28%，两项既有开发场景为 2.23% 和 0.80%，四路均保持来源隔离。[结构对照](../../../data/large-scale-validation/dev/l1/s05/round-five-diagnostics/small-pool/observer-attribution/structure-check/result.json) · [开发评分](../../../data/large-scale-validation/dev/l1/s05/round-five-diagnostics/small-pool/observer-attribution/paired-copy/development-results.json)

**结论与限制**　新增配对原生采集消耗 454.20 卡秒。结构投影、核验及宿主提取耗 382.16 秒，复用采集提取拷贝耗 73.71 秒。首份参考仍高于门槛，第五轮留出原分数维持 4/5。直接计时的读取证据待实采收口。[真机账本](../../../data/large-scale-validation/dev/l1/s05/round-five-diagnostics/small-pool/observer-attribution/native-accounting.json)

## 2026-11-27 — S05 直接原生调用取证与第六轮冻结

**范围**　状态读取和模型调用的宿主服务边界，及五组独立留出的启动。

**改动**　宿主适配器以原生函数入口及返回记录服务区间。拟合器以逐项证据优先级选择同坐标和同操作数的实测。冻结保留四份原模板及其 249 项来源核验，只更新共享库的成本发布。

**验证**　8 请求两份录制匹配 21 批，导出 238 次直接状态读取和 1,956 次追加配对拷贝，原生来源核验通过。四份既有库的原样本在冻结和当前拟合代码中的结果一致。最终候选覆盖固定小池图 252,927 项工作。首份和复测参考 TPOT MAPE 为 6.14%、1.26%，另两项开发对照为 2.21%、0.85%，四路来源隔离均通过。定向实采 306.32 卡秒，其中初始化 50.35 秒、预热 147.84 秒、有效观测 27.72 秒。新资产清单绑定 343 个运行文件及三份镜像，已抽取五份与历轮不同的请求。[直接计时](../../../data/large-scale-validation/dev/l1/s05/round-five-diagnostics/small-pool/observer-attribution/exact-native-call/direct-native-boundaries.json) · [固定图覆盖](../../../data/large-scale-validation/dev/l1/s05/round-five-diagnostics/small-pool/observer-attribution/exact-native-call/coverage-check.json) · [抽样清单](../../../data/large-scale-validation/holdout/rounds/s05-r6/results/s05/draw-integrity.json)

**结论与限制**　第五轮原结果与两份开发参考保留。基于已核验的采样修正及典型开发场景启动第六轮，首份参考仍记录超标。第六轮盲测开始时完整物理参考目录尚未创建，五案均须完成原定验收。[第六轮冻结清单](../../../data/large-scale-validation/holdout/rounds/s05-r6/freeze/s05/source-manifest.json)

## 2026-11-27 — S05 第六轮盲测封存

**范围**　五组新请求的隔离盲测与真实参考采集交接。

**改动**　封存五例的预测和库身份，按原冻结配置启动详细真实参考采集。

**验证**　五例首次预测均成功，最终库与初始库身份一致。五条执行记录均无参考挂载，首份参考在盲测封存后启动。盲测共用 168.60 秒宿主处理、零次采集、零卡秒。[盲测回执](../../../data/large-scale-validation/holdout/rounds/s05-r6/results/s05/blind.json)

**结论与限制**　纯热覆盖为 5/5，本轮盲测无新增自动补采实证。结构、固定图和自主精度仍待真实参考出齐后核验。

## 2026-11-27 — 后续阶段交接的静态核对

**范围**　趁 S05 真实参考采集期间，核对 S06、S08 和最终矩阵的实现入口。

**改动**　细化多卡 Decode 切口、物理通信、宿主成本的实施顺序。明确双来源比较复用同一次物理参考，真实流量使用 Azure 连续窗口及隔离行范围，排序组逐组给出比较结果。

**验证**　当前切口目录、局部模板发布及原生 Attention 入口均限制 TP1。现有分片运行提供逻辑 rank 与设备租约。矩阵按录制身份复用图，但验收摘要逐案例拒绝重复参考身份，构造器当前只生成合成请求。[切口目录](../../../src/python/prologue/integrations/frameworks/sglang/shared/configuration.py) · [验收摘要](../../../example/coarse_graph/acceptance.py) · [请求构造器](../../../example/coarse_graph/prepare_matrix.py)

**结论与限制**　本批仅修订后续实施方案。S05 按原冻结运行，L2 实现与实机验证在 S05 通过后开始。

## 2026-11-27 — S05 第六轮真实参考封存

**范围**　五组单卡用例的详细和轻量原生参考。

**改动**　十次录制均在盲测封存后执行，输出保留独立录制身份。按原生回执结算卡时，随后用详细参考生成真实图。

**验证**　十次录制均首次成功。同例两类录制的原生输入逐字节相同，物理卡一致，录制身份互异。两类共耗 1,719.01 卡秒，其中初始化 242.50、预热 856.62、有效观测 229.07、其余原生时间 390.83 卡秒。采集外围宿主处理 10.56 秒，参考文件保留约 1.21 GiB。[参考账本及摘要校验值](../../../data/large-scale-validation/holdout/rounds/s05-r6/logs/s05/reference-accounting.json)

**结论与限制**　窗口累计已结算 77,627.45 卡秒。投影导出、全轮存储峰值和三路验证结果待结算。

## 2026-11-27 — S05 第六轮独立结构核验

**范围**　以详细参考生成五张真实图，配合同例的盲测最终库做独立结构对照。

**改动**　投影并保留 SDK 导出，从详细记录提取执行条件，以冻结的预测入口展开目标结构。结构完成后启动固定图核验。

**验证**　五例均达 `full_structure_matched`，共匹配 12,527 个动作，原生来源一致、执行条件均完成提取，隔离 5/5。投影用 519.96 秒，SDK 导出共 881,997,589 字节。结构核验用 767.79 秒，两步新增卡时为零。[真实图回执](../../../data/large-scale-validation/holdout/rounds/s05-r6/results/s05/targets.json) · [结构报告](../../../data/large-scale-validation/holdout/rounds/s05-r6/results/s05/comparisons.json)

**结论与限制**　结构路径通过。固定图实测基线、预测成本覆盖和自主 TTFT/TPOT 评分仍待出齐。

## 2026-11-27 — S05 验收完成与 L2 交接

**范围**　第六轮五组独立留出的结算与 L1 全局复盘。

**改动**　封存三路报告及逐步账本。将多卡开发按本地设备、物理通信和物理组宿主三个职责展开，沿用冻结盲测和来源隔离流程。

**验证**　五组全过，匹配 12,527 个动作，固定图实测重放和 622,331 项成本覆盖通过，三路隔离 15/15。分块、混批、缓存、标准池组合和小池的自主 TTFT/TPOT MAPE 依次为 1.46%/2.16%、3.69%/2.79%、1.13%/2.68%、1.00%/3.89%、2.30%/3.96%。343 个运行文件和 Release 二进制摘要与冻结一致，Ruff、Python 模块规模、连续代码行和导入图检查通过。[独立结果](../../../data/large-scale-validation/holdout/rounds/s05-r6/results/s05/acceptance-summary.json) · [账本与清单](../../../data/large-scale-validation/holdout/rounds/s05-r6/logs/s05/acceptance-accounting.json)

**结论与限制**　本轮盲测初始库覆盖 5/5，零次自动补采。十次参考共耗 1,719.01 卡秒，宿主处理 2,242.46 秒，全轮保留 3.649 GiB，投影起每 20 秒采样的峰值为 3.683 GiB。固定图预测替换的偏差集中于详细参考宿主时延，保留作归因数据。S05 完成，进入 S06。[第六轮参考账本](../../../data/large-scale-validation/holdout/rounds/s05-r6/logs/s05/reference-accounting.json)

## 2026-11-27 — S06 TP2 分片 Attention 试采

**范围**　贯通多 rank 的 Attention 切口目录、采样规则、原生 ModelRunner 与成本发布。

**改动**　完整生产和局部模板发布都产出逐 rank 切口。探针沿用真实逻辑分片和单卡设备租约，只有目标 rank 执行测量。旧录制缺少当前释放字段，改用新 TP2 录制。首个局部会话中，文件 rendezvous 因缺少父目录停在初始化。补建会话目录后恢复运行，代码由探针在启动 worker 前创建目录。

**验证**　TP2 两个 rank 各生成一项 Attention 切口。单卡试采在 rank 1 的 8、15 token 上通过页/KV 读回，分别匹配四项设备任务，正式局部采样和双库更新均通过。Ruff 通过，首轮镜像内改动文件摘要与当时源码一致。[原生图模板](../../../data/large-scale-validation/dev/l2/s06/attention/tp2-current-production/result.json) · [局部资格](../../../data/large-scale-validation/dev/l2/s06/attention/tp2-local/result.json) · [增量发布](../../../data/large-scale-validation/dev/l2/s06/attention/tp2-publish.log)

**结论与限制**　两卡来源耗 301.21 卡秒，初始化、预热和有效观测为 26.02、67.73、15.72 秒墙时。局部试采耗 198.45 卡秒，初始化 144.77 秒含目录等待，预热 24.26 秒，四次准备、读回与执行共 0.052 秒。模板后处理 137.46 秒，SDK 导出约 161 MiB，局部 SDK 处理 6.25 秒，局部保留约 0.35 MiB、采样峰值约 2.02 MiB。补建目录的试采用于开发，正常启动及后续宽度仍需实测。[来源回执](../../../data/large-scale-validation/dev/l2/s06/attention/tp2-native-source/result.json) · [局部计时回执](../../../data/large-scale-validation/dev/l2/s06/attention/tp2-local/native/result.json)

## 2026-11-27 — S06 TP2 单卡正常入口复核

**范围**　复核更正初始化顺序后的镜像及 TP2 单卡采样和发布。

**改动**　重建推理与控制双镜像，从原模板库新开独立发布。用原两个真实工作查询走正式采样和发布入口。

**验证**　镜像探针内容与源码摘要一致。单张物理卡上两个逻辑 worker 成功初始化。仅 rank 1 执行定向 Attention，8、15 token 均通过读回和任务序列核验，C++ 成本覆盖和双库发布通过。实采一次，耗 94.52 卡秒，初始化 39.99 秒、预热 24.12 秒。SDK 处理 6.27 秒，保留约 0.35 MiB、采样峰值约 2.02 MiB。[局部回执](../../../data/large-scale-validation/dev/l2/s06/attention/tp2-clean/result.json) · [发布回执](../../../data/large-scale-validation/dev/l2/s06/attention/tp2-clean-publish.log)

**结论与限制**　无人工补目录。TP2 用例证实单卡定向执行与成本发布，测试源模板来自两张物理卡的详细录制。TP4/TP8 的本地切点和物理组通信、宿主采样继续取证。

## 2026-11-27 — S06 TP4/TP8 单卡切口与多 rank 物理通信

**范围**　将本地 Attention 取证扩至较宽 TP，并在真实 TP2/4/8 物理组验证同批通信补采。

**改动**　通信配方按原生批次、角色计划和通信操作数合批。每个参与者保留独立的编译后工作身份和成本表，平台时钟与原始端点由同一会话共同取证。

**验证**　TP4 rank 2 的 Attention 为 8 个 Query heads 和 2 个 KV heads，TP8 rank 7 为 4 个和 1 个。两例均在物理卡 4 上通过 8、15 token 的 KV 读回及发布，各耗 94.95、98.89 卡秒，初始化、预热、实际 Decode 和 SDK 处理分项在账本。物理通信三组各一次会话，14 项 AllReduce 查询都由 C++ 成本入口核验并发布，保留 1,022 次 AllReduce 和 14 次 AllGather 观测。[单卡账本](../../../data/large-scale-validation/dev/l2/s06/attention/wider-evidence.json) · [通信账本](../../../data/large-scale-validation/dev/l2/s06/physical-peers/grouped-evidence.json)

**结论与限制**　TP4/TP8 的完整模板来源及单卡局部采集共耗 2,147.39 卡秒，物理通信补采耗 1,768.57 卡秒，结算至 82,137.57 卡秒。完整来源导出约 627 MiB，未采样其峰值。局部单卡峰值低于 2.01 MiB，物理通信最大采样峰值为 178.38 MiB。当前通信发布覆盖一个真实 Prefill 几何，双来源结构、跨配置复用与宿主成本仍待完成。

## 2026-11-27 — S06 物理宿主取证与镜像配对

**范围**　在 TP2/4/8 的物理拓扑独立采集各 rank 宿主时间。

**改动**　按详细图的配置和请求录制宿主时间。启动时统一选取最新镜像，漏查了 TP2 详细来源早于镜像重建。TP2 的替代录制将直接读取详细来源的镜像摘要。

**验证**　三种拓扑的全部 14 个 rank 均封存 RAW 时间，以 4 秒周期采样存储峰值。三次真机共耗 2,062.70 卡秒，原始导出合计约 315 MiB，最大采样峰值约 180.41 MiB。逐宽度的初始化、预热、有效观测、导出及保留量已记账。提取器在 TP2 发现镜像不符并拒绝配对，TP4、TP8 的来源镜像已核对一致。[真机账本](../../../data/large-scale-validation/dev/l2/s06/host-timing/native-accounting.json) · [TP2 拒绝记录](../../../data/large-scale-validation/dev/l2/s06/host-timing/tp2-0-extract.log)

**结论与限制**　累计卡时 84,200.27 卡秒。保留 TP2 首次录制与费用，按详细来源使用的镜像补录该拓扑。原始宿主录制仍须经过逐 rank 配对和成本覆盖检查。

## 2026-11-27 — S06 TP2 通信跨容量复用

**范围**　在物理组、模型、实际批次和通信操作数相同的情况下变更 KV 池与并行请求容量。

**改动**　HCCL 通知先按实测设备时钟排序，再按原生通信组的秩顺序核对发送方、等待方和寄存器。原设备时钟约束仍独立保留。扩容对照只使用原容量的两张通信样本表训练。

**验证**　两个 rank 实际读回 512 页和 8 槽，原容量为 256 页和 4 槽。C++ 成本入口覆盖新录制的 148 次通信观测，两个 rank 按采样点计算的中位相对误差为 6.73% 和 6.18%。一次真机共耗 233.04 卡秒，逐 rank 的初始化、预热与读回、采集与导出分项和保留量已记账，采样峰值 45.11 MiB，原生退出至资格回执用 6.63 秒。[扩容对照及账本](../../../data/large-scale-validation/dev/l2/s06/physical-peers/tp2-larger-pool/accounting.json)

**结论与限制**　累计卡时 84,433.31 卡秒。实证范围为同组同操作数的容量变化。其余几何、错误操作数的拒绝和各 rank 的宿主拟合仍需核对。

## 2026-11-27 — S06 收口与 S07 交接

**范围**　独立宿主资格、物理通知和成本范围复核，及多卡开发回归交接。

**改动**　TP2 按详细录制中的原始镜像摘要补测。通信端点规则收口于平台，原生批次与参与者输入由框架配方提供，核心求值入口沿用原有范围约束。按这一边界展开 S07 的双来源回归。

**验证**　三个宽度的 14 个 rank 各匹配 13 批，647,006 个宿主区间及 13,468 个物理通信区间通过 `cost-points` 和 `work-costs`。旧通信发布用现行通知代码复算的 14 张表摘要一致，未测尺寸、改变规约、更换参与者均按合同拒绝。219 个 Python 模块的引用图、代码规模、函数连续行数及 Ruff 检查通过。宿主补录耗 279.83 卡秒，初始化、预热和观测耗 24.53、66.32、8.86 秒，原始导出 44.66 MiB、采样峰值 44.80 MiB。14 项宿主提取及拟合耗 304.41 秒，通信覆盖复核 188.08 秒，整图成本复核 74.84 秒。[宿主账本](../../../data/large-scale-validation/dev/l2/s06/host-timing/qualification-accounting.json) · [通信范围](../../../data/large-scale-validation/dev/l2/s06/physical-peers/communication-coverage.json) · [整图工作核验](../../../data/large-scale-validation/dev/l2/s06/work-costs/coverage.json)

**结论与限制**　S06 完成，窗口 6/14，累计 84,713.14 卡秒。通信实测覆盖当前配置实际出现的 AllReduce 与 AllGather。完整设备成本、双来源展开和自主精度由后续开发回归与隔离留出核对。

## 2026-11-27 — S07 双来源对照与分页模板发布

**范围**　Qwen3-8B TP2，四请求错峰到达并跨 16-token 页边界。

**改动**　将同步调用流身份的声明归并至框架来源布局。局部与完整录制使用相同原生调用标记识别调用流，原生通信的独立流保留各自身份。完整拓扑的局部调度采集由同一会话观测代表 rank，每个发布片段经过自身读回和正式模板查询。

**验证**　首批四次真机共 951.64 卡秒，含一张卡分轮观测两 rank、详细与轻量两份物理目标以及一次局部分配采集。采集阶段与存储分项见账本。新双镜像重投影两种来源共 251.46 秒，发布约 5.68 秒。旧单卡模板匹配 124 个动作后遇真实通信布局差异，旧完整模板匹配 336 个动作后缺四成员分页分配。修复流绑定后局部发布核验通过，前进一个动作至 rank 1 的同类缺项。固定图的实测基线对齐全部 14 个 token 发布，成本覆盖仍有缺项。[账本](../../../data/large-scale-validation/dev/l2/s07/qwen8-tp2/native-accounting.json) · [合库归因](../../../data/large-scale-validation/dev/l2/s07/qwen8-tp2/repairs/full-local/diagnostic/bound-stream/summary.json) · [片段来源](../../../data/large-scale-validation/dev/l2/s07/qwen8-tp2/repairs/full-local/local-source-evidence/source-evidence.json) · [固定图成本缺项](../../../data/large-scale-validation/dev/l2/s07/qwen8-tp2/targets/page-crossing/demand/summary.json)

**结论与限制**　两模式来源核验、发布冲突修复和固定图基线具备各自证据。下一步核对同会话取得的多 rank 片段，以及单卡通信子图与物理提交队列的装配。S07 尚未达到结构和自主预测门槛。

## 2026-11-27 — S07 同会话补齐两 rank，取得 TP4 单卡来源

**范围**　使用现有真实 TP2 目标复核多 rank 局部发布，另采集 TP4 单卡来源。

**改动**　补采库在一次原生调度切口记录实际代表 rank。各 rank 以自身配置核对读回、参数和模板查询，再共同发布。用现存 TP2/4/8 物理原图检查通信叶和调用者跨界条件。

**验证**　两个分页分配片段通过独立来源核验。完整拓扑对照的内容匹配由 336 个增至 509 个，来源仍隔离。TP2/4/8 的 13,468 次 AllReduce 和 AllGather 的跨叶非串行边均经命名端点。TP4 单卡分四个原生窗口实际记录并核验四 rank。两次真机共耗 537.02 卡秒。TP2 两 rank 初始化约 8.6–8.8 秒、准备约 46.34 秒、观测及导出约 10.26 秒，峰值 2.60 MiB。TP4 初始化 34.44 秒、预热 102.01 秒、观测 121.91 秒，投影 223.37 秒，采样峰值 167.09 MiB。其余账本分项随证据保留。[TP2 局部来源](../../../data/large-scale-validation/dev/l2/s07/qwen8-tp2/repairs/physical-peer-cut/source-evidence/source-evidence.json) · [TP2 硬件账本](../../../data/large-scale-validation/dev/l2/s07/qwen8-tp2/native-accounting.json) · [TP4 硬件账本](../../../data/large-scale-validation/dev/l2/s07/qwen8-tp4/sources/native-accounting.json)

**结论与限制**　TP2 两 rank 的分页缺项已收口。当前首个模板缺项出现在单请求跨页 Decode，初始来源未实测这一注意力页表分支。完整依赖和自主精度仍待验证。累计已结算 86,201.81 卡秒。[TP2 结构复验](../../../data/large-scale-validation/dev/l2/s07/qwen8-tp2/targets/page-crossing/structure-full-peer/comparison/actions.json) · [注意力分支](../../../data/large-scale-validation/dev/l2/s07/qwen8-tp2/repairs/physical-peer-cut/attention-branches.json)

## 2026-11-27 — AGY 深度架构审查与窗口阶段收口

**范围**　对分支 `large-scale-validation` 自 `8d10807` 起的全部 55 个修改文件、6 个新增文件及 L0–S07 实验账本（累计 86,201.81 卡秒）执行独立代码品质、隔离红线与特例浓度审查，并按用户“方案 A”裁决阶段收口。

**改动**　修正 `src/python/prologue/integrations/frameworks/sglang/control/flow.py` 中 `__init__` 1 处 26 行连续无空行代码墙；标记本窗口 `outline.md` 阶段完成提前收口并更新 `status.md`；将全部已验收成果按职责分层提交封存。

**验证**　AGY 独立机审全量通过：219 个业务 Python 模块 AST 0 导入环、0 缺失导入，> 550 NE 模块 = 0（最大 539 NE），Python 与修改的 C++ 文件函数体内 > 25 行无空行代码墙 = 0，四维隔离红线 0 越层（`engine/core/` 全窗口仅改 7 行、0 框架/平台字样，`sglang/` 0 `ascend`/`npu`，`abi/` 外 0 `sglang.srt.`，`ascend/reference/` 0 `analysis`），`ruff check src/ example/` 0 错误，活动源码 0 `.pyc`，`build/large-scale-scheduler-preparation/prologue` 晚于全部修改 `.cpp` 编译并在 S04–S07 实跑通过。

**结论与限制**　L0、L1（S05 第六轮 5 域独立留出 5/5 全过、TTFT 1.00%–3.69%、TPOT 2.16%–3.96%）、S06 及 S07 TP2 完整拓扑 546 动作对齐已安全落袋。审查确认 `recipes/scheduler.py`（512 NE）中为强状态调度机制手写逆向构造脚本的路线易滋生特例且投入产出比失衡；剩余 L2 多配置预测、L3 异构高级并行、L4 Azure 真实流量与 8 域总表转入新窗口 `2026-11-27_multi_config_prediction_validation`，改为“完整拓扑短流先出多配置预测 + 弱状态局部探针独立取证 + 冻结前预登记轻量复测 + M1/M2/M3 短里程碑暂停门控”。
