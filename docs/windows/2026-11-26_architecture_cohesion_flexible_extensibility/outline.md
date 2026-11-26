# 全仓逻辑交互重构、职责内聚拆分、代码可读性与柔性可扩展体系升级 — 窗口纲领（Outline）

**已关闭**（2026-11-26，完成）　本窗口 7/7 阶段（Stage 1–6 + Final Stage）全部完成并通过 AGY 独立终审与用户确认，按职责分层提交推送至 `origin/architecture-cohesion-extensibility` 后关闭，历史文档原地保留。

> **权威边界声明**：本文件为本窗口的**不可变战略纲领**，由 AGY 唯一维护，**Codex 只读严禁修改**。本文件定义目标、工程审美标尺、不可妥协红线、防偷懒门禁与资源预算；具体阶段拆解与推进路线见同目录 `plan.md`（由 AGY 提供初版，Codex 在每个阶段完成后须经 Bigger-Picture 全局 Overview 动态修订与展开）。

**开始日期**　2026-11-26（开发分支 `architecture-cohesion-extensibility`，源码起点 `cbea08e`；核心功能与精度对照基线 `b500cbe`）

**交付纪律**　本窗口采用 **Codex 全窗口端到端自主交付、不设中途暂停点、全流程严禁执行 `git commit` 与 `git push`** 的模式——全部改动保留在工作区，完成后交回 AGY 一次性全面终审，经终审通过与用户确认后再统一提交推送（绝不自行关窗，绝不触碰 `docs/note.md` 和 `.personal/`）。

---

## 一、核心目标与资深工程师六大标尺（全仓适用，非仅点名示例）

本窗口**彻底废除任何硬性文件数上限与机械减行指标**。Codex 必须以**资深优雅软件工程师**的标准，系统性通读并重整 `src/` 下的**每一个业务子系统**（`src/engine/`、`common/`、`host/`、`workers/`、`integrations/runtimes/pytorch/`、`integrations/platforms/ascend/`、`integrations/frameworks/sglang/{abi,shared,control,observation,acquisition}/`、`verification/`），用以下 **6 把标尺**统一拉齐全仓品质：

1. **数据驱动与可插拔，替代硬编码分发**：消除全仓冗长 `if/elif` 模式链、固定元组/名称硬编码、靠改旧代码 `matches()` 才能加特例的脆弱互斥与手写字段映射表，改为注册表、表驱动分发、优先级 + 自动回退链（Fallback）与可组合原语。
2. **统一公共内核 vs 灵活扩展策略**：消除平行状态机、平行图遍历/收集/报告拼装与重复任务组装/读回；沉淀唯一权威内核，只把真正变化的领域语义抽成清晰窄协议或轻量插件。
3. **外部边界校验一次，内部靠类型与所有权**：外部入口（CLI、原始录制、容器输入、外部库）保持严格 fail-closed；内部可信管线停止每跳重复 `keys() ==`、`type(x) is not int`、写后立刻读回重检与满屏裸 `dict` 对暗号，改用清晰数据类/领域对象与单一状态所有者。
4. **单一职责、适度粒度与浅层嵌套**：拆解过度合并的 600–1,286 行巨文件与上百行深层嵌套函数，理顺单向无环交互（同时不造 15 行无意义转发壳）。
5. **充裕的逻辑分段空行与高价值注释**：函数内部在参数解包、前置准备、核心计算、状态更新、结果组装、返回等不同逻辑阶段之间及控制流块前后留足清晰空行；为模块、核心类与非显而易见的设计意图/物理不变量/协议前提补足简练高信号的 docstring 与注释（严禁 `# add 1` 类复述语法的水注释）。
6. **克制优雅，拒绝过度设计**：简洁直白永远胜过花哨空壳；只在真实存在扩展维度或重复模式时引入抽象。

> **典型症状参考（仅作切入点，绝非工作上限）**：
> - 基线 `cbea08e` 全仓 255 文件 / 44,280 NE（Python 150f / 36,190L / 31,836 NE），含 **18 个 > 550 NE 的过载 Python 模块**（如 `verification/core/{protocol,connections,actions,state,source}.py`、`verification/fixed/fixed_reference.py`、`ascend/analysis/observation.py`、`ascend/reference/checks.py`、`sglang/observation/{layout,model,communications,scheduler_samples,observers,serving}.py`、`sglang/acquisition/{evidence,rules}.py`、`sglang/control/scheduler.py`），且空行仅 **12.0%**、`#` 注释仅 **2.3%**、> 25 行无空行代码墙 **141 处**（> 40 行 31 处）；
> - `LocalAcquisition` 死板互斥无回退、固定 9 元组硬编码、`sglang/observation/entry.py` 7 个 `--local-*` 与 `if/elif`、特例构造函数传参以及 `host/`/`common/`/`ascend/`/`pytorch/`/`sglang/`/`verification/`/`engine/` 中的同类硬编码与密集代码墙。

---

## 二、不可妥协的红线与非目标

- **非目标**：不实现 CUDA/vLLM/SGLang 第二版本的实机代码，不引入生产 mock/stub、独立测试工程或自测脚本；不触碰 `docs/note.md`、`.personal/`、`.clang-format`、`third_party/`、`deprecated/` 及既有 `data/` 资产。
- **科学/物理红线**：`TS_TASK_FLIP` 槽位证明、双锚点 RAW 解码、`source_clock` 的 **SymPy 有理数精确认证**（严禁浮点近似）、训练/验证来源隔离与输入差异阻断、`NativeCase` 与 `ascend/reference` 独立构建（`reference/` 0 导入 `analysis/`）、唯一逻辑动作锚点、PD/CPU 协议前提、固定图重算与自主闭环 TTFT/TPOT 口径。
- **物理分层与四维红线**：仓库根 5 个受跟踪目录、`src/{engine,probes,shared,python}` 0 多语言混装、纯 `COPY` Dockerfile、`engine/core/` 0 框架/平台字样、`sglang/` 0 `platforms.ascend`/`torch.npu`、`abi/` 外 0 `sglang.srt.`、`verification/` 0 导入 `host/cli`/`workers/` 且仅经窄适配口访问 `integrations/`。
- **构建、卫生与回归红线**：C++ `modeling` Release **0 error / 0 warning**；双镜像纯 `COPY` 重建实装 SHA-256 验真与原生探针加载；4 种打包形态与 5 域 16 子命令可用；`ruff check src/python example` **0 errors**，活动源码 0 `.pyc`，AST **0 环 / 0 缺失导入**；核心基线预测/拟合/成本与 `b500cbe` 一致（允许精简 `verification/` 与局部采集内部诊断明细 JSON/MD 结构，核心结论/计数/资产/数值必须一致）。

---

## 三、每阶段强制 Bigger-Picture Overview 机制

Codex 在推进 `plan.md` 时，**每完成一个阶段，必须执行一次完整的 Bigger-Picture 全局 Overview**（不可只盯上一阶段 diff）：
1. 重新俯瞰全仓架构拓扑、模块交互图、剩余过载文件与刚性硬编码分布；
2. 结合当前真实代码现状，**直接修改、细化、增补或重排 `plan.md` 中的后续阶段**（动态展开计划，而不是机械照搬初稿）；
3. 将本次全局审视结论与计划调整摘要记入 `log.md`，刷新 `status.md`，再进入下一阶段。

---

## 四、AGY 终审防偷懒门禁、有界预算与熔断规则

终审时 AGY 将独立运行脚本与抽检源码 Diff，**任一门禁不达标即整批打回返工**：

1. **全仓逐域审计矩阵**：交付 `data/architecture-cohesion-extensibility/audit-matrix.json`，逐一覆盖 `common`、`host`、`workers`、`pytorch`、`ascend`、`sglang/{abi,shared,control,observation,acquisition}`、`verification`、`src/engine` 全部业务子系统（改前/改后文件数、最大文件 NE、函数内 > 25 行无空行代码墙数、空行占比、注释占比及消除的硬编码清单；若某域未改须逐文件给出审计依据）。
2. **超大过载文件清零**：除纯常量/符号声明表外，**> 550 NE 的业务 Python 模块须清零**（确有极个别不可拆原子模块须在 `audit-matrix.json` 单列理由交 AGY 裁决），且 > 80 行嵌套长函数显著压降（`compare_actions` 单方法 ≤ 100 行）。
3. **空行呼吸感与高价值注释（防代码墙 + 动态柔性比例）**：**业务函数/方法体内 > 25 行连续无空行代码墙压至 0**（纯常量/符号表例外）；空行与注释行占总行数的比例采用**动态柔性参考区间**（不设僵硬下限，避免机械凑数：参考全仓空行约 **15%–24%**、注释与 docstring 约 **6%–15%**，薄胶水/数据定义可偏低，复杂协议/算法/不变量模块应偏高），核心由 AGY 终审抽检 Diff 的视觉呼吸感与注释含金量。
4. **真柔性零副作用扩展证明**：交付 `data/architecture-cohesion-extensibility/extensibility-proof.json` 与实际触发日志，证明新增采集规则/采集方式与验证检查项时，**核心分发器、旧规则 `matches()`、宿主编排与入库层改动 = 0 行**，且优先级与 `EvidenceGap` 自动回退链真实生效。
5. **工作区源码快照与 SHA-256 绑定**：因全流程不做 `git commit`，全部证据须绑定最终工作区 `src/` 的逐文件 SHA-256 清单与汇总哈希，严禁拿旧证据充数。
6. **有界真机闭环预算与全量离线对照**：
   - 全窗口累计真机上限：**单卡 ≤ 1,200s、后处理 ≤ 900s、存储 ≤ 8 GiB**（失败重试不清零，分项记账）；必须实测：① 升级后的柔性采集/回退链闭环（`library acquire` → `library publish --from-session` → 增量库 `predict run`）；② 最终工作区源码 + 双镜像 + C++ Release 的 Qwen3-8B TP1 `--service-reference` 全链路闭环（`capture run` → `capture project` → `cost extract` → `cost fit` → `library publish` → `predict run` → `verify {source, structure, fixed-graph, autonomous}`）。
   - 全量离线对照（基线 `b500cbe`）：Qwen TP1 四目标（211,290 轨迹、16 token 发布、4 份时延）与 4 空库首缺项、`library inspect`/`publish --from-collection`、678 样本 `cost fit`、四类 `cost extract`、43,276 项固定图 `work-costs`、留存 PD（5,355 动作 / 825 端口差异）、留存 TP1 自主闭环评分与 TP2/TP8 参考资格结论一致。
7. **失败熔断**：同一根因连续尝试 2 轮无新进展或真机预算耗尽时立即止损留证交回 AGY，严禁静默扩预算或降标。
