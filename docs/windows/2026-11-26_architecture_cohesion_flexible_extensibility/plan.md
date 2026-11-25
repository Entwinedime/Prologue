# 全仓逻辑交互重构、职责内聚拆分、代码可读性与柔性可扩展体系升级

> **协作与自主交付模式**：本窗口采用 **Codex 全窗口端到端自主交付、完成后统一交回 AGY 终审** 的协作模式（**不设中途暂停点**）。Codex 以**资深优雅软件工程师**的标准对**全仓所有子系统做全量架构与代码品质重整**（本文中列举的模块与问题仅作典型症状示例，绝非工作上限），在严守科学/物理不变量红线、有界真机预算与工程约束的前提下拥有充分的技术自主权；全窗口完成后分层提交并推送分支，交回 AGY 一次性全面终审（不要自行关窗）。

**开始日期**　2026-11-26（从已保留 R11 的基线 `cbea08e` 切出新开发分支 `architecture-cohesion-extensibility`；核心功能与精度对照基线仍为 `b500cbe`）

**目标**　彻底扭转此前增量式瘦身遗留的三类深层工程品质问题：
1. **过度合并导致职责混杂与交互紊乱**：拆解被硬性文件数指标挤压出的千行巨文件，恢复单一职责、清晰分层与简洁无环交互；
2. **全仓范围的“编码风格过硬”与刚性硬编码**：跳出局部点修，对全仓每个子系统系统性识别“统一核心逻辑”与“灵活可扩展策略”，以数据驱动、注册表、组合原语与清晰领域对象替代写死的 `if/elif` 分发、脆弱互斥条件、裸 `dict` 搬运与跨层重复防御；
3. **代码可读性与视觉呼吸感极差**：彻底消除缺乏空行的大块密集代码墙，**大幅增加逻辑分段空行与高价值注释/文档字符串**，让每一份源码都具备资深工程师手笔的清晰节奏与可维护性。

---

## 一、资深软件工程师全量重构准则（适用于全仓每一个子系统）

本窗口**彻底废除任何硬性文件数上限与机械减行指标**（空行与注释绝不是冗余，而是工程品质的核心组成）。Codex 须带着以下 **6 把架构与审美标尺**，系统性通读并重构 `src/` 下的**每一个子系统**——包括 `src/engine/`、`src/python/prologue/{common,host,workers,integrations/runtimes/pytorch,integrations/platforms/ascend,integrations/frameworks/sglang,verification}/`——而不只是局部修改几个被点名的文件：

| 标尺 | 要识别并消除的“硬/乱/闷”坏味道 | 期望达到的资深优雅形态 |
| --- | --- | --- |
| **1. 数据驱动与可插拔，替代硬编码分发** | 写死的 `if/elif` 模式链、固定元组规则列表、靠修改旧代码 `matches()` 才能加新特例的脆弱互斥、硬编码字段映射表。 | 注册表、表驱动分发、优先级与自动回退链（Fallback）、可组合原语；新增一种策略/规则/采集方式时尽量“增量插拔、零改旧核心”。 |
| **2. 统一核心逻辑 vs 灵活扩展策略** | 多处实现同一流程的平行变体（平行状态机、平行遍历/收集/报告拼装、重复任务组装与读回校验）。 | 把不变的公共流程沉淀为唯一权威内核，只把真正变化的领域语义抽成清晰窄协议或轻量插件。 |
| **3. 边界校验一次，内部靠类型与所有权** | 在内部可信调用链每一跳反复做 `keys() ==`、`type(x) is not int`、写后立刻读回重检、满屏裸 `dict` 传参对暗号。 | 外部边界（CLI、原始录制、容器输入、外部库）做一次严格 fail-closed 校验；内部使用清晰的数据类/领域对象与单一状态所有者，逻辑直白干净。 |
| **4. 单一职责、适度粒度与浅层嵌套** | 600–1,286 行混杂多职责的大文件、上百行深层嵌套函数、跨文件隐式共享可变状态袋；或反过来的 15 行无意义转发壳。 | 每个文件讲清一个完整故事（健康参考粒度约 150–450 非空行，超 ~500 行须审视拆解）；函数短小扁平（善用早返回/守卫子句），依赖单向无环。 |
| **5. 充裕的空行呼吸感与高价值注释** | 连续 30–100 行几乎没有空行的密集代码墙；缺少模块职责说明与关键不变量解释，人类工程师冷读极度吃力。 | **在函数内不同逻辑阶段（参数解包、前置准备、核心计算、状态更新、结果组装、返回）之间以及控制流块前后大幅插入空行**；为模块、核心类与非显而易见的设计取舍/物理不变量/协议前提撰写简练清晰的高信号注释与 docstring（解释“为什么”与“契约边界”，不写废话复述语法）。 |
| **6. 克制优雅，拒绝过度设计** | 为抽象而抽象的多层空壳工厂、形式主义包装。 | 简洁、直观、一眼看懂永远优于花哨炫技；只有在真实存在扩展维度或重复模式时才引入抽象。 |

> **典型症状示例（仅作切入参考，绝非全量清单）**：
> - **超大过载文件示例**（基线 `cbea08e` 共 22 个 ≥ 450 NE 模块）：`verification/core/{protocol,connections,actions,state,source}.py`（603–1,185 NE）、`verification/fixed/fixed_reference.py`（857 NE）、`ascend/analysis/observation.py`（933 NE）、`ascend/reference/checks.py`（794 NE）、`sglang/observation/{layout,model,communications,scheduler_samples,observers,serving}.py`（555–726 NE）、`sglang/acquisition/{evidence,rules}.py`（667–739 NE）、`sglang/control/scheduler.py`（666 NE）等。
> - **刚性硬编码链路示例**：`LocalAcquisition.select()` 死板互斥无回退、`SGLangLaunch.local_acquisition()` 固定 9 元组、`sglang/observation/entry.py` 硬编码 7 个 `--local-*` 与 `if/elif`、规则前置依赖靠构造函数特例传参而非统一 `AcquisitionContext`、`verification/` 中平行状态机与逐字节锁死的内部诊断拼装，以及 `host/`、`common/`、`ascend/`、`pytorch/`、`sglang/{shared,control,observation}/`、`src/engine/` 中所有同类硬编码与密集代码墙——Codex 须主动全量发掘并一并解决。

---

## 二、不可妥协的红线与非目标

- **非目标**：不实现 CUDA/vLLM/SGLang 第二版本的实机代码，不引入生产 mock/stub、独立测试工程或自测脚本；不触碰 `docs/note.md`、`.personal/`、`.clang-format`、`third_party/`、`deprecated/` 及既有 `data/` 资产。
- **不可妥协的科学/物理与架构红线**：
  1. **物理时钟与因果序**：`TS_TASK_FLIP` 槽位证明、双锚点 RAW 解码、`source_clock` 的 **SymPy 有理数精确认证**（严禁浮点近似替代）；
  2. **科学严谨性**：训练/验证来源隔离与输入差异阻断、`NativeCase` 与 `ascend/reference` 独立构建（`reference/` 严禁导入 `analysis/`）、动作配对仅认唯一逻辑锚点、PD/CPU 协议前提核验、固定图重算与自主闭环 TTFT/TPOT 评分口径；
  3. **顶层物理与四维边界**：仓库根保持 5 个受跟踪目录、`src/{engine,probes,shared,python}` 0 多语言混装、纯 `COPY` Dockerfile（0 构建期裁剪脚本）、`engine/core/` 0 框架/平台字样、`sglang/` 0 `platforms.ascend`/`torch.npu`、`abi/` 外 0 `sglang.srt.`、`verification/` 0 导入 `host/cli`/`workers/` 且仅通过窄适配口访问 `integrations/`；
  4. **公开入口与预算纪律**：所有真机采集经官方公开 CLI 执行、累计预算失败重试不清零，核心基线预测/拟合/成本数值与 `b500cbe` 一致（允许精简 `verification/` 与局部采集内部诊断明细 JSON/MD 结构，只要核心结论、计数、资产与数值一致）。

---

## 三、端到端自主推进路线（无中途暂停点）

Codex 可在守住上述准则与红线的前提下自主规划细化步骤与提交切分，按以下四个连贯环节一次走通：

### 环节 1：全仓 Bigger-Picture 交互盘点与重构蓝图
- 系统通读 `src/` 下全部子系统（Python 六大域 + C++ `src/engine/` 接口），产出全仓视角下的：
  - 数据流、控制流与状态所有权地图；
  - 统一核心逻辑 vs 可插拔扩展策略清单（覆盖全仓，不限于采集模块）；
  - 全量重整清单（哪些过载文件需拆分、哪些硬编码链需柔性化、哪些密集代码墙需重排空行与注释）。
- 将简洁明确的全局重构蓝图记录到本窗口 `log.md`。

### 环节 2：全仓柔性扩展架构升级与反僵硬重整
- 沿着环节 1 的全仓清单，逐子系统破除刚性硬编码、平行状态机与跨层重复防御：
  - 采集与补采链实现规则注册表 + 优先级 + 自动回退链 + 统一只读 `AcquisitionContext` + 容器 probe 注册派发与可复用切点/读回原语；
  - 同步推进 `host/`、`common/`、`workers/`、`pytorch/`、`ascend/`、`sglang/{abi,shared,control,observation,acquisition}/`、`verification/` 以及必要时 `src/engine/` 的逻辑解耦、表驱动/声明式化与领域类型提炼，确保“整仓同一水准的优雅与灵活”，不留死角。

### 环节 3：大文件内聚拆分、空行与注释品质重整、验证层收口与文档同步
- 将过度合并的偏重文件按内聚职责与清晰分层重新拆分（含完成 `verification/{core,structure,fixed,autonomous}` 整体梳理，保持 `compare_actions` 单方法 ≤ 100 行）；
- 对全仓改动及核心存量模块做一轮专门的**可读性与视觉呼吸感打磨**：大幅补足函数内逻辑段落空行、控制流间隔空行、模块/类 docstring 与关键不变量/设计意图注释；
- 同步更新 `README.md` 与静态设计文档中的目录拓扑、构建/运行命令、四维扩展边界与柔性扩展机制说明（严守文档跨层零引用）。

### 环节 4：全仓构建卫生、有界真机闭环实测与全量离线基线对照
- **静态与构建卫生**：
  - C++ `modeling` Release 编译 **0 error / 0 warning**；
  - `./docker/build_inference.sh` 纯 `COPY` 重建 `sglang` & `control` 双镜像，核验实装模块 SHA-256、0 宿主/CLI/验证模块泄露、原生探针加载正常；
  - 4 种打包形态（`source`/`wheel`/`target`/`editable`）与 `python3 -m prologue` 5 域 16 子命令 / `prologue-cli` 全部可用；
  - `ruff check src/python example` **0 errors**，活动源码与交付树 **0 `.pyc`**，AST **0 导入环 / 0 缺失导入**，全仓四维边界静态审计 0 越层。
- **有界真机闭环验证**（全窗口累计硬预算：单卡运行 ≤ **1,200s**、后处理 ≤ **900s**、存储 ≤ **8 GiB**；失败重试不清零，完整记账）：
  1. 空闲单卡经官方 CLI 实测升级后的柔性采集/回退链闭环（`library acquire` → `library publish --from-session` → 增量库 `predict run`）；
  2. 使用最终版源码、双镜像与 C++ Release 二进制，在空闲单卡完整跑通 Qwen3-8B TP1 `--service-reference` 端到端全链路：`capture run` → `capture project` → `cost extract` → `cost fit` → `library publish` → `predict run` → `verify {source, structure, fixed-graph, autonomous}`。
- **全量离线基线对照（对比 `b500cbe`）**：
  - Qwen3-8B TP1 四目标（211,290 轨迹、16 次 token 发布、4 份请求时延）与 4 个空库首缺项逐字节一致；
  - `library inspect`、`library publish --from-collection`、678 样本 `cost fit`、四类 `cost extract`（device/transport/host/batch）、`cost-points`、43,276 项固定图 `work-costs` 一致；
  - 留存 PD（配对 5,355 动作、825 处端口差异）、留存 TP1 自主闭环评分（1 请求 3 token、8,601 成本查询）与 TP2/TP8 参考资格结论与 `b500cbe` 一致。

---

## 四、交付门禁、防偷懒机审硬指标、熔断与停机条件

本窗口不设中途暂停点，为防止端到端自主推进中出现“只改点名示例、假拆文件、敷衍补空行、套壳假灵活或拿旧证据充数”，AGY 终审时将**独立运行脚本与逐文件抽检**核验以下 5 项硬性防偷懒门禁，**任一项不达标即整批打回返工，绝不迁就放行**：

1. **全仓逐域覆盖门禁（杜绝只改 2 个点名模块）**：
   - 必须产出 `data/architecture-cohesion-extensibility/audit-matrix.json`，逐一覆盖 `common`、`host`、`workers`、`runtimes/pytorch`、`platforms/ascend`、`frameworks/sglang/{abi,shared,control,observation,acquisition}`、`verification`、`src/engine` 全部业务子系统；
   - 每个子系统必须给出改前/改后文件数、最大文件 NE、函数内 > 25 行无空行代码墙数、空行占比、注释占比以及具体消除的硬编码/僵硬点清单（若某子系统判断无需结构改动，必须逐文件给出审计依据）。
2. **超大过载文件清零门禁（杜绝挑软柿子）**：
   - 基线 `cbea08e` 全仓存在 **18 个 > 550 NE** 的过载 Python 模块；交付时除纯数据/符号表文件外，**> 550 NE 的业务 Python 模块须清零**（确有极个别不可拆原子模块须在 `audit-matrix.json` 单列理由交 AGY 裁决），且 > 80 行嵌套长函数显著压降（`compare_actions` 单方法仍硬性 ≤ 100 行）。
3. **空行呼吸感与高价值注释机审门禁（杜绝敷衍塞空行或写水注释）**：
   - 基线 `cbea08e` 实测：Python 36,190 行中空行仅 4,354 行（**12.0%**）、`#` 注释仅 839 行（**2.3%**）、> 25 行连续无空行代码墙 **141 处**（> 40 行 31 处）；
   - 交付硬门禁：**业务函数/方法体内 > 25 行连续无空行代码墙压至 0**（全仓仅允许纯常量/符号定义表例外），全仓 Python 空行占比提升至 **≥ 18%**，每个业务模块、核心类与关键非平凡方法均有高信号 docstring / “为什么与不变量”注释（AGY 终审抽检源码 Diff，严禁 `# add 1` 类水注释）。
4. **真柔性零副作用扩展证明（杜绝套壳假可扩展）**：
   - 交付 `data/architecture-cohesion-extensibility/extensibility-proof.json` 与实际回退链验证日志：以“新增一种全新采集规则/采集方式 + 新增一种验证检查项”为例，结合真实源码路径与注册表证明核心分发器、旧规则 `matches()`、宿主编排与入库层改动 = 0 行，并实证优先级排序与 `EvidenceGap` 自动回退链生效。
5. **最终源码树绑定与 SHA-256 验真（杜绝旧证据充数）**：
   - 所有静态审计、双镜像实装、真机账本与离线对照产物均须记录最终提交的 `src` 树哈希并提供 SHA-256 清单，AGY 终审将独立复算验真。
6. **失败熔断**：同一根因连续尝试 2 轮无新进展，或真机累计预算耗尽时立即止损，保留完整证据与账本记入 `log.md`/`status.md` 交回 AGY 复盘，严禁静默扩预算或降标。
7. **最终交付**：将全窗口证据归档至 `data/architecture-cohesion-extensibility/`，更新本窗口 `status.md` 与 `log.md`，按清晰职责分层提交并推送至 `origin/architecture-cohesion-extensibility`（同时把已关闭窗口的 `topology-extensibility-slimming` 推齐）后停止，交回 AGY 终审（不自行关窗）。
