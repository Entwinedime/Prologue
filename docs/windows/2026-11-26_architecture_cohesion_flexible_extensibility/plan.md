# 全仓逻辑交互重构、职责内聚拆分与柔性可扩展体系升级

> **协作与自主交付模式**：本窗口采用 **Codex 全窗口端到端自主交付、完成后统一交回 AGY 终审** 的协作模式（**不设中途暂停点**）。Codex 拥有本窗口内从全局交互分析、模块职责拆合、接口与注册表重构到验证交付的完整技术自主权，但必须严格遵守既定科学/物理不变量红线、有界真机预算及工程约束；全窗口完成后分层提交并推送分支，交回 AGY 一次性全面终审（不要自行关窗）。

**开始日期**　2026-11-26（建议从当前已保留 R11 的基线 `cbea08e` 切出新开发分支 `architecture-cohesion-extensibility`；核心功能与精度对照基线仍为 `b500cbe`）

**目标**　针对上一窗口提前复盘暴露的两项核心结构性问题——**① 过度合并文件导致单文件职责过重与分层交互紊乱；② 表层有接口但内核逻辑依然刚性硬编码、缺乏全局视角下的职责与交互重梳**——由 Codex 从全局总览出发一次性完成全仓逻辑交互重整、大文件内聚拆分、柔性可扩展机制落地，以及遗留的三路验证整理、公开文档同步与端到端真机闭环终验。

---

## 一、核心诊断与重构原则

### 1. 本窗口要根治的两类问题

1. **过度合并导致的“巨文件 + 职责混杂 + 交互紊乱”**
   - 基线 `cbea08e`（全仓 255 文件 / 44,280 NE，其中 Python 150f / 31,836 NE）中存在 **22 个 ≥ 450 NE 的偏重 Python 模块**，其中最突出的包括：
     - `verification/core/protocol.py`（1,185 NE / 1,286L）、`connections.py`（1,159 NE / 1,246L）、`actions.py`（978 NE / 1,079L）、`state.py`（849 NE / 935L）、`source.py`（603 NE / 662L）
     - `verification/fixed/fixed_reference.py`（857 NE / 943L）、`fixed_context.py`（534 NE / 609L）、`verification/structure/execution_inputs.py`（558 NE / 592L）
     - `integrations/platforms/ascend/analysis/observation.py`（933 NE / 1,024L）、`reference/checks.py`（794 NE / 848L）、`host/services.py`（558 NE / 630L）
     - `integrations/frameworks/sglang/acquisition/evidence.py`（739 NE / 838L）、`rules.py`（667 NE / 737L）、`bindings.py`（482 NE / 566L）
     - `integrations/frameworks/sglang/observation/{layout,model,communications,scheduler_samples,observers,serving}.py`（555–726 NE）与 `control/scheduler.py`（666 NE）
   - 本窗口**彻底废除硬性文件数上限与机械减行指标**：禁止为压文件数把不相关职责塞入同一文件；改以**单一职责、清晰分层、显式所有权与单向无环交互**为准绳（推荐健康单文件粒度 ~150–450 NE；凡显著超过 ~500 NE 的模块须审视是否混杂多职责并合理拆解）。

2. **“表层可扩展、逻辑仍僵硬”的硬编码与跨层冗余**
   - **采集与补采链僵硬**：宿主侧 `LocalAcquisition.select()` 强制唯一互斥、无优先级与自动回退链（新增特化规则须回头改旧规则的 `matches()` 避让，特化规则前置条件不足时无法自动降级到更通用采集方式）；规则注册写死为固定 9 元组；容器侧 `sglang/observation/entry.py` 写死 7 个 `--local-*` CLI 枚举与 `if/elif` 分发链；规则前置依赖靠构造函数特例传参而非统一只读上下文；多条规则各自重复私有 `job.json` 组装与冗长读回校验。
   - **全局交互缺乏“统一核心 vs 灵活扩展”的顶层抽象**：长期增量演进导致 `verification/`、`sglang/`、`ascend/`、`host/` 间残留平行状态机、重复的跨层字段/类型/键集重检（偏离“外部边界校验一次、内部靠所有权”原则）以及把内部诊断明细当核心契约锁死的冗余样板。

### 2. 明确非目标与不可逾越的红线

- **非目标**：不实现 CUDA/vLLM/SGLang 第二版本的实机代码，不引入生产 mock/stub、独立测试工程或自测脚本；不触碰 `docs/note.md`、`.personal/`、`.clang-format`、`third_party/`、`deprecated/` 及既有 `data/` 资产。
- **不可妥协的科学/物理与架构红线**：
  - 物理时钟与因果序：`TS_TASK_FLIP` 槽位证明、双锚点 RAW 解码、`source_clock` 的 **SymPy 有理数精确认证**（严禁浮点近似替代）；
  - 科学严谨性：训练/验证来源隔离与输入差异阻断、`NativeCase` 与 `ascend/reference` 独立构建（`reference/` 严禁导入 `analysis/`）、动作配对仅认唯一逻辑锚点、PD/CPU 协议前提核验、固定图重算与自主闭环 TTFT/TPOT 评分口径；
  - 顶层物理与四维边界：仓库根保持 5 个受跟踪目录、`src/{engine,probes,shared,python}` 0 多语言混装、纯 `COPY` Dockerfile（0 构建期裁剪脚本）、`engine/core/` 0 框架/平台字样、`sglang/` 0 `platforms.ascend`/`torch.npu`、`abi/` 外 0 `sglang.srt.`、`verification/` 0 导入 `host/cli`/`workers/` 且仅通过窄适配口访问 `integrations/`；
  - 预算与公开入口：所有真机采集经官方公开 CLI 执行、累计预算失败重试不清零，核心基线预测/拟合/成本数值与 `b500cbe` 一致。

---

## 二、自主推进路线（Codex 端到端闭环，中途不停机）

Codex 可在达成上述目标与红线的前提下自主调整内部拆分粒度与实施顺序，建议按以下四个连贯环节一次走通：

### 环节 1：全仓 Bigger-Picture 交互总览与重整方案（先看全局，再动手）
- **为什么做**：避免继续陷入局部 diff 式的增量修补，先真正看清全仓的数据流、控制流、状态所有权与跨模块调用关系。
- **核心产出**：梳理并明确全仓六大主干（`host/`、`common/`、`workers/`、`integrations/frameworks/sglang/`、`integrations/platforms/ascend/`、`verification/` 及 `src/engine/` 接口）中：
  1. 哪些是**唯一统一的公共内核逻辑**（只应有一份实现与一个所有者）；
  2. 哪些是**应通过注册表/协议/组合原语灵活扩展的策略与实现**；
  3. 哪些过度合并的大文件需要按职责重新拆分、哪些跨模块交互需要理顺或收窄接口。
- 将这份简洁的架构重整决策记入本窗口 `log.md`，作为后续重构的锚点。

### 环节 2：柔性可扩展体系升级（重点破除“太硬”的硬编码链路）
- **采集与自动补采柔性化**：
  - 将 `LocalAcquisition` 升级为**声明式规则注册 + 优先级排序 + 自动回退链（Fallback Chain）**：高优先级特化微基准在 `prepare()` 遇到前置条件不足（`EvidenceGap`）时自动无缝降级尝试下一级更通用规则（如 `local-worker` 或完整/分片采集），新增特化规则**无需修改任何旧规则的 `matches()`**；
  - 为规则引入统一只读 `AcquisitionContext`（收敛 `release`、`sampling_sources`、已有局部会话、`deployment` 等前置依赖），消除特例构造函数传参；
  - 容器侧将 `sglang/observation/entry.py` 的硬编码 `--local-*` 枚举与 `if/elif` 分发升级为**统一注册表驱动的 probe 派发**（保留兼容入口或显式映射），并提炼可复用的局部切点执行与通用读回原语，使新增采集规则/采集方式显著轻量、可组合；
  - 保持宿主 `host/acquisition/` 与 `host/libraries/` 框架无关。
- **全仓反僵硬清理**：
  - 按“外部边界 fail-closed 校验一次、内部靠所有权传递”原则，清理跨层重复的内部字段类型/键集/写后二次读回重检；
  - 释放 `verification/` 与局部采集内部诊断明细 JSON/MD 的旧格式逐字节锁定（核心结论、计数、资产与数值精度保持一致），消除平行状态机与硬编码特例分支。

### 环节 3：大文件内聚拆分、分层交互梳理与验证层收口
- **重平衡过重单文件**：系统审视基线中 ≥ 500 NE 的 20 余个过重模块（涵盖 `verification/core/`、`verification/{structure,fixed,autonomous}/`、`ascend/{analysis,reference,host}/`、`sglang/{observation,acquisition,control}/`），按**内聚职责与清晰单向依赖**拆解过载文件、清理跨文件隐式状态耦合，同时避免重新制造无意义的 15 行空壳微文件；
- **完成 `verification/` 整体梳理**：基于统一 `engine.py` 内核完成 `core/` + `{structure,fixed,autonomous}/` 的清晰组织（保持 `compare_actions` 单方法 ≤ 100 行，保留全套科学验证红线），补齐上一窗口提前结束时未完成的验证层整理；
- **公开文档同步**：同步更新 `README.md` 与静态设计文档中的目录拓扑、构建/运行命令、四维扩展边界与柔性规则扩展说明（严守文档跨层零引用）。

### 环节 4：全仓构建、有界真机闭环实测、全量离线回归与终审交付
- **静态与构建卫生**：
  - C++ `modeling` Release 编译 **0 error / 0 warning**；
  - `./docker/build_inference.sh` 纯 `COPY` 重建 `sglang` 与 `control` 双镜像，核验实装模块 SHA-256、0 宿主/CLI/验证模块泄露、探针原生库加载正常；
  - 4 种打包形态（`source`/`wheel`/`target`/`editable`）与 `python3 -m prologue` 5 域 16 子命令 / `prologue-cli` 全部可用；
  - `ruff check src/python example` **0 errors**，活动源码与交付树 **0 `.pyc`**，AST **0 导入环 / 0 缺失导入**，全仓四维边界静态审计 0 越层。
- **有界真机闭环验证**（全窗口累计硬预算：单卡运行 ≤ **1,200s**、后处理 ≤ **900s**、存储 ≤ **8 GiB**；失败重试不清零，完整记账）：
  1. **柔性采集与补采闭环**：在空闲单卡上经官方 CLI 实测升级后的注册表/优先级/回退链采集流程（`library acquire` → `library publish --from-session` → 增量库 `predict run`，至少实测覆盖一条局部规则并验证回退/选择机制）；
  2. **端到端全链路终验**：使用最终版源码、双镜像与 C++ Release 二进制，在空闲单卡完整跑通 Qwen3-8B TP1 `--service-reference`：`capture run` → `capture project` → `cost extract` → `cost fit` → `library publish` → `predict run` → `verify {source, structure, fixed-graph, autonomous}`。
- **全量离线基线对照（对比 `b500cbe`）**：
  - Qwen3-8B TP1 四目标（211,290 条执行记录、16 次 token 发布、4 份请求时延）与 4 个空库首缺项逐字节一致；
  - `library inspect`、`library publish --from-collection`、678 样本 `cost fit`、四类 `cost extract`（device/transport/host/batch）、`cost-points`、43,276 项固定图 `work-costs` 一致；
  - 留存 PD（配对 5,355 动作、825 处端口差异）、留存 TP1 自主闭环评分（1 请求 3 token、8,601 成本查询）与 TP2/TP8 参考资格结论与 `b500cbe` 一致。

---

## 三、交付门禁与停机条件

1. **架构质量门禁（非机械数字，以工程清晰度为准）**：
   - 全仓模块职责清晰、无“千行大杂烩文件”与混乱反向/循环交互；
   - 采集规则支持优先级 + 自动回退 + 注册表 + 统一上下文 + 容器 probe 注册派发，新增一种采集规则/方式无需修改核心分发与旧规则互斥条件；
   - 核心科学/物理不变量与四维隔离红线 100% 完好。
2. **失败熔断**：同一根因连续尝试 2 轮无新进展，或真机累计预算耗尽时立即止损，保留完整证据与账本记入 `log.md`/`status.md` 交回 AGY 复盘，严禁静默扩预算或降标。
3. **最终交付**：将全窗口证据归档至 `data/architecture-cohesion-extensibility/`，更新本窗口 `status.md` 与 `log.md`，按清晰职责分层提交并推送至 `origin/architecture-cohesion-extensibility` 后停止，交回 AGY 终审（不自行关窗）。
