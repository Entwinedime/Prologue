# 多拓扑、复杂调度与真实流量大范围验证及局部补采泛化提速 — 分层执行计划（Plan）

> **动态计划维护说明**：本文件由 AGY 制定精细分层的 5 层 14 阶段初始蓝图（L0–L4 / S01–S14），每阶段均写清职责边界、代码落点、模型配置、验证输入与量化门禁。本计划受同目录只读纲领 [outline.md](outline.md) 约束；**Codex 在每完成一个阶段后，必须执行一次全仓 Bigger-Picture Overview**，依据真实架构、误差归因与卡时数据在文末追加演进记录，并动态细化、拆分或增补后续子阶段（阶段总数可随实测根因自然展开，但不得删减纲领覆盖维度或弱化门禁）。

**起点与纪律**　分支 `large-scale-validation`（起点 `8d10807`，行为对照基线 `b500cbe`），**全流程严禁执行 `git commit` 与 `git push`**，所有开发产物落于 `data/large-scale-validation/dev/`，所有冻结留出产物落于 `data/large-scale-validation/holdout/`，严禁混目录或跨集泄漏。

---

## 一、五层架构与阶段总览表

| 层级 | 阶段范围 | 物理卡规模 | 聚焦模型与配置域 | 核心解决的问题（路线二 + 路线一协同） | 参考卡时预算 |
| --- | --- | --- | --- | --- | ---: |
| **L0　底座清债** | S01–S02 | 0–3 卡 | 既有留存 PD / TP1 / TP2 / TP8 | 环境盘点 + 根治 PD 825 端口差异、CANN 1ns 重叠与 TP8 参考资格 | ≤ 1,800s |
| **L1　复杂调度与补采提速** | S03–S05 | 1 卡 | `Qwen3-8B` TP1（标准池 + 小池背压）× Chunked / Mixed / Prefix Cache + 8–32 请求并发 | 同会话多切点合批提速（冷启 < 50%）+ 复杂调度局部配方 + 宿主成本 + L1 冻结留出 < 6% MAPE | ≤ 9,000s |
| **L2　多卡 TP 与跨模型** | S06–S08 | 1 / 2 / 4 / 8 卡 | `Qwen3-8B` TP2 / TP4 / TP8 + `Mistral-7B-v0.3` TP2 | 单卡分片多 rank 泛化 + 并发模板边界修复 + 物理通信分组补采 + L2 冻结留出 < 6% MAPE | ≤ 18,000s |
| **L3　异构高级并行** | S09–S11 | 2–8 卡 | `Qwen3-8B` PD (TP1/TP2) + `Mistral-7B` PP3 / TP2×PP3 + `Qwen3-30B-A3B` TP4×EP4 | PD / PP / MoE 局部配方插件扩展 + 跨阶段/单边传输/变长通信归因 + L3 冻结留出 < 6% MAPE | ≤ 21,600s |
| **L4　真实流量、压测与总终验** | S12–S14 | 1–8 卡 | 全部 8 域 + Azure 2024 真实流量与 32→128→256 请求压力档 | 长窗口压测 + 误差驱动模型族演进 + `prepare_matrix`/`matrix` 8 域总冻结留出验收 + 文档/品质终检 | ≤ 21,600s |

---

## 二、Layer 0：环境核验与历史结构/资格债清零（S01–S02）

### S01　主机资源、模型路径与三项历史遗留根因精准诊断
- **为什么做**：在投入任何多卡真机预算前，先确认本机 3 款模型权重、Azure 2024 trace 数据源、8 卡 NPU 可用性与双镜像状态，并从既有证据精准剖析三项历史遗留的根因，避免带病进入大规模验证。
- **具体盘点与诊断边界**：
  1. 核验 `/models/Qwen3-8B`（36 层 BF16）、`/root/models/markov-next/Mistral-7B-v0.3`（32 层 BF16）、`/root/models/markov-next/Qwen3-30B-A3B`（48 层 128 专家 top_k=8 BF16）、8 张 NPU（`/dev/davinci0..7`）空闲状态及宿主机 Azure 2024 CSV 路径；
  2. 剖析留存 `pd-structure` 5,355 动作中 11 次 `prefill/0 run_batch` 产生 **825 处 `ports` 差异**的精确根因（追踪 `sglang/observation/{serving,kv_scopes,call_ports}.py`、`src/engine/core/templates/`、`src/engine/adapters/sglang/` 与 `verification/structure/`）；
  3. 剖析历史长 PD trace 中 CANN 合并 trace 终点四舍五入产生 **1ns 相邻任务重叠**（而 CANN 原生任务表端点精确相等）在 `ascend/trace_store.py` / `ascend/analysis/` / `ascend/reference/` 的处理路径；
  4. 剖析 TP8 留存参考 `unverified` 的具体阻断原因，以及 R09 暴露的并发 `allocation_prefix` 与 `block_table_strided=0/1` 模板边界条件。
- **验收与交接**：
  - 产出 `data/large-scale-validation/dev/l0/diagnosis.json`（含主机模型/NPU/CSV 盘点及三项根因定位，0 新增真机卡时）；
  - 交接：`8d10807` 基线 + 留存 PD/TP8 证据 → 三项根因定位报告 → S02 定点修复。

### S02　根治 PD 825 端口差异、CANN 1ns 重叠与 TP8 参考资格缺陷
- **为什么做**：把已知结构与参考资格缺陷在单卡/离线层彻底清零，确保后续所有层的结构对照与固定图基线建立在 0 历史污染的底座之上。
- **代码落点与边界**：
  - `src/python/prologue/integrations/frameworks/sglang/`（PD prefill `run_batch` 端口/绑定边界）与 `src/engine/adapters/sglang/`（如涉及模板端口归一化）；
  - `src/python/prologue/integrations/platforms/ascend/{trace_store.py,analysis/,reference/}`（优先使用 CANN 原生任务表精确端点消除合并 trace 的 1ns 舍入重叠，严禁靠人为加时间容差掩盖真实因果倒置）；
  - 针对 TP8 参考资格缺口完成对应解析或采集侧修正（不放宽 `source_clock` SymPy 认证或任何隔离判据）。
- **验收门禁**：
  - 留存 PD 结构验证从 **825 处端口差异降为 0 差异**（5,355/5,355 `full_structure_matched`）；
  - CANN 1ns 舍入重叠回归用例通过独立 `verify source`；
  - 离线基础回归不退化（`b500cbe` Qwen TP1 4 目标 211,290 轨迹、16 token 发布、43,276 项固定图工作与 678 样本拟合一致）；
  - 交接：S01 根因报告 → 0 历史结构债的代码与双镜像底座 → Layer 1。

---

## 三、Layer 1：单卡复杂调度 + 局部补采泛化与冷启提速（S03–S05）

### S03　同会话多切点合批与常驻 Worker 复用（局部采集提速）
- **为什么做**：历史实测局部 attention 会话冷启动卡时占同点完整服务的 **75.5%**（目标 < 50%），根本原因在于相同模型/域/配置的多个缺口会反复启容器和冷加载 `ModelRunner`。在进入大规模补库前先打通提速底座，可为 L1–L4 节省数倍卡时。
- **代码落点与设计边界**：
  - `src/python/prologue/host/acquisition/{rules.py,replenishment.py}`：扩展 `session_key` 兼容分组与多 query 合批调度，使同 `(framework, revision, model, domain_structure)` 下多个兼容局部切点合并进入同一次 `NativeAttempt`；
  - `src/python/prologue/integrations/frameworks/sglang/observation/{probes/,model_samples.py,scheduler_samples.py}`：让 `--local-attention`、`--local-worker`、`--local-host-pair` 与 `--local-copy` 支持单次预热后按批连续执行多个切点/长度组合，切点间仅做原生逻辑 cache flush（保留已加载权重与编译内核），并分别记录各切点的独立观测与总冷启分摊账本；
  - **边界红线**：严禁引入跨任务驻留守护进程或破坏单任务原子隔离，合批严格限定在同一任务会话与同一预算账本内。
- **验收门禁**：
  - 在空闲单卡上用相同多点 Decode attention + 局部批次缺口实测对比“同会话合批局部采集”vs“同点完整 serving 采集”，实证局部采集总卡时占比 **< 50%**（记录于 `data/large-scale-validation/dev/l1/speedup-benchmark.json`）；
  - 跨切点独立读回资格校验 100% 通过，0 状态串扰；
  - 交接：L0 底座 → 低冷启合批局部采集引擎 → S04 与全部后续波次。

### S04　扩展复杂调度机制局部补采配方（Prefix Cache / Chunked Prefill / Mixed Batch）
- **为什么做**：当前 [scheduler.py](../../../src/python/prologue/integrations/frameworks/sglang/acquisition/recipes/scheduler.py) 写死了 `impl["cache"]["disabled"] and not any(impl["scheduling"].values())`，导致启用 Prefix Cache、Chunked Prefill 或 Mixed Batch 时无法通过局部规则自动补采，只能被迫退回完整服务；且历史记录显示复杂调度下“续块前宿主时序与调度期 CPU 成本”仍是误差主因。
- **代码落点与插件边界**：
  - 严格通过 `common/components.py` 包贡献机制，在 `src/python/prologue/integrations/frameworks/sglang/acquisition/recipes/` 与 `observation/probes/` 新增或扩展以下局部配方（核心分发器 0 改动）：
    1. **Prefix Cache 局部配方**：构造含共享前缀的冷写入 → 命中复用 → 小池容量驱逐（`evict_from_tree_cache`）三种最小局部边界状态，输出对应结构片段与命中/驱逐 CPU 与设备成本样本；
    2. **Chunked Prefill & Mixed Batch 局部配方**：解除 `LocalSchedulerBatch` 的非分块/非混批限制，扩展 `priming()` 支持按 `max_prefill_tokens` 构造中间 chunk 续块与 Prefill+Decode 混批边界，同时覆盖续块前宿主 CPU 调度切口（配合 `paired_host` 双遍独立计时）；
    3. **并发页边界与 `allocation_prefix` 补采**：覆盖单成员/多成员 `allocation_prefix` 与 `block_table_strided=0/1` 页边界模板缺口。
- **验收门禁**：
  - 新增/扩展配方全部由包贡献自动发现，优先级与 `EvidenceGap` 自动回退链通过验证；
  - 在单卡上经官方 `library acquire` + `library publish --from-session` 实测覆盖 Prefix Cache（命中+驱逐）、Chunked Prefill、Mixed Batch 及其组合的局部结构与宿主/设备成本补采；
  - 交接：S03 提速引擎 → 复杂调度局部补采插件集 → S05 开发回归与留出验收。

### S05　L1 开发回归、固定图误差归因与 L1 冻结独立留出验收（1 卡）
- **为什么做**：把 `Qwen3-8B TP1` 下的复杂调度机制从“结构跑通”推向“冷库自动补采 → 热库零卡复用 → 冻结独立留出 < 6% MAPE”。
- **具体实验矩阵（`Qwen3-8B` BF16，1 卡）**：
  - **配置集合**：① 标准池 + Chunked Prefill；② 标准池 + Mixed Batch；③ 标准池 + Prefix Cache；④ 标准池 + 三机制全开（`cache + chunk + mixed`）；⑤ 小 KV 池 + 三机制全开（强制触发缓存驱逐与队列背压）；
  - **Dev 集**：单请求 + 4–16 请求错峰到达与共享前缀组合，从空库跑通 `predict explore --deployment` 自动局部补采 → 热库零卡复预测 → `verify {source, structure, fixed-graph}`，按固定图误差归因补齐续块前宿主时序、同步搬运与 Attention 成本盲区；
  - **Holdout 集**：冻结 L1 源码与发布库，用独立未见请求（8–32 请求持续到达、混合前缀命中/淘汰、跨页分块长度）在物理单卡采集独立详细参考与轻量参考，运行三路验证。
- **L1 硬验收门禁**：
  - 5 组配置独立留出结构全部 `full_structure_matched`（0 差异）；
  - 固定图实测基线全部通过，热库在换请求/换容量下实现 0 新增卡时复预测；
  - 独立留出自主闭环 **TTFT MAPE < 6% 且 TPOT MAPE < 6%**，所有 `training` 隔离核验通过；
  - 交接：L1 冻结报告与低冷启复杂调度补采体系 → Layer 2。

---

## 四、Layer 2：多卡 TP2 / TP4 / TP8 与跨模型物理通信闭环（S06–S08）

### S06　多卡 TP 局部补采泛化、物理通信分组补采与多卡宿主成本
- **为什么做**：当前 [attention.py](../../../src/python/prologue/integrations/frameworks/sglang/acquisition/recipes/attention.py) 限制 `logical_tp == 1`，且多卡 TP 的跨 rank 宿主开销与按物理操作数分组的 HCCL 通信补采尚未形成完整自动化闭环。
- **代码落点与边界**：
  - `recipes/attention.py` & `observation/probes/model.py`：解除 `logical_tp == 1` 限制，支持通过单卡顺序分片对 TP2/TP4/TP8 任意代表 rank（含 rank ≥ 2）定向执行 Decode attention 局部采样与 KV 读回；
  - `recipes/services.py`（`LocalPeerWork`）& `platforms/ascend/host/collectives.py`：强化多卡物理通信（AllReduce / AllGather / ReduceScatter）按参与者集合、归约操作与端点分组的自动补采、HCCL 通知寄存器收紧与同操作数跨配置复用；
  - 补齐多卡 TP 下各 rank 宿主调度/同步提交 CPU 成本的独立采样与拟合覆盖（禁止无依据直接把 TP1 宿主样本硬复制到 TP8）。
- **验收门禁**：
  - 单卡顺序分片成功采样 TP2/TP4/TP8 所需代表 rank 局部结构与本地设备样本；
  - 物理 2/4/8 卡通信样本按操作数分组拟合后，在 C++ `prologue cost-points` / `work-costs` 上完整覆盖目标多卡通信特征坐标；
  - 交接：L1 底座 → 多卡 TP 本地分片 + 物理通信补采体系 → S07。

### S07　L2 开发回归与双来源一致性对照（`Qwen3-8B TP2/TP4/TP8` + `Mistral-7B-v0.3 TP2`）
- **为什么做**：验证“单卡分片建库 + 物理通信样本”与“完整多卡建库”在结构与固定图上的一致性，同时通过 `Mistral-7B-v0.3` 检验跨模型架构（GQA/层数/词表差异）的零特判泛化。
- **具体验证边界（Dev 集）**：
  - `Qwen3-8B` TP2（常规 + Chunked/Mixed/Prefix Cache 组合）、TP4、TP8（16-token 与 32-token 页两种布局）；
  - `Mistral-7B-v0.3` TP2（跨模型自动解释与冷建库 → 热复用）；
  - 执行双来源对照（单卡分片来源 vs 完整多卡来源的 `verify structure`），并在完整多卡真值上跑 `verify fixed-graph` 定位所有通信争用、小包延迟与多卡宿主偏差。
- **验收门禁**：
  - Dev 集 4 个多卡配置双来源结构 0 差异，TP8 参考 `source_clock` 与全 rank 因果核验 `consistent`；
  - 固定图实测基线全部通过，高误差工作项已定位并补齐对应样本或特征表达；
  - 交接：L2 Dev 归因与修补后的多卡库 → S08 冻结留出。

### S08　L2 冻结独立留出验收（2 / 4 / 8 卡真实物理拓扑）
- **为什么做**：在完全隔离的物理多卡真值上验证多卡 TP 与跨模型的自主预测精度。
- **验证输入与门禁**：
  - 冻结 L2 源码与发布库，另抽独立留出请求（覆盖单请求、多请求并发错峰、跨页边界），分别在物理 2 卡（Qwen TP2 + Mistral TP2）、4 卡（Qwen TP4）、8 卡（Qwen TP8）上采集独立详细参考与轻量参考；
  - **L2 硬验收门禁**：全套 4 个多卡配置结构 0 差异、固定图合格、来源隔离通过、**各配置独立留出 TTFT 与 TPOT MAPE 均 < 6%**（同时单列“单卡分片建库 + 物理通信样本”与“完整多卡建库”两组精度的直接对照）；
  - 交接：L2 多卡 TP 冻结验收报告 → Layer 3。

---

## 五、Layer 3：异构高级并行 — PD 分离、PP 流水线与 MoE-EP（S09–S11）

### S09　PD 分离、PP 流水线与 MoE-EP 局部补采插件扩展
- **为什么做**：把新建立的包贡献插件体系推向最复杂的三类异构并行拓扑，实现“缺什么切口就补什么局部配方”，而不是每次都重跑整套异构大集群。
- **代码落点与插件边界**（全部以独立模块贡献至 `sglang/acquisition/recipes/`、`sglang/observation/probes/`，核心调度器 0 改动）：
  1. **PD 分离配方**（`Qwen3-8B Prefill TP1 / Decode TP2`）：扩展 `LocalPrebuilt` 支持启用 prefix cache 时的接收端池几何，补齐 Prefill/Decode 双实例局部初始化绑定、KV 单边传输与首 token 元数据通知的物理样本补采配方；
  2. **PP 流水线配方**（`Mistral-7B-v0.3 PP3` & `TP2×PP3`）：新增覆盖首/中/末 stage 代表角色的局部层/输入输出切口采样，以及相邻 stage 间激活与输出 token 环的跨卡点对点传输采样；
  3. **MoE-EP 配方**（`Qwen3-30B-A3B TP4×EP4`，48 层 128 专家 top_k=8）：新增覆盖条件 EP 路由与 TP 对齐 padding 的局部路由器/专家层切口采样（利用已有按层主机暂存支持单卡分片），以及 DeepEP Dispatch/Combine 变长通信的物理多卡采样。
- **验收门禁**：
  - 三类插件零改动接入 `LocalAcquisition` 与 worker 探针注册表；
  - 各类插件均经官方 `library acquire` → `library publish --from-session` 完成至少 1 项真实局部补采闭环；
  - 交接：L2 底座 → 异构并行局部补采插件全集 → S10。

### S10　L3 开发回归与固定图误差归因（PD / PP3 / TP2×PP3 / TP4×EP4）
- **为什么做**：在完整物理异构拓扑上验证结构保真度并完成固定图误差归因，攻克单边 KV 传输、流水线气泡/共识唤醒与 MoE 专家不均衡通信的建模精度瓶颈。
- **具体验证边界（Dev 集）**：
  - `Qwen3-8B` **PD（Prefill TP1 / Decode TP2，3 卡）**：涵盖逐请求准入、双成员并发准入、KV+元数据两路通知与可选 Decode prefix cache；
  - `Mistral-7B-v0.3` **PP3（3 卡）** 与 **TP2×PP3（6 卡）**：涵盖多微批流水交接、stage 间主机/设备搬运端点与末级共识唤醒；
  - `Qwen3-30B-A3B` **TP4×EP4（4 卡）**：涵盖 `balanced`、`concentrated`、`alternating` 三种显式专家路由模式及 TP 对齐 padding。
- **验收门禁**：
  - 四种异构拓扑在 Dev 集上结构验证 **0 差异**（含 PD 从历史 825 差异彻底清零，PP 微批共识唤醒与 MoE 逐层 router/expert 配对全通过）；
  - 固定图实测基线通过并完成逐工作类误差归因，暴露的跨 stage 传输、PD 异步通知等待或 DeepEP 变长通信成本短板完成修补；
  - 交接：L3 Dev 归因与修补后的异构库 → S11。

### S11　L3 冻结独立留出验收（3 卡 PD / 3&6 卡 PP / 4 卡 MoE-EP）
- **为什么做**：验证异构高级并行拓扑在完全隔离留出负载下的端到端结构与精度达标。
- **验证输入与门禁**：
  - 冻结 L3 源码与双库发布，在完整物理拓扑（3 卡 PD、3 卡 PP3、6 卡 TP2×PP3、4 卡 TP4×EP4）上用隔离留出请求执行 `verify {source, structure, fixed-graph, autonomous}`；
  - **L3 硬验收门禁**：四种异构拓扑独立留出结构全匹配、固定图合格、来源隔离通过、**TTFT 与 TPOT MAPE 均 < 6%**（若某超边界子项确受原生上游/硬件时钟限制，按纲领熔断保留完整原生证据并单列资格边界，严禁手调系数）；
  - 交接：L3 异构并行冻结验收报告 → Layer 4。

---

## 六、Layer 4：真实流量、高并发压力、模型族演进与 8 域总终验（S12–S14）

### S12　真实流量（Azure 2024 Trace）与高并发压力开发回归及模型族按需演进
- **为什么做**：短请求达标不代表线上真实流量达标；长上下文（2K–8K+）、偏态长短混合到达、32→128→256 请求持续并发和小 KV 池驱逐背压往往会暴露长窗口异步边界、设备等待取证与非线性成本特征瓶颈。
- **具体验证与代码边界**：
  1. **压力负载构造**：通过 [azure_workload.py](../../../example/coarse_graph/azure_workload.py) 从真实 Azure 2024 inference trace CSV 截取代表窗口（若宿主机未预置 CSV，则在 `data/large-scale-validation/dev/l4/` 明示来源并构造保留偏态分布与真实突发到达间隔的可复现长流负载），按 **32 → 128 → 256 请求**三级阶梯递增；
  2. **长窗口稳定性修复**：在 Dev 压测中定位并修复长窗口录制超时、小 KV 池反复驱逐/重算时的事件回收与在途数据保护问题、以及大队列下调度器宿主开销漏记问题；
  3. **误差驱动的性能模型族演进（严格按证据触发）**：检查 L1–L4 固定图误差归因汇总——仅当现有 [cost_families.json](../../../src/shared/cost_families.json) 的 4 个模型族在长上下文 Attention 交互、多卡通信争用/小包固定开销或大队列宿主开销上存在可复现的系统性偏差时，同步在 `src/shared/cost_families.json` + `src/python/prologue/host/performance/` + `src/engine/core/performance/` 登记新通用模型族，通过整组坐标留出择优验证后启用（禁止按单案例过拟合）。
- **验收门禁**：
  - 32 / 128 请求真实流量（及条件允许下的 256 请求压力档）完整通过 `verify {source, structure, fixed-graph, autonomous}`，无小池超时、无内存/事件泄漏；
  - 若新增性能模型族，C++ Release 0 warning 编译通过、Python 与 C++ 双端求值逐纳秒一致且留出误差实证下降；
  - 交接：经高并发真实流量锤炼的最终源码与模型族 → S13 总冻结矩阵。

### S13　全仓统一冻结与 8 域总独立留出矩阵验收（`prepare_matrix.py` + `matrix.py`）
- **为什么做**：把全窗口所有成果放到同一份不可变冻结快照下，经官方矩阵工具一次性完成跨模型 × 跨拓扑 × 复杂调度 × 真实流量的总留出验收，产出无懈可击的最终精度总表。
- **执行流程（严格遵循冻结隔离协议）**：
  1. 冻结最终工作区 `src/`（生成 `source-manifest.json` 汇总 SHA-256），在 `modeling` 容器编译最终 C++ Release 二进制，用 `./docker/build_inference.sh` 纯 `COPY` 重建最终 `sglang` 与 `control` 双镜像；
  2. 以冻结清单摘要作为唯一 `--seed`，用 [prepare_matrix.py](../../../example/coarse_graph/prepare_matrix.py) 一次性生成 8 域 + 真实流量压力档的独立留出输入（严禁看结果重新抽样）：
     - `qwen8-tp1`、`qwen8-tp1-cache-chunk-mixed`、`qwen8-tp2`、`qwen8-tp8`、`qwen8-pd-tp1-tp2`、`mistral7-pp3`、`mistral7-tp2-pp3`、`qwen30-tp4-ep4` 及真实流量/并发配置排序组；
  3. 用 [matrix.py](../../../example/coarse_graph/matrix.py) 依次完成全矩阵 `sources → detailed/light/targets → compare → fixed → autonomous`；
  4. 汇总产出 `data/large-scale-validation/holdout/accuracy-summary.json`（逐域逐案例列出结构匹配、固定图基线误差、成本替换误差、自主 TTFT/TPOT MAPE、跨配置排序一致性及来源隔离状态）。
- **验收门禁**：
  - 全部 8 域及真实流量留出案例通过来源隔离核验、结构全匹配、固定图合格、**TTFT 与 TPOT MAPE < 6%** 且跨配置性能排序与物理实测一致；
  - 累计物理卡时 ≤ 72,000 卡秒、后处理 ≤ 18,000 秒、存储 ≤ 256 GiB，分项账本完整。

### S14（Final Stage）　代码品质终检、公开文档同步、证据归档与零提交交审
- **为什么做**：确保经历大范围功能与验证演进后，代码库依然维持上一窗口建立的高架构品质，文档与真实能力 100% 对齐，证据链完备可独立验真。
- **核验与收尾清单**：
  1. **代码品质零回退自检**：全仓业务 Python **> 550 NE 过载模块 = 0**、Python & C++ 函数体内 **> 25 行无空行代码墙 = 0**、空行占比与高价值注释维持健康可读、AST **0 环 / 0 缺失**、四维隔离 **0 越层**、`ruff check src/python example` **0 错误**、活动源码 **0 `.pyc`**、4 种打包形态与 5 域 16 子命令可用、双镜像实装 SHA-256 逐文件匹配；
  2. **基础离线回归核对**：核对 `b500cbe` 四目标 211,290 轨迹、16 token 发布、678 样本拟合与 43,276 固定图工作（若因修复 PD 825 端口差异或新增模型族产生预期内的合理变化，须逐项给出变化归因与前后对照）；
  3. **公开文档同步**：更新 `README.md` 与静态设计文档中的已验证配置表、新局部补采配方、性能模型族与验证结论边界（严守文档跨层零引用）；
  4. **终审归档**：将 `source-manifest.json`、`evidence-manifest.json`、L0–L4 分层预算总账、`speedup-benchmark.json`、`accuracy-summary.json` 归档至 `data/large-scale-validation/`，刷新 `status.md` 与 `log.md`，**保持全流程零 `git commit` / 零 `git push`**，停止并交回 AGY 一次性全面终审（不要自行关窗）。

---

## 七、阶段演进与 Bigger-Picture Overview 记录（由 Codex 每阶段动态追加）

> Codex 每完成一个阶段（S01..S14 及任何动态拆出的子阶段），必须在此追加一段「阶段后全局 Overview 与后续计划修订」，从全局视角总结刚完成阶段对架构、误差归因、补采效率与剩余风险的影响，并据此动态修订或增补上方后续阶段的具体安排。
