# 顶层物理分层、四维扩展架构收口与不变量锁死深度瘦身状态

**更新日期**　2026-11-26

**进度**　**已关闭（提前结束）**。已交付 12/14 轮（85.7%），完成 4/5 个阶段（M1–M4 全部验收通过，M5 交付 R11 `cbea08e`）。分支为 `topology-extensibility-slimming`，功能对照基线为 `b500cbe`。

**当前交接**　经用户确认提前终止本窗口：保留已提交的 R11（`cbea08e`），丢弃未提交的 R12 半成品工作区；本窗口产出的顶层物理分层、纯 `COPY` 镜像与四维接口解耦作为已稳定基础保留，而瘦身过程中因硬性文件数指标导致的超大文件职责混杂、层次交互扰动，以及局部采集/验证/观测中仍存在的刚性硬编码逻辑，统一转入下一自主窗口 `2026-11-26_architecture_cohesion_flexible_extensibility` 全局重构解决。[M4 验收汇总](../../../data/topology-extensibility-slimming/m4/acceptance.json)

| 阶段 | 轮次 | 已交付 | 状态 |
| --- | --- | ---: | --- |
| M1　物理分层与纯净打包 | R01–R02 | 2/2 | 完成 |
| M2　源码归口、平台与框架接口、作业路由 | R02b、R03–R05 | 4/4 | 完成 |
| M3　Ascend、PyTorch 与公共代码瘦身 | R06–R07 | 2/2 | 完成 |
| M4　SGLang 实现瘦身 | R08–R10 | 3/3 | 完成 |
| M5　验证代码瘦身与全仓验收 | R11–R13 | 1/3 | 暂停（提前结束，R12–R13 转入新窗口） |

**R11 结果**　验证核心从 21 个 Python 文件收至 8 个。11 组录制各做来源和输入两项对照，22 项与基线一致。验证全目录现有 31 个文件、10,097 个非空行。R12 继续处理结构、固定图和自主闭环三条路径。[对照明细](../../../data/topology-extensibility-slimming/m5/r11/source-comparison.json)

**业务源码规模**　按 Git 源码树统计 `src/` 中的业务文件，Python 文件数包含包标记。M4 较 M3 共减少 69 个 Python 文件、1,056 个非空行。按 AST 导入语句计，减少 303 个导入非空行，其余减少 753 行，含定义、逻辑与注释。

| 时点 | 业务文件 | 物理行 | 非空行 | Python 文件 | Python 物理行 | Python 非空行 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 基线 `b500cbe` | 346 | 51,240 | 45,037 | 244 | 37,114 | 32,708 |
| M1 `24ff96b` | 351 | 51,368 | 45,149 | 246 | 37,121 | 32,716 |
| M2 `d48605f` | 368 | 52,339 | 46,030 | 263 | 38,079 | 33,586 |
| R06 `4c207bd` | 351 | 52,292 | 45,982 | 246 | 38,032 | 33,538 |
| M3 `5f33571` | 336 | 51,877 | 45,630 | 231 | 37,617 | 33,186 |
| R08 `2fbd8c4` | 314 | 51,695 | 45,478 | 209 | 37,435 | 33,034 |
| R09 `1aa58f2` | 285 | 51,292 | 45,128 | 180 | 37,032 | 32,684 |
| R10 / M4 `934c46f` | 267 | 50,731 | 44,574 | 162 | 36,471 | 32,130 |
| R11 | 255 | 50,450 | 44,280 | 150 | 36,190 | 31,836 |

**M2 验收范围**　SGLang/Ascend 采集镜像和控制镜像分别核对了 124、72 个已安装模块。Qwen3-8B TP1 四个既有目标的 211,290 条执行记录、16 次 token 发布、四份请求时延及 43,276 项固定图成本查询均与基线逐字节一致。M2 新增物理卡时为零。硬件代际、CUDA、SGLang 新版本与 vLLM 完成接口级静态审计，实机兼容性另行验证。[M2 验收证据](../../../data/topology-extensibility-slimming/m2/acceptance.json)

**M3 交付**　Ascend 共 16 个 Python 文件，PyTorch 共 4 个，公共包共 7 个。PyTorch 保留全部 29 条显式守卫。投影和核验从原始录制分别建索引，各自判断语义。[源码规模](../../../data/topology-extensibility-slimming/m3/r07/source-measurements.json) · [接口与依赖](../../../data/topology-extensibility-slimming/m3/r07/extensibility-audit.json)

**真机与离线**　R06 在单卡采集了 TP2 的两个逻辑角色。R07 在物理卡 4 上采集 Qwen3-8B TP1，最终镜像得到 165 个设备样本、591 个 CPU 样本，独立模板在真实时间注入下完整匹配 107 个动作。留存的五组拓扑和成本数据有 63 份主要产物与基线逐字节一致。TP8 和三段 PP 的历史图保留原核验结论，双方分别从原录制重新投影后，来源核验均为 `consistent`。[M3 验收汇总](../../../data/topology-extensibility-slimming/m3/acceptance.json) · [R07 真机账本](../../../data/topology-extensibility-slimming/m3/r07/budget.json) · [多拓扑对照](../../../data/topology-extensibility-slimming/m3/r07/offline-comparison.json)

**M4 交付**　按包含包标记的同一口径，SGLang 从 M3 的 105 个文件降至 R09 的 54 个、R10 的 36 个。R10 将局部采集从 22 个文件、2,647 个非空行收敛到 4 个文件、2,096 个非空行，保留九条规则。单卡实测 8 B、16 B 两个同步拷贝点，增量发布后以独立成本完成预测。四目标的 211,290 条轨迹、16 次 token 发布、四份请求时延与 43,276 项固定图成本和基线一致。[源码统计](../../../data/topology-extensibility-slimming/m4/r10/source-measurements.json) · [单卡账本](../../../data/topology-extensibility-slimming/m4/r10/budget.json) · [离线对照](../../../data/topology-extensibility-slimming/m4/r10/offline-comparison.json)

**证据边界**　R09 单卡分轮录制支持逐 rank 建库；与并发真值比较时，TP2、PD 分别在比较 88、288 个动作后到达模板未覆盖的工作条件。R10 的新增真机证据覆盖同步拷贝规则。交付 Git 树和活动源码的字节码文件均为零；本地受保护的 `deprecated/` 与历史 `data/` 分别留有 382、137 个缓存。[R09 结构对照](../../../data/topology-extensibility-slimming/m4/r09/review.json) · [R10 核验范围](../../../data/topology-extensibility-slimming/m4/r10/bytecode-scope.json)
