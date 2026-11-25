# 顶层物理分层、四维扩展架构收口与不变量锁死深度瘦身状态

**更新日期**　2026-11-25

**进度**　M1、M2、M3 均已完成并通过 AGY 审查放行，已交付 8/14 轮（57.1%），共 3/5 个阶段（60%）。分支为 `topology-extensibility-slimming`。功能对照基线为 `b500cbe`，M3 业务终版为 `5f33571`（证据与文档 `c080478`）。

**当前交接**　AGY 已完成 M3 审查放行并基于 `5f33571` 实测数据校准 M4 目标（`frameworks/sglang/` 105f → ≤ 38f，务实净减 ≥ 700 非空行：R08 ≥ 150、R09 ≥ 350、R10 ≥ 200），经用户确认准予启动 M4（R08–R10）；M4 完成后须在 M4/M5 边界立即暂停交回 AGY 审查。

| 阶段 | 轮次 | 已交付 | 状态 |
| --- | --- | ---: | --- |
| M1　物理分层与纯净打包 | R01–R02 | 2/2 | 完成 |
| M2　源码归口、平台与框架接口、作业路由 | R02b、R03–R05 | 4/4 | 完成 |
| M3　Ascend、PyTorch 与公共代码瘦身 | R06–R07 | 2/2 | 完成 |
| M4　SGLang 实现瘦身 | R08–R10 | 0/3 | 待做 |
| M5　验证代码瘦身与全仓验收 | R11–R13 | 0/3 | 待做 |

**业务源码规模**　按 Git 对象统计 `src/` 中的业务文件，Python 文件数包含包标记。最新提交较 M2 减少 32 个业务 Python 文件、400 个非空行。

| 时点 | 业务文件 | 物理行 | 非空行 | Python 文件 | Python 物理行 | Python 非空行 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 基线 `b500cbe` | 346 | 51,240 | 45,037 | 244 | 37,114 | 32,708 |
| M1 `24ff96b` | 351 | 51,368 | 45,149 | 246 | 37,121 | 32,716 |
| M2 `d48605f` | 368 | 52,339 | 46,030 | 263 | 38,079 | 33,586 |
| R06 `4c207bd` | 351 | 52,292 | 45,982 | 246 | 38,032 | 33,538 |
| M3 `5f33571` | 336 | 51,877 | 45,630 | 231 | 37,617 | 33,186 |

**M2 验收范围**　SGLang/Ascend 采集镜像和控制镜像分别核对了 124、72 个已安装模块。Qwen3-8B TP1 四个既有目标的 211,290 条执行记录、16 次 token 发布、四份请求时延及 43,276 项固定图成本查询均与基线逐字节一致。M2 新增物理卡时为零。硬件代际、CUDA、SGLang 新版本与 vLLM 完成接口级静态审计，实机兼容性另行验证。[M2 验收证据](../../../data/topology-extensibility-slimming/m2/acceptance.json)

**M3 交付**　Ascend 共 16 个 Python 文件，PyTorch 共 4 个，公共包共 7 个。PyTorch 保留全部 29 条显式守卫。投影和核验从原始录制分别建索引，各自判断语义。[源码规模](../../../data/topology-extensibility-slimming/m3/r07/source-measurements.json) · [接口与依赖](../../../data/topology-extensibility-slimming/m3/r07/extensibility-audit.json)

**真机与离线**　R06 在单卡采集了 TP2 的两个逻辑角色。R07 在物理卡 4 上采集 Qwen3-8B TP1，最终镜像得到 165 个设备样本、591 个 CPU 样本，独立模板在真实时间注入下完整匹配 107 个动作。留存的五组拓扑和成本数据有 63 份主要产物与基线逐字节一致。TP8 和三段 PP 的历史图保留原核验结论，双方分别从原录制重新投影后，来源核验均为 `consistent`。[M3 验收汇总](../../../data/topology-extensibility-slimming/m3/acceptance.json) · [R07 真机账本](../../../data/topology-extensibility-slimming/m3/r07/budget.json) · [多拓扑对照](../../../data/topology-extensibility-slimming/m3/r07/offline-comparison.json)
