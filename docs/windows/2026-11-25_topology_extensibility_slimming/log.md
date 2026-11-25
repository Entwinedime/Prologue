# 顶层物理分层、四维扩展架构收口与不变量锁死深度瘦身日志

## 2026-11-25 — 窗口开启：基线盘点、深层目录规划与瘦身方案立项

**范围**　从已整合推送的 `main` 基线 `b500cbe` 切出新开发分支 `topology-extensibility-slimming`，完成顶层物理分层、四维扩展解耦与激进不变量锁死瘦身的 5 阶段 / 13 轮总体规划。

**改动**　新建本窗口 `plan.md`、`status.md` 与 `log.md`，明确 `engine/`（纯 C++23 仿真引擎）、`probes/`（纯底层原生探针/Patch/Shell）、`shared/`（跨语言单一真源静态资产）与 `python/prologue/{common,host,workers,integrations,verification}` 顶层物理结构，规划 `DeviceRuntime`、`PlatformBackend` 与 `frameworks/sglang/abi/` 多版本声明式 Hook 适配层，并为 M3–M5 设定逐子系统真机不变量清单与硬性非空行净减门槛（总底线 ≥ 7,500 非空行、冲刺 ≥ 10,000 行、Python 文件数 244 → ≤ 130）。本轮仅建立窗口计划文档，未修改业务源码。

**验证**　基于 `b500cbe` 实测全仓基线指标：业务源码 346 文件 / 51,240 物理行（Python 244f / 37,114L / 32,708 非空行；C++ 93f / 13,978L / 12,204 非空行；其他 9f / 148L）；8 个核心子目录存在多语言混装；三运行域可达交集仅 12 个共享模块、控制容器与宿主 CLI 除共享模块外交集为 0；`sglang/` 存在 40 文件 98 处 `ascend`/`torch_npu`/`hccl` 耦合与 51 文件 235 处 126 个 `sglang.srt.*` 私有符号直连；13 处 `AdapterRegistry.select()` 中 12 处无参走默认。

**结论与限制**　窗口基线数据与改造路线已锚定，等待用户启动 Codex 进入 M1（R01–R02）。本窗口对 NVIDIA CUDA、vLLM 与 SGLang 多版本仅落地抽象契约、既有实现迁移与零副作用静态推演，不编写新后端/新框架实机代码，全程保持新增物理卡时为 0。
