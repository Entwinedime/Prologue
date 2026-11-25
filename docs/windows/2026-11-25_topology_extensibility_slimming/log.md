# 顶层物理分层、四维扩展架构收口与不变量锁死深度瘦身日志

## 2026-11-25 — 窗口开启：基线盘点、深层目录规划与瘦身方案立项

**范围**　从已整合推送的 `main` 基线 `b500cbe` 切出新开发分支 `topology-extensibility-slimming`，完成顶层物理分层、四维扩展解耦与激进不变量锁死瘦身的 5 阶段 / 13 轮总体规划。

**改动**　新建本窗口 `plan.md`、`status.md` 与 `log.md`，明确 `engine/`（纯 C++23 仿真引擎）、`probes/`（纯底层原生探针/Patch/Shell）、`shared/`（跨语言单一真源静态资产）与 `python/prologue/{common,host,workers,integrations,verification}` 顶层物理结构，规划 `DeviceRuntime`、`PlatformBackend` 与 `frameworks/sglang/abi/` 多版本声明式 Hook 适配层，并为 M3–M5 设定逐子系统真机不变量清单与硬性非空行净减门槛（总底线 ≥ 7,500 非空行、冲刺 ≥ 10,000 行、Python 文件数 244 → ≤ 130）。本轮仅建立窗口计划文档，未修改业务源码。

**验证**　基于 `b500cbe` 实测全仓基线指标：业务源码 346 文件 / 51,240 物理行（Python 244f / 37,114L / 32,708 非空行；C++ 93f / 13,978L / 12,204 非空行；其他 9f / 148L）；8 个核心子目录存在多语言混装；三运行域可达交集仅 12 个共享模块、控制容器与宿主 CLI 除共享模块外交集为 0；`sglang/` 存在 40 文件 98 处 `ascend`/`torch_npu`/`hccl` 耦合与 51 文件 235 处 126 个 `sglang.srt.*` 私有符号直连；13 处 `AdapterRegistry.select()` 中 12 处无参走默认。

**结论与限制**　窗口基线数据与改造路线已锚定，等待用户启动 Codex 进入 M1（R01–R02）。本窗口对 NVIDIA CUDA、vLLM 与 SGLang 多版本仅落地抽象契约、既有实现迁移与零副作用静态推演，不编写新后端/新框架实机代码，全程保持新增物理卡时为 0。

## 2026-11-25　M1 交付

**范围**　开发分支 `topology-extensibility-slimming`。从 `b500cbe` 的独立源码快照构建旧可执行文件和控制包。两个版本读取同一份 Qwen3-8B TP1 发布库、工作负载、固定图和 678 行性能样本。证据入口为 `data/topology-extensibility-slimming/m1/acceptance.json`。

**改动**　`ba07034` 将 C++ 执行引擎、原生探针和跨语言数据分开。预测工具及局部图工具通过同一注册表选择框架，SGLang 负责 worker 的状态与位置绑定。C++ 和宿主包共用 `shared/cost_families.json`。`24ff96b` 建立独立的 Python 运行域，移除旧 `src/` 和 AST 镜像裁剪脚本。

**验证**　modeling 容器完成 Release 全目标构建（49/49 步，0 错误、0 警告）。15 个 C++ 子命令的帮助与基线逐字节相同，实测输入导出的 662 个动作全部成功展开，动作与局部子图输出逐字节相同。宿主源码、由 sdist 构建的 wheel、定向安装和可编辑安装均能加载唯一的成本族目录。五个 CLI 域的 16 个子命令与基线一致。示例的隔离入口仅挂载 Python 与共享数据即可加载正式 CLI。静态审计 250 个模块、983 条内部导入边，循环、缺失导入、缺失包标记、核心越层包含及第一方字节码残留均为 0，Ruff 通过。

在已提交代码的干净检出中通过 `docker/build_inference.sh` 构建双镜像，复用相同官方 SGLang 修订的安装缓存，重新编译 Ascend 和 Gloo 探针。使用阶段专用镜像标签，避免宿主标准标签的周期清理打断核验。采集镜像 `3c71643190ec` 安装 110 个第一方 Python 模块，控制镜像 `44e14ccb042c` 安装 60 个。两份文件清单与源码 SHA-256 完全一致，约定的宿主、验证和采集编排模块泄漏为 0。采集、控制和插件入口均可导入，两个动态库的指定符号均可加载。原生依赖的 Torch CMake 配置提示缺少可选静态 Kineto 库，探针编译和符号核验通过。清单为 `sglang-installed.json`、`control-installed.json`，正式构建记录为 `images.log`。

固定发布库 114 个文件的哈希保持一致。标准池与大池分别配单请求、双请求，四种组合全部生成预测。211,290 条执行轨迹、16 次 token 发布和四份请求时延与基线逐字节相同，另与留存轨迹作了交叉核对。整轮及目标结果仅剔除宿主实际耗时、查询计时和输出占用字节三项运行统计后比较，其余字段完全一致。`library inspect`、`cost fit`、两组 `cost-points` 和 43,276 项 `work-costs` 的完整文件逐字节相同，其中两版均有 15,223 项缺少价格。各项哈希和运行统计见 `regression.json`。实机采集与其他拓扑不在本阶段验收范围，新增物理卡时为 0。

下表按 Git 对象统计，每格依次是文件数、物理行、非空行。`.hpp.in` 和嵌套的 CMake 清单计入辅助文件。开窗记录中的 C++ 非空行 12,204 包含头模板的 4 行，表内将其与辅助文件一并统计一次。

| 范围 | 基线 | R01 后 | R02 后 | M1 净变化 |
| --- | --- | --- | --- | --- |
| 业务 Python | 244 / 37,114 / 32,708 | 244 / 37,121 / 32,714 | 246 / 37,121 / 32,716 | +2 / +7 / +8 |
| C++ 源文件和头文件（含探针） | 93 / 13,978 / 12,200 | 95 / 14,030 / 12,240 | 95 / 14,030 / 12,240 | +2 / +52 / +40 |
| 业务辅助文件 | 9 / 148 / 129 | 10 / 217 / 193 | 10 / 217 / 193 | +1 / +69 / +64 |
| **业务合计** | **346 / 51,240 / 45,037** | **349 / 51,368 / 45,147** | **351 / 51,368 / 45,149** | **+5 / +128 / +112** |
| 示例 Python | 4 / 724 / 638 | 4 / 725 / 639 | 4 / 725 / 640 | 0 / +1 / +2 |
| Docker Python 裁剪脚本 | 1 / 52 / 44 | 1 / 52 / 44 | 0 / 0 / 0 | −1 / −52 / −44 |
| 根构建配置及其余 Docker 文件 | 11 / 576 / 494 | 11 / 511 / 433 | 11 / 512 / 436 | 0 / −64 / −58 |

**结论与限制**　M1 完成，停在 M2 入口，等待 AGY 审查与用户放行。

## 2026-11-25 — AGY 审查 M1 与顶层 `src/` 归口计划修订

**范围**　独立复核 M1（`b0796e9..aa49d03`）全部交付，并根据用户对仓库根目录整洁度的要求修订本窗口计划。

**改动**　AGY 修订 `plan.md` 与 `status.md`：在 M2 开头增设前置切片 `R02b`，将 `engine/`、`probes/`、`shared/`、`python/` 统一收归 `src/` 下（`src/{engine,probes,shared,python}`），使仓库根仅保留 `src/`、`docker/`、`example/`、`docs/`、`third_party/` 五个受跟踪目录，同时保持 `src/` 内部 100% 语言与运行域物理隔离；窗口总轮次调整为 14 轮。本轮仅修订计划文档，未修改业务源码。

**验证**　独立静态核验 M1 交付：`engine/` 与 `probes/` 0 个 `.py`、`python/` 0 个非 `.py`、`engine/core/` 0 越层 `#include` 与 0 框架/平台字样、246 个 Python 模块 0 导入环与 0 跨层违例、旧 `src/` 与 `package_runtime.py` 彻底清除、纯 `COPY` 模拟 `sglang`/`control` 双镜像 110/60 模块 0 泄露 0 断链、`ruff check python example` 0 错误、`.pyc` 0 残留；[regression.json](../../../data/topology-extensibility-slimming/m1/regression.json) 确认新增物理卡时为 0 且四目标 211,290 条轨迹与基线逐字节一致。

**结论与限制**　M1 验收通过，准予进入 M2（`R02b + R03–R05`）。`R02b` 仅涉及目录归口与 5 处构建/打包/示例路径更新，不影响业务语义；后续阶段请沿用标准四段日志格式并同步更新双 Dockerfile 新增 `COPY` 路径。

## 2026-11-25 — R02b 源码目录归口

**范围**　从已同步的 `733656a` 开始 M2，把四个交付目录移入 `src/`。

**改动**　`5485a5e` 完成目录迁移，更新根构建、宿主打包、两个镜像和示例的路径。容器内安装路径与 Python 包名保持原值。

**验证**　Git 记录保留文件迁移关系，根目录有五个受跟踪目录，变更通过 `git diff --check`。

**结论与限制**　R02b 实现已提交。双镜像构建与离线产物对照在 M2 集中执行。

## 2026-11-25　M2 设备接口、框架版本与作业路由

**范围**　R03–R05 处理三个选择发生的位置。框架调用设备能力时使用平台接口，代码访问 SGLang 内部符号时使用版本声明，宿主和 worker 根据同一份作业声明装配实现。

**改动**　`fe24246` 抽出设备操作和宿主侧观测接口，CPU 控制进程只读取平台描述。`d68d144` 将原生符号与可安装、可恢复的 hook 集中到 `4e155ecc` 版本声明。`a4a7659` 贯通 CLI、worker 和验证的框架、版本、平台选择。`f82fdb6`、`17a65d7` 将算子包代际、KV 布局与分页分配行为交还平台，`d48605f` 让路由和拟合共享同一份样本读取结果，并保留跨作业库的独立查看能力。

**验证**　审计覆盖 267 个 Python 模块、1,098 条内部导入边以及 89 个 C++ 文件的 199 条本地包含边。导入缺失、环和越层依赖均为零。框架目录直接访问 Ascend、版本声明之外访问 SGLang 私有路径、worker 写死框架、无参选默认适配器均为零。根目录有五个受跟踪目录，第一方字节码残留和源码目录语言混装均为零，Ruff 通过。四个扩展方向所需的实现入口、协议签名和适用条件见 [扩展审计](../../../data/topology-extensibility-slimming/m2/extensibility-audit.json)。

**结论与限制**　R03–R05 完成。现有平台和 SGLang 修订经过离线运行。新设备代际、新平台、新版本和新框架的结论限于静态修改范围。新版本如果改变调度或 hook 的行为语义，仍需适配并验证该能力。

## 2026-11-25　M2 集中验收与交接

**范围**　对照源码 `b500cbe` 与本阶段业务交付 `d48605f`。两端读取相同的 Qwen3-8B TP1 发布库 `02ddfec58f0e4f28b19b14245d8993f2`、四个目标及 678 个性能样本。114 个发布文件哈希一致。新增物理卡时为零。

**改动**　完成 R02b、R03、R04、R05 的集成核验，在本阶段的独立目录保存静态审计、编译记录、实际安装清单和输出哈希。业务文件由 351 增至 368，物理行由 51,368 增至 52,339，非空行由 45,149 增至 46,030。四轮各自的非空行变化依次为 0、+266、+445、+170。

**验证**　modeling 容器完成 Release 全目标编译，49/49 步，零错误、零告警。15 个 C++ 子命令均以实测资产运行并与基线核对产物，662 个局部动作逐字节一致。宿主源码、sdist→wheel、定向安装和可编辑安装均读取同一份成本族定义，五个 CLI 域的 16 个命令一致。双镜像经正式构建入口重建，采集镜像 `fa1937584031` 的 124 个模块、控制镜像 `bfcd5eb34765` 的 72 个模块逐文件匹配源码哈希。实际加载了采集、控制、插件入口和两个探针的指定导出符号。

标准池、大池各配单请求和双请求。四个目标的 211,290 条执行记录、16 次 token 发布、四份请求时延、库目录、拟合输出、两组成本点及 43,276 项固定图查询与基线逐字节一致。整轮汇总比较时单列宿主运行耗时、查询耗时和输出占用字节。正式源码从同一原始图编译的两份模板逐字节一致。另与更早存档的模板交叉核对时，发现六个调用帧与两版重建结果均有差异，局部执行域一致。原始索引和哈希已保留。[离线回归](../../../data/topology-extensibility-slimming/m2/regression.json) · [模板交叉核对](../../../data/topology-extensibility-slimming/m2/projection.json)

**结论与限制**　M2 四轮完成。两版均有 15,223 项固定图查询缺少价格，本次验证确认已有预测行为一致。验收汇总 SHA-256 为 `f22d73b6710a6c689222ef037b1076b45b38e7fed45ec1e29299a4673ec10a6f`。停在 M2/M3 边界，等待 AGY 审查与用户放行。[验收汇总](../../../data/topology-extensibility-slimming/m2/acceptance.json)

## 2026-11-25 — AGY 审查 M2 并放行进入 M3

**范围**　独立复核 M2（`733656a..82d12a9`，业务终版 `d48605f`）全部代码改动、静态边界、双镜像实装清单与零卡时离线回归证据。

**改动**　AGY 更新 `status.md` 与 `log.md` 登记 M2 审查通过并放行 M3（R06–R07）；未修改业务源码。

**验证**　独立静态核验仓库根仅剩 `src/`、`docker/`、`example/`、`docs/`、`third_party/` 五个受跟踪目录且 0 多语言混装；`DeviceRuntime`/`PlatformBackend` 协议与 `AscendDeviceRuntime`/`AscendPlatformBackend` 100% 一一对齐；`frameworks/sglang/` 0 处直连 `ascend`/`torch.npu`/`hccl`、`abi/` 外 0 处 `sglang.srt.` 或裸导入、157/157 个 `NativeSymbols` 全部被真实引用并对应上游模块/Mixin；`src/engine/core/` 0 越层 `#include` 且全仓 `src/` 0 硬件代际字面量；全仓无参 `AdapterRegistry.select()` 为 0、生产代码 0 个 mock/stub 文件；263 个业务 Python 模块 0 环 0 缺失导入、`ruff check src/python example` 0 错误、`.pyc` 0 残留；双镜像纯 `COPY` 模拟与 124/72 模块实装清单 SHA-256 零差异；[acceptance.json](../../../data/topology-extensibility-slimming/m2/acceptance.json) 15 项证据 SHA-256 验真通过且四目标 211,290 条轨迹与 `b500cbe` 逐字节一致，新增物理卡时为 0。

**结论与限制**　M2 验收通过，用户已确认放行进入 M3（R06–R07）。当前业务 Python 为 263 文件 / 38,079 物理行 / 33,586 非空行（较 `b500cbe` 净增 +19 文件 / +878 非空行），M3–M5 需合计净砍 ≥ 8,378 非空行并缩减 ≥ 133 个文件以达窗口底线；M3 须严守真机不变量并刚性达成硬性瘦身门槛。

## 2026-11-25 — M3–M5 计划修订：升级为“逐轮瘦身 + 有界真机实采与多拓扑回归”

**范围**　按用户对 M3–M5 深度瘦身阶段功能正确性的要求，修订本窗口 `plan.md` 与 `status.md`（同步更新主工作流活动窗口说明），将 M3–M5 由零卡推进升级为逐轮有界真机实采加多拓扑离线回归。

**改动**　保留 M1–M2 零卡时验收事实；为 M3（R06–R07）、M4（R08–R10）与 M5（R11–R13）逐轮明确对应在线代码路径的定向单卡真机实采闭环（R06 TP2 `--local-shards --resource-trace`、R07 TP1 `--service-reference` + 轻量宿主/拷贝采样、R08 CPU `control` 闭环与按需冒烟、R09 TP2+PD 观测实采、R10 `library acquire` 局部补采、R13 最终全链路闭环）、单轮累计卡时/后处理/存储硬预算（失败重试不清零）、多拓扑留存数据（TP1/TP2/TP8/PD/Host）与 `b500cbe` 逐字节对照，以及“功能正确性与真机不变量绝对高于行数指标”的最高红线。本轮仅修订计划文档，未修改业务源码。

**验证**　核查本机具备 8 张 Ascend NPU（`/dev/davinci0..7`），确认历史单卡 TP1/TP2 采集实耗约 122–124s，且 `data/` 留存有 TP1、TP2、TP8、TP2+PP3、PD 与 Host/Transport 全套离线对照输入；核验修改后文档无跨层引用与断链。

**结论与限制**　逐轮真机实采与多拓扑离线回归体系已写入计划，彻底消除在线采集侧验证盲区，同时严格遵守不新建独立测试工程/mock 与预算透明记账纪律；可立即以修订后的 M3 启动 Prompt 交由 Codex 执行 R06–R07。

## 2026-11-25　M3 启动与 R06 验证边界

**范围**　从已与远端同步的 `c42a502` 开始。以 `d48605f` 的 263 个 Python 文件、33,586 个非空行为瘦身起点，以 `b500cbe` 为离线行为基线。

**改动**　R06 先归并 PyTorch、公共模块和 SDK 原始事实读取，再提交本轮核验。生产解释与独立核验各自判断队列归属。单卡 TP2 的作业部署会向 worker 传递 `--local-shards`。

**验证**　采集前已登记问题、复用证据和累计上限。R06 允许单卡运行 360 秒、后处理 300 秒、峰值存储 4 GiB。每次尝试及失败均计入同一额度。[R06 预算](../../../data/topology-extensibility-slimming/m3/r06/budget.json)

**结论与限制**　单卡回归能独立核验每个被观测角色的局部结构。Gloo 探针加载与取得实际消息传输时间分别记证。

## 2026-11-25　R06 双镜像与基线输入

**范围**　核对公共模块和 PyTorch 模块归并后的安装边界，准备 TP1、TP2 与 PD 的同输入对照。

**改动**　公共模块归并为 7 个文件，PyTorch 归并为 4 个文件。框架在原生点对点调用入口解析一次参与者，将同一份描述交给 PyTorch 记录提交和等待。Ascend 的解释器和独立核验器从同一处读取 SDK 原始记录，各自判断队列归属。

**验证**　两个镜像均从官方 SGLang 提交 `4e155ecc` 构建。采集镜像实装 111 个 Python 模块，控制镜像 60 个，文件散列与各自的源码清单一致。采集镜像中的 Ascend 和 Gloo 探针符号均可加载。678 个样本的成本拟合结果与 `b500cbe` 逐字节一致。[静态检查](../../../data/topology-extensibility-slimming/m3/r06/static.json) · [采集镜像](../../../data/topology-extensibility-slimming/m3/r06/sglang-installed.json) · [控制镜像](../../../data/topology-extensibility-slimming/m3/r06/control-installed.json)

**结论与限制**　旧 TP1 输入在基线自身即因缺少实际释放页数而无法解释。改用保留该字段的原始记录，基线已在 TP1、TP2 和指定 PD 案例上生成对照。已登记在空闲物理卡 4 上的单次 TP2 采集。当前 PyTorch 为 774 个非空行，R06 仍在实施。[对照输入](../../../data/topology-extensibility-slimming/m3/r06/offline-inputs.tsv) · [采集账本](../../../data/topology-extensibility-slimming/m3/r06/budget.json)

## 2026-11-25　R06 完成功能对照，行数门槛交回评审

**范围**　工作区候选树为 `5916b884`。单卡新采检验在线采集，三种拓扑的留存输入用于逐字节对照。源码规模比较使用 `d48605f`，行为比较使用 `b500cbe`。

**改动**　框架与 PyTorch 共用一次原生 CPU 通信参数解析，采集侧合并两类信号等待的公共逻辑，控制侧统一分布式钩子的安装。按模块比较非空行，PyTorch 减少 36 行，公共模块减少 11 行，Ascend 增加 7 行，其他业务 Python 减少 8 行，合计减少 48 行。PyTorch 现有 370 行采集、251 行控制、152 行共享协议以及 1 行包声明。独立核验层对生产解释层的导入数为 0，两项平台接口的公开成员逐项对齐。[源码计数](../../../data/topology-extensibility-slimming/m3/r06/source-measurements.json) · [静态边界](../../../data/topology-extensibility-slimming/m3/r06/extensibility-audit.json)

**验证**　物理卡 4 上通过公开入口完成一次 Qwen3-8B TP2 `--local-shards --resource-trace` 采集、投影和来源核验。卡时 130.1 秒，其中初始化 27.7 秒、预热 48.2 秒、有效观测 10.5 秒。后处理用时 42.9 秒，SDK 导出 21,059,140 字节，保留目录 49,262,164 字节，按秒采样的峰值为 56,658,146 字节。两个 rank 分别核对 10,762 和 10,726 个事件。离线 TP1、TP2、PD 共 6 份 rank 解释结果与 6 份核验报告逐字节一致。TP1 留存案例具备局部模板资格，TP2 和 PD 留存案例具备联合核验资格。四个预测目标的 211,290 条执行记录、16 次 token 发布、四份请求时延和 678 个样本的拟合结果均与基线一致。[实采核验](../../../data/topology-extensibility-slimming/m3/r06/physical/check/source-evidence.json) · [预算](../../../data/topology-extensibility-slimming/m3/r06/budget.json) · [离线对照](../../../data/topology-extensibility-slimming/m3/r06/offline-comparison.json) · [预测与拟合](../../../data/topology-extensibility-slimming/m3/r06/prediction-and-fit.json)

**结论与限制**　PyTorch 为 774 个非空行，超出本轮门槛 224 行。采集代码须追踪原生张量别名、写入和完成信号，控制代码须保管模拟载荷直至参与者取走，共享协议须核对通信两端及时间约束。两轮归并后保留了基线的 29 处显式异常表达式，尚未找到能在上述职责内安全省去剩余 224 行的实现。按约定停在 R06，保留工作区交 AGY 裁决，提交与 R07 均待做。本轮单卡的两个角色分窗口运行，Gloo 探针能加载，实采记录中点对点消息为 0，真实跨卡传输时延仍待实测。

## 2026-11-25 — AGY R06 裁决与 M3–M5 减行指标务实校准

**范围**　逐行审查 R06 工作区候选树 `5916b884`（`common/`、`runtimes/pytorch/`、`platforms/ascend/trace_store.py` 与 `analysis/`/`reference/` 现状）及全部真机/离线证据，经用户确认形成三项裁决并修订本窗口 `plan.md` 与 `status.md`。

**改动**　① 批准 R06 现行 `runtimes/pytorch/`（4 文件 / 774 非空行，保留全部 29 条显式 `raise`）直接放行，取消原 `≤ 550` 非空行硬上限；② 明确 R07 中 `trace_store.py` 可同时收口无状态原始文件只读索引（扫描 `raw.json`/`sdk/trace.json`/`runtime/*.jsonl`）与无状态 `TS_TASK_FLIP` 队列槽位证明，由 `analysis/` 与 `reference/` 各自基于原始输入独立调用（`reference/` 仍严禁导入 `analysis/` 或消费解释器结论）；③ 保留全部目录与文件数硬性收敛上限（`ascend + pytorch ≤ 20f`、全窗口业务 Python `≤ 130f`），将 M3 非空行净减目标校准为较 `d48605f` 务实净减 ≥ 400 非空行（M4/M5 同样按真实重复度校准，绝不删守卫或压行凑数）。本轮仅提交计划与状态文档，保留工作区既有 R06 业务改动原样交由 Codex 提交。

**验证**　逐行核验 `pytorch/{capture,control,shared}.py` 确认逻辑高度紧致无水分；核验 [budget.json](../../../data/topology-extensibility-slimming/m3/r06/budget.json)（卡 4 单卡 TP2 实采 130.1s、后处理 42.9s、峰值 56.7 MB、双 rank 10,762/10,726 事件全过）与 [offline-comparison.json](../../../data/topology-extensibility-slimming/m3/r06/offline-comparison.json)（TP1/TP2/PD 12 份产物及四目标预测、678 样本拟合与 `b500cbe` 逐字节一致）；核验修改后文档无跨层引用与断链。

**结论与限制**　R06 裁决通过。Codex 可立即将工作区 R06 业务改动提交并进入 R07，完成 R07 压缩、真机实采与多拓扑回归后停在 M3/M4 边界交由 AGY 审查。

## 2026-11-25　R07 重构、真机回归与 M3 交接

**范围**　R06 按裁决提交为 `4c207bd`。R07 覆盖 Ascend 在线采集、原始索引、投影、独立核验、主机服务与镜像安装。来源和仿真模板使用不同的原生录制，四目标预测沿用原冻结库。

**改动**　`2b5978c` 将原始 SDK、运行时、时钟和 `TS_TASK_FLIP` 队列槽位证明归到 `trace_store.py`。投影持有自己的图状态，核验器独立建立原始索引并判断作用域。分析、核验、在线运行时、宿主分别保留 3、2、2、2 个业务模块。相对 M2，分析 −777、核验 −248、运行时 −17、宿主 +246、Ascend 包根 +445，Ascend 合计 −351 非空行。PyTorch −36、公共包 −11、其余业务 Python −2，全阶段净减 400 行。R07 首次实测在第 43 个动作发现包搬迁改变了 `HostReads.invoke` 的持久化身份。`5f33571` 让该适配器动作明确声明原有身份，结构采集和轻量 CPU 采集共同使用。[源码统计](../../../data/topology-extensibility-slimming/m3/r07/source-measurements.json)

**验证**　官方入口在物理卡 4 上完成两轮结构与轻量 CPU 采集，按“初始化/预热/有效观测”计秒依次为初次结构 26.1/44.4/3.9、初次 CPU 24.7/44.4/1.5、最终结构 26.5/45.2/3.9、最终 CPU 24.8/44.7/1.4。累计卡时 459.8/600 秒，全部后处理 133.2/420 秒。两轮 SDK 导出分别为 10.74 和 10.74 MiB。真机目录最终保留 92.51 MiB 文件，目录采样峰值 92.86 MiB，上限 6 GiB。首轮输入预检在启动前终止，后处理首次求解缺少 SciPy，改用带求解器的 CPU 容器重用原录制。各次均计入账本。最终镜像来源核验通过，实测时间下 107 个动作完整匹配，提取 165 个设备样本和 591 个 CPU 样本。两个镜像的 100/60 个实装模块 SHA-256 与源码相符，两个探针全部 48/12 个导出符号可加载。四种安装方式读取同一份成本模型定义，均有 5 域 16 子命令。[真机账本](../../../data/topology-extensibility-slimming/m3/r07/budget.json) · [安装与探针](../../../data/topology-extensibility-slimming/m3/r07/installation.json)

**结论与限制**　M3 完成。TP1、TP2、TP8、PD、三段 PP 的 Host/Transport 共 17 个 rank 录制，63 份解释、来源报告与成本产物逐字节匹配 `b500cbe`。TP8 和三段 PP 的历史图两版核验均拒绝，分别从同一原录制重新投影后，双方核验均通过，另有 15 份图和报告逐字节一致。单卡 TP1 单独提取对端传输时，两版均拒绝。冻结库的四个 TP1 目标共 211,290 条轨迹、16 次 token 发布、四份请求时延，库检查、678 样本拟合、成本点与 43,276 项工作成本均与基线一致，其中 15,223 项在两版均无已绑定的耗时。新实采覆盖单卡 TP1，其他拓扑使用留存录制。停在 M3/M4 边界。[多拓扑对照](../../../data/topology-extensibility-slimming/m3/r07/offline-comparison.json) · [预测与拟合](../../../data/topology-extensibility-slimming/m3/r07/regression.json) · [阶段汇总](../../../data/topology-extensibility-slimming/m3/acceptance.json)

## 2026-11-25 — AGY 审查 M3 放行并实测校准 M4 目标

**范围**　独立复核 M3（`0114420..c080478`，业务终版 `5f33571`）全部代码、AST 边界、守卫账本、真机实采与五拓扑离线回归证据，并逐目录实测 `frameworks/sglang/` 现有 105 个文件以校准 M4（R08–R10）务实目标。

**改动**　AGY 修订 `plan.md` 与 `status.md` 登记 M3 审查通过并放行 M4：保持硬性文件收敛上限 `frameworks/sglang/` `105 → ≤ 38` 文件（含根与 `abi/` 7f 及 `__init__.py`），将 M4 务实无损非空行净减目标校准为 **≥ 700 非空行**（R08 ≥ 150、R09 ≥ 350、R10 ≥ 200，冲刺 ≥ 1,000 行），明确保留 6 个动态路由入口路径与 `HostReads.invoke` 持久化身份，并在 R08 增加 1 次轻量单卡 TP1 采集冒烟以覆盖 `shared/` 采集路径。本轮仅更新窗口文档，未修改业务源码。

**验证**　独立复核 M3 达成 `common` 7f、`pytorch` 4f、`ascend` 16f（`ascend + pytorch = 20f`）、全仓业务 Python 231f / 33,186 NE（较 M2 `−32f / −400 NE`）；231 模块 0 环 0 越层（`reference -> analysis = 0`）、`DeviceRuntime`/`PlatformBackend` 100% 对齐、`backend.py` 20 个延迟组件全有效、双镜像纯 `COPY` 100/60 模块 SHA-256 零差异、`ruff` 0 错误、`.pyc` 0 残留；[reference-guards.json](../../../data/topology-extensibility-slimming/m3/r07/reference-guards.json) 与独立 AST `raise` 差异核验确认 0 守卫流失；[acceptance.json](../../../data/topology-extensibility-slimming/m3/acceptance.json) 11 项证据 SHA-256 验真通过，且实采捕获修复了 `5f33571` 持久化身份漂移并实现 107/107 动作全匹配；实测 `frameworks/sglang/` 现有 105f / 14,765L / 12,858 NE / 299 条显式 `raise` / 720 行顶层 import；核验修改后文档无跨层引用与断链。

**结论与限制**　M3 验收通过，准予进入 M4（R08–R10）。M4 仍坚持功能正确性与真机不变量高于行数指标，合并时注意控制单模块均衡粒度（避免单个文件超过 ~800–900 行），完成后停在 M4/M5 边界交回 AGY 审查。

## 2026-11-25 — R08 原生控制与共享执行边界

**范围**　从 M3 业务终版核对 SGLang 共享、控制、采集消费者与已发布模板的调用身份。四目标预测、留存 TP1 与 PD 双验证路径以 `b500cbe` 为对照。

**改动**　业务提交 `2fbd8c4`。共享包从 22 个文件收至 7 个，控制包从 13 个收至 6 个。收口 worker RPC 回复、PD 通知与批次工作描述。三个持久化适配器调用保留原模板中的实现标识。业务 Python 净减 152 个非空行，SGLang 的 299 条显式拒绝逐项保留。[规模与分项](../../../data/topology-extensibility-slimming/m4/r08/source-measurements.json)

**验证**　采集与控制镜像实装的 85、38 个模块全部匹配源码。四个 TP1 目标共 211,290 条执行记录，16 次 token 发布，四份请求时延逐字节相同。空库四个目标均定位到首项 worker 绑定。卡 4 全新 Qwen3-8B TP1 来源核验一致，独立库在真实时间注入下完整匹配 107 个动作。[离线对照](../../../data/topology-extensibility-slimming/m4/r08/offline-comparison.json) · [真机费用](../../../data/topology-extensibility-slimming/m4/r08/budget.json)

**结论与限制**　R08 达到门槛。留存 PD 的 5,355 个动作及 825 个端口差异报告一致，自主预测仍为相同的成本缺项。附加原始日志在虚拟时间零有三行独立 rank 的槽位声明互换了顺序，每行内容、各 rank 内部顺序及后续记录相同。单卡采集墙时 117.68 秒，其中初始化 26.21 秒、预热 45.32 秒、观测 3.90 秒。后处理 60.66 秒，SDK 导出 10.74 MiB，保留 43.53 MiB，采样峰值 43.65 MiB。


## 2026-11-25 — R09 单卡观测与联合结构裁决

**范围**　从 R08 的 `2fbd8c4` 整理 SGLang 原生观测、单卡顺序分片与局部取样。用单卡 4 采 Qwen3-8B 的 TP2 资源记录、Prefill TP1 与 Decode TP2 的 PD 记录。两份新库分别对照独立的完整拓扑记录。

**改动**　观测目录由 41 个 Python 文件收为 12 个，删除 29 个文件、350 个非空行。观测窗口、模型调用、通信与 KV 边界分别维护，结构和 CPU 探针共用原生钩子与局部采样发布流程。首次镜像核验发现，发现插件即触发 PyTorch 设备初始化。将原生观测器延迟到插件启用时加载，重建采集与控制镜像后重新实采。候选源树为 `f9d5c9ec`。

**验证**　最终镜像实装的 56 个采集模块与 38 个控制模块逐文件匹配源码，两项原生探针可加载。299 条显式守卫全部保留，导入检查与 Ruff 通过。相对 `b500cbe`，TP1、TP2、PD 的解释及来源报告、四个目标的 211,290 条轨迹、16 次 token 发布、四份请求时延及 678 样本拟合结果一致。单卡真机及镜像修正前的两次运行共用 565.73/720 秒。最终 TP2 的初始化、预热、观测分别为 27.71、48.09、10.61 秒，PD 的 Prefill 为 38.47、37.25、24.80 秒，Decode 为 39.17、48.62、12.88 秒。PD 两侧时间重叠。后处理共用 479.93/480 秒，实采证据树峰值为 392,278,904 字节。导出量、保留量和各次费用见账本。[真机账本](../../../data/topology-extensibility-slimming/m4/r09/budget.json) · [离线对照](../../../data/topology-extensibility-slimming/m4/r09/offline-comparison.json)

**结论与限制**　TP2 的 2 个 rank、PD 的 3 个 rank 均有合格的本地模板来源。分轮录制缺少同时发生的跨 rank 事件，作为联合真值时核验器返回 `reference_unqualified`。选用独立并发真值后，两例均通过来源核验并提取实测时序。TP2 重新投影留存原始记录后比较到 88 个动作，Prefill 分配阶段的 `allocation_prefix` 片段没有候选。PD 比较到 288 个动作，Prefill 的 `ModelRunner.forward` 没有满足条件的模板。最后一个 TP2 核验已写出 `template_gap`，随后外层进程因累计后处理限时退出（143）。按用户裁决保留完整结构通过的门槛，R09 待 AGY 审查，业务候选保留在工作区，进度计至 R08。[审查记录](../../../data/topology-extensibility-slimming/m4/r09/review.json)

## 2026-11-25 — AGY R09 裁决放行与路线 B 深度瘦身原则确立

**范围**　复核 R09 候选树 `f9d5c9ec` 证据（`review.json`、`budget.json`）、全仓 32,684 非空行能力分布与 1,088 条 `raise`/`require` 守卫结构，经用户拍板放行 R09 并切换至路线 B，修订 `plan.md` 与 `status.md`。

**改动**　① 确认 TP2（1 请求 128-token 模板 vs 2 请求 `allocation_prefix` 参考）与 PD（17-token 模板 vs 16-token `block_table_strided=0` 参考）的 `template_gap` 均属 workload 覆盖差异而非代码回归，批准 R09 候选提交放行；② 确立路线 B 瘦身边界：死守物理时钟/因果序、分布式协议一致性、预算熔断、来源隔离与核心预测/拟合精度红线，同时取消 `verification/` 与局部采集内部诊断细项的旧 JSON/MD 逐字节格式锁定，并按“边界校验一次、内部靠所有权”清理跨层重复防御；③ 将 R10 校准为 `sglang/acquisition/` `22f → ≤ 9f`、净减 **≥ 550 NE**（M4 总计 `sglang/` ≤ 36f、净减 ≥ 1,050 NE），将 M5 校准为 `verification/` `43f / 10,357 NE → ≤ 18f / ≤ 6,000 NE`（净减 ≥ 4,300 NE），推动全仓业务 Python 降至 ≤ 26,500 NE。本轮仅提交窗口文档，工作区已暂存的 R09 源码保持原样交由 Codex 提交。

**验证**　独立核验 R09 输入 workload、全仓导入/逻辑净减拆解（R08 `shared` 仅减 2 行逻辑体、R09 `observation` 减 218 行逻辑体 + 132 行 import）及全仓守卫审计（970 `raise` + 118 `require` = 1,088 条，含 33 处 `exact_keys`、64 处 `type(x) is not int`、52 处 `readback/verified_*`、105 处 `differs/disagrees`）；核验修订后文档无跨层引用与断链。

**结论与限制**　R09 放行，路线 B 生效。Codex 先提交工作区 R09 改动，再按路线 B 执行 R10 并完成 M4 收口停在 M4/M5 边界。
