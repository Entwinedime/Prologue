# 构建环境、公共入口与日常开发指南

本文维护 Prologue 的稳定环境、已交付命令行入口、典型操作流程与日常排障导航。构建成功或命令存在不代表模型、结构或性能资格通过；本文件仅记录仓库实际交付的公开入口，不保留已废弃的细图 CLI、旧回放命令或兼容别名。

## 一、容器环境与构建

### 1. 镜像职责与运行边界

| 镜像标签 | 核心职责 | Docker Runtime |
| --- | --- | --- |
| `prologue-modeling:ubuntu24.04` | C++23 核心编译（GCC 13 / CMake≥3.20 / Ninja / nlohmann-json≥3.11 / OpenSSL）、静态检查与纯预测仿真 | 显式 `runc` |
| `prologue-sglang-env:ubuntu22.04` | 官方 SGLang 推理软件基础层，不含项目业务观察代码 | —（构建基座） |
| `prologue-sglang:ubuntu22.04` | 官方框架、当前采集包、Ascend/PyTorch 原生采集桥与观察资源 | 真实采集/后处理显式 `ascend` |
| `prologue-control:ubuntu22.04` | 与 `sglang` 同上游软件来源的无设备原生控制与元数据代理 | 显式 `runc` |

- **显式 Runtime 与无网络纯预测**：CPU 协调器、`modeling`、`control` 与独立验证均显式使用 `runc`，避免继承主机默认加速器 runtime；纯预测保持无设备、无网络，不加载模型权重、采集插件或目标验证数据。真实 NPU 采集及需 SDK 环境的离线导出显式使用 `ascend`；原生采集桥在推理镜像内单独按 C++23 构建，不链入 `prologue_core`。
- **上游版本一致性**：Docker Compose 配置位于 [`docker/compose/inference.yml`](../../docker/compose/inference.yml)。执行 [`./docker/build_inference.sh`](../../docker/build_inference.sh) 要求 `third_party/sglang` 工作树干净，自动写入 `PROLOGUE_SGLANG_REVISION` 与 `org.prologue.sglang.revision`，先构建 `sglang` 再构建继承它的 `control`。采集清单、后处理、预测与控制端逐级核对真实上游提交与 Torch 版本（不维护内部格式版本号或旧格式回退）。构建代理监听宿主回环时可设 `PROLOGUE_BUILD_NETWORK=host`。
- **Python 与 C++ 基线**：Python 源码与宿主/容器入口统一保持 Python≥3.10 语法基线（宿主调度需 Docker CLI）；C++23 关闭编译器扩展。

### 2. 标准构建命令

```bash
# 1. 初始化官方 SGLang 子模块并构建基础与推理/控制镜像
git submodule update --init third_party/sglang
docker build -t prologue-sglang-env:ubuntu22.04 docker/images/base/sglang
./docker/build_inference.sh
docker compose -f docker/compose/inference.yml build modeling

# 2. 在 modeling 容器内构建 C++23 目标（统一构建目录：build/coarse-graph）
mkdir -p build/coarse-graph
docker run --rm --runtime runc --network none \
  -v "$PWD:/repo:ro" -v "$PWD/build/coarse-graph:/build" \
  prologue-modeling:ubuntu24.04 bash -c \
  'cmake -S /repo -B /build -DCMAKE_BUILD_TYPE=Release && cmake --build /build --parallel 8'
```

正式 CMake 目标定义在 [`CMakeLists.txt`](../../CMakeLists.txt)：静态库 `prologue_core`、`prologue_sglang`，可执行文件 `prologue` 与 `prologue_local_graphs`。删除或变更目标须同步清理源文件、Python 导入、Docker COPY、示例与说明。

### 3. 运行时发布、锁与恢复红线

- 程序仅消费显式输入与本次发布资产，严禁扫描历史 `data/` 自动补库；同一任务目录由 `.lock` 保证单一写者。
- 结果经临时文件原子落盘（C++ 侧使用同目录硬链接），输入输出互不覆盖；失败、超时或中断不发布空成功文件。
- 原生采集、离线解释、模板发布与纯预测分阶段交接：CPU 后处理失败优先复用已闭合的原生事实，不重启原生 Engine；旧资产不静默升级格式，契约变更后显式重建。

## 二、已交付命令速查表

### 1. Python 宿主入口

命令格式为 `python3 -m prologue <命令域> <操作>`。安装宿主包后也可使用 `prologue-cli`。宿主运行在 Python 3.10 及以上环境中。开发容器通过 `PYTHONPATH=src/python` 使用当前源码。框架与平台的具体实现由宿主注册表选择。[宿主装配](../../src/python/prologue/host/cli/cli.py) · [适配器注册](../../src/python/prologue/host/registry.py)

| 命令域 | 操作与用途 |
| --- | --- |
| `predict` | `run` 预测一个目标，`explore` 探索配置与负载组合 |
| `library` | `acquire` 回答原始缺项，`publish` 从局部会话或已有训练集合发布，`inspect` 只读查看发布或存储 |
| `capture` | `run` 采集跨 rank 原生证据，`project` 将已闭合的记录生成模板或实测粗图 |
| `cost` | `extract` 按 `--kind device/transport/batch/host` 选择样本来源，`fit` 拟合模型，`acquire` 在预算内补充成本 |
| `verify` | `source` 核对来源，`inputs` 提取实测条件，`local-graphs` 核对局部展开。三路正式验证为 `structure`、`fixed-graph`、`autonomous` |

### 2. C++23 核心入口（[`src/engine/tools/main.cpp`](../../src/engine/tools/main.cpp)）与辅助脚本

| 程序与子命令 | 职责 |
| --- | --- |
| `prologue project <facts> <coarse> [--evidence]` | 将原生观测事实投影为粗粒度执行图 |
| `prologue templates <coarse> <templates>` | 将粗图编译为来源子图模板库 |
| `prologue link-templates <source-libraries> <out-dir>` | 连接同一次采集中相容的局部实现 |
| `prologue query-template` / `join-observation` / `template-evidence` / `supplement-template` | 模板库缺项查询、局部观测合并与局部证据补充编译 |
| `prologue instantiate <templates> <invocation> <coarse>` | 实例化单次动作调用为目标粗图 |
| `prologue action-layouts <observed-coarse> <layouts>` | 导出动作切图的来源端点映射，供受控执行输入使用 |
| `prologue work-identities` / `work-costs` / `cost-points` | 生成工作身份坐标、按图或按点批量执行 C++23 成本查询（支持 `--bindings` 溯源） |
| `prologue execute <coarse> <given-timing> <execution> [--trace]` | 执行单张给定图（须声明 `initial_inputs: "ready"`，支持 `work_ns` 与 `boundaries`） |
| `prologue replay <fixed-execution> <services> <event-times>` | 固定真实图多场景服务时长重算 |
| `prologue serve <templates-dir> <task> <result> [--trace] [--execution-inputs \| --costs]` | 应用内部核心仿真入口（标准输入输出接控制协议） |
| `prologue_local_graphs actions \| expand` | 给定真实动作的离线局部构图验证程序 |
| 容器内诊断入口 | `python3 -m prologue.workers.capture capture\|interpret\|export\|sampling-cuts`、`python3 -m prologue.workers.control`、`/opt/prologue/export-profile.sh` |
| 示例与矩阵入口（[`example/coarse_graph/`](../../example/coarse_graph/)） | `run.py`（四目标探索示例）、`prepare_matrix.py`、`matrix.py` |

## 三、核心工作流操作手册

### 1. 配置集探索与自动局部补库（`explore`）

- **只读预测探索**：读取固定发布目录，以只读挂载在 CPU 上预测全部配置 × workload 组合；使用相同 `--output` 可幂等恢复（保留已完成目标），任一组合缺项时其余继续运行且进程返回非零。

```bash
python3 example/coarse_graph/run.py --libraries /path/to/qwen-release --timeout 180
```

- **自动局部补库模式**：传入 `--deployment` 与可写长期库根目录，系统先用同一对固定库预测全部目标，归并可共用初始化的缺项后执行局部取证并增量发布，随后**仅重算所选 worker 模板、初态或成本域发生变化的目标**。存储额度覆盖 `acquisition/` 与新写入库文件；四项额度均为零时仍可查询现有库并输出缺项；卡时按单调时钟乘实际参与卡数累计（含超时清理）。

```bash
python3 -m prologue predict explore --input /data/targets.json --output /data/exploration \
  --deployment /data/placement.json --libraries /data/libraries --binary /build/prologue \
  --timeout 300 --attempt-limit 2 --card-budget-seconds 400 \
  --postprocessing-budget-seconds 120 --storage-bytes 3221225472
```

`explore --prediction-workers 4` 可让四个配置与请求组合在 CPU 上同时预测，默认依次执行。每个组合读取同一份发布并单独保存结果。所有目标返回后，协调器统一归并缺项、安排采集和发布。

### 2. 局部采集、增量发布与历史资产导入

- **局部缺项采集（`library acquire`）**：直接接收预测输出的原缺项查询；可共用一次初始化的多个查询合并传入 `--query`。局部会话收据记录实际设备、累计卡时、保留字节及按秒采样的磁盘峰值。

```bash
PYTHONPATH=src/python python3 -m prologue library acquire \
  --query /path/to/gap.json --deployment /path/to/deployment.json \
  --output /path/to/local-session --timeout 60 --budget-seconds 60 --attempt-limit 1
```

  宿主按查询和部署选出适用规则，依优先级尝试。在方法明确返回证据缺口时才尝试低优先级候选。同优先级无法唯一选择时停止，候选产物不完整或与查询冲突时按错误处理。会话记录实际获得资格的录制以及各次采集费用。

  1. **同步张量搬运**：按模板工作执行同卡张量操作并核对同步 ACL 区间与目标数据；样本经 `cost fit --samples ... --gap ...` 复核。
  2. **稠密 TP1 Decode Attention**：额外传 `--libraries <发布根>`，复用单个 `ModelRunner` 跑 Prefill 写 KV 与指定 Decode attention，回读 K/V 并逐段核对设备任务与计时；支持零新增尝试从原生记录恢复。对照完整服务开销时使用 `capture run --attention-sites <文件>`。
  3. **PD `prebuilt_prepare`**：接收端按已准入成员顺序与长度预备本地 KV 并执行批准备，核对页表写入与 K/V 读回，产物 `structure/receiver.json` 可直接投影编译。
  4. **稠密 TP 词嵌入**：按 `model_tokens` 合批同阶段同词元数查询共用初始化，单卡顺序观测并经主机交换分片，完整模式在设备间求和，输出 `structure/embedding-<rank>.json`。
  5. **Worker 初态和调度器批次。** 初始化读取真实角色、槽位、KV 容量和预热后状态。局部批次按需求中的队列、成员和已生成 token 数在原生调度器准备前序状态，再观察选批、模型前向或结果发布。这个批次入口覆盖关闭前缀缓存、采用常规调度的稠密模型。
  6. **批内服务时间。** CPU 服务用相同原生调用的独立轻量计时。设备工作优先查询已记录的同一工作及特征，配对采集也可复用其中的结构记录。跨卡通信使用真实对端同时参与的物理计时，单列其来源和适用的操作数。

并发 Prefill 首次需要实测时，采集器从这批成员的实际长度和可运行步数生成有限的局部输入。它测量最长成员单独执行的 Prefill。有两个可继续的成员组时，还测量它们合批后的 Decode 前向与结果处理。不同探针共享一次有额度的采集任务，收据列出 ModelRunner 启动数、卡时和各产物的原生测量位置。模板编译从指定的合并入口抽取子图。冷任务结束前，协调器核对这些实测切口涉及的同步搬运，将仍缺的载荷交给已有的就绪张量采样器。额度耗尽时，结果单列尚未定价的训练工作。

- **增量发布到长期库（`library publish`）与局部批次提样（`cost extract --kind batch`）**：`library publish` 核对候选后原子切换双库并返回固定发布目录（重复提交同证据返回现有发布，旧读者不受影响；已核对的单批次结构会话亦可回答对应设备区间成本查询）。

```bash
python3 -m prologue library publish --libraries /data/libraries \
  --from-session /data/session --query /data/query.json \
  --binary /build/prologue --timeout 300

python3 -m prologue cost extract --kind batch --session /training/local-batch \
  --binary /build/prologue --timeout 300 --output /training/samples/batch.json
```

- **从旧训练集合导入发布**：通过宿主 CLI 执行，并按原绝对路径挂载历史采集目录；发布根 `release.json` 固定双库入口与原生来源身份，所有运行时引用指向发布根内文件。

```bash
PYTHONPATH=src/python python3 -m prologue library publish \
  --from-collection /path/to/training/collection --costs /path/to/training/costs.json \
  --libraries /path/to/qwen-release
```

### 3. 原生采集（`capture run`）与离线生产（`capture project`）

```bash
PYTHONPATH=src/python python3 -m prologue capture run \
  --input <job.json> --deployment <deployment.json> \
  --output <采集任务目录> --timeout 900 --budget-seconds 1800 --attempt-limit 2

PYTHONPATH=src/python python3 -m prologue capture project \
  --capture <已发布的原生采集任务目录> --output <生产目录> \
  --binary <modeling构建的prologue> --budget-seconds 600
```

- **采集预算与容器恢复**：`--timeout` 在宿主协调器与容器内同时生效；`--budget-seconds` 与 `--attempt-limit` 跨恢复累计且失败不清零，每次命令至多启动一次原生尝试。Docker 容器保留至应用核验退出状态后主动清理；正常退出并读取闭合 `source/capture.json` 后发布 `result.json`。
- **分组提交与单卡重放**：可选 `submission_groups` 按组串行提交、组间排空重置共同 RAW 起点（记录于 `request-delivery.json`）。单卡模式（`{"acquisition_mode": "single_device", "physical_devices": [4]}`）下每个 Engine 启动一次共用预热，随后按代表角色顺序开启设备 profiler 分窗重放（`profile_windows`），PP 暂存张量、EP 超显存按层暂存完整存储并单独计采集开销，协议记录 `protocol/worker-<rank>/` 全员发布。
- **离线生产与 SDK 隔离**：`--product templates`（默认）编译并连接同次采集内相容局部实现，成功后默认清理中间件；`--product graphs` 仅保留实测粗图供验证对照（不生成模板）；`--keep-intermediates` 保留 trace、facts、graph 及证据。Ascend SDK 仅在原始记录的私有副本上运行导出，绝不修改原始来源目录。

### 4. 成本样本导出与离线拟合

```bash
# 导出物理通信样本（可重复 --rank/--work-identities/--output 共用同次校时）
python3 -m prologue cost extract --kind transport \
  --production /training/collection/assets/0 --rank 0 \
  --work-identities /training/assets/0/sources/0/0-work.json \
  --output /calibration/tp2-rank-0.json

# 拟合离线成本资产（可选 --check / --demand / --gap 走 C++23 复核）
python3 -m prologue cost fit --samples /calibration/*.json \
  --binary /build/prologue --output /training/costs.json
```

- `cost extract --kind host` 配对详细观测与 `capture run --host-timing` 观测提取主机 CPU 服务样本。
- `cost acquire` 支持 `--production`、`--samples`、`--demand`、`--gap` 与 `--transport-samples`，按累计预算自动补采主机/通信样本并输出 `costs.json`（保留明细需加 `--keep-intermediates`）。

### 5. 单目标预测（`predict`）与执行明细排障

```bash
PYTHONPATH=src/python python3 -m prologue predict run \
  --libraries /path/to/qwen-release --input <prediction.json> \
  --binary <modeling构建的prologue> --output <新输出目录> --timeout 180
```

- 入口调度无网络的 `modeling` 与 `control` 双容器（带 `--init`），`core.log` / `control.log` 首行记录完整命令，结束后发布 `result.json`；结构缺口落 `template-gap.json`，成本缺口落 `cost-gap.json`。
- **追查单项工作估值**：传 `--execution-trace` 流式写出 `execution.jsonl`（运行中写 `execution.partial.jsonl`，结束后原子发布，不改变精简结果）。展开图中通过 `work_ns` 与 `model_bindings` 下标定位参与者成本域 `models`，再经 `training_rows` 追回原始样本源与行号；`prologue cost-points` 与 `work-costs` 加 `--bindings` 返回同一绑定下标。

#### `execution.jsonl` 记录种类速查

| `kind` | 含义 |
| --- | --- |
| `resources` / `resource` | `clock_epoch_ns`、各实例容量、worker 与请求顺序；核心资源服务的实际查询/修改与失效页/槽位 |
| `binding` / `state_binding` | 图实际使用的请求 KV 页映射（`[物理页号, 页内起, 止)`）与持久请求字段行绑定 |
| `context_value` / `graph` | 共享工作上下文表；实例化粗图、局部到执行事件 ID 映射、工作耗时与插入时刻 |
| `condition_used` / `boundary_time` / `logical_boundary` | 已核对受控时间条件、具名边界 `[begin_ns, end_ns]` 约束与纯因果逻辑端点 |
| `edge` / `completion` | 执行器实际连接的前驱/后继及延迟；事件实际模拟完成时刻（流接收/等待另带 `activation`） |
| `notification_declared` / `notification_ready` | 跨参与者通知登记与实际就绪交付（含前驱事件与时刻） |
| `completion_read` / `state_query` | 实际完成信号读取（布尔结果与顺序）及原生适配器状态查询汇总 |
| `input_read` / `end` | 实际请求拉取记录（含空读）；核心事件循环正常闭合标记（缺 `end` 仅作停止前缀） |

### 6. 独立验证与冻结矩阵命令速查

三路独立验证的完整算法、时钟认证与资格判据详见 [verification.md](verification.md)，常用公开调用如下：

```bash
# 来源证据独立检查（宿主源码，时钟约束求解使用带 SciPy 的无设备环境）
PYTHONPATH=src/python python3 -m prologue verify source \
  --production <生产目录> --capture <原始采集目录> [--libraries <双库目录>] --output <报告目录>

# 单案例执行结构验证（串联参考核验、受控输入、预测与 compare_actions）
PYTHONPATH=src/python python3 -m prologue verify structure \
  --capture <采集目录> --reference <真实图生产目录> --libraries <双库目录> \
  --input <目标输入.json> --binary <prologue> --output <验证目录> --timeout 180

# 单案例自主闭环评分
PYTHONPATH=src/python python3 -m prologue verify autonomous \
  --capture <轻量采集目录> --libraries <双库目录> --input <目标输入.json> \
  --binary <prologue> --output <验证目录> --timeout 180

# 冻结矩阵抽样与分阶段运行
PYTHONPATH=src/python python3 example/coarse_graph/prepare_matrix.py \
  --root <运行根> --specification <覆盖声明.json> --stage <阶段> --seed <冻结种子>
python3 example/coarse_graph/matrix.py \
  --root <运行根> --matrix <固定矩阵.json> --phase sources|detailed|light|targets|compare|fixed|autonomous
```

## 四、Ascend 与 PD 部署环境要点

- **物理设备映射**：完整模式按各实例 `TP×PP` 之和从升序 `physical_devices` 依次分配逻辑设备 `0, 1, ...`（[Ascend 可见设备顺序](https://www.hiascend.com/doc_center/source/zh/canncommercial/850/maintenref/envvar/envref_07_0028.html)）。用 `--device` 挂载宿主卡（如卡 4/5）时，容器内 `ASCEND_RT_VISIBLE_DEVICES` 须使用驱动重新枚举的 `0,1`，禁止直接传宿主编号。
- **PD 传输环境变量**：Ascend PD 两侧须配同一 `ASCEND_MF_STORE_URL=tcp://<可达地址>:<空闲端口>`（由 Prefill Engine 启动配置服务，不另启第二份），`ASCEND_MF_TRANSFER_PROTOCOL` 支持 `sdma`（默认）与 `device_rdma`（后者另需真实硬件与网络条件）。部署前应先核对设备与传输设置，避免以完整模型启动排查环境。
- **单卡主机计时权限**：`--host-timing` 单卡容器启用 `PERFMON` 并放开 `memlock` 限额以映射每线程 perf 环形缓冲区，仍只挂载声明的物理卡。
- **单机 RAW 时钟锚点**：单机投递、接收与资源操作统一使用 `CLOCK_MONOTONIC_RAW`；SDK 导出分别按 `host/start_info` 与对应 `device/start_info` 的 `collectionTimeBegin`/`clockMonotonicRaw` 锚点还原，严禁混用两类偏移。

## 五、问题定位导航与开发修改节奏

### 从哪条路径判断问题

| 需要回答的问题 | 优先入口与依据 | 排查顺序 |
| --- | --- | --- |
| 真实时间下能否展开出一致的执行结构？ | `python3 -m prologue verify structure` | 核对参考材料、`condition_inputs` 覆盖与消费，定位首个动作、模板绑定、依赖或资源效果差异 |
| 同一执行图上的性能成本是否准确？ | `python3 -m prologue verify fixed-graph`（调用 `work-costs` + `replay`） | 核对单工作自身服务测量资格 → 确认实测基线通过 → 查单元误差与关键链传播 |
| 使用独立成本后的端到端结果如何？ | `python3 -m prologue verify autonomous` + `schedule` | 固定图基线通过后，比较自主运行与固定图的首处时序、控制或状态分歧 |

### 日常开发边界与验证纪律

- **目录与依赖红线**：框架/平台差异严格收敛在 `integrations/`，共同核心、观测合同与发布逻辑不导入具体实现；单案例验证归 `src/python/prologue/verification/`，场景与矩阵编排归 `example/`；`deprecated/` 仅作本地参考，严禁进入 Git、CMake、Python import、安装或 fallback；`third_party/` 保持官方上游干净工作树。
- **按职责链修改与验证**：从目标公开行为倒推输入、输出与唯一所有者，先确定合并、拆分、重写与旧入口删除范围，集中改完受影响全链并静态复查后，再在 modeling 容器统一编译与运行必要公开入口；优先复用已有原生记录做离线核对。
- **不新增独立测试脚手架**：不新建独立 test 工程、CTest、fixture 或 mock；评价工具的受控反例留在 `example/`，严禁进入业务分支。
