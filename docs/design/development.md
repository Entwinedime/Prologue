# 构建环境与入口

C++23 核心用 Ubuntu 24.04、GCC 13、CMake、Ninja 构建。建模镜像 `prologue-modeling:ubuntu24.04` 提供工具和依赖，应用镜像 `prologue-app:ubuntu24.04` 安装当前 CMake 产物和 Python 包。构建过程只读本次源码，应用镜像从安装目录启动。Python 保持 3.10 语法基线。

当前公开入口 `prologue plan <task.json>` 在 CPU 容器中读取一份任务。Python 装配模型与配置，C++ 校验请求并生成稳定到达顺序，应用输出配置与 workload 的组合。模型位置在此阶段是部署输入。任务计划成功表示输入合格，原生运行、模板、成本和精度分别取证。

观察镜像 `prologue-observer:ubuntu22.04` 从已固定的官方 SGLang 源码和 Ascend 软件环境安装当前观察包，使用框架插件记录现行进程中的批次和模块调用。详细模式经原生 profiler 留下提交及设备证据，轻量模式从同一发布边界记录请求的 token 数和时刻。观察目录由一次会话独占，完成后写入清单。`prologue compile <observation>` 在无设备的应用容器内解释详细记录并编译模板，输出可追溯的定义、调用实例和设备工作归属。设备计时与主机 RAW 时间各自保留，交给后续时钟核对。

## 层次与构建边界

- `src/prologue/app/` 维护公开命令及对应的跨语言装配，`src/prologue/core/` 负责统一业务判定。任务级字段与 SGLang 配置分别在入口和框架适配侧解释。
- 主机用于编辑、Git 与容器调度。共同 C++ 目标在建模环境构建。设备采集桥随原生 SDK 单独构建和加载，正式纯预测只读取明确的输入与发布资产。
- 官方 SGLang 保留在 `third_party/sglang`。现有 Ascend 依赖层由 `docker/images/base/sglang` 构建。接入原生运行时记录实际提交、镜像 ID 和部署身份。模型、编译输出、采集数据与归档留在本地。
- 实验矩阵和请求来源组织留在 `example/`，可复用的单案例核验由正式验证模块维护。新的命令与镜像随各自的可用实现同步进入操作说明。
