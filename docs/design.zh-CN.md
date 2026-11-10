# XInfer 设计文档（评审稿 v1）

[English](design.md) | 简体中文

- **范围**：整体架构、核心接口、首批模型与硬件、开发流程

依据的现有代码与记录：

- embodied-serving main `e910f50`：FleetServe 运行时、论文实验记录和模型文档；
- HarriX main `9e633ac`；
- vla-engine 原型 `a774bb0`（2026-04）；
- pipeline-overlap-vla 研究；
- vLLM、SGLang 当前 main 源码。

Wall-OSS 各项测量的条件见性能和验证指南。

---

## 0. 已定事项

| 事项 | 决定 |
|---|---|
| 项目名 | XInfer |
| 仓库 | GitHub 上的 X-Square-Robot/x2infer。 |
| kernel | 沿用现有 xcompute 仓库，由算子组负责；XInfer 通过 ops 层的 xcompute 绑定接入 |
| 算子绑定 | xcompute 默认使用 TVM-FFI；有明确的 PyTorch 编译、导出需求时，按需增加 `torch.library` 适配并复用底层实现（§14.2） |
| 部署场景 | 每个加速器运行一个模型实例，为一台或多台机器人提供服务。 |
| 内置模型 | Wall-OSS |
| 首批硬件 | RTX 5090（sm120）、Jetson Thor（sm110） |
| GPU 进程布局 | 默认 `colocated`：每 GPU 一个 Engine，Vision / Prefill / Action 同进程；v1 只实现这一布局，`split` 暂不实现 |
| 阶段与部署边界 | 逻辑阶段与进程映射分离；StageGraph 声明计算与依赖，阶段的进程归属由运行时负责（§9.3） |
| 硬件方向 | v1 只实现首批硬件；其他芯片经 Platform 插件与算子后端接入（§13、§18.3）。不提供图编译类执行后端（§9.4） |
| 不含 | SLO 调度器（FleetServe-T、central-ssbp 等）；只保留调度策略插件接口 |

---

## 实现状态（2026-11-04，main `b4f1ac8`）

本文其余各节描述目标设计。下表列出各设计项在 main 上的实现情况；“未实现”的条目在实现前仍按原文评审。

| 设计项 | 状态 | 现状 |
|---|---|---|
| §4 包间依赖方向 | 已实现 | `tests/unit/test_layering.py` 检查（含 `frontend/`、`observability/`）；`engine/` 不依赖 `ops/`、`frontend/` 两条尚未加入检查 |
| §4 设备边界 | 未实现 | `engine/` 直接调用 `torch.cuda`，并按 `Platform.is_cuda` 分支；`test_layering.py` 只检查 `platforms/interface.py`、`models/interface.py` 不 import torch |
| §5 取消与单请求错误 | 部分实现 | 未发射的取消请求直接丢弃，已发射的跑完后丢弃结果；阶段执行、接纳时的上传或取批时出错，只让相应的请求失败，并释放其 slot。引擎线程或完成线程停止时 `xinfer-serve` 以状态 1 退出；设备粘滞错误的识别（`Platform.classify_error`）与 DRAINING 未实现 |
| §6.3 三条调度规则 | 已实现 | 上游发射后请求即进入下游队列，下游 lane 在 GPU 上等待事件；零等待组批；完成线程只等最终事件（`xinfer/engine/core.py`） |
| §6.4 组批精度 | 已实现 | Wall-OSS 参考组合允许 action 阶段使用 B1、B2、B4 和 B8。优化组合、vision 和 prefill 只使用 B1。 |
| §7.1 接纳 | 部分实现 | `submit` 返回 Future，拒绝时为 `AdmitError`（`queue_full`、`per_robot_limit`、`stopped`）；`admission_timeout` 未实现 |
| §7.2–§7.4 阶段队列、组批、发射 | 已实现 | FIFO；兼容键由运行时按阶段实际读取的输入推导；序列长度不同的交接张量左填充后组批；合批用 torch 拼接，不经 Platform 拷贝原语 |
| §7.5 策略插件 | 未实现 | FIFO 写在引擎内 |
| §8.1 slot 状态机 | 部分实现 | `SlotPool` 只做接纳计数和代数检查（FREE / ACTIVE / RELEASING）；交接张量由请求持有，每个阶段复制一份 |
| §8.2 显存规划器 | 未实现 | |
| §8.3 图管理 | 部分实现 | 每条通道有自己的图池和静态缓冲。Wall-OSS 默认按 16 token 对 prompt 分桶，预热覆盖每条通道可达的宽度与允许的批大小。 |
| §9.1 Engine 线程 | 部分实现 | engine thread、completion thread 与 device-image thread（在自己的 stream 上解码、resize 带 `DeviceImage` 输入的请求的视图）三个线程；OMP / MKL / NUMBA 线程数默认 1；预热后冻结 GC 未实现 |
| §9.2 Frontend 进程 | 已实现 | 默认 2 个进程，SO_REUSEPORT，每个进程 4 个处理线程；共享内存加 pipe 通知（不是 ZMQ）；不初始化 CUDA，但 import torch。进程内调用由 `Policy` 提供 |
| §10 配置解析 | 部分实现 | 显式值 > 调优表 > 模型 / 平台规则 > 默认值，启动时打印每项来源（docs/configuration.md）；目前没有随包发布的调优表；`xinfer tune`、显存规划器参与推导、`Platform.derive` / `validate` 未实现 |
| §11 模型接入 | 已实现 | 模型声明阶段、交接张量、能力、输入输出处理和形状桶。前端通过模型类的 `build_processors` 构建处理器，引擎预热时调用模型的 `pad` 钩子。 |
| §11.4 内置模型 | 已实现 | Wall-OSS 提供参考和优化两种 BF16 组合，RTX 5090 上的优化组合通过 quick100 门禁。 |
| §12 量化 | 部分实现 | 按模块组的 `QuantConfig`，BF16 与 MXFP8（xcompute，sm120）；量化预设由模型选项 `quantization` 指定，默认 BF16，不按（模型, 平台）解析；NVFP4（`nvfp4`、`nvfp4_mse`，即 Wall-X NVFP4 服务的两种配方）只有 torch 参考实现，没有设备后端，也没有模型预设使用；`xinfer eval quant` 未实现，精度门禁见 docs/precision.md |
| §13 Platform | 部分实现 | 只有设备能力描述和 CUDA / CPU 解析（`xinfer/platforms/`）；stream、event、拷贝原语、`register_host`（现由 `frontend/shm.py` 直接调用 `cudaHostRegister`）、`classify_error`、`derive` / `validate` 和插件激活未实现；Thor（sm110）未验证 |
| §14 算子 | 部分实现 | `OpSpec` 与契约、注册表、构建期选择和选择表已实现；后端为 xcompute（TVM-FFI）和 flash-attn；`xinfer.ops.conformance` 未实现 |
| §15 协议与前端 | 部分实现 | WebSocket + msgpack，兼容 x2robot 客户端；`xinfer.Policy` 进程内 API；Client SDK 与兼容适配未实现 |
| §16 可观测性与基准工具 | 部分实现 | 服务通过 [/metrics](metrics.zh-CN.md) 导出 Prometheus 指标。使用[压测工具](benchmark.zh-CN.md)测单请求延迟和闭环容量，或用 [xinfer-profile](profiling.zh-CN.md) 分析算子、排队及各阶段执行时间。 |
| §17 测试 | 部分实现 | CPU 单元与契约测试；GPU 与资产测试需要本地资产，手动运行；CI（`.gitlab-ci.yml`）只跑 ruff 与 CPU 测试 |
| §18.1 仓库结构与命令行 | 与规划不同 | 目录以仓库为准；服务入口为 `xinfer-serve`（即 `python -m xinfer.entrypoints.serve`），profiling 入口为 `xinfer-profile`（`ops`）；统一的 `xinfer` 命令未实现 |

---

## 1. 背景：VLA serving 与 LLM serving 的差别

| 方面 | LLM serving | VLA serving |
|---|---|---|
| 请求形态 | 一次 prefill + 长时间自回归 decode | 一次性计算：视觉编码 → VLM prefill → 动作去噪（固定步数），没有长 decode |
| 到达模式 | 用户随机到达 | 每台机器人闭环、周期性发送，每台通常只有一个在途请求；下一次发送 = max(上次发送 + 周期, 响应到达) |
| 状态寿命 | KV 跨多轮 decode 存活，需要分页管理 | 前缀 KV 只活到本请求去噪结束；按请求分配 slot 即可 |
| 形状 | 序列长度变化大 | 相机数和分辨率固定，prompt 长度有界，适合形状桶加 CUDA Graph |
| 瓶颈 | 显存带宽、KV 容量 | 小 batch 下 GPU 利用率低；CPU 预处理与发射开销；阶段之间计算量差别大 |
| 指标 | TTFT、TPOT、吞吐 | 单请求延迟；单卡容量 N\*（给定延迟口径下能服务的机器人数） |

所以 VLA 不能直接套用 vLLM/SGLang 的 token 级调度和分页 KV。实测上，FleetServe 在 RTX 5090 的同口径对比中，容量显著高于 SGLang 和 vLLM-Omni（附录 A）。

---

## 2. 目标与非目标

### 2.1 目标

| 目标 | 验收口径 | 现有参考 |
|---|---|---|
| Thor 单请求 | Thor 上 Wall-OSS，同输入 | 不高于 HarriX `mxfp8-thor` |
| 单卡容量最高 | 5090，统一输入，客户端 P50 < 150 ms，三遍 | FleetServe FIFO 路径的同口径容量（基线数字需按 §20 M1 复核） |
| 单请求与满载结果一致 | 同一输入单独运行和满载运行 | 两者都通过该模型的精度门禁；阶段算子都是确定性时输出逐位相同（§6.4） |
| 接入新模型容易 | 骨干和动作头已有构件时 | 只写模型文件和权重映射，不改运行时 |
| 接入新硬件和量化容易 | 新平台或新量化方法 | 只实现平台层、算子后端或 QuantMethod，不改模型文件 |

### 2.2 非目标（v1）

- **SLO 调度器。** 核心只留策略插件接口（§7.5）。
- **训练、RL、在线换权重。**
- **多卡张量并行。** 首批模型单卡放得下；多卡用数据并行副本加 Router。
- **GPU 阶段拆进程（`split`）。** v1 只实现 `colocated`，不实现阶段间 CUDA IPC、copy-ACK 或 MPS 生命周期；CPU Frontend 多进程仍在范围内。保留职责边界，不预建远程执行框架（§9.3）。
- **默认开启改变数值的优化。** 例如近似阶段重叠（pipeline-overlap-vla 的代理 KV 方案在 LIBERO 上成功率约 95% vs 97%）或减少去噪步数。量化作为显式配置提供，带精度门禁。
- **自回归动作 token 模型**（FAST、OpenVLA 类）。阶段抽象为 decode 阶段预留接口，v1 不实现。

---

## 3. 术语

| 术语 | 含义 |
|---|---|
| Stage（阶段） | 模型声明的一段计算，例如 `vision`、`prefill`、`action`。代码单位 |
| StageGraph | 模型声明的计算阶段 DAG，不声明进程归属。首批模型按 vision → prefill → action 的逻辑链组织，具体阶段与交接由各模型声明 |
| Lane | 某阶段的一个并发执行槽：一条 stream + 一组静态工作区 + 一个图池 |
| 形状桶（bucket） | 阶段输入可取的离散形状，例如 Wall-OSS 的 prompt 宽度取 16 的倍数 |
| Slot | 按请求分配的交接存储，存放跨阶段张量（如前缀 KV），带 generation |
| Handoff | 阶段之间交接的张量集合，由 HandoffSpec 声明 |
| 链式发射 | 上游阶段发射时，把下游阶段一起发射，靠 GPU 事件衔接（§6） |
| 调优表 | 按（模型, 平台, 量化）存放的实测最优资源配置（§10） |

---

## 4. 总体架构

```text
 机器人侧 Client SDK（触发规则、RTC、chunk 对齐、计时）
        │ WebSocket + msgpack
 ┌──────▼──────────────────────────────────────────────────────────┐
 │ Frontend 进程 ×N（纯 CPU，看不到 GPU）                            │
 │   协议与会话 · InputProcessor（解码/resize/tokenize/归一化）       │
 │   OutputProcessor（反归一化、表示转换、RTC 对齐）                  │
 └──────┬─────────────────────────────────────────▲────────────────┘
        │ 共享内存 slot（lease + generation）+ 通知  │
 ┌──────▼─────────────────────────────────────────┴────────────────┐
 │ Engine 进程（每 GPU 一个）                                        │
 │   Scheduler：接纳 · 阶段队列 · 组批 · 链式发射 · 取消 · 策略插件    │
 │   StageExecutor ×3：每阶段 k 条 lane；GraphManager                │
 │   Memory：只读权重 · 请求 slot 池 · 图池预算 · 显存规划器          │
 │   完成线程：观察最终事件，交还结果                                  │
 └──────┬──────────────────────────────────────────────────────────┘
        │ 阶段函数
 ┌──────▼──────────────────────────────────────────────────────────┐
 │ models/ · layers/ · quantization/ · loader/                      │
 └──────┬──────────────────────────────────────────────────────────┘
        │ ops.<name>(...)
 ┌──────▼──────────────────────────────────────────────────────────┐
 │ platforms/（cuda: sm120, sm110；cpu 参考）· ops/（契约 + 参考实现）│
 │ xcompute（现有算子仓库，按平台交付高性能实现）                     │
 └─────────────────────────────────────────────────────────────────┘
 可选：Router（会话亲和、失败重试）· 实例生命周期 · 健康检查
```

### 依赖方向

逻辑阶段与进程映射分离。v1 的三个 GPU 阶段都在同一 Engine 内，各自拥有队列、lane、stream 和图池；阶段独立推进不要求阶段独立进程。模型通过 StageGraph 和 HandoffSpec 声明计算及交接契约，不持有进程、IPC 或调度器内部句柄。

两条原则：

- **层间依赖尽量少。** 新增跨包依赖要先写进下表并说明理由。
- **抽象的代价放在构建期和启动期。** 算子选择、配置解析、图捕获都在启动时完成；发射、完成观察和交接这些热路径只经过薄的一层调用，不做按调用的分派、校验或额外拷贝，开销按 §6.5 测量。

| 模块 | 可以依赖 | 不得依赖 |
|---|---|---|
| `engine/` | `platforms/`、模型接口（抽象类）、`observability.annotate`（阶段 span） | 任何具体模型、量化方法、kernel 包；`config/`、`layers/`、`loader/`、`ops/`、`frontend/` |
| `models/` | `layers/`、`ops/`、`quantization/` 接口、`loader/` | `engine/` 内部、具体平台 |
| `layers/`、`quantization/` | `ops/` | 具体 kernel 包（只能经 ops 注册） |
| `ops/` | `platforms/` | 模型、`observability/` |
| `frontend/` | 模型的 Input/OutputProcessor | `engine/` 等运行时内部；`torch.cuda` 与任何设备运行时（Frontend 进程不初始化 CUDA 或其他设备运行时） |
| `config/` | `platforms/`、`engine/`（`EngineConfig`）、已构建的模型 | `frontend/`、`layers/`、`loader/` |
| `loader/` | 无 | `models/`、`layers/`、`engine/` |
| `observability/` | `ops/`、`platforms/` | `engine/`、`models/`、`layers/`、`frontend/`、`quantization/`、`loader/`、`config/` |
| `entrypoints/` | 以上任何包 | 无（其他包都不依赖它） |

启动顺序是：构建模型（算子在此时选定）→ `config.resolve` 读取已构建模型的 `capabilities()` 与平台能力 → 创建 Engine。所以已构建模型支持的能力（例如批大小）经模型接口交给 engine，engine 不依赖 ops，也不依赖 config；Frontend 的共享内存与 Engine 之间由 `entrypoints/serve.py` 衔接，engine 不依赖 frontend。

`test_layering.py` 检查上表中包之间的依赖方向，以及 `platforms/interface.py`、`models/interface.py` 不 import torch。下面的设备边界是目标，尚未实现，也未检查。

**设备边界（目标）。** 设备运行时 API（`torch.cuda`、`torch.npu` 等）、设备字符串和架构判断（`sm_`、`gfx` 等）只允许出现在 `platforms/` 下的具体平台文件和 `ops/backends/`。ROCm 版 PyTorch 也用 `torch.cuda` 接口，所以边界按平台文件划分，不按 API 名划分。核心代码按 Platform 的能力判断，不按平台名判断（§13.1）。现状：`engine/` 直接使用 torch 张量和 `torch.cuda` 的 stream、event、graph 与 pinned 内存，并按 `Platform.is_cuda` 分支；`frontend/shm.py` 直接调用 `cudaHostRegister`；各模型的 `processing.py` 输出 torch CPU 张量，`frontend/` 因此 import torch，但不初始化 CUDA。

---

## 5. 请求生命周期

```text
Client ──obs──▶ Frontend: 校验 IOSchema → InputProcessor → 写入共享内存 slot → 通知 Engine
Engine Scheduler: 接纳（分配请求 slot）→ H2D（uint8 图像、token、state）
   vision lane:  encode → 写 slot.image_embeds             (event v)
   prefill lane: wait(v) → prefill → 写 slot.prompt_kv …   (event p)
   action lane:  wait(p) → gather → N 步去噪 → D2H 到 pinned (event a)
完成线程: wait(a) → 原始动作交回 Frontend → OutputProcessor → Client
回收: 每个 lane 在其阶段事件后空闲；slot 在最后一个读者的事件后回收（generation +1）
```

### 所有权规则

- **权重**：只读，加载并完成量化转换后冻结。
- **Slot**：归请求所有。生产阶段写入，消费阶段只读；最后一个读者的事件完成后才回收，回收时 generation 加 1，过期句柄会被检测到。
- **Lane 工作区**：归 lane 所有。阶段输出若需跨阶段使用，必须写入 slot，不能留在 lane 工作区。
- **图输出**：下一次 replay 之前，必须已复制或已交接。

### 取消与错误

- **取消/断连**：还没发射的请求直接丢弃；已发射的让 GPU 跑完、丢弃结果，slot 按事件回收。
- **单请求错误**（输入校验失败、形状不在桶内）：只让该请求失败。
- **设备粘滞错误**（CUDA sticky error，或其他平台上同类的不可恢复错误；由 `Platform.classify_error` 区分请求级与设备级）：实例进入 DRAINING，由监督进程重启。

---

## 6. 统一执行模型：单请求与机队用同一套实现

### 6.1 结论

只有一套分阶段结构。单请求路径是机队路径在没有排队时的自然表现，不另写代码，也不做模式切换。

### 6.2 依据：各类开销分别来自哪里

| 开销 | 影响 | 依据 |
|---|---|---|
| 阶段之间经过主机中转（观察完成 → 入队 → 再发射） | 单请求延迟 | 完成观察、阶段交接与返回 |
| 等待凑批 | 单请求时延 | 即使有空闲通道，请求也要等待。 |
| 整个请求独占一条通道 | 机队吞吐 | 各阶段无法在空闲时调度其他已就绪的请求。 |
| 各阶段同步推进 | 机队吞吐 | 阶段无法在工作就绪时独立推进。 |

前两项属于"阶段之间怎么衔接"，后两项属于"按什么粒度执行"，两者互不冲突：保留分阶段的粒度，只改衔接方式。

### 6.3 三条调度规则

1. **零等待组批。** 发射时把队列中已有的兼容请求一起带上，不为凑批等待。空载时自然是 B1，有排队时自然成批。
2. **下游空闲时链式发射。** 发射阶段 s 时，如果下游阶段队列为空且有空闲 lane，就立即把下游也发射出去，用设备侧事件等待（`stream.wait_event`，CUDA 上即 `cudaStreamWaitEvent`）衔接；链式路径不等待主机观察中间完成再提交下游。下游已有排队时正常入队，保持 FIFO 和组批。
3. **结果返回只等待最终完成。** 完成线程等待最后一个阶段的事件并返回结果；调度线程仍非阻塞检查各阶段事件，用于及时回收 lane、推进非链式请求和维护 slot 生命周期。

**效果：** 1 台机器人时，每个请求从 vision 到 action 一次发射完，就是单请求最快路径。机器人一多，下游有了排队，自动变成流水线加组批。每个请求按发射时刻的队列状态各自决定，从空载到满载是连续的。

**不做整链图。** 所有部署都是单卡服务多台机器人，把整条链捕获成一张图会取消阶段间的流水和组批；单请求的最快路径靠链式发射。HarriX 的 `mxfp8-thor` 用的是整调用图，Thor 的 CPU 也较弱；若 M0 测出 Thor 上"三段图 + 事件"明显慢于整调用图（§6.5），再评估只在空载时使用的整链图。

### 6.4 正确性契约：单独运行与满载运行

- **同一套 kernel。** 两条路径使用同一批阶段图。算子不要求确定性（§14.1）：满载下的结果要像单独运行的结果一样通过该模型的精度门禁；某个阶段的算子都是确定性时，两者逐位相同。
- 模型通过精度门禁后，才为该阶段声明大于 1 的批大小。组批可能改变浮点舍入。Wall-OSS 参考组合支持 action 组批，优化组合保持 B1。
- **适用范围。** 逐位相同只在同一平台和同一组算子实现内成立，不跨芯片；跨平台与参考实现的对齐用容差，每个平台保存自己的 golden。

### 6.5 M0 / M1 需要量化的风险

以下测量均使用 `colocated`；容量与组批风险在 M1 的多 lane 运行时和机队基准具备后验证，不要求实现 `split` 对照。

| 风险 | 测量方法 | 里程碑 |
|---|---|---|
| 链式发射预占下游 lane，可能降低吞吐 | 链式开/关两组，比较同口径容量 | M1 |
| 链式发射的请求不再参与下游组批 | 临界负载附近，比较组批率与容量 | M1 |
| Thor 上"三段图 + 事件"比整调用图慢 | 同输入比较单请求延迟 | M0 |
| 主机发射开销 | 测"请求就绪 → 整条链发射完"的耗时，目标在 100 µs 量级 | M0 |

---

## 7. 调度器

### 7.1 接纳

- 在途请求数不超过 slot 数；默认每台机器人至多一个在途请求。
- 接纳和分配一步完成，返回 `AdmitReady(slot)`、`AdmitDelay` 或 `AdmitFail(code)`，避免"先探测、后分配"的竞态（vla-engine 的教训）。
- 队列有界。错误码区分 `queue_full`、`per_robot_limit` 和 `admission_timeout`。

### 7.2 阶段队列

`q_vision → q_prefill → q_action`，默认均按到达 FIFO。这与 FleetServe 现在的 `greedy-fair` 行为一致，论文中称为 FIFO 基线。

### 7.3 组批

- compat key 由 HandoffSpec 和模型声明计算，key 相同的请求才能组批。
- 只用已捕获的 batch 大小，不补假行。
- 零等待：只带上已经在队列里的请求。
- v1 中，action 阶段把多个 slot 的上下文 gather 到 lane 的批工作区。以后可以改为按 slot 索引读取的 kernel，省掉这次复制。gather 由 Platform 的异步拷贝原语（支持跨步长，§13.1）完成。

### 7.4 发射

- 某阶段有空闲 lane 就发射，并按 §6.3 判断是否链式发射。
- 每次发射记录事件；lane 在本阶段事件完成后释放。

### 7.5 策略插件

```python
class StagePolicy(Protocol):
    def select(self, stage: str, queue: StageQueue, free_lanes: int,
               now: float) -> list[Dispatch] | Hold: ...
```

- **职责划分**：策略只决定"发射哪些请求、现在发还是等"。lane 分配、链式发射、组批拼装和回收都由运行时完成。
- **开源版**：只带 `fifo`。
- **内部 SLO 调度器**：作为插件实现，不修改核心。这样也避免了 embodied-serving `global_scheduler.py` 里 SLO 代码与通用代码交织的问题。

---

## 8. 显存与所有权

### 8.1 Slot 状态机

沿用 embodied-serving `kv_cache/batch_manager.py` 已验证的状态机：

```text
FREE → WRITING → READY → READING → RELEASING(等事件) → FREE (generation+1)
```

### 8.2 显存规划器

启动时检查：

```text
权重 + Σ lane 工作区 + Σ 图池 + slot 数 × slot 大小 + 余量 ≤ 可用显存
```

- 如果显存检查失败，规划器应报告可行的通道数和 slot 数。
- Thor 是统一内存，预算按平台给出的可用值计算。

### 8.3 图管理

- 图按（阶段, lane, 形状桶, batch）捕获。同一 lane 内不同形状共享图池，lane 之间不共享。
- 预热时捕获所有声明的组合后封存。运行中遇到未声明的形状默认报错，可配置为 eager 回退并告警。
- 需要 workspace 的算子必须在预热阶段完成首次分配（§14.1）。
- 形状桶的粒度由平台决定：模型声明合法范围，平台在 padding 浪费与捕获次数之间取舍。

---

## 9. 进程与线程模型

### 9.1 Engine 进程（每 GPU 一个，默认 `colocated` 布局）

v1 只实现 `colocated`。Vision、Prefill、Action 共享一个 Engine 进程，执行资源和队列仍按阶段独立管理；不同请求的 GPU 工作可在不同 stream 上重叠。同一请求始终遵循 StageGraph 的依赖顺序。

| 线程 | 职责 |
|---|---|
| Engine 线程（唯一） | 接纳、队列、组批、链式判断、阶段事件的非阻塞完成检查，以及全部 CUDA 提交（H2D、图 replay、事件）。调度和发射放在同一线程，省掉一次线程切换 |
| 完成线程 | 阻塞等待最终事件（等待时不持 GIL），交还结果 |

- **线程卫生**（所有平台默认，不藏在某个硬件配置里，也不经 Platform）：进程启动时把 OMP/MKL/NUMBA 线程数设为 1，预热后冻结 GC。依据：NUMBA 线程超订曾导致 14 台时超时。
- **发射热路径**：CPU 前后处理交给 Frontend；Engine 线程不执行阻塞的 GPU 完成等待，预热后不做显存分配或图捕获。M0 测提交开销，M1 测 Engine 线程占用、锁等待，以及请求可运行后到实际提交的延迟；单进程容量必须实测验收。

### 9.2 Frontend 进程

- 默认 N 个进程，用 SO_REUSEPORT 共享端口，不初始化 CUDA。
- 每个进程用多个处理线程（默认 4）同时处理不同连接的请求：图像解码和 resize 释放 GIL，连接在进程间分布不均不会成为瓶颈。InputProcessor 与 OutputProcessor 因此须可并发调用。
- 通过共享内存 slot（lease + generation）加 ZMQ/管道通知与 Engine 通信。
- Engine 进程一侧把共享内存映射一次，并经 `Platform.register_host` 注册为 pinned，H2D 直接从中读取。可分页内存上的异步拷贝会阻塞发射线程，还多一次主机拷贝。
- 单机器人部署可以改用进程内线程，省掉跨进程开销。

### 9.3 逻辑阶段与进程映射分离；`split` 暂缓

- **模型边界**：StageGraph 只声明阶段及依赖，HandoffSpec 只声明跨阶段张量布局；模型文件不感知阶段所在进程或传输方式。
- **运行时边界**：阶段队列、组批和 lane 按逻辑阶段管理；slot 归请求所有，其读写与回收遵循阶段依赖。执行资源的组装与交接由运行时负责；v1 将全部 GPU 阶段固定映射到同一 Engine，通过请求 slot 和 GPU event 交接。
- **当前范围**：不实现 `split`，不增加进程布局切换参数、远程 StageExecutor、阶段间 IPC 通道或 MPS 启停代码。CPU Frontend 的共享内存和通知通道不受此限制。仅保留上述职责边界，不预建空实现。
- **后续扩展**：若实测表明 colocated 在消除阻塞和热路径开销后仍受 CPU 提交瓶颈限制，再单独评审进程拆分。可考察 FleetServe 的 Vision + Prefill / Action 两组映射，复用逻辑阶段和数值契约；不纳入当前 M0–M3 的实现或验收承诺。

各阶段位于同一进程时可直接使用请求 slot，并通过设备事件衔接发射。应在目标负载下测量该布局的时延和容量。

### 9.4 执行方式

每个阶段在 stream 上直接调用模型的阶段函数（经 ops 提交），并按 §8.3 捕获为图。不提供图编译类执行后端（追踪导出后交给 TensorRT 或 NPU 编译器），只提供图编译运行时的平台不在设计范围内；也不把多个阶段合成一张图（§6.3）。

---

## 10. 配置解析

### 10.1 参照 vLLM / SGLang

两者都**没有命名 profile**：

- 参数扁平，大多默认 None 或 auto，启动时只填用户没设的项。
- vLLM 的 `get_batch_defaults()` 按显存档位和使用场景给出默认值。
- SGLang 的 `resolve_once()` 依次运行设备、平台、显存、模型几类 handler。
- vLLM 的 `--performance-mode interactivity|throughput` 只调整推导出的默认值（捕获哪些 batch 的图、批量上限），不切换代码路径。
- kernel 级的实测调优表随仓库发布（vLLM 的 fused MoE 有数百个按设备名命名的 JSON）。

### 10.2 解析顺序（高优先级在前）

1. **显式值**：总是生效；与平台能力冲突时直接报错，不静默改写。
2. **调优表**：按（模型, 平台, 量化）存放实测最优的 lane 数等资源配置。由 `xinfer tune` 生成，随仓库发布，记录提交、日期和测量口径。
3. **推导规则**：根据平台能力、模型声明（阶段图、批大小、形状桶）和显存规划器推导。形状桶粒度和默认 `QuantConfig` 同样按（模型, 平台）在这一层解出，不写在模型文件里。caps 表达不了的平台限制由 `Platform.derive` 与 `Platform.validate` 提供（§13.1），结果在启动报告中标为 `platform:<name>`；核心规则里不写厂商特例。
4. **固定默认值**：与负载无关。

不设按负载推导的意图参数（如服务的机器人数）。实例服务的机器人数随时变化，而引擎本身随到达的请求自适应：只把已在队列中的请求组批，空闲的 lane 不占算力，CUDA graph 按实际出现的键捕获。因此在途请求上限（对应 vLLM 的 `max_num_seqs`，它同样是固定的默认上限）、各阶段允许的 batch 集合、lane 数和 frontend 进程数，都取对各种负载都合适的上限或固定值（docs/configuration.md）。

链式发射、零等待组批等运行时行为不做成配置，始终开启，只保留消融用的调试开关。

v1 的 GPU 进程布局固定为 `colocated`，不由调优表或负载状态切换为 `split`。

### 10.3 示例与审计

```bash
xinfer serve --model wall_oss --checkpoint CKPT                     # 默认值
xinfer serve --model wall_oss --checkpoint CKPT --lanes vision=1    # 显式覆盖单项
```

启动时打印并落盘解析结果，每一项都标出来源：

```text
max_running_requests  32         default
lanes.vision          2          tuned:wall_oss/sm120-mxfp8@<commit>
batch_sizes.vision    [1]        model:declares B1 only
batch_sizes.action    [1,2,4,8]  model:declared sizes
cudagraphs            True       platform:supports graph capture
frontends             2          default, ≤ cpus/2 = 12
frontend_workers      4          default, frontends × workers ≤ cpus/2 = 12
```

### 10.4 不用命名 profile 的理由

- 组合会随"模型×硬件×部署"成倍增长，而且容易过时。
- 值的来源不可见。
- 与硬件无关的设置容易被藏进硬件 profile。例如 embodied-serving 的 `rtx5090` profile 里设置了 `OMP_NUM_THREADS=1`，这本该是所有平台的默认值。

---

## 11. 模型接入

### 11.1 模型作者实现什么

| 组件 | 职责 | 运行位置 |
|---|---|---|
| `IOSchema` | 输入：相机名与分辨率、state 维度、指令、RTC 字段及合法范围；输出：动作维度与表示 | 协议校验 |
| `InputProcessor` | 解码并缩放视图，分词并归一化状态。带连接状态的处理器可实现 `connect()`。模型可以返回 `DeviceImage`，由引擎在首个阶段前解码并缩放。 | Frontend |
| `VLAModel` | 声明 StageGraph、形状桶（合法范围）、HandoffSpec、Capabilities；实现各阶段前向 | Engine |
| `OutputProcessor` | 反归一化、表示转换、RTC 对齐 | Frontend |
| `WeightMapping` | checkpoint 键到参数的映射 | 加载 |

```python
class WallOSS(VLAModel):
    stage_graph = StageGraph(
        Stage("vision",  "encode_images"),
        Stage("prefill", "prefill", depends_on="vision"),
        Stage("action",  "denoise", depends_on="prefill"),
    )
    handoffs = (
        HandoffSpec("vision", "prefill", {"image_embeds": TensorSpec(batch_axis=0)}),
        HandoffSpec("prefill", "action", {
            "prompt_kv": TensorSpec(batch_axis=2, seq_axis=4)}),  # 模型按所选 attention 后端声明轴（§11.2）
    )
    def shape_buckets(self, cfg): ...          # 计划：合法范围，如 prompt 宽度 ≤ 200、步长 16；实际桶由平台选（§8.3）
    pad_prompt = staticmethod(pad_prompt)      # 把准备好的 prompt 填充到桶宽度；基类的 pad() 调用它
    def capabilities(self): ...                # RTC、batch 大小、各阶段调用的算子
    def stage_outputs(self, stage, b): ...     # 各阶段输出缓冲的形状与 dtype，由运行时分配

    def encode_images(self, b: StageBatch) -> None: ...   # 写 b.out["image_embeds"]
    def prefill(self, b: StageBatch) -> None: ...         # 写 b.out["prompt_kv"]
    def denoise(self, b: StageBatch) -> None: ...         # 整段 N 步，写 b.out["actions"]
```

**注册是惰性的。** 模型以名字加两个"模块:类"字符串注册：一个指向 processing（IOSchema、Input/OutputProcessor），一个指向 model。

```python
register_model("wall_oss",
               processing="xinfer.models.wall_oss.processing:WallOSSInputProcessor",
               model="xinfer.models.wall_oss.model:WallOSS")
```

Frontend 只导入 processing，不导入模型代码；Engine 才导入 model。一个模型的导入失败不影响其他模型。树外模型经 `xinfer.models` 入口用同样的形式注册。

### 11.2 阶段函数约束

- 每个形状桶内形状静态，输出只写入运行时提供的 slot 或工作区。
- 只在当前 stream 上提交；不做主机同步；预热后不分配显存。
- 不持有跨请求的可变状态。
- 去噪整段执行（N 步在一个图里）。步数写在 Capabilities 中，为以后的步级调度留接口。

`HandoffSpec` 声明每个张量的批次轴和序列轴。运行时按声明收集和分发阶段输出；生产者和消费者的布局须一致。

### 11.3 可复用构件

| 类别 | v1 | 之后 |
|---|---|---|
| VLM 骨干 | Qwen2.5-VL | — |
| 视觉塔 | Qwen2.5 ViT | — |
| 动作头 | 带自适应 RMS 归一化及前缀 K/V 注意力的动作专家 | DiT 交叉注意力头（GR00T 类） |
| 采样器 | flow-matching Euler；RTC（冻结前缀） | — |
| 预处理 | 多相机图像、归一化器、prompt 模板 | — |

### 11.4 内置模型

Wall-OSS 使用 Qwen2.5-VL 视觉塔、prefill 专家和流匹配动作专家。张量布局、执行阶段和数值配置见 [Wall-OSS 设计](models/wall_oss_design.zh-CN.md)。

---

## 12. 量化

- **`QuantConfig`**：方法，加上按模块的策略。例如 Wall-OSS 的 `mxfp8` 预设为 `{vision.blocks: mxfp8, vision.merger: mxfp8, prefill: mxfp8}`，其余模块（动作专家）为默认的 BF16。默认值按（模型, 平台）解析（§10.2），不写死在模型文件里；HarriX 的 Thor 配置 ViT 用 BF16 就是平台差异。
- **`QuantMethod` 接口**：
  - `create_weights`；
  - `process_weights_after_loading`：转换、打包、swizzle，必须在权重冻结之前完成；
  - `apply`：调用 ops 中对应格式的 GEMM。
- **两类来源**：加载时在线量化（MXFP8）；离线预量化 checkpoint（QAT 产出的 NVFP4，M2）。
- **新增方法**：一个 QuantMethod，加上（需要时）对应格式的算子后端，不改模型文件。
- **精度门禁** `xinfer eval quant`（按模型、平台、方法分别判定）：
  - 对比 BF16 参考，报告开环动作误差和相对 BF16 的漂移；
  - 用一组扰动重复测量报告噪声底；
  - 未通过门禁的方法标为 experimental。
- **v1 范围**：BF16、MXFP8。NVFP4 在 M2。

---

## 13. 硬件平台

### 13.1 Platform 接口

```python
class Platform:
    name: str                 # "cuda-sm120", "cuda-sm110", "cpu"
    caps: PlatformCaps        # supported_dtypes, unified_memory, memory, compute_units,
                              # power_limit, graph_capture, stream_priorities, supported_quant
    # 必选契约：引擎实际依赖的全部原语
    def stream(self, priority=0): ...             # 异步执行队列；stream.wait_event(evt) 在设备侧等待
    def event(self): ...                          # evt.query() 非阻塞；evt.synchronize() 阻塞且不持 GIL
    def alloc(self, nbytes, kind): ...            # kind: device | pinned_host，返回 torch 张量
    def copy_async(self, dst, src, stream): ...   # H2D / D2H / D2D，支持跨步长
    def register_host(self, mem): ...             # 把已映射的主机内存（Frontend 共享内存）注册为可 DMA，H2D 直接读它（§9.2）
    def memory_budget(self): ...
    def device_count(self): ...                   # 可用设备数；serve 为每个设备启动一个 Engine
    def device_info(self, i): ...                 # 名称、uuid、总内存，写入启动报告
    def bind_env(self, i): ...                    # 子进程启动前要设置的环境变量（可见设备等）
    def classify_error(self, exc): ...            # 请求级错误 vs 设备粘滞错误（§5）
    # 配置参与（§10.2）
    def derive(self, cfg): ...                    # 补全 caps 表达不了的默认值，每项带来源 platform:<name>
    def validate(self, cfg): ...                  # 平台特有的限制，冲突时报错
    # 可选能力
    def graph(self, pool=None): ...               # 图捕获与回放（caps.graph_capture，§8.3）
```

平台在构建时声明能力。运行时按能力推导配置；遇到不支持的显式配置直接报错。

- **必选契约要求异步执行。** 调度器的单线程发射、完成线程和链式发射都依赖异步 stream 与设备侧事件等待；同步阻塞的平台不在设计范围内。
- **dtype 与量化是能力。** `supported_dtypes` 缺少模型所需的 dtype（例如部分 NPU 没有 bf16），或 `supported_quant` 不含所选量化方法时，在配置解析阶段报错，不等到模型构建。
- **外部平台**通过 `xinfer.platforms` 入口注册（§18.3），内置的 cuda 和 cpu 走同一机制。
- `ipc`、`mps`、`spatial_partition` 在 v1 没有代码读取（§9.3），需要时再加入。
- **核心按能力判断，不按身份判断。** Platform 不提供 `is_cuda()` 一类的身份谓词；`engine/`、`layers/`、`models/` 只读 caps，未声明的 caps 取保守值（不支持）。Platform 的每个方法都必须有核心里的调用方和契约测试，否则不加（SGLang 的平台接口里有一批核心并未调用的方法，核心仍硬编码 `torch.cuda`，覆盖无效）。

### 13.2 首批平台

| 能力 | RTX 5090（sm120） | Jetson Thor（sm110） |
|---|---|---|
| 典型部署 | 机队，单卡多台 | 机队，单卡多台 |
| 内存 | 32 GB 独立显存 | 统一内存：H2D 可以零拷贝或映射 |
| CPU | 服务器 CPU（测试容器配额 24 核） | ARM，较弱：发射开销占比更高，机队下 Frontend 预处理也更容易成为瓶颈 |
| MXFP8 GEMM | xcompute（TVM-FFI） | xcompute `mxfp8-thor` |
| MPS | 支持（FleetServe split 布局使用）；v1 colocated 不依赖 | 待确认；v1 colocated 不依赖 |
| 已知参考 | FleetServe、HarriX `mxfp8-sm120` | HarriX `mxfp8-thor`（ViT BF16、整调用图、RTC 关闭） |

AMD GPU 也是目标平台，接入时机与型号待定（§22）。Platform 接口按 HIP 的差异设计：HIP Graph、没有 MPS、CU mask；可用的量化格式由型号决定，经 `supported_quant` 表达。

### 13.3 对外稳定面与厂商自证

XInfer 作为开源项目接入不同厂商的芯片，厂商包在树外维护，所以要写明什么是稳定的，以及厂商怎样证明自己接对了。

- **稳定接口面**：Platform、`ops` 的算子签名与数值契约、VLAModel 与 HandoffSpec。这个集合带一个 API 版本号；插件声明所需的版本范围，不符时在激活阶段报错，并说明双方版本。不做迁移层和兼容垫片，破坏性变更升主版本。
- **厂商自证**：
  - `xinfer.platforms.conformance` 检查 Platform 契约：`wait_event` 次序、`query` 不阻塞、`synchronize` 释放 GIL、`copy_async` 跨步长、`register_host` 后的拷贝不阻塞主机、`alloc` 的两种 kind、错误分类；
  - `xinfer.ops.conformance`（§14.1）检查算子实现。
- **树外路径本身要被测试。** CI 把 CPU 模拟平台打成独立的树外风格包安装一遍，经入口点走完发现、激活和契约测试；否则第一个厂商会先踩到入口点路径上的问题。
- **支持等级**（写入 `docs/platforms/`）：Tier 1 由主仓库 CI 覆盖（5090、Thor）；Tier 2 由厂商维护并通过上述套件，需标明所用的 API 版本；其余为社区维护。

---

## 14. 算子与 kernel 仓库

### 14.1 算子契约（在 XInfer 仓库 `xinfer/ops/`）

每个算子定义以下内容：

- **签名**：输出写入调用方提供的 `out`；需要 workspace 时由调用方提供。
- **torch 参考实现**。
- **数值契约**：与参考逐位一致还是给定容差；是否确定性。

确定性只需声明，不作要求。算子在多次运行之间可以有舍入量级的差异，例如 stream-K GEMM 在并发负载下部分和以不同的顺序相加。模型对算子的要求是能通过该模型的精度门禁（docs/precision.zh-CN.md）；对 VLA 模型来说主要是对照录制轨迹的开环门禁。声明 `deterministic=False` 是告诉测试和读者不要期待逐位可复现。

调用约定：

- 只在当前 stream 上提交；不隐式同步；不在内部分配显存。
- 需要 workspace 的算子在预热中完成首次分配。现在 xcompute MXFP8 按 stream 懒分配 workspace，在图捕获中会报错，这条约定由此而来。
- 上述 `out`、workspace、当前 stream、无隐式同步与分配约定不因 TVM-FFI 或 `torch.library` 绑定而改变。

验证要求：

- 每种绑定均需验证与参考实现的数值一致性、所声明的确定性，以及所声明的 CUDA Graph 捕获与重放能力；绑定方式本身不保证这些性质或性能收益。
- 引入 `torch.library` 适配时，额外使用 `torch.library.opcheck` 验证注册契约，并测试实际支持的编译或导出路径，覆盖输入修改与别名声明、FakeTensor／形状推导等行为。`opcheck` 不替代数值一致性测试。

### 14.2 后端注册与选择

```python
@ops.register("gemm_mxfp8", platform="cuda", arch=("sm120",), backend="xcompute",
              graph_safe=True, deterministic=True)
def _gemm_mxfp8_sm120(x, xs, w, ws, out, workspace): ...
```

- 在模型构建时选定实现，不在每次调用时分派。
- 选择结果和库版本、哈希写入引擎元数据，启动时打印"算子选择表"。
- 缺少实现时显式失败；也可配置为回退到参考实现，并告警。

**算子绑定策略：**

- **默认路径**：xcompute 通过 TVM-FFI 接入，由 `xinfer/ops/backends/xcompute.py` 统一包装并注册到 XInfer 的算子注册表。
- **按需适配**：当某个算子有明确的 `torch.compile`、`torch.export` 等 PyTorch 集成需求时，再增加 `torch.library` 注册。包装层复用已有底层实现，可通过 TVM-FFI 调用同一份 xcompute kernel，不要求维护两套 kernel。
- **职责边界**：`torch.library` 适配负责算子签名、输入修改与别名声明，以及所需的 FakeTensor／形状推导规则；硬件实现选择和数值策略仍由 XInfer 的算子注册表与契约管理。
- **统一调用面**：模型和 layer 只调用 XInfer 的算子接口，不直接依赖绑定方式。注册到 XInfer 不等于注册为 PyTorch 自定义算子。
- **交付范围**：不要求每个算子同时提供两种入口，也不要求默认执行路径经过两层包装。当前 xcompute 已采用 TVM-FFI；`torch.library` 适配随具体需求实现和验证，不预建占位模块，不声称已具备完整双路径。

### 14.3 分工

| | XInfer 仓库（框架组） | xcompute 仓库（算子组） |
|---|---|---|
| 内容 | 算子签名、参考实现、一致性测试套件、注册机制 | 各平台高性能实现（CUDA/CuTe/Triton/HIP），按平台出 wheel |
| 测试 | 模型级对齐、运行时一致性 | 在 CI 中运行 `xinfer.ops.conformance`（数值、确定性、图捕获），并给出 microbenchmark |
| 交付 | 每个模型的算子清单与形状 | 每个实现附理论下限表（roofline） |
| 版本 | 声明兼容的 xcompute 版本范围；启动时校验构建指纹 | 语义化版本；ABI 变化要升主版本 |

工具：`xinfer profile ops --model wall_oss` 按模型输出每个算子的耗时与理论下限，供算子组排定优先级。

厂商 NPU 的实现不要求并入 xcompute：作为独立包，经 `xinfer.ops`、`xinfer.platforms` 入口注册即可（§18.3）。

### 14.4 Wall-OSS 算子清单

- GEMM：BF16、MXFP8；以及融合量化（RMSNorm + quant、SiLU-mul + quant）。
- 注意力：
  - 视觉双向（Qwen ViT window/full）；
  - 前缀 prefill；
  - 动作对前缀的联合注意力（按行有效长度）。
- 其他：RMSNorm / adaRMS、RoPE / M-RoPE、patch embed、uint8 图像 → 归一化 patch、flow Euler 步（含 RTC）。

---

## 15. 协议、前端与客户端 SDK

- **协议**：WebSocket + msgpack，与现有服务一致。
  - 请求：元数据，加上观测（按 IOSchema 命名的多相机 JPEG 或原始图像、state、指令、可选 RTC 字段：上一 chunk、`inference_delay`）。
  - 响应：动作 chunk、动作空间描述、分段计时。
- **兼容适配（M3，可选）**：LeRobot 异步推理。
- **离线 API**：`xinfer.Policy(model, checkpoint).infer(obs)`。它在进程内使用同一个 Engine 和同样的配置解析，用于评测（如 LIBERO）和机载集成。
- **Client SDK**（独立小包）：
  - 真实机器人触发规则：下一次发送 = max(上次发送 + 周期, 响应到达)；
  - RTC delay 估计（例如最近 10 次延迟的 P75）；
  - chunk 按采集时刻对齐；
  - 客户端响应延迟计时。

---

## 16. 可观测性与基准工具

- **每请求分段计时**：接纳等待、各阶段排队与 GPU 时间、交接、返回。
- **指标与追踪**：Prometheus 指标；Perfetto trace 导出。
- **基准工具**：
  - `xinfer bench latency`：单请求延迟。
  - `xinfer bench fleet`：虚拟机器人机队（200 Hz 控制、模型步、执行长度、触发规则、RTC），搜索单卡容量 N\*，三遍重复。
  - `xinfer tune`：扫描资源配置，生成调优表。
  - `xinfer eval`：与参考实现做逐位/容差对比，报告开环轨迹误差；`eval quant` 见 §12。
- **主延迟口径**：客户端响应延迟（D-17）。

这几项工具决定了外部能否复现"单卡多少台"，与引擎同等重要。

---

## 17. 正确性与测试

| 层级 | 内容 | 运行环境 |
|---|---|---|
| 契约测试 | 用假模型覆盖接纳、slot 生命周期、取消、组批、链式发射、错误路径；检查依赖方向（设备边界随设备抽象补上，§4）。CPU 平台用线程模拟异步 stream 与 event | CPU，每个 MR |
| 树外插件路径 | 独立的 dummy 平台包经入口点完成发现、激活与契约测试（§13.3） | CPU，每个 MR |
| 算子一致性 | 每个后端实现对照参考实现；各绑定遵守 §14.1 的执行与验证契约 | xcompute 仓库 CI（GPU） |
| PyTorch 算子适配（引入时） | `torch.library.opcheck`、实际支持的编译／导出路径，以及适配后的数值一致性（§14.1） | 主仓库 CI，按声明支持的设备执行 |
| 模型对齐 | 冻结输入、噪声和参考输出，与原始实现对比 | GPU，每个 MR（小样本） |
| 运行时一致性 | 链式 vs 非链式、B1 vs Bn、空载 vs 满载回放：输出通过精度门禁；算子确定时 B1 输出逐位相同 | GPU，夜间 |
| 容量回归 | 按（模型, 平台）维护阈值 | GPU，夜间或每周 |

---

## 18. 仓库与开发流程

- XInfer 通过 §14 中的算子契约、绑定和一致性检查接入 xcompute。
- **沿用 embodied-serving 的规范**：
  - MR 描述必须写明问题、改动、验证和限制；
  - 实验原始数据不进 Git；
  - 性能声明附口径和证据位置；
  - 默认值必须有同口径证据。
- **语言**：代码、注释和对外文档用英文；内部评审文档在公开前可以用中文。
- **CI**：每个 MR 和 `main` 都跑 CPU 检查（ruff、CPU 测试，`.gitlab-ci.yml`）；GPU（5090、Thor）计划按夜间跑，尚未接入。

### 18.1 XInfer 仓库结构

以下为计划中的完整目录结构，与现有代码的目录不同（现状见“实现状态”）。

```text
XInfer/
├── README.md
├── CONTRIBUTING.md                 # MR 规范、证据规则（沿用 embodied-serving）
├── pyproject.toml                  # 包名待定（§22）；extras: test / bench / kernels
├── .gitlab-ci.yml                  # 每个 MR 跑 CPU 检查；5090、Thor 夜间跑 GPU
├── docs/
│   ├── design.md                   # 本文档
│   ├── architecture/               # 子系统设计：scheduler、memory、ops 契约、配置解析
│   ├── models/                     # 每个模型三页：使用 / 设计 / 验证
│   ├── platforms/                  # rtx5090.md、thor.md
│   ├── guides/                     # 接入新模型、新平台、新量化方法、新 kernel
│   └── results/                    # 结果索引与简短验证报告（原始数据在外部存储）
├── xinfer/
│   ├── __init__.py                 # 公共 API：Policy、serve
│   ├── entrypoints/
│   │   ├── cli.py                  # xinfer serve | bench | tune | eval | profile
│   │   ├── serve.py                # 解析配置 → 按 Platform.device_count / bind_env 为每个设备启动 Engine，并启动 Frontend → 监督生命周期
│   │   └── policy.py               # 进程内 API：Policy(model, ckpt).infer(obs)
│   ├── config/
│   │   ├── schema.py               # 配置 dataclass（严格，未知字段报错）
│   │   ├── resolve.py              # 解析流水线：显式 > 调优表 > 推导 > 默认（§10）
│   │   ├── rules.py                # 推导规则：平台能力、模型声明、显存规划；不含厂商特例（由 Platform.derive 提供）
│   │   └── report.py               # 打印并落盘每项取值与来源
│   ├── frontend/                   # Frontend 进程，不初始化 CUDA
│   │   ├── server.py               # WebSocket 服务（SO_REUSEPORT）、会话
│   │   ├── protocol.py             # msgpack 编解码；按 IOSchema 校验
│   │   ├── worker.py               # 主循环：调用模型的 Input/OutputProcessor
│   │   ├── shm.py                  # 共享内存 slot（lease + generation）
│   │   └── channel.py              # 与 Engine 的通知通道
│   ├── engine/                     # 只依赖 models/interface.py，不认识具体模型
│   │   ├── engine.py               # 组装、预热、封存、ready 状态
│   │   ├── request.py              # 请求对象与状态机
│   │   ├── scheduler/
│   │   │   ├── scheduler.py        # 调度/发射主循环（单线程）
│   │   │   ├── admission.py        # AdmitReady/Delay/Fail、每机器人上限、有界队列
│   │   │   ├── batching.py         # 零等待组批、compat key
│   │   │   ├── chaining.py         # 下游空闲时链式发射
│   │   │   └── policy.py           # StagePolicy 协议 + fifo
│   │   ├── stage/
│   │   │   ├── executor.py         # StageExecutor：lane 集合与发射
│   │   │   ├── lane.py             # stream + 工作区 + 图池
│   │   │   ├── graphs.py           # 图捕获、封存、有界缓存
│   │   │   └── completion.py       # 完成线程
│   │   └── memory/
│   │       ├── slots.py            # slot 池与状态机
│   │       ├── handoff.py          # 按 HandoffSpec 做 gather / scatter
│   │       ├── weights.py          # 只读权重冻结
│   │       └── planner.py          # 显存规划器
│   ├── models/
│   │   ├── interface.py            # VLAModel、StageGraph、HandoffSpec、Capabilities、
│   │   │                           # IOSchema、InputProcessor、OutputProcessor
│   │   ├── registry.py             # 惰性注册（名字 → 模块:类 字符串）与发现，含插件入口
│   │   ├── processing.py           # 各模型输入处理共用：视图并发、分段 tokenizer
│   │   ├── wall_oss/
│   │   │   ├── config.py           # 从 checkpoint 配置得出结构参数
│   │   │   ├── model.py            # 阶段图与三个阶段函数（组装 layers）
│   │   │   ├── processing.py       # IOSchema、Input/OutputProcessor
│   │   │   └── weights.py          # 权重映射
│   ├── layers/                     # 跨模型复用的构件，只调用 ops
│   │   ├── linear.py               # 量化感知 Linear（委托 QuantMethod）
│   │   ├── attention.py            # 视觉双向 / 前缀 prefill / 联合注意力
│   │   ├── norm.py                 # RMSNorm、adaRMS
│   │   ├── rotary.py               # RoPE、M-RoPE
│   │   ├── mlp.py
│   │   ├── image_pack.py           # uint8 视图 → 归一化图像 / patch（设备上）
│   │   ├── vision/                 # qwen2_5_vit.py
│   │   ├── backbones/              # qwen2_5.py
│   │   └── action/                 # wallx_expert.py、flow.py（含 RTC）
│   ├── quantization/
│   │   ├── base.py                 # QuantMethod、QuantConfig、按模块策略匹配
│   │   ├── bf16.py
│   │   ├── mxfp8.py
│   │   └── nvfp4.py                # M2
│   ├── loader/
│   │   ├── safetensors.py          # 分片读取、dtype 转换
│   │   └── mapping.py              # 权重映射规则引擎（重命名、拆分、合并）
│   ├── platforms/
│   │   ├── interface.py            # Platform、PlatformCaps
│   │   ├── detect.py               # 识别 sm120 / sm110 / cpu，并发现经 xinfer.platforms 入口注册的外部平台
│   │   ├── cuda.py                 # stream、event、图、显存预算；按架构的能力表
│   │   ├── cpu.py                  # 参考平台（测试用）：线程模拟异步 stream 与 event
│   │   └── conformance.py          # 平台契约套件，厂商在自己的 CI 中运行（§13.3）
│   ├── ops/
│   │   ├── spec.py                 # 算子签名与数值契约
│   │   ├── registry.py             # 后端注册、构建时选择、选择表
│   │   ├── reference/              # torch 参考实现：gemm、attention、norm、rope、image、sampler
│   │   ├── backends/
│   │   │   └── xcompute.py         # 绑定并注册 xcompute 实现（现用 TVM-FFI）
│   │   └── conformance/            # 一致性测试套件（xcompute 仓库 CI 也运行）
│   │       ├── cases.py            # 各算子的形状与 dtype 用例，取自首批模型
│   │       └── checks.py           # 数值、确定性、图捕获检查
│   ├── observability/
│   │   ├── timing.py               # 每请求分段计时
│   │   ├── metrics.py              # Prometheus
│   │   ├── trace.py                # Perfetto 导出
│   │   └── roofline.py             # 每算子耗时与理论下限
│   ├── benchmarks/
│   │   ├── latency.py              # xinfer bench latency
│   │   ├── fleet.py                # xinfer bench fleet：虚拟机器人机队、容量搜索
│   │   ├── tune.py                 # xinfer tune：扫描并生成调优表
│   │   └── eval.py                 # xinfer eval：对齐、开环误差、量化门禁
│   └── tuned/                      # 调优表，随包发布
│       ├── wall_oss/sm120-mxfp8.json   # 取值 + 提交 + 日期 + 测量口径
│       └── wall_oss/sm110-mxfp8.json
├── client/                         # 独立小包 xinfer-client，不依赖 torch
│   ├── pyproject.toml
│   └── xinfer_client/
│       ├── client.py               # WebSocket 客户端
│       ├── trigger.py              # 触发规则：max(上次发送 + 周期, 响应到达)
│       ├── rtc.py                  # RTC delay 估计、chunk 对齐
│       └── timing.py               # 客户端响应延迟
├── tests/
│   ├── unit/                       # 无 GPU：调度、slot、组批、链式、配置解析
│   ├── contract/                   # 假模型端到端；test_layering.py 检查依赖方向
│   ├── plugin/                     # 树外风格的 dummy 平台包：经入口点完成发现、激活与契约测试（§13.3）
│   ├── models/                     # GPU：与参考实现对齐（资产按 manifest 从外部拉取）
│   ├── consistency/                # GPU：链式/非链式、B1/Bn、空载/满载 都通过精度门禁
│   └── fixtures/                   # 小型确定性 fixture，注明使用它的测试
├── examples/                       # 5090 与 Thor 机队部署、离线 Policy
├── tools/
│   ├── check_layering.py
│   └── fetch_assets.py             # 拉取 checkpoint 与参考输出并校验哈希
└── docker/
    ├── Dockerfile.sm120
    └── Dockerfile.thor             # 基于 JetPack/L4T
```

### 18.2 xcompute 仓库与接入边界

沿用现有 xcompute 仓库及其工程结构，由算子组维护；不另建算子仓库。XInfer 的接入与交付要求如下：

| 项 | 约定 |
|---|---|
| 实现 | xcompute 提供各平台的 GEMM、量化融合、注意力、norm/RoPE、图像处理等高性能算子，范围由首批模型的算子清单驱动 |
| 绑定 | 默认由 XInfer 的 `ops/backends/xcompute.py` 通过 TVM-FFI 包装并注册；按明确的 PyTorch 集成需求增加复用底层实现的 `torch.library` 适配（§14.2） |
| 验证 | xcompute 自有测试之外，在其 CI 中运行 `xinfer.ops.conformance`，交付 microbenchmark 与 roofline 表；XInfer 为新增的 `torch.library` 适配补充 §14.1、§17 的验证 |
| 版本与发布 | 按平台交付构建产物，记录库版本、哈希和支持能力；XInfer 校验兼容范围与构建指纹（§14.3） |
| 第三方依赖 | 由 xcompute 维护来源与许可记录，公开发布前完成审查 |

### 18.3 扩展方式：插件入口

XInfer 不直接 import 外部扩展，而是在启动时按 Python entry point 发现它们。vLLM 也用插件入口接入树外的模型和平台。

| 入口组 | 用途 | 例子 |
|---|---|---|
| `xinfer.ops` | 注册算子实现 | xcompute 绑定，以及其他厂商算子包 |
| `xinfer.models` | 树外模型 | 内部未公开的模型 |
| `xinfer.policies` | 调度策略 | 内部 SLO 调度器 |
| `xinfer.platforms` | 注册 Platform | 厂商 NPU 包；内置的 cuda、cpu 走同一机制 |
| `xinfer.quantization` | 注册 QuantMethod | 厂商量化格式 |

这样，内部 SLO 调度器、未公开的模型和受 NDA 约束的芯片 SDK 绑定都放在私有包里，不需要 fork 开源仓库。

**激活协议**（vLLM 与 SGLang 的做法一致）：

- **入口函数**不接收参数、可重入。平台入口在当前环境不支持时返回 `None`，支持时返回 Platform 类的全限定名字符串；发现阶段不导入厂商 SDK。
- **一次只激活一个平台。** 多个同时激活时报错，要求用 `XINFER_PLATFORM` 指定；指定后只加载该入口，不可用则启动失败。`XINFER_PLUGINS` 按名字过滤其余入口组。
- **失败策略。** 被选中平台的发行包里的入口（platform、ops、quantization）加载失败即中止启动，不回退到别的平台；其他包的入口加载失败只告警。
- **启动报告**打印激活的平台、各入口所属的包与版本，以及双方的 API 版本（§13.3）。

### 18.4 里程碑与目录的对应

M0–M3 均以 `colocated` 为 GPU 执行布局，不包含 split 执行后端或阶段间 IPC/MPS 实现；`frontend/` 的 CPU 多进程通信仍按计划实现。

| 里程碑 | 主要落在 |
|---|---|
| M0 | `platforms/`、`ops/`、`layers/`、`models/wall_oss`、引擎阶段与 slot、Policy 和基准测试入口 |
| M1 | `frontend/`（多进程）、`engine/scheduler/` 全部、`memory/planner.py`、`config/resolve.py`、`benchmarks/fleet.py`、`tune.py`、`tuned/`、`client/` |
| M2 | `quantization/nvfp4.py`、精度门禁和扩展模型所用的构件 |
| M3 | 兼容适配、Router、文档、Docker 镜像、GitHub 发布 |

### 18.5 关键文件的迁移来源

| XInfer 文件 | 来源 |
|---|---|
| `engine/memory/slots.py` | embodied-serving `kv_cache/batch_manager.py` 的状态机 |
| `engine/scheduler/admission.py` | embodied-serving `global_scheduler.py` 的通用接纳部分；vla-engine 的 Admit 三态 |
| `engine/scheduler/batching.py` | embodied-serving `multiprocess/batching.py`（`BatchIssuer` 规则） |
| `frontend/shm.py`、`worker.py` | embodied-serving 的共享内存传输与前端工作进程 |
| `models/wall_oss/` | HarriX `mxfp8-sm120` / `mxfp8-thor`，拆为共享 layers 与模型文件 |
| `ops/backends/xcompute.py` | embodied-serving `adapters/wallx/mxfp8_ops.py`、HarriX 各 track 的 `ops.py` |
| `config/schema.py` | HarriX `launch_config.py` 的严格加载方式 |

---

## 19. 与现有代码的关系

**选择新建仓库，不在现有仓库上演进：**

- embodied-serving：adapters 中有移植来的私有代码，SLO 代码与通用调度交织在一起。
- HarriX：依赖未在 setup.py 中声明的 `wall_x` 和内部 xcompute 构建，内核按"模型×GPU"逐个移植。

### 迁移来源

| 来源 | 迁移内容 |
|---|---|
| embodied-serving | `global_scheduler` 中的通用部分（接纳、队列链、lane 上限、slot 分配与回收、失败排空）；slot 状态机；`BatchIssuer` 规则；frontend 进程池与共享内存；cluster 层 |
| HarriX | Executor/Engine 分离（prepare → engine → decode）；构建时能力校验；预热后封存图；严格的配置加载；`mxfp8-sm120` / `mxfp8-thor` 的 4B 实现 |
| vla-engine | AdmitReady/Delay/Fail；slot handle generation；预热完成前 `/ready` 返回 503；分层精度阈值配置 |

迁移的是逻辑阶段、组批和所有权规则；本版不迁移 GPU 阶段的 ActionProcess、CUDA IPC 传输及私有 MPS 启停实现。CPU Frontend 进程池与共享内存按原计划保留。

### 迁移时不带过来的耦合

- RTC delay 范围硬编码（`3 <= delay <= 8`）；
- 假设固定的 handle 结构；
- Wall-X 命名的线程和环境变量；
- 硬编码的 5.6 Hz；
- 热路径里导入消融门控；
- 相机名和双臂写死在通用类型里（HarriX）。

---

## 20. 里程碑与验收

| 里程碑 | 内容 | 验收 |
|---|---|---|
| M-1 评审与建仓 | 主仓库 kc/XInfer 已创建（私有），算子仓库确定为 xcompute；本文档评审及算子契约待完成；CPU CI 已接入 | 仓库安排已确定；本里程碑其余事项完成后记录评审与验收结论 |
| M0 数据面 | 平台、算子后端、模型层、Wall-OSS、单通道发射和基准测试 | Wall-OSS 通过精度门禁，并记录目标平台的时延与图发射开销。 |
| M1 机队运行时 | 多通道发射、零等待组批、前端工作进程、slot 生命周期与配置解析 | 测量负载下的机队容量、组批精度和发射开销。 |
| M2 量化与扩展 | NVFP4 与模型扩展 | 记录各模型及量化方法的数值结果。 |
| M3 发布 | 文档、Client SDK、兼容适配、Router；许可审批；GitHub 发布 | 外部用户按文档能复现单请求延迟与单卡容量 |

---

## 21. 主要风险

| 风险 | 影响 | 应对 |
|---|---|---|
| 链式发射在临界负载附近降低容量 | 统一模型的前提 | M1 在 colocated 多 lane 路径测量；必要时调整链式发射条件，保持单一路径 |
| 单进程发射成为瓶颈（Python 开销、GIL） | 机队容量 | M1 测发射线程占用、锁等待与提交延迟；优先消除热路径阻塞和 CPU 重活，必要时评估下沉提交逻辑。未达容量门槛则继续优化并记录限制，不以尚未实现的 split 作为兜底；拆进程需另行评审 |
| Thor 的图、统一内存与 CPU 提交开销和 5090 不同 | Thor 性能 | 按平台能力推导配置；Thor 单独做 M0 测量。Thor 也按机队部署，M1 在 Thor 上单独测容量和 Frontend 的 CPU 占用。v1 不依赖 MPS，MPS 能力确认不阻塞本版 |

---

## 22. 待决策

1. **对外名称冲突**：PyPI 上 `xinfer` 已被 dnth/xinfer（计算机视觉推理框架）占用，Xorbits 的 Xinference 名字也相近。仓库名仍为 XInfer，pip 包名在公开前确定。
2. **模型权重**：对外分发 checkpoint 文件须单独取得授权。
3. **PyTorch 算子适配范围与版本**：默认 TVM-FFI、按需增加 `torch.library` 的策略已确定（§0、§14.2）；与算子组确认首批需要适配的算子、实际支持的编译／导出路径及 PyTorch、TVM-FFI、xcompute 版本范围。
4. **许可证**：在 M3 前审批。
5. **对外稳定面与支持等级**：API 版本号的起点与破坏性变更规则，Tier 2 的准入标准（§13.3）。
6. **AMD GPU 的接入时机与型号**：是否进入 v1；型号决定可用的量化格式和 kernel 来源（§13.2）。

---

## 附录 A：证据索引

均为 RTX 5090；"T-xxx" 为 embodied-serving `papers/fleetserve/experiments/` 中的记录。

| 数据 | 出处 | 说明 |
|---|---|---|
| 4B 整请求并发饱和 39.9 vs FIFO 42.8 req/s | G1c（T-37） | |
| ViT 第 2 份 +25%，第 3 份无增益 | T-76 | |
| SM 预留无端到端收益 | SM_SHARING_STUDY_20261013 | 旧 4B |
