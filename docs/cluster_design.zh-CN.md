# 集群服务设计

[English](cluster_design.md) | 简体中文

本页设计如何用一台或多台机器上的多张 GPU 服务同一个机器人机队，包括：
- 单卡 worker 前面的 router；
- 故障转移、worker 的加入和退出、健康检查；
- 可插拔的路由策略。

这是一份待评审的设计，worker（§4）、router（§5）和 Kubernetes 示例部署文件（§6）已经实现，进度在[设计](design.zh-CN.md)的状态表里跟踪。它是设计 §2.2（"多卡用数据并行副本加 Router"）和 §4 中可选 Router 的展开。

## 1. 目标与范围

| 目标 | 标准 |
|---|---|
| 用多张 GPU 服务机队 | k 个 worker 组成的池，容量至少为单卡容量的 0.95 × k 倍 |
| 故障时继续服务 | worker、其 GPU 或其所在机器故障时，每台机器人至多丢一个响应（有状态的连接重新开始一次，§4.5），且没有机器人连接被断开 |
| 加卡、减卡不需要重启 | worker 可以加入和退出正在运行的池；排空后退出的 worker 不产生失败的响应 |
| 健康检查 | 挂掉、挂起或正在排空的 worker 不会收到请求 |
| 可插拔的路由策略 | 内置若干策略，更多策略通过 entry point 注册 |
| 单卡路径不变 | 单个 worker 的单请求延迟和容量不变 |

不在范围内：
- 张量并行；
- 不同阶段放在不同 GPU 上（设计 §9.3）；
- 在一个进程组里跑多张卡的 `xinfer-serve`。即使一台机器有多张卡，一个 worker 也只对应一张卡，单机和多机都由 router 服务；
- 核心中的 SLO 调度器：它们可以作为路由策略插件实现；
- 自建进程托管或服务注册：由 Kubernetes 提供（§6）。

## 2. 设计依据的负载特点

| 特点 | 结论 |
|---|---|
| engine 不在请求之间保存状态（没有跨请求的 KV cache） | 无状态池中任何 worker 都能服务任何请求，请求也可以换 worker 重试 |
| 模型可以在前端的 InputProcessor 里保存按连接的状态（`connect`，设计 §11），例如实时分块（RTC）plan；内置模型都没有 | 这类模型的请求在整个连接期间必须留在同一个 worker 上（亲和） |
| 机器人闭环发送，每台只有一个请求在途；单卡服务 4–15 台机器人（[性能](performance.zh-CN.md)） | 负载按机器人数和在途请求数计算；准入按池限制机器人数 |
| 请求体大（三路相机画面）且对延迟敏感 | router 转发消息时不解码；它增加的延迟是验收标准之一（§8） |

## 3. 架构

```text
 机器人 ──WebSocket + msgpack──▶ xinfer-router ×2（无状态；每个池一个监听端口）
                                   │ 逐请求（无状态池）或按连接（有状态）
                ┌──────────────────┼──────────────────┐
                ▼                  ▼                  ▼
          worker：一张 GPU 上的 xinfer-serve（不变的单卡服务：前端 + engine）
          GET /healthz /ready /info /metrics；SIGTERM 触发排空
 服务发现：Kubernetes headless Service（DNS 只返回就绪的 worker）或静态列表
```

| 术语 | 含义 |
|---|---|
| worker | 一张 GPU 上的一个 `xinfer-serve` 进程组：它的前端和 engine |
| 池 | 服务同一模型配置的一组 worker；同一个池的机器人连接 router 的同一个端口，池内每个 worker 必须发送相同的元数据帧 |
| router | `xinfer-router`：接住机器人的连接，把每个请求转发给池中的一个 worker |

## 4. worker 的改动

### 4.1 接口

| 接口 | 返回 |
|---|---|
| `GET /healthz`、`GET /ready` | worker 在监听时返回 200。engine 无法继续服务的 worker 会退出（§4.3），排空中的 worker 停止监听（§4.2），所以连接被拒就说明 worker 未就绪 |
| `GET /info` | JSON：模型、模型选项、量化、数值组合（numerics）、平台、XInfer 版本、元数据帧的哈希、`connection_state`（模型的 InputProcessor 是否保存按连接的状态）、`capacity`（机器人数，来自 `--capacity`：按 GPU 型号和模型配置实测的单卡容量，见[性能](performance.zh-CN.md)；未设置时为 null） |
| `GET /metrics` | 不变（[指标](metrics.zh-CN.md)） |

### 4.2 排空

SIGTERM 使 worker 排空（[服务](serving.zh-CN.md)）：
- 它的前端停止监听，之后收到的请求回答 `unavailable`，由 router 换一个 worker 重试（§5.5）；
- 前端回答完已收到的请求，然后以代码 1001 关闭连接；
- 这些请求都回答完后 worker 退出，最长等 `--drain-timeout`。

在 Kubernetes 上，`terminationGracePeriodSeconds` 必须大于 `--drain-timeout`。

### 4.3 看门狗与请求超时

- **engine 看门狗**即带 `--hang-timeout` 的 `Engine.health`（[服务](serving.zh-CN.md)）：有请求准入超过该时限仍未完成，或者设备报告粘滞错误时，worker 以状态 1 退出，与 engine 线程停止时相同。
  - 挂起的 kernel 可能在进程退出后仍在运行。
  - GPU 的恢复交给平台（§6）。
- **请求超时。** engine 在 `--request-timeout` 内没有完成的请求，前端以状态 `timeout` 回答，便于 router 重试。

### 4.4 机器人身份

请求在顶层字段 `robot_id` 中携带机器人 id，与 `state`、`views` 同级。worker 用它跨连接按机器人限制在途请求；没有这个字段的请求仍按连接限制。

### 4.5 故障转移后有状态的连接

按连接保存状态的模型（例如它返回过的实时分块 plan），在机器人的连接转到另一个 worker 时会丢失这些状态。新 worker 按新连接应答，由模型的协议告诉机器人重新开始（例如用普通请求重新锚定）。要避免这一点，需要让状态跟着机器人走，并修改客户端协议。

## 5. router

### 5.1 连接与转发

机器人连接 router 的方式与现在连接服务端相同：协议不变，只是把服务端地址换成 router 的地址。一个 router 进程在一个端口上服务一个池（[router](router.zh-CN.md)）。

router 接住机器人的 WebSocket：
- 它发送池的元数据帧，取自第一个 worker。只有 `/info` 中元数据哈希相同的 worker 才能加入池。
- 它原样转发每条请求消息，再把 worker 的回答发回。它只读取顶层字段，不解码其他字段：回答的 `status`，以及亲和池中每条连接第一个请求的 `robot_id`（§4.4）。
- 每条机器人连接对它用到的每个 worker 各有一条自己的连接，每条连接同时只承载一个请求。

### 5.2 路由

| 池 | 规则 |
|---|---|
| `connection_state` 为 false（Wall-OSS、Wall-X 2B） | 逐请求：router 发给它的在途请求数（§5.7）相对其 `capacity` 最少的 worker；相同时轮转 |
| `connection_state` 为 true | 亲和：机器人连接在第一个请求时，按 `robot_id` 在未满容量的 worker 中做 rendezvous 哈希放置（没有 id 时选相对容量机器人数最少者），并一直留在那里；该 worker 故障时重新放置这条连接，机器人在新连接上重新开始（§4.5） |

### 5.3 服务发现

- **来源。** 池的 worker 来自静态列表（`host:port`，用于开发和没有 Kubernetes 的机器），或来自每秒解析一次的 DNS 名称（`dns:<name>:<port>`）。Kubernetes headless Service 只返回就绪的 Pod。
- **加入与离开。** 新地址的 `/info` 返回池的元数据哈希后即加入；不一致的 worker 不加入，并记录日志。DNS 不再返回的地址离开池。

### 5.4 健康检查

router 被动检查 worker：
- 连接错误、上游连接关闭，或者回答的状态属于可重试状态（§5.5），都记为失败；一次成功清零。
- 连续 3 次失败后，该 worker 5 s 内不再收到请求。之后下一个请求作为试探，再失败一次就再暂停。
- 所有 worker 都被暂停时，router 把请求发给它们全部，而不是拒绝所有机器人。

在 Kubernetes 上，就绪探针会把不回答 `/ready` 的 worker 从 DNS 中去掉。

### 5.5 重试

只有同时满足以下条件，请求才会换一个 worker 重试。对亲和池，这个 worker 就是连接新放置到的 worker（§5.2）：
- 机器人还没有收到这个请求的任何回复；
- 池中还有没为这个请求试过的 worker：每个 worker 最多试一次，所以几个 worker 同时排空时请求仍能得到服务；
- 收到请求不到 `--retry-deadline-ms`（默认 30 ms）。

否则 router 立即把失败返回给机器人，由机器人发送新的请求。在 RTC 场景下机器人等待期间仍在异步执行，延迟太高的动作已经没有用了。

| 重试 | 不重试 |
|---|---|
| 连接失败、上游连接关闭、`timeout`、`queue_full`、`unavailable` | `invalid_request`、模型协议错误（`error_info`）、`internal_error` |

### 5.6 准入

每个 worker 都报告 `capacity` 时，池的容量是它们之和。超出容量的新机器人连接在握手阶段以 HTTP 503 拒绝；还没有 worker 加入时，连接同样被拒绝。

### 5.7 负载信号

router 统计自己发给每个 worker 的在途请求数，从收到请求到 worker 回答为止，因此包含预处理。只有一个 router 服务这个池时，这个数是精确的。有多个 router 时，每个 router 只看到自己那部分流量；worker 不另外上报负载。

### 5.8 可用性与数据面

- **router 故障。** router 无状态，以两个或更多副本运行在同一个地址后面，亲和哈希在每个副本中都相同。router 故障时，它的机器人重新连接这个地址。
- **数据面。** router 用 Python asyncio 实现。如果它增加的延迟达不到 §8 的标准（例如请求是原始相机帧），数据面改用编译型语言实现。
- **需要时再加：** 其他路由策略（例如 router 很多时的二选一）、router 指标、主动健康检查，以及立即回答 `overloaded`。

## 6. Kubernetes 部署

| 对象 | 设置 |
|---|---|
| worker Deployment（每个池一个） | 每个 Pod `nvidia.com/gpu: 1`；按前端需要设置 CPU 请求；`startupProbe` 访问 `/ready`，时长足够完成预热；`readinessProbe` 每秒访问 `/ready`；`livenessProbe` 访问 `/healthz`；`terminationGracePeriodSeconds` 大于排空时间（§4.2）；`minReadySeconds` 大于 router 经 DNS 发现新 worker 的时间（几秒），滚动升级时等 router 发现了新 worker 再排空下一个旧 worker；checkpoint 放在只读卷上；`hostNetwork` 可选，用来避开 overlay 网络的延迟 |
| headless Service（每个池一个） | 选中池的 worker；`publishNotReadyAddresses: false`；它的名称就是 router 服务发现用的名称 |
| router Deployment 和 Service | 两个副本；机器人连接的是这个 Service 的地址，按现在暴露服务端地址的方式暴露 |
| PodDisruptionBudget | 同一个池同时至多中断一个 worker |
| GPU 健康 | GPU Operator 的 DCGM 健康检查把故障 GPU 标为不可调度，重启的 worker 不会再落到它上面 |

部署文件和镜像作为示例提供（[router](router.zh-CN.md#kubernetes)）。在 Kubernetes 之外，同样的 router 和 worker 用静态列表运行。

| 操作 | 过程 |
|---|---|
| 加卡 | 增加 worker 副本数；新 Pod 预热完成后就绪，经服务发现加入 |
| 摘除一个 worker | Pod 收到 SIGTERM 后排空（§4.2） |
| 滚动升级 | Deployment 逐个替换 worker（没有空闲 GPU 时 `maxSurge: 0`），每个旧 worker 先排空；新 worker 就绪满 `minReadySeconds` 后才替换下一个 |
| worker 崩溃或 GPU 挂起 | 存活探针失败，Kubernetes 重启 Pod；router 的被动检查在几次失败的请求之内停止向它路由 |
| 机器故障 | 其上的 Pod 被重新调度；在此期间 router 的检查把它们摘除 |

## 7. 故障处理

| 故障 | 发现方式 | 机器人看到的 | 恢复 |
|---|---|---|---|
| worker 进程退出 | 上游连接关闭 | 请求被重试，至多一个迟到或失败的响应；有状态的连接在新 worker 上重新开始（§4.5） | Kubernetes 重启 Pod |
| GPU 挂起 | 看门狗（§4.3）、请求超时 | 同上 | 看门狗使 worker 退出；Pod 重启；故障 GPU 被标为不可调度 |
| worker 过载 | 在途请求数、`queue_full` | 请求被发给其他 worker | — |
| 机器故障 | 主动和被动检查 | 与其上 worker 退出相同 | Pod 被重新调度 |
| router 故障 | 机器人连接断开 | 机器人重新连接 Service 地址 | Kubernetes 重启 router |
| 池已满 | 准入 | 新机器人在握手时被拒绝 | 增加 worker |

## 8. 验证与验收

C1、C2 阶段在一台有三张 RTX 5090 的机器上用静态列表测试。C3 阶段在单节点 k3s 集群上测试，两个 worker Pod 共用一张 RTX 5090（NVIDIA device plugin 的 time-slicing）：只检验机制，不测容量。

| 标准 | 测量 |
|---|---|
| 单卡路径不变 | 单个 worker 的单请求延迟和容量，与 main 对比，在轮间波动以内 |
| router 延迟 | 容量负载下、三路相机 JPEG 请求，有无 router 的客户端响应延迟：P50 至多增加 1 ms |
| 扩展 | 按仓库的压测口径，k 个 worker 的容量至少为单卡容量的 0.95 × k 倍 |
| worker 故障 | 在负载下杀掉一个 worker：每台机器人至多丢一个响应，且没有机器人连接被断开；报告 router 停止向它路由所用的时间 |
| 排空与滚动升级 | 没有失败的响应 |
| router 逻辑 | 用桩 worker 的 CPU 单元测试：路由、重试、熔断、服务发现、准入 |

## 9. 阶段

| 阶段 | 内容 |
|---|---|
| C1 worker | §4：`/info`、请求超时、机器人身份 |
| C2 router | §5：转发、两种路由模式、服务发现（静态、DNS）、被动健康检查、重试、准入 |
| C3 Kubernetes | §6：示例部署文件、探针、滚动时的排空、GPU 健康；在 GPU 集群上验证 |
| C4 | 按需：DNS 太慢时改用 `EndpointSlice` 服务发现、编译型数据面、§5.8 中需要时再加的部分、SLO 策略插件 |

## 10. 相关系统

依据 vLLM `9b38c83` 和 SGLang `62eab14` 的源码：

| | vLLM 数据并行 | SGLang router（`sgl-router`、`sgl-model-gateway`） | 本设计 |
|---|---|---|---|
| 拓扑 | 每个前端连接每个 engine；engine 负载经协调进程送到前端（变化时推送，最快每 100 ms 一次） | 独立的 Rust 进程，位于各自独立的 worker 前面 | 独立的 router，位于各自独立的单卡 worker 前面 |
| engine 故障 | 整个服务退出 | 熔断；`sgl-model-gateway` 另外每 60 s 检查一次 `/health` | 该 worker 连续失败 3 次后被暂停，其请求被重试 |
| 负载均衡 | 等待加运行数最少者，每次选择后在本地调整 | 二选一（power of two choices）、按缓存感知等 | §5.2 |
| 成员管理 | 规模固定 | 静态列表、Kubernetes；gateway 另有 workers 接口 | 静态列表、DNS（Kubernetes headless Service） |
| 排空 | `--shutdown-timeout` | 就绪返回 503，经过排空窗口，再等在途请求 | 就绪返回 503，经过排空窗口，再等在途请求 |
