# 服务

[English](serving.md) | 简体中文

`python -m xinfer.entrypoints.serve`（或 `xinfer-serve`）在一块 GPU 上通过 WebSocket + msgpack 服务一个模型，与 x2robot 桌面客户端兼容。

## 进程布局

```text
robot ──WebSocket + msgpack──▶ frontend process ×N (CPU; SO_REUSEPORT on one port), a processing thread:
                                 decode msgpack → InputProcessor
                                   ("gpu": only the JPEG headers are read; each view is a DeviceImage)
                                 lease a shared-memory buffer; write the tensors and the DeviceImage data
                                 │ manifest over a pipe
                                 ▼
engine process (GPU)
  bridge thread     one per frontend: map the buffer (no copy) → Engine.submit → waiting queue
  engine thread     admission: FIFO, a free slot; upload the tensors, record inputs_ready
                      │ no DeviceImage           │ DeviceImage values
                      │                          ▼
                      │                    device-image thread: decode and resize on its own
                      │                    stream; its event becomes inputs_ready
                      ▼                          │
                    first-stage queues ◀─────────┘ (handed back through the engine thread)
  lanes             wait for inputs_ready on the GPU → vision → … → final stage
  completer thread  final event complete → future resolved → outputs over the pipe
                                 │
                                 ▼
robot ◀──WebSocket── frontend: release the buffer → OutputProcessor → response
```

准入、组批和各阶段见 [engine.md](engine.zh-CN.md)，设备图像线程见[设备图像](engine.zh-CN.md#设备图像)。

- **前端**从不初始化 CUDA。每个前端拥有一个共享内存缓冲区环；缓冲区在其响应返回后复用。每个前端在 `--frontend-workers` 个线程（默认 4）上处理请求，因此不同连接的请求同时处理；单个连接仍然一次只发送一个请求。
- **引擎进程。** 加载模型，运行引擎（见 [engine.md](engine.zh-CN.md)），映射这些缓冲区并将其注册为锁页（pinned）主机内存，使上传可直接读取它们。在 `image_preprocess_device="gpu"` 时，前端只读取 JPEG 文件头；`DeviceImage` 值中的 JPEG 字节和原始帧经缓冲区传递，其大小写在清单（manifest）中，引擎直接从缓冲区解码 JPEG（[配置](configuration.zh-CN.md#在-gpu-上做图像预处理)）。
- **启动。** 前端在引擎完成预热之后才开始监听，因此端口打开即表示已就绪。使用 `--warmup` 时，每个前端先让预热请求经过模型的 InputProcessor，并用引擎对这些请求的输出运行 OutputProcessor，这样最先接入的机器人不必承担首次调用的开销（如 numba 编译）。这些调用使用专为预热构建的处理器；服务用的处理器不会从中继承任何状态。

## 协议

- 建立连接时，服务端发送模型的元数据帧。Wall-OSS 的为：`{"batch_enabled": true, "model": "wall-x", "action_horizon": 16, "action_dim": 26, "rtc_ee_chunk_dim": 14, "env": "x2robot"}`。
- 每个请求是一条 msgpack 消息，数组采用 msgpack-numpy 布局。同一连接的请求逐个应答，由该连接自己的输入处理器处理：有按连接状态的模型在连接存续期间保留这些状态。响应包含模型的字段，以及 `status`（`"ok"`）和 `server_timing`：
  - `infer_ms`：从请求解码完成到响应编码完成；
  - `prepare_ms`、`engine_ms`、`output_ms`；
  - `frontend`：处理该请求的前端进程。
- 请求可以在顶层字段 `robot_id` 中携带机器人 id，与 `state`、`views` 同级。这时引擎按机器人 id 跨连接限制在途请求数（`--per-robot-inflight`）；没有时按连接限制。
- 失败的请求以 `{"status": ..., "error": "<message>"}` 应答，不带 `server_timing`；连接保持打开。机器人需要据此处理的协议错误还带有 `error_info`：

  | `status` | 含义 |
  |---|---|
  | `invalid_request` | 消息无法解码，或被模型的输入处理拒绝 |
  | `queue_full` | 引擎的准入队列已满（`--max-queued-requests`）；请稍后重试 |
  | `per_robot_limit` | 该机器人已有 `--per-robot-inflight` 个请求在途 |
  | `unavailable` | 服务正在关闭 |
  | `timeout` | 引擎没有在 `--request-timeout` 内完成该请求 |
  | `internal_error` | 服务在处理该请求时出错（引擎或输出处理） |

  凡 `status` 不是 `"ok"` 的响应都应计为失败。
- 同一端口上的 `GET /healthz`（或 `/ready`）返回 200。
- `GET /info` 以 JSON 返回所服务的模型：`model`、`quantization`、`numerics`、`platform`、`version`、`model_options`、`capacity`（`--capacity`，或 null）、`metadata_sha256`（元数据帧的哈希）和 `connection_state`（模型的输入处理器是否保存按连接的状态）。
- 同一端口上的 `GET /metrics` 返回引擎和所有前端的 Prometheus 指标（[指标](metrics.zh-CN.md)）。

## 运行

```bash
python -m xinfer.entrypoints.serve --model wall_oss --checkpoint /path/to/checkpoint \
  --port 33985 --warmup flow.msgpack rtc.msgpack \
  --model-options '{"quantization": "bf16", "processing": {"instruction_override": null}}'
```

| 选项 | 含义 |
|---|---|
| `--model-options` | 传给 `from_checkpoint` 和 `build_processors` 的 JSON（Wall-OSS：`processing` = `ProcessingConfig` 的字段） |
| `--frontends`, `--frontend-workers`, `--shm-buffers`, `--shm-buffer-mib` | 前端进程数（默认 2）、每个前端的处理线程数（默认 4）和每个前端的共享内存缓冲区数；缓冲区大小 |
| `--lanes`, `--max-batch`, `--max-running-requests`, `--max-queued-requests`, `--per-robot-inflight` | 引擎设置（[engine.md](engine.zh-CN.md)）；会经过检查，从不改写 |
| `--enforce-eager` | 以 eager 模式运行所有阶段，不使用 CUDA graph（默认：平台支持捕获时使用 CUDA graph）；`--warmup` 请求文件（JSON 或 msgpack，每个文件一个请求或一个列表）：在开始服务之前，按模型从中推导出的每个形状桶（一个 flow 请求和一个 RTC 请求即可覆盖 Wall-OSS），在每个通道和每个允许的批大小上捕获它们的 graph，并由每个前端让它们经过模型的处理器 |
| `--tuned` | 调优表选择：`auto`、`none` 或路径 |
| `--resolved-config`, `--dry-run` | 写出解析后的设置及其来源；打印这些设置后退出 |
| `--profile-dir` | 记录引擎的 span，停止时把每个请求的引擎分段写到该目录；用于剖析运行（`xinfer-profile load --serve`，[剖析](profiling.zh-CN.md#经过-xinfer-serve--serve)），记录随请求数增长 |
| `--disable-metrics` | 不记录 Prometheus 指标；`GET /metrics` 返回 404（[指标](metrics.zh-CN.md)） |
| `--hang-timeout` | 已准入的请求最多可用的秒数，超过则认为引擎挂起（默认 30）；0 关闭该检查 |
| `--request-timeout` | 引擎处理一个请求最多可用的秒数，超过则机器人收到 `timeout`（默认 10；0 表示不限制）；引擎仍会把该请求处理完 |
| `--capacity` | 本服务服务的机器人数，即按其 GPU 和模型配置实测的容量（[性能](performance.zh-CN.md)）；由 `GET /info` 报告 |
| `--drain-timeout` | 收到 SIGTERM 后前端回复已收到请求最多可用的秒数（默认 10） |
| `--reference-ops`, `--op-backends` | 只使用 torch 参考算子；为每个算子按后端、库或两者固定实现（`op=cuda`、`op=xcompute`、`op=cuda:xcompute`、`op=reference`），固定的实现优先于 `--reference-ops` |

SIGTERM 让服务排空：前端停止接受连接（包括健康检查），对之后的消息返回 `unavailable`，回复已收到的每个请求后以代码 1001（going away）关闭连接；此后才到达前端的消息得不到响应。然后服务释放 graph 和锁页缓冲区并以 0 退出，若这些请求在 `--drain-timeout` 秒内没有回复完则以 1 退出。SIGINT 的处理相同，但终端里的 Ctrl-C 也会发给前端，使它们立即停止。服务进程不在了的前端（例如被 OOM killer 杀掉）也以同样方式排空并退出。服务每 0.5 s 检查一次 `Engine.health()`。当引擎无法继续服务（引擎线程或完成线程已停止、设备报告 CUDA 粘滞错误，或者有请求准入超过 `--hang-timeout` 秒仍未完成）时，服务停止前端并以 1 退出，不等待设备，以便其监管进程重启它。挂起的 kernel 在服务退出后仍会继续运行（GPU 会释放它的显存），因此重启后的服务与它共用 GPU，直到它结束或 GPU 被重置。

## 压测

`benchmarks/robot_load.py` 以闭环方式让虚拟机器人向服务发送请求。见 [benchmark.md](benchmark.zh-CN.md)。
