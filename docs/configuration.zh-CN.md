# 配置解析

[English](configuration.md) | 简体中文

`xinfer-serve` 和 `Policy` 不使用具名配置档（docs/design.md §10），也不接受负载参数。一个实例服务的机器人数量会随时间变化，引擎会根据到达的请求自行适应：
- 批只由已在队列中的请求组成（零等待组批）；
- 没有工作的通道（lane）保持空闲；
- 只为实际出现的键和批大小捕获 CUDA graph。

因此每项设置都是适用于任何负载的固定选择或上限。它在启动时解析一次，并连同其来源一起报告。优先级从高到低：

1. **显式（explicit）**：运维人员设置的命令行参数或函数参数。它会对照平台和模型进行检查，但从不被改写。冲突即报错：模型未声明的批大小、未加 `--allow-non-invariant-batching` 却对算子非批次不变的阶段组批、在不支持捕获的平台上启用 CUDA graph（仅限 API），或未知的阶段。
2. **调优表（tuned table）**：针对（模型、架构、量化）组合的实测值，来自 `xinfer/tuned/<model>/<arch>-<quantization>.json` 或 `--tuned <path>`。每张表记录提交、日期、方法和证据；来源显示为 `tuned:<table>@<commit>`。表可以设置 `lanes`。目前没有随附任何表：只有当实测值与默认值不同时才添加表。
3. **模型 / 平台规则**：模型声明的内容（批大小、所选算子下的批次不变性）和平台支持的能力（graph 捕获、CPU 配额）；来源为 `model:<rule>` 和 `platform:<rule>`。
4. **默认（default）**：固定值。

代码：`xinfer/config/`（`resolve.py`、`tuned.py`）。

## 设置与规则

| 设置 | 无显式值时 | 说明 |
|---|---|---|
| `per_robot_inflight` | 1 | 每个机器人未完成的请求数 |
| `max_running_requests` | 32 | 在途的已准入请求数。这是一个上限：槽位不预分配内存，且每个机器人只有一个请求时，只有 32 台机器人才会达到它 |
| `max_queued_requests` | 256 | 已提交、等待准入的请求数 |
| `lanes.<stage>` | 调优值，否则为 2 | 一个通道是一条带有自己 graph 池的 stream；第二个通道让两个请求可以同时运行该阶段 |
| `stream_priorities.<stage>` | CUDA 上最终阶段为 −1（Wall-OSS：`action`），其他为 0 | 该阶段各通道的 CUDA stream 优先级，越小越高（`--stream-priorities action=0`、`Policy(stream_priorities={...})`） |
| `batch_sizes.<stage>` | 若该阶段在所选算子下批次不变，则为模型声明的所有大小；否则为 `[1]` | 允许更大的批大小从不会延迟请求，因为批只由已排队的请求组成 |
| `cudagraphs` | 平台支持捕获时开启 | |
| `frontends` | 2，最多为容器配额 CPU 数的一半 | 每个前端会在引擎进程中增加一个桥接线程 |
| `frontend_workers` | 4，且 `frontends × frontend_workers` 不超过配额 CPU 数的一半 | 每个前端的处理线程数；不同连接的请求同时处理 |
| `shm_buffers` | 每个前端 `ceil(max_running_requests / frontends)` 个 | 每个运行中的请求一个缓冲区；缓冲区全部被租用的前端会等待。默认：2 × 16 个 `--shm-buffer-mib`（32 MiB）缓冲区，锁页 |

stream 优先级只是对就绪 GPU 工作的提示：它不会抢占正在运行的 kernel，也不会绕过事件等待，因此它只是把延迟在阶段之间转移，而不是消除延迟。未知阶段、非整数、设备无法表示的值，以及在不支持 stream 优先级的平台上设置非零值，都会报错（`--dry-run` 解析这些值时不创建通道，因此不检查设备支持的范围）。

为什么取这些值：
- **通道数 2。** 在 Wall-OSS MXFP8（RTX 5090）上，两个 action 通道的容量为 7 台，一个通道为 5 台（action 阶段组批）；不组批时 2/2/2 优于 1/1/1（饱和时 31.2 对 27.2 req/s）；三个及以上通道未测量。证据：`xinfer-action-batching-20261101.tar.gz` 和 `xinfer-flash-attn-20261101.tar.gz`（哈希见 docs/models/wall_oss_validation.md）。
- **两个前端。** 对 x2robot dummy 客户端的请求（一路 720×1280 原始视图和两路 640×480 JPEG 视图；三路原始视图时为 1.7 ms），Wall-X 输入处理每个请求耗时 7.1 ms，因此 8 台机器人以 5.6 Hz 发送时，一个前端约有三分之一的时间处于忙碌（`xinfer-wallx-input-processing-v3-20261105.tar.gz`，sha256 `04df60bd…`）。下面的对比是在输入处理每个请求耗时 22.2 ms 时测得的，当时 8 台机器人下一个进程约 100 % 的时间处于忙碌。低负载时更多前端没有帮助：4 台机器人时（2026-11-01，action 阶段组批之前；各跑一次），服务端 P50 / P99 在 1 个前端时为 94.8 / 123.8 ms，2 个时为 95.9 / 133.0 ms，4 个时为 119.9 / 144.0 ms。启用 action 组批后，4 台机器人在 1 个和 2 个前端下端到端时延相同（汇总 P50 分别为 85–87 和 86–88 ms）；1 个前端时请求排队等待输入处理（P50 28 ms 对 21 ms），2 个前端时则改为在引擎中等待（`xinfer-config-defaults-20261103.tar.gz`，sha256 `6f85d5eb…`）。
- **每个前端 4 个处理线程。** 图像解码和 resize 会释放 GIL，因此一个进程可以同时准备多个请求。
- **最终阶段优先。** 优先级相同时，最终阶段的就绪工作要排在后续请求的 vision 和 prefill kernel 之后。在 RTX 5090 上（仓库压测口径，同一代码树上的成对实例），action 优先级 −1 对比 0：
  - Wall-OSS MXFP8，8 台机器人：服务端汇总 P50 为 120.9 / 123.4 对 137.1 / 142.4 ms，P99 为 167.5 / 162.3 对 182.2 / 203.4 ms；
  - 1 台和 4 台机器人（Wall-OSS MXFP8）时没有可测量的差异。

  达到或超过饱和时，它可能带来代价：
  - Wall-OSS BF16，5 台机器人（过载）：中位数低 42 ms、吞吐高 15 %，但 P99 高 24–37 ms；
  - Wall-OSS MXFP8，10 台机器人（过载）：中位数低 13 ms、P95 更低、吞吐高 10 %，但 150 ms 以内的请求更少（2–11 % 对 28–32 %）。

  `--stream-priorities action=0` 可恢复相同优先级。证据：`xinfer-stream-priorities-20261104.tar.gz`（sha256 `b642823fb11b337e6dd5f49ec35108c1e7fafb529c6f96a283704a7de5ff2ec7`）。

## Prompt 形状桶

Wall-OSS 使用 `xinfer/models/interface.py` 中的 `BucketSpec`。启动报告中的 `shape_buckets.prompt` 显示生效的策略。默认值为 `processing.prompt_bucket=16`。

| 模型选项 | 实际策略 |
|---|---|
| 正数 `prompt_bucket` | 向上取整到该粒度，上限为模型的 prompt 长度上限 |
| `prompt_bucket=0` | 精确的有效长度，不做桶填充 |

Wall-OSS 加载时会检查 action 算子的批次不变性。这些算子保证 prompt 填充前后的输出逐位一致。

引擎在 `xinfer/engine/cudagraph.py` 中展开 `meta["prompt_len_range"]` 并调用模型的 `pad` 钩子，按从大到小的顺序预热可达宽度。日志报告各宽度和每个阶段的 graph 数量。精确长度策略只预热所提供的 prompt 宽度，其他长度按配置的预热后 graph 未命中策略处理。

## 在 GPU 上做图像预处理

这是每个内置模型的处理选项 `processing.image_preprocess_device`，按入口分别设置：

- `xinfer serve`：`--model-options '{"processing": {"image_preprocess_device": "gpu"}}'`；默认值为 `"cpu"`。
- `Policy(..., image_preprocess_device=...)`，是 `Policy` 自己的参数（不放在 `processing` 中）：默认值 `None` 取平台的默认：能在设备上解码和 resize 视图时（有 torchvision 的 CUDA 设备，即平台能力 `device_images`）为 `"gpu"`，否则为 `"cpu"`。在不具备该能力的平台上指定 `"gpu"` 会抛出 `ConfigError`。

| 取值 | 视图在哪里解码和 resize | 数值 |
|---|---|---|
| `"cpu"`（`xinfer serve` 的默认值） | 在前端：OpenCV 解码，加上模型的 OpenCV / Pillow resize，一个请求的各路视图并行处理 | 与参考服务逐位一致 |
| `"gpu"`（`Policy` 在 CUDA 上的默认值） | 在设备上，由引擎完成。前端把每路 JPEG 视图的字节（其他编码在 CPU 上解码）或原始帧作为 `DeviceImage` 发送，附带从 JPEG 文件头读出的目标尺寸和模型的 resize 方案；在第一个阶段之前，引擎用一次 torchvision `decode_jpeg` 调用（nvJPEG）解码该请求的全部 JPEG，并以 FP32 resize 各视图（[引擎](engine.zh-CN.md#设备图像)） | 不逐位一致：解码出的 JPEG 在锐利的色度边缘附近与 libjpeg-turbo 的结果不同（在机器人相机视图上平均相差 0.1–0.2 级，少数像素最多相差几十级），FP32 resize 也与 Pillow 的不同（torch 的 bicubic 用 a = −0.75，Pillow 用 −0.5），因此动作会不同 |

`"gpu"` 需要 CUDA 设备和 torchvision ≥ 0.19（`pip install -e '.[gpu-image]'`）；否则每个请求都会失败，`xinfer serve --warmup` 会在启动时失败。解码器为第一个请求创建，使用 `--warmup` 时即为一个预热请求。视图形状不变，因此两种模式下的 CUDA graph 相同。

Wall-OSS MXFP8 在 RTX 5090 上启用 CUDA graph，使用 `benchmarks/request_latency.py` 运行 50 次的单请求 P50（ms）：

| 模型、请求 | `"cpu"`：总计（准备 / 引擎） | `"gpu"`：总计（准备 / 引擎） |
|---|---|---|
| Wall-OSS，x2robot dummy 客户端（一路 1280×720 原始前视图，两路 640×480 噪声 JPEG 腕部视图） | 40.4 (7.4 / 32.8) | 37.4 (3.1 / 34.2) |
| Wall-OSS，三路机器人相机 JPEG 视图（1280×720 和 640×480） | 39.0 (6.1 / 32.8) | 35.7 (1.9 / 33.6) |

前端改为把解码和缩放交给引擎进程的设备图像线程，在该线程的 stream 上每个请求耗时 1.1–1.4 ms（[引擎](engine.zh-CN.md#设备图像)）。以 8 台机器人运行 Wall-OSS MXFP8 时，两种模式均达到 44.0–44.3 req/s。GPU 预处理的每轮 P50 为 83–93 ms，CPU 为 83–88 ms。相对 BF16 参考，GPU 预处理的 quick100 B1 GT-MSE 降低了 6.13%，CPU 预处理降低了 5.01%。证据见 [Wall-OSS 验证](models/wall_oss_validation.zh-CN.md)。

尚未纳入解析：
- 内存规划器（design §8.2）；
- `xinfer tune`（调优表需根据基准测试结果手工编写）；
- 进程内前端（design §9.2）。

## 示例：RTX 5090 上的 Wall-OSS

`xinfer-serve --model wall_oss ... --dry-run`（optimized 组合）：

```text
shape_buckets.prompt          {"granularity": 16, "max_length": null, "sizes": []} model:bucket policy
per_robot_inflight            1          default
max_running_requests          32         default
max_queued_requests           256        default
lanes.vision                  2          default
lanes.prefill                 2          default
lanes.action                  2          default
stream_priorities.vision      0          default
stream_priorities.prefill     0          default
stream_priorities.action      -1         default:final stage
allow_non_invariant_batching  False      default
batch_sizes.vision            [1]        model:declares B1 only
batch_sizes.prefill           [1]        model:declares B1 only
batch_sizes.action            [1]        model:declares B1 only
cudagraphs                    True       platform:supports graph capture
frontends                     2          default, ≤ cpus/2 = 12
frontend_workers              4          default, frontends × workers ≤ cpus/2 = 12
shm_buffers                   16         default:one per running request
```

参考组合（`--model-options '{"numerics": "reference"}'`）声明 action 批次 `[1, 2, 4, 8]`：使用批次不变的算子时四种都可用，使用 `--reference-ops` 时解析为 `[1]`（`model:not batch-invariant`）。

## 默认设置下的实测结果

各模型在这些设置下的单卡容量见[性能](performance.zh-CN.md)。
