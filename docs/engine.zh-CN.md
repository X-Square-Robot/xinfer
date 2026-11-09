# 引擎

[English](engine.md) | 简体中文

`xinfer.engine.Engine` 在一块 GPU 上运行一个模型，所有阶段都在同一个进程内（docs/design.md §6–§9）。本页介绍它的行为和设置。

## 线程

| 线程 | 工作 |
|---|---|
| 引擎线程 | 准入（最多 `max_running_requests` 个）、各阶段的 FIFO 队列、组批，以及各阶段的 CUDA 提交（上传、graph 回放、事件） |
| 完成线程 | 等待每个最终阶段的事件，不做任何拷贝，然后完成该请求的 future |
| 设备图像线程 | 在自己的 CUDA stream 上，为已准入且带有 `DeviceImage` 输入的请求解码并 resize 这些输入（[见下文](#设备图像)） |

调用方使用 `submit(prepared, robot=...) -> Future`（线程安全）或 `run(prepared)`。`StageGraph` 校验模型恰好有一个终止阶段；请求完成时，在该阶段之后释放其槽位，且只释放一次。

`xinfer/engine/core.py` 中的请求路径如下：

```text
submit() -> waiting queue
engine thread (_engine_thread):
  _run_engine_loop() -> _step()
    _admit_requests()
    _select_batch(stage) -> _execute_stage_batch(stage, lane, batch)
completion thread:
  _completion_loop() -> wait for final event -> resolve Future
```

`_select_batch` 把选中的请求从阶段队列中移除。`_execute_stage_batch` 提交异步 CUDA 工作并记录其事件；它返回并不代表 GPU 已执行完成。

如果等待最终阶段事件时抛出异常，该批请求失败，完成线程继续运行。`Engine.health()` 返回引擎无法继续服务的原因（引擎线程或完成线程已停止），否则返回 None；`xinfer-serve` 会轮询它（[服务](serving.zh-CN.md)）。

## 调度规则

1. **链式分发（dispatch）。** 请求的上游阶段一经*分发*，请求就进入该阶段的队列；通道（lane）在 GPU 上等待上游的事件。通道空闲时，请求无需主机往返即可跑完所有阶段，这就是单请求路径。有负载时队列被填满，各阶段形成流水线。
2. **零等待组批。** 一次分发取队列中已有的兼容请求，不超过该阶段的上限，选用允许的最大批大小，且从不填充空行。兼容指该阶段读取的每个输入形状相同（批维除外）且主机端取值相同。只有当模型声明该阶段批次不变（`Capabilities.batch_invariant`，由其算子的契约推导）或设置了 `allow_non_invariant_batching` 时，该阶段才会组批。每个阶段的第一次分发单独运行，因为要借此得知该阶段读取哪些输入。
   - **不同长度。** 若交接张量的 `TensorSpec` 声明了 `seq_axis`（Wall-OSS：prompt 的 K/V），同批请求之间该张量的长度可以不同。组批时每一行左填充到最长者的长度，阶段会收到 `<name>_lengths`（`int32 [B]`，B1 时也有）。见 `xinfer/engine/batching.py`。在 CUDA 上，长度从锁页（pinned）内存异步上传：通道可能已经在等待上游事件，阻塞式上传会卡住发起提交的线程。
   - **延迟合并。** 输入在阶段读取时才合并。阶段不读取的输入，或无法共享的输入（形状或主机端取值不同），不会合并进批。
   - **计数器。** `stats["dispatch_<stage>_b<N>"]` 按批大小统计分发次数，`stats["padded_dispatch_<stage>"]` 统计对不同长度做过填充的批数。
3. **下游优先。** 引擎线程每一轮先在有空闲槽位时准入等待中的请求，再从最后一个阶段到第一个阶段依次分发，使下游工作先于上游阶段新增的工作发出。

通道的 stream 在创建时排在当前 stream 上已排队的工作（如权重加载和转换）之后，因此第一次分发不会读到仍在写入的权重。一个通道同一时间只运行一次分发；其事件完成后通道重新空闲。后续阶段要读取的输出，会在该通道下一次分发之前从通道的缓冲区拷贝到请求自有的张量中。

最终阶段的结果在通道的第二个 stream（`Lane.copy_stream`）上拷贝到锁页主机缓冲区，这个 stream 从不被捕获：释放锁页缓冲区时会在用过它的 stream 上记录事件，而在捕获期间记录的事件会导致下一次锁页内存分配失败。

如果一次分发在已有工作入队（包括部分到主机的拷贝）之后失败，通道的 stream 会等待其拷贝 stream 并记录一个事件；通道和这些请求的槽位（持有其张量）在该事件之后才释放。

## 设备图像

准备好的输入可以是 `DeviceImage`（`xinfer.models.interface`）而非张量：即到达时原样的相机视图（JPEG 字节或 RGB 帧），附带每个 resize 步骤之后的尺寸以及每一步的 `interpolate` 模式。输入处理器在 `processing.image_preprocess_device="gpu"` 时发送这种输入（每个内置模型；[配置](configuration.zh-CN.md#在-gpu-上做图像预处理)）。在请求的第一个阶段之前、CUDA graph 之外，引擎把其中每一个转成 `[h, w, 3]` uint8 视图，各阶段以同一名称读取它，与 CPU 预处理时相同。这些视图与 CPU 路径的形状相同，因此 graph 及其键不变，变长的 JPEG 字节也不会成为 graph 的键。

准入时，请求的张量照常上传，其 `DeviceImage` 值留在主机内存中。带设备图像的请求随后进入**设备图像线程**，而不是其起始阶段的队列。该线程按准入顺序处理：让自己的 CUDA stream 等待该请求的上传，用一次批量的 torchvision `decode_jpeg` 调用解码该请求的全部 JPEG，上传其帧，以 FP32 resize 各视图（四舍五入到 uint8），然后记录一个事件，该事件即成为请求的 `inputs_ready` 事件。随后引擎线程把请求放入其起始阶段的队列，这些阶段的通道在 GPU 上等待该事件：引擎线程从不等待解码，一个请求的解码与它之前各请求的阶段重叠执行。不带设备图像的请求不经过该线程。视图归请求所有，直到请求在最终阶段之后完成，因此不会有阶段读到已释放的内存。如果解码抛出异常，只有该请求失败（其张量保留到该线程的事件之后）；`shutdown` 会排空并停止该线程。预热请求走同一条路径。

解码器（`xinfer.engine.device_images`）在该线程上、处理第一个带设备图像的请求时创建。没有 CUDA 设备或没有 torchvision 时，每个带设备图像的请求都会以此为原因失败；在 `xinfer serve --warmup` 下，预热请求由相同的处理器准备，因此服务会在启动时失败。

一个线程既足够也必要：torchvision 每个进程只有一个 nvJPEG 解码器，其解码调用会阻塞调用线程。对 baseline JPEG，它使用 nvJPEG 的硬件后端，运行在 GPU 的 JPEG 引擎上（NVJPG，RTX 5090 具备）：熵解码和 IDCT 都在那里执行，CPU 只解析文件头并提交批次，唯一的 CUDA kernel 是每张图约 2 µs 的 RGBA 到 RGB 转换。Progressive JPEG 则走 nvJPEG 的混合路径，Huffman 解码在 CPU 上、IDCT 在 CUDA kernel 中执行。在 RTX 5090 上（torchvision 0.24.1），三路机器人相机视图的一次调用阻塞约 0.7 ms，原因在于等待：torchvision 用进程级锁串行化调用，并在批次前后同步其解码 stream，nvJPEG 的硬件路径则在一个事件上同步；该线程的 CPU 时间都花在这些等待的自旋上。该调用会释放 GIL，因此不占用引擎线程的任何时间。JPEG 字节留在主机内存，因为 nvJPEG 从那里读取码流（torchvision 不接受 CUDA 张量作为输入）。

## 预热

`Engine.warmup(requests)`（`Policy.warmup`、`xinfer-serve --warmup`）先应用模型专用的 `warmup_requests`（如有），再通过模型的 `pad` 钩子展开 `model.shape_buckets()`：同类请求可能落入的每个形状桶（shape bucket）各一个变体（在该请求的指令下，状态文本可能产生的每个 prompt 宽度：Wall-OSS 默认按 16 token 分桶）。它单独运行每个变体，从而在第一个通道上以批大小 1 捕获其 graph；每个通道保留这些捕获的输入。随后 `capture_batch_sizes`（`xinfer/engine/cudagraph.py`）在该阶段的每个通道上、以每个允许的批大小运行这些输入，合并方式与服务时组批相同，因此服务时能找到这些形状所需的 graph。最大的输入先运行，因为增大通道的静态缓冲区会丢弃使用旧缓冲区的 graph。长度不等的混合批会填充到其中最长的成员，因此为每个长度预热即可覆盖各种混合。所以每类请求各一个就够了（Wall-OSS：一个 flow 请求和一个 RTC 请求；每条指令各一组）；模型不会推导出的形状（其他指令、相机布局或 RTC 行数）仍需各自的预热请求，否则会在首次使用时捕获并复用。预热会记录所选的桶宽度、耗时、每个阶段保留的 graph、它们的捕获耗时以及之后预留的显存。

每个通道最多保留 `CUDAGraphConfig.max_graphs`（64）个 graph。action 阶段的通道保留的最多：每条指令下，Wall-OSS 每个通道需要 3–4 个 prompt 桶 × flow/RTC × 4 种批大小 = 24–32 个 graph。

`Engine.warmup` 返回每个预热请求的输出（取自其第一个变体），`xinfer-serve` 把它们交给前端，用于前端自己的预热（[serving.md](serving.zh-CN.md)）。

## 设置（`EngineConfig`）

| 字段 | 默认值 | 含义 |
|---|---|---|
| `lanes` | 每阶段 1 个 | 每个阶段的通道数（stream + graph 池），例如 `{"prefill": 2, "action": 2}` |
| `stream_priorities` | 每阶段 0 | 一个阶段所有通道的 CUDA stream 优先级；整数越小优先级越高 |
| `max_batch` | 每阶段 1 | 每个阶段的批上限；受 `Capabilities.batch_sizes` 和批次不变性门槛限制 |
| `max_running_requests` | 8 | 在途的已准入请求数 |
| `max_queued_requests` | 256 | 已提交、等待准入的请求数（超出则 `AdmitError("queue_full")`） |
| `per_robot_inflight` | 1 | 每个机器人 id 未完成的请求数（`AdmitError("per_robot_limit")`） |
| `cudagraphs` | `None` | 设为 `CUDAGraphConfig()` 时回放 CUDA graph（见 `xinfer.engine.cudagraph`） |
| `allow_non_invariant_batching` | false | 对算子非批次不变的阶段也组批（会改变数值） |

通道的 stream 优先级在创建通道时确定；上传 stream 和拷贝 stream 保持优先级 0。优先级只改变 GPU 执行就绪工作的顺序：准入、组批和事件等待都不变，graph 仍在同一个通道 stream 上捕获和回放。

`xinfer-serve` 和 `Policy` 依据调优表（目前尚未随附任何表）、模型与平台规则以及与负载无关的固定默认值来确定这些设置（[配置](configuration.zh-CN.md)）；显式指定的值仍然优先。

## 保证与测试

- 使用批次不变的算子时，请求单独运行与在任意批中运行的结果逐位一致：`test_wall_oss_parity.py::test_action_batches_equal_single_requests` 以不同 prompt 长度和 RTC 延迟对 Wall-OSS action 阶段做 B2..B8 组批，并把每个请求与其单独运行的结果比较。
- 批大小为 1 时，请求的结果与并发无关：`tests/models/test_wall_oss_parity.py` 在每阶段 2 个通道、启用 graph 的情况下运行 21 个并发请求，并与相同请求单独运行的结果逐位比较。
- `tests/unit/test_engine.py` 覆盖并发提交、组批与按行拆分、兼容性键、批次不变性门槛、准入上限与背压、取消以及逐请求失败。
- `tests/unit/test_cudagraph.py::test_result_freed_during_a_capture` 在通道正在捕获时于另一个线程释放一个结果。
- `tests/unit/test_host_copy.py` 用一个 GPU 闸门挡住失败分发的部分拷贝，并检查在此期间其通道和槽位不会被复用。
- `tests/unit/test_batching_async.py` 让上游生产者保持未完成，并检查合并不等长的批不会阻塞发起提交的线程。
- `tests/unit/test_engine.py` 覆盖设备图像线程：先前请求的解码被挂起时，不带设备图像的请求能够完成；只有带设备图像的请求会到达解码器，且解码器在该线程上只创建一次；解码遵循准入顺序；解码抛出异常只让其所属请求失败；没有 CUDA 设备时，带设备图像的请求失败而其他请求照常服务。`test_device_images_are_decoded_off_the_lanes_and_replay_one_graph`（`tests/unit/test_cudagraph.py`）检查图像在被该线程于自己的 stream（而非通道的 stream）上解码之前一直留在主机内存，并检查不同长度的 JPEG 和一帧原始图像回放同一个 graph，且输出与 eager 引擎一致。`tests/unit/test_shm.py` 检查 `DeviceImage` 数据从前端经共享内存而非管道传递。
- `tests/unit/test_cudagraph.py` 中的 `test_warmup_captures_every_lane_and_batch_size`、`test_warmed_lengths_cover_batches_of_mixed_lengths` 和 `test_warmup_requests_cover_every_bucket` 在预热后以 `on_miss_after_warmup="error"` 服务并发请求。
