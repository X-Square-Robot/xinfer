# 性能剖析

[English](profiling.md) | 简体中文

算子级性能剖析由以下几部分组成：

| 部分 | 位置 | 提供 |
|---|---|---|
| 标注 | `xinfer.observability.annotate` | torch profiler 和 Nsight Systems trace 中带名字的 host 区间 |
| 剖析用算子表 | `annotate.profiled_ops` | 每次算子调用外的一个 span，带调用的形状 |
| 算子代价 | `xinfer.observability.op_cost` | 一次调用的 FLOPs、读写字节数和计算管线 |
| 引擎 span | `xinfer.engine` | 每次阶段发射外的 `stage::<stage>` span、其中通道等待外的 `wait` span、每次准入上传外的 `admit` span |
| roofline 下限 | `xinfer.observability.roofline` | 由调用的代价和 GPU 峰值得到的下限 |
| 算子剖析 | `xinfer-profile ops`、`xinfer.observability.op_profile` | 模型每种算子调用形状在各阶段的 GPU 时间、代价和下限 |
| 负载剖析 | `xinfer-profile load`、`xinfer.observability.load_profile` | 闭环机器人负载下每个请求在引擎中的时间：准入，以及每个阶段的排队、GPU 等待与执行和批大小 |

## 标注

```python
from xinfer.observability import annotate

annotate.configure(annotate.kineto, annotate.nvtx)  # 启动时调用一次
with annotate.span("vision", batch_size=2):
    ...
```

- `span(name, **args)` 是调用点唯一的 API。参数是紧凑的值（str、int、float 或 bool），不含张量数据。
- `configure(*sinks)` 选定之后所有 span 使用的 sink。不配置 sink 时（默认），`span` 返回同一个共享的空上下文；每次调用不做任何判断。
- 内置 sink 用 `label(name, args)` 命名区间：名字之后是每个参数的 `key=value`，以空格分隔，例如 `op::linear x=bfloat16[1,816,2048] weight=bfloat16[6144,2048]`。同一个解析器可以读两种 trace。
  - `kineto`：`torch.profiler.record_function`，即 torch profiler trace 中的 `user_annotation` 事件；
  - `nvtx`：供 Nsight Systems 使用的 NVTX 区间（`nsys profile --trace=cuda,nvtx`）；需要 CUDA 版的 torch。
- sink 是任何返回上下文管理器的可调用对象 `sink(name, args)`，因此其他 sink（例如进程内记录器）通过 `configure` 加入，不用改调用点。
- span 标记的是 host 代码。CUDA graph 回放不运行 host 代码：捕获时记录的 span 标记的是捕获本身而不是之后的回放，所以要在 eager 模式下标注。
- 引擎把每次阶段发射放在 `span("stage::<stage>", batch_size=B, lane=i, requests="<id>,...")` 中：输入合并、阶段本身（或其 graph 回放）以及输出的拷贝。其中 `span("wait")` 覆盖通道等待该批输入和上游阶段的过程。每次准入的上传放在 `span("admit", request=<id>, robot=<robot>)` 中。不开启标注时它们都是共享的空上下文。

## 剖析用算子表

```python
from xinfer import ops
from xinfer.observability import annotate

profiled = annotate.profiled_ops(ops.registry.build_table(platform))
with ops.use_ops(profiled):
    model = model_cls.from_checkpoint(checkpoint, device=platform.device, ops=profiled)
```

- 在剖析用算子表下构建的模型，每次算子调用都在 `span("op::<operator>", **args)` 中运行，参数按实现的参数名命名：张量记为 `dtype[shape]`，host 端序列记为 `[a,b,...]`，其他值原样记录；`None` 省略。
- 算子选择、契约和选择表与普通算子表相同。
- 在普通算子表下构建的模型直接绑定所选实现的函数对象本身，因此不剖析时没有包装，也没有开销。`profiled_ops` 使用 `OpTable.wrapped(wrap)`：它复制一张表，把每个可调用对象换成 `wrap(implementation)`。

## 算子代价（`op_cost`）

`op_cost` 给出一次算子调用的 FLOPs、读写字节数和计算管线，它们是这次调用 roofline 下限的输入：在给定 GPU 上，计算时间与访存时间中较大的那个。它只属于性能剖析；`OpSpec` 和算子后端都不带代价。

```python
from xinfer.observability.op_cost import op_cost

cost = op_cost("linear", args, kwargs, outputs)  # OpCost(flops, read_bytes, write_bytes, pipe)
```

`op_cost(name, args, kwargs, outputs)` 接收一次调用的参数和输出，即用 `OpTable.wrapped` 做的包装函数看到的那些。

- **flops**：只计矩阵乘，一次乘加计 2。epilogue、融合的 norm、softmax 和逐元素运算不计。
- **read_bytes**、**write_bytes**：每个张量按存储格式计一次：BF16 或 FP32；E4M3 元素和 E8M0 scale（MXFP8）；打包的 E2M1 元素和 E4M3 scale（NVFP4）；scale 按 swizzle 布局计，含其 padding。cache 只计读或写的那些行。中间结果不计。
- **pipe**：按输入 dtype 为 `bf16`、`fp16` 或 `fp32`，或 `mxfp8`、`nvfp4`，纯数据搬运为 `memory`。

做矩阵乘的算子有显式公式（`FORMULAS`），按参考实现的参数名接收算子参数：

| 算子 | FLOPs | 管线 | 说明 |
|---|---|---|---|
| `linear` | 2·M·N·K | 输入 dtype | |
| `gemm_epilogue`、`gemm_glu_epilogue` | 2·M·N·K | 输入 dtype | 读残差、gate 行、行类别、行缩放和 modulation；`glu` 输出 `[M, N/2]` |
| `qkv_rope_epilogue`、`qkv_rope_kvcache` | 2·M·N·K | 输入 dtype | 读 RoPE 表和偏置；每行写 N 列，K 和 V 作为 cache 的新行；不读 cache |
| `flow_velocity_euler` | 2·T·N·K，T 个 token 各经过一个 `[N, K]` head | 较宽的操作数 dtype | flow head 权重为 FP32 时是 `fp32`；Euler 步不计 |
| `conv2d` | 2·B·H'·W'·O·C·kH·kW | 输入 dtype | 无 padding，单组 |
| `mxfp8_gemm`、`mxfp8_gemm_rope`、`mxfp8_gemm_silu_mul`、`mxfp8_gemm_residual` | 2·M·N·K | `mxfp8` | 读 rope 表和残差；输出 BF16，`silu_mul` 为 `[M, N/2]` |
| `nvfp4_gemm` | 2·M·N·K，K 为打包宽度的 2 倍 | `nvfp4` | |
| `causal_attention` | 4·D·B·H·L(L+1)/2 | 输入 dtype | 精确的因果工作量 |
| `varlen_attention` | 4·D·H·Σ l² | 输入 dtype | 精确，由 host 端的分段边界算出 |
| `prefix_attention` | 4·D·B·H·S² | 输入 dtype | 按 padding 布局：有效长度是设备数据 |
| `action_attention`、`masked_joint_attention` | 4·D·B·H·L·(P + L) | 输入 dtype | 按 padding 布局；读长度、固定行数或 mask |
| `kvcache_attention` | 4·D·B·H·L·S | 输入 dtype | 按 padding 布局；读 S − L 列 cache，把自身的 key 和 value 写入两个 cache |
| `prefix_kvcache_attention` | 4·D·B·H·M·(new_pos + M) | 输入 dtype | 按 padding 布局：前缀长度和 `flash_mask` 是设备数据；读 cache 的 [0, new_pos + M) 行 |
| `chunk_gated_delta_rule` | 每个长 C = 64 的 chunk 每个 head：2C²(3Dk + 2Dv) + 6C·Dk·Dv + (C−1)C(2C−1)/3 | 输入 dtype | 分块形式，长度补齐到 C 的倍数 |

其余算子（norm、逐元素算子、旋转位置编码、逐通道的 `causal_conv1d`、各量化算子）使用通用代价：`memory` 管线上 0 FLOPs，每个张量参数读一次，每个输出张量写一次。代价只读取形状、dtype 和 host 端的值，因此也能在由 span 参数构造的 `meta` 张量上计算。

局限：

- 按 padding 布局的 attention 计入 padding 布局中的每个 key；按长度计算的 kernel 工作量更少。
- `nvfp4_quantize(mse=True)` 是加载时的权重转换，每个块搜索 126 个 scale；这里按数据搬运计。
- 参考实现实际搬运的数据多于计数（mask、重复的 KV head、中间结果）。

## roofline 下限

```python
from xinfer.observability.roofline import PEAKS, roofline

bound = roofline(op_cost(name, args, kwargs, outputs), PEAKS["NVIDIA GeForce RTX 5090"])  # Bound(seconds, limiter)
```

- `seconds = max(flops / peaks[pipe], bytes / peaks["dram"])`；计算项较大时 `limiter` 为 `compute`，否则为 `memory`。`memory` 管线没有计算项；某个管线没有峰值时抛出 `KeyError`。
- 没有 L2 限制项：一次调用的数据是否在 L2 中取决于之前的调用，单次调用的代价无法说明。所有字节都按 DRAM 带宽计，所以输入仍在 L2 中的调用可以快于它的下限。
- 下限不含启动延迟，因为一个算子可能运行多个 kernel。

`PEAKS` 按 GPU 名称（`torch.cuda.get_device_name()`）存放实测峰值：

| GPU | BF16 | FP32 | MXFP8 | NVFP4 | DRAM |
|---|---|---|---|---|---|
| NVIDIA GeForce RTX 5090 | 220.1 TFLOP/s | 66.2 TFLOP/s | 574.3 TFLOP/s | 1116.4 TFLOP/s | 1522 GB/s |

2026-11-06 在 driver 595.91.07、torch 2.9.1+cu130 下按稳态测得：每一项先连续负载约两秒，再取 20 次 CUDA event 计时的中位数。此时 BF16 GEMM 运行在 575 W 功耗上限，SM 时钟为 2617–2632 MHz（最高 3090 MHz）。

- BF16、FP32：`torch.mm` 计算随机 8192 × 8192 矩阵（FP32 关闭 TF32），2N³ / 时间。
- MXFP8：该次测量使用的 xcompute `mxfp8_gemm` 后端，8192³，`fast_accum=False`，即该算子的默认值。MXFP8 线性层的 `fast_accum="auto"` 对至多 16 行或至少 8192 个输出的调用开启快速累加，这些调用可以快于这个峰值。
- NVFP4：`torch._scaled_mm` 计算 NVFP4 操作数，每 16 个元素一个 E4M3 scale，8192³。
- DRAM：1 GiB 的 `copy_`，读写字节数 / 时间。

短时突发可以运行在高于稳态的时钟上，所以调用也可能因此快于它的下限。

## 算子剖析（`xinfer-profile ops`）

```bash
xinfer-profile ops --model wall_oss --checkpoint /path/to/checkpoint \
  --requests requests.msgpack --requests-per-run 2 --out profile
```

这个命令适用于所有已注册的模型；`--model` 只选择运行哪一个。它需要 CUDA，以及能记录 CUDA 活动的 torch profiler：在 driver 595.91.07 下，torch 2.9.1+cu130 记录不到 CUDA 活动，torch 2.10 可以。trace 中没有 CUDA 活动时，命令会给出这一说明并停止。

1. **构建。** 模型在剖析用算子表下构建，再包装一层，记录每种调用形状的 `OpCost`（`op_profile.CostRecorder`）。
2. **预热。** `--requests` 中的每个请求以 eager 模式运行两次：完成惰性初始化、自动调优，并记录代价。
3. **剖析。** `--requests-per-run` 个请求（默认为文件中的全部请求，循环使用）以 eager 模式在 torch profiler 下运行，记录所有线程的 CPU 和 CUDA 活动，使用 kineto sink。
4. **归属。** 每个 GPU 活动（kernel、内存拷贝或置值）按其启动调用的 correlation id，归到启动时所在的最内层 `op::` span 和 `stage::` span（`op_profile.attribute`）。在阶段内但不在任何算子 span 内的活动计入该阶段的 “(outside operators)” 行：输入合并、交接拷贝和拷回 host。
5. **下限。** 每种调用形状的下限使用该 GPU 在 `PEAKS` 中的峰值，或 `--peaks` 给出的 JSON 对象 `{pipe: FLOP/s, "dram": bytes/s}`。两者都没有时省略下限各列。
6. **graph 回放。** 之后同样的请求在 CUDA graph 回放（即服务路径）下运行，得到每个阶段的 GPU 区间。

| 文件 | 内容 |
|---|---|
| `ops.md`、`ops.csv`、`ops.json` | 每个阶段每种算子调用形状一行：调用次数、µs/调用、总 µs、占该阶段 GPU 时间的比例、GFLOP/调用、MB/调用、下限 µs/调用、效率（下限 / µs）、限制项和代价来源（`formula` 或 `generic`）；均为每请求的值 |
| `stages.md` | 每个阶段每请求：host 区间、eager GPU 区间 = 算子 + 算子外 − 重叠 + 空闲，以及 graph 回放下的 GPU 区间 |
| `trace.json`、`trace-graphs.json` | eager 和 graph 回放的 kineto trace（可用 Perfetto 打开）；eager trace 中 kernel 位于各自的 `op::` span 之下 |
| `meta.json` | 源码提交和哈希、GPU、时钟和功耗上限、torch、CUDA 和 driver 版本、算子选择、模型选项、请求文件的 sha256 和所用峰值 |

- 覆盖率写在输出的开头几行，是落在算子 span 和阶段 span 内的 GPU 活动占比；阶段之前的活动（输入上传）两者都不在。
- eager 发射在 kernel 之间要等 Python，所以阶段的 eager GPU 区间大部分是空闲。算子时间是 kernel 时间，不含这些间隙；graph 回放区间才是服务时的情况。
- 没有显式公式的算子使用通用的数据搬运代价（`generic`），所以下限是访存下限。

## 负载剖析（`xinfer-profile load`）

此命令需要通道引擎的准入和阶段 span，不支持使用显式完成的 RPU 配置；RPU 推理使用进程内 `Policy` 接口。

```bash
xinfer-profile load --model wall_oss --checkpoint /path/to/checkpoint --requests observation.msgpack \
  --robots 8 --out artifacts/wall-oss-load
```

多台机器人共用引擎时，请求的时间花在引擎的哪里：这是一次只跑一个请求的 `xinfer-profile ops` 看不到的。

1. **引擎。** 模型和一个按 `xinfer-serve` 解析出的默认值（通道、批大小、CUDA graph）构建的 `Engine`，像 `--warmup` 一样用 `--requests` 的请求预热。
2. **负载。** `--robots` 台虚拟机器人（默认 8 台）轮流提交准备好的请求，每台 `--requests-per-robot` 个（默认 80），同时最多一个在途：机器人在上一个请求发出 `--period-ms`（默认 180）毫秒后发出下一个，若响应到得更晚则在响应到达时发出。每台机器人的前 `--warmup-requests` 个（默认 20）不计入统计。`--start spread`（默认）时第 `r` 台机器人比第一台晚 `r / N` 个周期发出第一个请求，就像相互独立的机器人；`--start together` 时所有机器人同时发出第一个请求，只要响应在周期内到达，它们之后就一直同步，这是对队列最不利的情况。结果取决于这种到达方式：`benchmarks/robot_load.py` 压测 `xinfer-serve` 的结果介于两者之间。
3. **记录。** 一个 span sink（`load_profile.SpanRecorder`）记录引擎的 `admit`、`stage::` 和 `wait` span：host 时间，以及在 CUDA 上于各 span 两端、在通道的 stream 上记录的计时 event。event 在负载结束后才读取，所以运行中不做任何同步。一次派发的第一个 event 在空闲通道上大约于派发的 host 时刻执行，由此把 GPU 时间换算到 host 时钟上。

每个请求的分段（ms）：

| 分段 | 时间 |
|---|---|
| `admit_ms` | 提交到准入：等待 slot |
| `inputs_ms` | 准入时的上传 |
| `<stage>.delay_ms` | 从阶段的输入就绪（首个阶段为上传结束，否则为上游阶段在 GPU 上完成）到它在 GPU 上开始 |
| `<stage>.run_ms` | 阶段在 GPU 上执行（CPU 上为 host 时间） |
| `complete_ms` | 最终阶段在 GPU 上完成到拿到结果：拷回 host 和完成线程 |
| `total_ms` | 提交到拿到结果 |

阶段串成一条链时（现有模型都是如此），这些分段相加等于 `total_ms`，所以尾部各分段多出的时间相加等于总时间多出的时间。`requests.csv` 中还有每个阶段与上游执行重叠的时间，用来查延迟从哪里来：

| 列 | 时间 |
|---|---|
| `<stage>.queue_ms` | host 上从上传或上游派发结束到本阶段派发：等待空闲通道 |
| `<stage>.wait_ms` | 通道在 GPU 上等待输入和上游阶段 |

链式派发下，请求的上游阶段一经派发，它就进入下游阶段的队列；而通道要等它的派发在 GPU 上完成才空闲。所以某阶段的排队和等待发生在上游阶段执行期间，只有体现在 `delay_ms` 中的部分才真正耗时。

| 文件 | 内容 |
|---|---|
| `requests.csv` | 每个请求一行：机器人、序号、各分段，以及每个阶段的批大小、排队和等待时间 |
| `summary.md`、`summary.json` | 相加等于 `total_ms` 的各分段的 P50/P95/P99、每个阶段各批大小的请求数、吞吐，以及对 `total_ms` 处于 P95 和 P99 及以上的请求，各分段均值减去中位带（P45–P55）均值 |
| `meta.json` | 命令、模型选项、请求文件的 sha256、负载设置、解析出的配置、算子选择，以及在 CUDA 上的 GPU、版本和源码 |

- 进程内没有前端进程：请求只准备一次，输出不做处理，所以输入输出处理、网络和进程间传输不在这些时间里。`--serve` 会把它们加进来（见下）。
- 带 `DeviceImage` 输入时，设备上的解码计入首个阶段的延迟。

### 经过 `xinfer-serve`（`--serve`）

加 `--serve` 时，命令会为这次运行启动一个 `xinfer-serve`（同样的模型、checkpoint 和选项，用 `--requests` 预热，另加 `--serve-args`），并带上 `--profile-dir <out>/engine`；虚拟机器人变成 WebSocket 客户端，轮流发送 `--requests` 中的原始请求。这样每个请求都像机器人的请求一样，经过前端的输入和输出处理、共享内存和管道。`--profile-dir` 让服务的引擎进程记录同样的 span，并在服务停止时把每个请求的引擎侧写到 `engine/engine-requests.json`。

每台机器人保持一条连接、同时只有一个请求在途，并在上一台连上之后才连接，所以每个前端按机器人顺序给连接编号；引擎通过连接识别请求，由此把每个请求的客户端时间、`server_timing` 和引擎 span 对上。每个请求的分段（ms）：

| 分段 | 时间 |
|---|---|
| `network_ms` | 客户端时间减去 `server_timing.infer_ms`：网络和消息编解码 |
| `prepare_ms`、`output_ms` | 前端的输入和输出处理（`server_timing`） |
| `handoff_ms` | `server_timing.engine_ms` 减去引擎从准入到最终阶段完成的区间：共享内存缓冲区、管道、等待 slot、拷回 host 和完成 |
| `inputs_ms`、`<stage>.delay_ms`、`<stage>.run_ms` | 与进程内相同 |
| `total_ms` | 客户端从发出到收到 |

和进程内一样，这些分段相加等于 `total_ms`；`requests.csv` 中还有各阶段的排队和等待时间。

负载结束后命令会停止服务；`serve.log` 中有服务的日志和解析出的配置。
