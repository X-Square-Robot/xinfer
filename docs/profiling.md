# Profiling

English | [简体中文](profiling.zh-CN.md)

Operator profiling builds on these pieces:

| Piece | Where | Gives |
|---|---|---|
| Annotations | `xinfer.observability.annotate` | named host regions in torch profiler and Nsight Systems traces |
| Profiled op tables | `annotate.profiled_ops` | a span around every operator call, with the call's shapes |
| Operator cost | `xinfer.observability.op_cost` | FLOPs, bytes read and written, and compute pipe of one call |
| Engine spans | `xinfer.engine` | a `stage::<stage>` span around every stage dispatch, a `wait` span around its lane's waits, an `admit` span around each admission's upload |
| Roofline bound | `xinfer.observability.roofline` | the lower bound of a call from its cost and the GPU's peaks |
| Operator profile | `xinfer-profile ops`, `xinfer.observability.op_profile` | GPU time, cost and bound of every operator call shape of a model, per stage |
| Load profile | `xinfer-profile load`, `xinfer.observability.load_profile` | each request's time in the engine under closed-loop robot load: admission, per stage queue, GPU wait and run, batch size |

## Annotations

```python
from xinfer.observability import annotate

annotate.configure(annotate.kineto, annotate.nvtx)  # once, at startup
with annotate.span("vision", batch_size=2):
    ...
```

- `span(name, **args)` is the one call-site API. Arguments are compact values (str, int, float or bool), never tensor
  data.
- `configure(*sinks)` chooses the sinks of every later span. Without sinks, the default, `span` returns one shared
  no-op context; nothing is decided per call.
- The built-in sinks name the region `label(name, args)`: the name, then `key=value` per argument, separated by
  spaces, for example `op::linear x=bfloat16[1,816,2048] weight=bfloat16[6144,2048]`. One parser reads both traces.
  - `kineto`: `torch.profiler.record_function`, a `user_annotation` event in torch profiler traces;
  - `nvtx`: an NVTX range for Nsight Systems (`nsys profile --trace=cuda,nvtx`); needs a CUDA build of torch.
- A sink is any callable `sink(name, args)` that returns a context manager, so another sink, such as an in-process
  recorder, is added through `configure` without changing call sites.
- Spans mark host code. A CUDA graph replay runs no host code: spans recorded during capture mark the capture, not
  the replays, so annotate eager runs.
- The engine runs every stage dispatch inside `span("stage::<stage>", batch_size=B, lane=i, requests="<id>,...")`: the
  merge of the inputs, the stage (or its graph replay) and the copies of its outputs. Inside it, `span("wait")` covers
  the lane's waits for the batch's inputs and upstream stages. Each admission's upload runs inside
  `span("admit", request=<id>, robot=<robot>)`. With annotation off these are the shared no-op context.

## Profiled op tables

```python
from xinfer import ops
from xinfer.observability import annotate

profiled = annotate.profiled_ops(ops.registry.build_table(platform))
with ops.use_ops(profiled):
    model = model_cls.from_checkpoint(checkpoint, device=platform.device, ops=profiled)
```

- Every operator call of a model built under the profiled table runs inside `span("op::<operator>", **args)`, the
  arguments named by the implementation's parameters: tensors as `dtype[shape]`, host sequences as `[a,b,...]`,
  other values as they are; `None` is omitted.
- Selections, contracts and the selection report are those of the plain table.
- A model built under a plain table binds the selected implementations' function objects themselves, so profiling
  adds no wrapper and no overhead when it is off. `profiled_ops` uses `OpTable.wrapped(wrap)`, which copies a table
  with each callable replaced by `wrap(implementation)`.

## Operator cost (`op_cost`)

`op_cost` gives the FLOPs, bytes read and written, and compute pipe of one operator call, the inputs of the call's
roofline bound: the larger of its compute time and its memory time on a given GPU. It belongs to profiling only;
`OpSpec` and the operator backends carry no cost.

```python
from xinfer.observability.op_cost import op_cost

cost = op_cost("linear", args, kwargs, outputs)  # OpCost(flops, read_bytes, write_bytes, pipe)
```

`op_cost(name, args, kwargs, outputs)` takes a call's arguments and outputs, as a wrapper made with
`OpTable.wrapped` sees them.

- **flops**: matrix products only, a multiply-add counting two. Epilogues, fused norms, softmax and elementwise work
  are not counted.
- **read_bytes**, **write_bytes**: each tensor once, as stored: BF16 or FP32; E4M3 elements and E8M0 scales (MXFP8);
  packed E2M1 elements and E4M3 scales (NVFP4); scales in the swizzled layout with its padding. Of a cache, only the
  rows read or written count. Intermediates are not counted.
- **pipe**: `bf16`, `fp16` or `fp32` by input dtype, `mxfp8`, `nvfp4`, or `memory` for data movement only.

The operators that do matrix products have explicit formulas (`FORMULAS`), which take the operator's arguments under
the reference implementation's parameter names:

| Operators | FLOPs | Pipe | Notes |
|---|---|---|---|
| `linear` | 2·M·N·K | input dtype | |
| `gemm_epilogue`, `gemm_glu_epilogue` | 2·M·N·K | input dtype | residual, gate rows, row classes, row scales and modulation are read; `[M, N/2]` output for `glu` |
| `qkv_rope_epilogue`, `qkv_rope_kvcache` | 2·M·N·K | input dtype | RoPE tables and bias are read; writes the N columns of each row, K and V as new cache rows; the caches are not read |
| `flow_velocity_euler` | 2·T·N·K, each of T tokens through one `[N, K]` head | wider operand dtype | `fp32` with the FP32 flow-head weights; the Euler step is not counted |
| `conv2d` | 2·B·H'·W'·O·C·kH·kW | input dtype | no padding, one group |
| `mxfp8_gemm`, `mxfp8_gemm_rope`, `mxfp8_gemm_silu_mul`, `mxfp8_gemm_residual` | 2·M·N·K | `mxfp8` | rope table and residual are read; BF16 output, `[M, N/2]` for `silu_mul` |
| `nvfp4_gemm` | 2·M·N·K, K twice the packed width | `nvfp4` | |
| `causal_attention` | 4·D·B·H·L(L+1)/2 | input dtype | exact causal work |
| `varlen_attention` | 4·D·H·Σ l² | input dtype | exact, from the host-side segment bounds |
| `prefix_attention` | 4·D·B·H·S² | input dtype | padded: the valid lengths are device data |
| `action_attention`, `masked_joint_attention` | 4·D·B·H·L·(P + L) | input dtype | padded; lengths, pinned rows or mask are read |
| `kvcache_attention` | 4·D·B·H·L·S | input dtype | padded; reads the S − L cache columns, writes the own keys and values into both caches |
| `prefix_kvcache_attention` | 4·D·B·H·M·(new_pos + M) | input dtype | padded: prefix lengths and `flash_mask` are device data; reads cache rows [0, new_pos + M) |
| `chunk_gated_delta_rule` | per chunk of C = 64 and head: 2C²(3Dk + 2Dv) + 6C·Dk·Dv + (C−1)C(2C−1)/3 | input dtype | the chunked form over the length padded to C |

Every other operator (norms, elementwise operators, rotary embeddings, the depthwise `causal_conv1d`, the quantizers) has the generic cost: 0 FLOPs on the `memory` pipe,
every tensor argument read once and every output tensor written once. Costs read only shapes, dtypes and host-side
values, so they also run on `meta` tensors built from a span's arguments.

Limitations:

- Padded attention counts every key of the padded layout; a length-aware kernel does less work.
- `nvfp4_quantize(mse=True)`, the weight conversion at load time, searches 126 scales per block; it is counted as
  data movement.
- Reference implementations move more data than the count (masks, repeated KV heads, intermediates).

## Roofline bound

```python
from xinfer.observability.roofline import PEAKS, roofline

bound = roofline(op_cost(name, args, kwargs, outputs), PEAKS["NVIDIA GeForce RTX 5090"])  # Bound(seconds, limiter)
```

- `seconds = max(flops / peaks[pipe], bytes / peaks["dram"])`; `limiter` is `compute` when the compute term is the
  larger, else `memory`. A `memory` pipe has no compute term; a pipe without a peak raises `KeyError`.
- There is no L2 limiter: whether a call's data is in L2 depends on the calls before it, which its cost does not
  tell. All bytes are charged at DRAM bandwidth, so a call whose inputs are still in L2 can beat its bound.
- Launch latency is not part of the bound, since an operator may run several kernels.

`PEAKS` holds measured peaks by GPU name (`torch.cuda.get_device_name()`):

| GPU | BF16 | FP32 | MXFP8 | NVFP4 | DRAM |
|---|---|---|---|---|---|
| NVIDIA GeForce RTX 5090 | 220.1 TFLOP/s | 66.2 TFLOP/s | 574.3 TFLOP/s | 1116.4 TFLOP/s | 1522 GB/s |

Measured on 2026-11-06 with driver 595.91.07 and torch 2.9.1+cu130, at steady state: each entry after about two
seconds of continuous load, the median of 20 runs timed with CUDA events. The BF16 GEMM then ran at the 575 W power
limit with the SM clock at 2617–2632 MHz (maximum 3090 MHz). Evidence archive:
`xinfer-profiling-foundation-20261106.tar.gz` (sha256 `a2e7b27836e6b1410dade4835cd5904dcafaaab1ecb0f334589a4ee637f5779f`).

- BF16, FP32: `torch.mm` of random 8192 × 8192 matrices (FP32 with TF32 off), 2N³ / time.
- MXFP8: the registered `mxfp8_gemm` (xcompute `wallx_sm120`) on 8192³ operands with `fast_accum=False`, the
  operator's default. The MXFP8 linear layers' `fast_accum="auto"` turns fast accumulation on for calls with at most
  16 rows or at least 8192 outputs, which can run faster than this peak.
- NVFP4: `torch._scaled_mm` of NVFP4 operands with E4M3 scales per 16 elements, 8192³.
- DRAM: `copy_` of 1 GiB, bytes read and written / time.

Short bursts can run at higher clocks than this steady state, so a call can also beat its bound that way.

## Operator profile (`xinfer-profile ops`)

```bash
xinfer-profile ops --model wall_oss --checkpoint /path/to/checkpoint \
  --requests requests.msgpack --requests-per-run 2 --out profile
```

The command works for every registered model; `--model` only selects which one runs. It needs CUDA and a torch whose
profiler records CUDA activity: with driver 595.91.07, torch 2.9.1+cu130 records none and torch 2.10 does. A trace
without CUDA activity stops the command with that message.

1. **Build.** The model is built under the profiled op table, wrapped once more to record the `OpCost` of every call
   shape (`op_profile.CostRecorder`).
2. **Warm up.** Every request of `--requests` runs twice eagerly: lazy initialization, autotuning and the costs.
3. **Profile.** `--requests-per-run` requests (default: all, cycling through the file) run eagerly under
   torch.profiler with CPU and CUDA activity on every thread, with the kineto sink.
4. **Attribute.** Every GPU activity (kernel, memory copy or set) goes to the innermost `op::` span and the
   `stage::` span around its launch, through the launch's correlation id (`op_profile.attribute`). Activity in a
   stage but outside every operator span is the stage's "(outside operators)" row: input merges, handoff copies and
   the copy to the host.
5. **Bound.** Each call shape's bound uses `PEAKS` for the GPU, or `--peaks`, a JSON object `{pipe: FLOP/s, "dram":
   bytes/s}`. Without either the bound columns are omitted.
6. **Graph replay.** The same requests then run under CUDA graph replay, the serving path, for each stage's GPU span.

| File | Contents |
|---|---|
| `ops.md`, `ops.csv`, `ops.json` | per stage, one row per operator call shape: calls, µs/call, total µs, share of the stage's GPU time, GFLOP/call, MB/call, bound µs/call, efficiency (bound / µs), limiter and cost source (`formula` or `generic`); per request |
| `stages.md` | per stage and request: host span, eager GPU span = operators + outside − overlap + idle, and the GPU span under graph replay |
| `trace.json`, `trace-graphs.json` | the eager and the graph replay kineto traces (Perfetto); in the eager one kernels sit under their `op::` spans |
| `meta.json` | source commit and hash, GPU, clocks and power limit, torch, CUDA and driver, the operator selection, model options, the request file's sha256 and the peaks |

- Coverage, in the outputs' first lines, is the share of GPU activity inside operator and stage spans; activity
  before a stage (input uploads) is in neither.
- Eager dispatch waits on Python between kernels, so most of a stage's eager GPU span is idle. Operator times are
  kernel times and do not include those gaps; the graph replay span is what serving sees.
- Operators without an explicit formula have the generic data-movement cost (`generic`), so their bound is a memory
  bound.

## Load profile (`xinfer-profile load`)

```bash
xinfer-profile load --model wall_oss --checkpoint /path/to/checkpoint --requests observation.msgpack \
  --robots 8 --out artifacts/wall-oss-load
```

Where a request's time goes in the engine when several robots share it: what `xinfer-profile ops` does not see,
since it runs one request at a time.

1. **Engine.** The model and an `Engine` with `xinfer-serve`'s resolved defaults (lanes, batch sizes, CUDA
   graphs), warmed up with the requests of `--requests` like `--warmup`.
2. **Load.** `--robots` virtual robots (default 8) submit the prepared requests in turn, `--requests-per-robot` each
   (default 80), at most one in flight: a robot sends its next request `--period-ms` (default 180) after the
   previous one, or when the response arrives if that is later. The first `--warmup-requests` (default 20) of each
   robot are left out of the statistics. With `--start spread` (default) robot `r` sends its first request `r / N`
   of a period after the first robot, as independent robots would; with `--start together` all send it at once and
   stay in step while the responses come within the period, the worst case for the queues. The times depend on
   this arrival pattern: `benchmarks/robot_load.py` against `xinfer-serve` lies between the two.
3. **Record.** A span sink (`load_profile.SpanRecorder`) keeps the engine's `admit`, `stage::` and `wait` spans:
   host times and, on CUDA, timing events on the lane's stream at their ends, read after the load, so nothing
   synchronizes while it runs. A dispatch's first event runs on its free lane at about the dispatch's host time,
   which places the GPU times on the host clock.

Segments per request (ms):

| Segment | Time |
|---|---|
| `admit_ms` | submission to admission: waiting for a slot |
| `inputs_ms` | the upload at admission |
| `<stage>.delay_ms` | from the stage's inputs being ready (the upload's end for a first stage, else its upstream stages done on the GPU) to its start on the GPU |
| `<stage>.run_ms` | the stage running on the GPU (host time on the CPU) |
| `complete_ms` | the final stage done on the GPU until the result: the copy to the host and the completion thread |
| `total_ms` | submission to the result |

Along a chain of stages, as in every current model, these add up to `total_ms`, so the segments' excess in the tail
adds up to the total's. `requests.csv` also has each stage's times that overlap the upstream run, for finding where
a delay comes from:

| Column | Time |
|---|---|
| `<stage>.queue_ms` | on the host, from the upload's or the upstream dispatches' end to the dispatch: waiting for a free lane |
| `<stage>.wait_ms` | the lane waiting on the GPU for the inputs and upstream stages |

With chained dispatch a request enters a stage's queue as soon as its upstream stages are dispatched, and a lane
stays busy until its dispatch is done on the GPU. A stage's queue and wait times therefore run while its upstream
stages run, and cost time only where they show up in `delay_ms`.

| File | Contents |
|---|---|
| `requests.csv` | one row per request: robot, index, every segment, each stage's batch size, queue and wait times |
| `summary.md`, `summary.json` | P50/P95/P99 of the segments that add up to `total_ms`, the requests per stage and batch size, the throughput, and for the requests at or above P95 and P99 of `total_ms` each segment's mean minus its mean over the median band (P45–P55) |
| `meta.json` | the command, model options, the request file's sha256, the load settings, the resolved configuration, the operator selection and, on CUDA, the GPU, versions and source |

- In process there are no frontend processes: requests are prepared once and their outputs are not processed, so
  input and output processing, the network and the transfers between processes are not in the times. `--serve`
  adds them (below).
- With `DeviceImage` inputs the decoding on the device is in the first stage's delay.

### Through `xinfer-serve` (`--serve`)

With `--serve` the command starts `xinfer-serve` for the run (the same model, checkpoint and options, warmed up with
`--requests`, plus `--serve-args`) with `--profile-dir <out>/engine`, and the robots become WebSocket clients sending
the wire requests of `--requests` in turn, so every request goes through a frontend's input and output processing,
the shared memory and the pipes as robots' requests do. `--profile-dir` makes the service's engine process record the
same spans and write each request's engine side to `engine/engine-requests.json` when it stops.

Each robot keeps one connection with one request in flight and connects after the previous robot, so each frontend
numbers its connections in robot order; the engine knows a request by that connection, which joins each request's
client time, `server_timing` and engine spans. Segments per request (ms):

| Segment | Time |
|---|---|
| `network_ms` | the client's time minus `server_timing.infer_ms`: the network and the message encoding |
| `prepare_ms`, `output_ms` | the frontend's input and output processing (`server_timing`) |
| `handoff_ms` | `server_timing.engine_ms` minus the engine's span from admission to the final stage done: the shared-memory buffer, the pipes, waiting for a slot, the copy to the host and the completion |
| `inputs_ms`, `<stage>.delay_ms`, `<stage>.run_ms` | as in process |
| `total_ms` | sent to received, on the client |

These add up to `total_ms` as in process; `requests.csv` also has the stages' queue and wait times.

The command stops the service after the load; `serve.log` holds its log and resolved configuration.
