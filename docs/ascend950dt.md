# Ascend 950DT

English | [简体中文](ascend950dt.zh-CN.md)

WALL-OSS-0.5 and WALL-OSS-1.0-Nano use the shared operator registry and engine on Ascend 950DT. The AOT kernels build from this repository into `xinfer_kernels_ascend.so`. WALL-OSS-0.5 also compiles in-tree JIT kernels during warmup.

## Runtime

Install matching PyTorch and TorchNPU and source CANN before importing TorchNPU. Select the device with
`--device npu:0` or `Policy(device="npu:0")`. With `ASCEND_RT_VISIBLE_DEVICES`, the index is relative to the
visible set. The platform's backend name is `ascend`; its torch device is `npu:N`.

WALL-OSS-0.5 uses the existing checkpoint and request formats. The service processes images on the CPU by default.
With PyACL DVPP V2 available, select device images using the top-level model option
`--model-options '{"image_preprocess_device":"gpu"}'`, or `Policy(image_preprocess_device="gpu")`.
`Policy` defaults to device images when supported. Do not put this option inside `processing`.
Unsupported JPEG profiles, including progressive JPEGs, log a warning and use host decoding with device
resize. Record the image path in precision and performance results.

JPEG decoders live in the platform modules. `platforms.devices.jpeg_decoder_for(device)` lazily selects a
decoder exposing `device`, `__call__(jpegs)` and `close()`. The platform owns vendor resources and stream
ordering. The engine preserves the input mapping, uploads raw frames and applies the common resize steps
to both raw and decoded images.

## Build

Build on aarch64 with the CANN 9.3 Toolkit and matching Ascend 950DT operator package, compatible PyTorch and TorchNPU, Ascend TileLang, CMake 3.24 or later, and TVM-FFI. Set `CANN_ROOT` to the installed Toolkit directory.

```bash
CANN_ROOT=/path/to/cann-9.3
source "$CANN_ROOT/set_env.sh"
export BISHENG_HOME="$CANN_ROOT"
python -m pip install -e ".[kernels,models,serve]"
cmake -S xinfer/ops/backends/ascend/src -B build/kernels-ascend \
  -DPython_EXECUTABLE="$(command -v python)" \
  -DXINFER_ASCEND_SOC=ascend950dt \
  -DXINFER_CANN_RUNTIME="$CANN_ROOT" \
  -DXINFER_BISHENG_HOME="$CANN_ROOT"
cmake --build build/kernels-ascend --parallel "${BUILD_JOBS:-4}"
```

The build writes `xinfer_kernels_ascend.so` beside the Python backend. The loader records the library's SHA-256 and supported SoC in operator-selection metadata.

| Variable | Meaning |
|---|---|
| `XINFER_ASCEND_KERNELS_LIBRARY` | optional path to `xinfer_kernels_ascend.so`; otherwise the package locates its library |
| `XINFER_ASCEND_KERNELS` | `0` disables the in-tree native implementations |

An absent implicit library leaves native operators unregistered, so their next available implementations run. A missing or broken explicit library is an error. Unsupported call layouts use the operator's torch fallback before native launch; an error from an admitted native launch propagates without a torch retry.

## Run

```bash
source examples/ascend/wall_oss_0_5.env
xinfer-serve --model wall-oss-0.5 --checkpoint /path/to/checkpoint --device npu:0 \
  --model-options "$ASCEND_MODEL_OPTIONS" --op-backends "$ASCEND_OP_BACKENDS" \
  --warmup flow.msgpack rtc.msgpack --port 33985
```

For WALL-OSS-1.0-Nano, source [wall_oss_1_0_nano.env](../examples/ascend/wall_oss_1_0_nano.env) and use `--model wall-oss-1.0-nano` with its checkpoint and requests. Both recipes use BF16 and CPU image processing. WALL-OSS-1.0-Nano uses the shared RTC gripper limit of 5.0. Stage graphs capture during warmup. Use `--resolved-config FILE` to inspect selected operators and service settings. In-tree operators report `ascend:kernels`; selection does not prove native coverage for every call shape.

The official TorchNPU implementations remain opt-in. For example, `--op-backends gemm_epilogue=ascend:npu_linear` selects the official linear implementation for that operator.

## Graphs

Graph capture is enabled by default when the installed TorchNPU exposes its graph APIs, as on CUDA.
A request's first unseen shape is captured lazily; representative `--warmup` requests can capture the reachable
shape buckets, lanes and batch combinations before serving. Each capture first runs the shape eagerly on the
capture stream, then captures fixed buffers. After warmup, `CUDAGraphConfig.on_miss_after_warmup` applies to
new shapes: `"capture"` (the default) captures once and replays subsequent requests, `"eager"` runs them without
capture, and `"error"` rejects them. `--enforce-eager` or `Policy(cudagraphs=False)` disables graphs.

When no native implementation supports the call, `ascend:torch` attention uses the same FP32 matmul, softmax
and matmul composition in eager execution and capture. This avoids the SDPA side stream that can prevent
capture from closing. CPU/CUDA and explicit reference tables retain the original SDPA reference.
Without native operator pins, `--reference-ops` selects the original references; pair it with `--enforce-eager` on NPU, or use
`Policy(reference_ops=True, cudagraphs=False)`, because the reference attention may not support capture.

## Stream priorities

The engine passes `priority` through the same `Stream(device=..., priority=...)` interface on CUDA and NPU.
It uses the active runtime's stream-priority capability and reported numeric range. When that range is
known, the final stage's preferred priority −1 is bounded by it: a runtime with range 0–7 keeps the final stage
at 0. An unknown range keeps all defaults at 0. Explicit unsupported or out-of-range values are errors; a runtime
advertising priority support that changes a requested priority or cannot report it also fails lane creation.

TorchNPU 2.10.0.post4 and 2.10.0.post7 leave priority range queries and readback unimplemented, and their
managed stream pools create native streams at priority 0. These runtimes therefore advertise no stream-priority
capability and reject nonzero requests. CANN's native priority support does not establish that TorchNPU honors
its constructor argument; see the [post4 stream binding](https://github.com/Ascend/pytorch/blob/5dd8ef3f9b375b5ae4a83538d5785754148c3302/torch_npu/csrc/npu/Stream.cpp#L90-L100),
[post7 stream binding](https://github.com/Ascend/pytorch/blob/3363ea59d49f3e08cfd8d4a1f532b756ccda0682/torch_npu/csrc/npu/Stream.cpp#L103-L113)
and [post7 pool creation](https://github.com/Ascend/pytorch/blob/3363ea59d49f3e08cfd8d4a1f532b756ccda0682/torch_npu/csrc/core/npu/NPUStream.cpp#L254-L268).
The post7 `ExternalStream` wrapper cannot replace a lane stream: its documented restrictions include event
recording/waiting, allocator stream tracking and graph capture; see the [upstream contract](https://github.com/Ascend/pytorch/blob/3363ea59d49f3e08cfd8d4a1f532b756ccda0682/torch_npu/npu/streams.py#L112-L177).

## Benchmark

Use the shared [benchmark protocol](benchmark.md#performance-table) with the same recipe for latency and load.

```bash
source examples/ascend/wall_oss_0_5.env
python benchmarks/performance.py --name "WALL-OSS-0.5 · BF16 · Ascend 950DT" \
  --model wall-oss-0.5 --checkpoint /path/to/checkpoint --device npu:0 \
  --model-options "$ASCEND_MODEL_OPTIONS" --op-backends "$ASCEND_OP_BACKENDS" \
  --latency-requests flow.msgpack rtc.msgpack --warmup flow.msgpack rtc.msgpack \
  --robots 3 4 5 --out artifacts/perf-wall-oss-0-5-ascend
```

For WALL-OSS-1.0-Nano, measure [single-request latency](benchmark.md#single-request) with its recipe and captured request set. Bind the test processes to the device's local CPU NUMA node.

## Validation

The Ascend host tests run without an NPU, CANN, or the native library:

```bash
XINFER_ASCEND_KERNELS=0 pytest -q tests/ascend
```

These tests check loader, layout, workspace, lifetime, and recipe contracts; they do not execute Ascend kernels or establish device numerics. On an exclusively reserved 950DT, build the native library and run both recipes. Compare fixed prepared inputs and explicit noise between eager and graph execution, then exercise real requests through the WebSocket service. Check lazy capture on a cold request without warmup and on a new shape after warmup, followed by replay of that shape.

Run the [precision gate](precision.md) and [benchmarks](benchmark.md), recording source and library identity,
composition, operator selection, image path, checkpoint identity and graph counters. Keep accelerator traces
and experiments outside the inference package. Measure request latency and robot capacity without a profiler.
Select the NPU with `benchmarks/performance.py --device npu:0`; put image settings in its existing
`--model-options`. For isolated backend or eager comparisons, use the existing options of `Policy` or
`xinfer-serve`.
