# Ascend 950DT

English | [简体中文](ascend950dt.zh-CN.md)

Wall-OSS and Wall-X 2B use the shared operator registry and engine on Ascend 950DT. The AOT kernels build from this repository into `xinfer_kernels_ascend.so`. Wall-OSS also compiles in-tree JIT kernels during warmup.

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

The build writes `xinfer_kernels_ascend.so` beside the Python backend.

## Run

```bash
source examples/ascend/wall_oss.env
xinfer-serve --model wall_oss --checkpoint /path/to/checkpoint --device npu:0 \
  --model-options "$ASCEND_MODEL_OPTIONS" --op-backends "$ASCEND_OP_BACKENDS" \
  --warmup flow.msgpack rtc.msgpack --port 33985
```

For Wall-X 2B, source [wallx2b.env](../examples/ascend/wallx2b.env) and use `--model wallx2b` with its checkpoint and requests. Both recipes use BF16 and CPU image processing. Wall-X 2B uses the shared RTC gripper limit of 5.0. Stage graphs capture during warmup. Use `--resolved-config FILE` to inspect selected operators and service settings. In-tree operators report `ascend:kernels`.

## Benchmark

Use the shared [benchmark protocol](benchmark.md#performance-table) with the same recipe for latency and load.

```bash
source examples/ascend/wall_oss.env
python benchmarks/performance.py --name "Wall-OSS · BF16 · Ascend 950DT" \
  --model wall_oss --checkpoint /path/to/checkpoint --device npu:0 \
  --model-options "$ASCEND_MODEL_OPTIONS" --op-backends "$ASCEND_OP_BACKENDS" \
  --latency-requests flow.msgpack rtc.msgpack --warmup flow.msgpack rtc.msgpack \
  --robots 3 4 5 --out artifacts/perf-wall-oss-ascend
```

For Wall-X 2B, measure [single-request latency](benchmark.md#single-request) with its recipe and captured request set. Bind the test processes to the device's local CPU NUMA node.
